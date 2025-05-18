"""
LLM 통합 및 자가 개선 시스템 모듈

Claude 3.7 Sonnet 기반 LLM 통합 및 자가 개선 시스템을 제공합니다. 이 시스템은 코드 생성 및
최적화를 지원하며 RAG를 통한 컨텍스트 강화 기능을 제공합니다.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# 선택적 의존성 임포트
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None
    logging.warning("yaml 패키지가 설치되지 않았습니다. YAML 관련 기능이 제한됩니다.")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None
    logging.warning(
        "sentence-transformers 패키지가 설치되지 않았습니다. 임베딩 기능이 제한됩니다."
    )

try:
    import chromadb  # type: ignore
except ImportError:
    chromadb = None
    logging.warning("chromadb 패키지가 설치되지 않았습니다. 벡터 DB 기능이 제한됩니다.")

from analysis.mcp_init import mcp
from analysis.tools.atoms.llm_tools import (
    ClaudeClient,
    create_completion,
    explain_code,
    generate_code,
    load_prompt_template,
    refine_code,
    review_code,
    save_generated_code,
)
from common.helpers import load_config
from common.logging_config import configure_logging

# 로깅 설정
configure_logging()
logger = logging.getLogger("llm_integration")


def get_palantir_root() -> str:
    """Palantir 프로젝트 루트 디렉토리 경로를 반환합니다."""
    # 현재 파일의 디렉토리를 기준으로 프로젝트 루트를 찾습니다
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "config")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


class LocalKnowledgeRAG:
    """로컬 지식 베이스 RAG 시스템 클래스"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: RAG 시스템 구성 파일 경로
        """
        if not yaml:
            raise ImportError(
                "yaml 패키지가 필요합니다. pip install pyyaml로 설치해주세요."
            )
        if not SentenceTransformer:
            raise ImportError(
                "sentence-transformers 패키지가 필요합니다. pip install sentence-transformers로 설치해주세요."
            )
        if not chromadb:
            raise ImportError(
                "chromadb 패키지가 필요합니다. pip install chromadb로 설치해주세요."
            )

        self.config = load_config(config_path)
        self.embeddings_model = None
        self.vector_db = None
        self.collection = None

        self._initialize_vector_db()
        logger.info("로컬 지식 베이스 RAG 시스템이 초기화되었습니다.")

    def _initialize_vector_db(self) -> None:
        """설정 파일을 기반으로 ChromaDB 벡터 저장소를 초기화한다."""
        if chromadb is None:
            raise ImportError("chromadb package is required for vector DB")

        db_cfg = self.config.get("vector_db", {})
        persist_dir = os.path.join(
            get_palantir_root(), db_cfg.get("persist_directory", "vector_db")
        )

        self.vector_db = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.vector_db.get_or_create_collection(
            name=db_cfg.get("collection_name", "project_knowledge"),
            metadata={"hnsw:space": db_cfg.get("similarity_metric", "cosine")},
        )
        logger.info("벡터 데이터베이스 초기화 완료: %s", persist_dir)

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """주어진 텍스트 리스트에 대한 임베딩을 생성한다."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers package is required for embedding generation"
            )

        if self.embeddings_model is None:
            emb_cfg = self.config.get("embeddings", {})
            model_name = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            device = emb_cfg.get("device", "cpu")
            self.embeddings_model = SentenceTransformer(model_name, device=device)

        batch = self.config.get("embeddings", {}).get("batch_size", 32)
        embeddings = self.embeddings_model.encode(texts, batch_size=batch)
        return embeddings.tolist()

    async def retrieval_augmented_generation(
        self, query: str, top_k: int | None = None
    ) -> str:
        """RAG 방식으로 쿼리에 대한 응답을 생성한다."""
        if self.collection is None:
            raise RuntimeError("Vector database is not initialized")

        query_emb = self._generate_embeddings([query])[0]
        k = top_k or self.config.get("search", {}).get("top_k", 5)
        results = self.collection.query(query_embeddings=[query_emb], n_results=k)
        documents = results.get("documents", [[]])[0]
        context = "\n".join(documents)

        llm_cfg_path = os.path.join(get_palantir_root(), "config", "llm.yaml")
        client = ClaudeClient(llm_cfg_path)
        prompt = f"{query}\n\n자료:\n{context}"
        response = await create_completion(client, prompt)
        return response

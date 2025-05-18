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
    logging.warning("sentence-transformers 패키지가 설치되지 않았습니다. 임베딩 기능이 제한됩니다.")

try:
    import chromadb  # type: ignore
except ImportError:
    chromadb = None
    logging.warning("chromadb 패키지가 설치되지 않았습니다. 벡터 DB 기능이 제한됩니다.")

from analysis.tools.atoms.llm_tools import (
    ClaudeClient, create_completion, explain_code, generate_code, 
    load_prompt_template, refine_code, review_code, save_generated_code
)
from analysis.mcp_init import mcp
from common.helpers import load_config

# 로깅 설정
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
            raise ImportError("yaml 패키지가 필요합니다. pip install pyyaml로 설치해주세요.")
        if not SentenceTransformer:
            raise ImportError("sentence-transformers 패키지가 필요합니다. pip install sentence-transformers로 설치해주세요.")
        if not chromadb:
            raise ImportError("chromadb 패키지가 필요합니다. pip install chromadb로 설치해주세요.")
            
        self.config = load_config(config_path)
        self.embeddings_model = None
        self.vector_db = None
        self.collection = None
        
        self._initialize_vector_db()
        logger.info("로컬 지식 베이스 RAG 시스템이 초기화되었습니다.")

# ... existing code ... 
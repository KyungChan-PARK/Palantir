import os
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
import yaml
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/embedding_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self, cache_dir: str, max_size_mb: int = 1000):
        """
        임베딩 캐시 초기화
        
        Args:
            cache_dir (str): 캐시 디렉토리 경로
            max_size_mb (int): 최대 캐시 크기 (MB)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_mb * 1024 * 1024  # MB를 바이트로 변환
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """캐시 메타데이터 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"files": {}, "total_size": 0}
    
    def _save_metadata(self):
        """캐시 메타데이터 저장"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def _get_cache_key(self, text: str) -> str:
        """텍스트의 캐시 키 생성"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """캐시에서 임베딩 가져오기"""
        cache_key = self._get_cache_key(text)
        if cache_key in self.metadata["files"]:
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                return np.load(str(cache_file))
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """임베딩을 캐시에 저장"""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        # 캐시 크기 확인 및 관리
        embedding_size = embedding.nbytes
        while self.metadata["total_size"] + embedding_size > self.max_size:
            if not self.metadata["files"]:
                break
            # 가장 오래된 캐시 항목 제거
            oldest_key = next(iter(self.metadata["files"]))
            oldest_file = self.cache_dir / f"{oldest_key}.npy"
            if oldest_file.exists():
                self.metadata["total_size"] -= self.metadata["files"][oldest_key]["size"]
                oldest_file.unlink()
            del self.metadata["files"][oldest_key]
        
        # 새 임베딩 저장
        np.save(str(cache_file), embedding)
        self.metadata["files"][cache_key] = {
            "size": embedding_size,
            "timestamp": str(np.datetime64('now'))
        }
        self.metadata["total_size"] += embedding_size
        self._save_metadata()

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 cache_dir: Optional[str] = None):
        """
        임베딩 생성기 초기화
        
        Args:
            model_name (str): 사용할 Sentence Transformer 모델 이름
            cache_dir (Optional[str]): 캐시 디렉토리 경로
        """
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache(cache_dir) if cache_dir else None
        logger.info(f"임베딩 생성기 초기화 완료: {model_name}")
    
    def generate_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        텍스트 리스트에 대한 임베딩 생성
        
        Args:
            texts (List[str]): 임베딩을 생성할 텍스트 리스트
            use_cache (bool): 캐시 사용 여부
            
        Returns:
            np.ndarray: 생성된 임베딩 벡터 배열
        """
        if not use_cache or not self.cache:
            return self._generate_embeddings_direct(texts)
        
        embeddings = []
        texts_to_generate = []
        text_indices = []
        
        # 캐시에서 임베딩 가져오기
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                texts_to_generate.append(text)
                text_indices.append(i)
        
        # 캐시에 없는 텍스트에 대한 임베딩 생성
        if texts_to_generate:
            new_embeddings = self._generate_embeddings_direct(texts_to_generate)
            for text, embedding in zip(texts_to_generate, new_embeddings):
                self.cache.put(text, embedding)
                embeddings.append(embedding)
        
        # 원래 순서대로 정렬
        sorted_embeddings = [None] * len(texts)
        for i, embedding in zip(text_indices, embeddings):
            sorted_embeddings[i] = embedding
        
        return np.array(sorted_embeddings)
    
    def _generate_embeddings_direct(self, texts: List[str]) -> np.ndarray:
        """캐시 없이 직접 임베딩 생성"""
        try:
            embeddings = self.model.encode(texts)
            logger.info(f"{len(texts)}개의 텍스트에 대한 임베딩 생성 완료")
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        단일 텍스트에 대한 임베딩 생성
        
        Args:
            text (str): 임베딩을 생성할 텍스트
            use_cache (bool): 캐시 사용 여부
            
        Returns:
            np.ndarray: 생성된 임베딩 벡터
        """
        if use_cache and self.cache:
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                return cached_embedding
        
        embedding = self._generate_embeddings_direct([text])[0]
        
        if use_cache and self.cache:
            self.cache.put(text, embedding)
        
        return embedding
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str) -> None:
        """
        생성된 임베딩을 파일로 저장
        
        Args:
            embeddings (np.ndarray): 저장할 임베딩 벡터 배열
            output_path (str): 저장할 파일 경로
        """
        try:
            np.save(output_path, embeddings)
            logger.info(f"임베딩 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"임베딩 저장 중 오류 발생: {str(e)}")
            raise
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """
        저장된 임베딩 파일 로드
        
        Args:
            input_path (str): 로드할 임베딩 파일 경로
            
        Returns:
            np.ndarray: 로드된 임베딩 벡터 배열
        """
        try:
            embeddings = np.load(input_path)
            logger.info(f"임베딩 로드 완료: {input_path}")
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 로드 중 오류 발생: {str(e)}")
            raise

def main():
    # 설정 파일 로드
    with open("config/rag.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 임베딩 생성기 초기화
    generator = EmbeddingGenerator(
        model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_dir=config.get("cache", {}).get("directory")
    )
    
    # 테스트용 텍스트
    test_texts = [
        "이것은 테스트 문장입니다.",
        "임베딩 생성 모듈 테스트 중입니다.",
        "RAG 시스템의 핵심 구성 요소입니다."
    ]
    
    # 임베딩 생성 (캐시 사용)
    embeddings = generator.generate_embeddings(test_texts, use_cache=True)
    
    # 임베딩 저장
    output_dir = "data/embeddings"
    os.makedirs(output_dir, exist_ok=True)
    generator.save_embeddings(embeddings, os.path.join(output_dir, "test_embeddings.npy"))

if __name__ == "__main__":
    main() 
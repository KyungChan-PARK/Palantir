"""
인덱서 클래스 - 문서 인덱싱 및 검색을 위한 모듈

이 모듈은 문서의 효율적인 검색을 위한 인덱싱 기능을 제공합니다.
전체 텍스트 검색, 필터링, 패싯 검색 등의 기능을 구현합니다.
"""

import os
import json
import logging
import datetime
import shutil
import re
import hashlib
from pathlib import Path
import numpy as np
from collections import defaultdict

# 로깅 설정
logger = logging.getLogger(__name__)

class Indexer:
    """
    문서 인덱싱 및 검색을 위한 클래스
    간소화된 구현으로, 실제 검색 엔진 없이 메모리 내 인덱스 사용
    """
    
    def __init__(self, config=None):
        """
        Indexer 초기화
        
        Args:
            config (dict, optional): 인덱서 설정
        """
        self.logger = logger
        self.config = config or {}
        
        # 인덱스 디렉토리
        self.index_dir = self.config.get('index_dir', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'index'))
        os.makedirs(self.index_dir, exist_ok=True)
        
        # 인덱스 파일
        self.index_file = os.path.join(self.index_dir, 'document_index.json')
        
        # 인덱스 로드
        self.index = self._load_index()
        
        # 변경 여부 (commit 필요 여부)
        self.changed = False
    
    def _load_index(self):
        """
        인덱스 로드
        
        Returns:
            dict: 인덱스 데이터
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"인덱스 로드 중 오류 발생: {str(e)}")
        
        # 인덱스가 없거나 로드 실패 시 새 인덱스 생성
        return {
            'metadata': {
                'created': datetime.datetime.now().isoformat(),
                'last_modified': datetime.datetime.now().isoformat(),
                'document_count': 0
            },
            'documents': {},
            'terms': defaultdict(list),
            'metadata_fields': defaultdict(dict)
        }
    
    def _save_index(self):
        """
        인덱스 저장
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 백업 생성
            if os.path.exists(self.index_file):
                backup_file = f"{self.index_file}.bak"
                shutil.copy2(self.index_file, backup_file)
            
            # 메타데이터 업데이트
            self.index['metadata']['last_modified'] = datetime.datetime.now().isoformat()
            self.index['metadata']['document_count'] = len(self.index['documents'])
            
            # JSON으로 변환하기 위해 defaultdict를 dict로 변환
            index_copy = self.index.copy()
            index_copy['terms'] = dict(index_copy['terms'])
            index_copy['metadata_fields'] = dict(index_copy['metadata_fields'])
            
            # 인덱스 저장
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_copy, f, ensure_ascii=False, indent=2)
            
            self.changed = False
            return True
        except Exception as e:
            self.logger.error(f"인덱스 저장 중 오류 발생: {str(e)}")
            return False
    
    def commit(self):
        """
        변경사항 커밋 (인덱스 저장)
        
        Returns:
            bool: 성공 여부
        """
        if self.changed:
            return self._save_index()
        return True
    
    def _tokenize(self, text):
        """
        텍스트를 토큰으로 분리
        
        Args:
            text (str): 토큰화할 텍스트
            
        Returns:
            list: 토큰 목록
        """
        if text is None:
            return []
        
        # 소문자 변환
        text = text.lower()
        
        # 토큰화 (간소화된 구현)
        tokens = re.findall(r'\w+', text)
        
        # 중복 제거
        return list(set(tokens))
    
    def add_document(self, doc_id, content, metadata=None):
        """
        문서 추가
        
        Args:
            doc_id (str): 문서 ID
            content (str): 문서 내용
            metadata (dict, optional): 문서 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 메타데이터 기본값
            metadata = metadata or {}
            
            # 기존 문서가 있으면 제거
            if doc_id in self.index['documents']:
                self.remove_document(doc_id)
            
            # 텍스트 토큰화
            tokens = self._tokenize(content)
            
            # 문서 추가
            self.index['documents'][doc_id] = {
                'content': content[:1000],  # 내용 미리보기 (처음 1000자)
                'metadata': metadata,
                'indexed_at': datetime.datetime.now().isoformat(),
                'token_count': len(tokens)
            }
            
            # 역인덱스 업데이트
            for token in tokens:
                self.index['terms'][token].append(doc_id)
            
            # 메타데이터 필드 인덱스 업데이트
            for field, value in metadata.items():
                # 숫자 및 문자열 값만 인덱싱
                if isinstance(value, (int, float, str, bool)):
                    if field not in self.index['metadata_fields']:
                        self.index['metadata_fields'][field] = {}
                    
                    # 값을 문자열로 변환
                    str_value = str(value)
                    
                    if str_value not in self.index['metadata_fields'][field]:
                        self.index['metadata_fields'][field][str_value] = []
                    
                    self.index['metadata_fields'][field][str_value].append(doc_id)
            
            self.changed = True
            return True
        except Exception as e:
            self.logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            return False
    
    def remove_document(self, doc_id):
        """
        문서 제거
        
        Args:
            doc_id (str): 문서 ID
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 문서가 없으면 종료
            if doc_id not in self.index['documents']:
                return False
            
            # 역인덱스에서 제거
            for token, docs in self.index['terms'].items():
                if doc_id in docs:
                    docs.remove(doc_id)
            
            # 빈 토큰 제거
            self.index['terms'] = {k: v for k, v in self.index['terms'].items() if v}
            
            # 메타데이터 필드 인덱스에서 제거
            for field, values in self.index['metadata_fields'].items():
                for value, docs in values.items():
                    if doc_id in docs:
                        docs.remove(doc_id)
            
            # 문서 제거
            del self.index['documents'][doc_id]
            
            self.changed = True
            return True
        except Exception as e:
            self.logger.error(f"문서 제거 중 오류 발생: {str(e)}")
            return False
    
    def update_document(self, doc_id, content=None, metadata=None):
        """
        문서 업데이트
        
        Args:
            doc_id (str): 문서 ID
            content (str, optional): 새 문서 내용
            metadata (dict, optional): 새 문서 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 문서가 없으면 종료
            if doc_id not in self.index['documents']:
                return False
            
            # 기존 문서 가져오기
            doc = self.index['documents'][doc_id]
            
            # 내용 업데이트
            if content is not None:
                # 기존 문서 제거
                self.remove_document(doc_id)
                
                # 새 내용으로 추가
                return self.add_document(doc_id, content, metadata or doc['metadata'])
            
            # 메타데이터만 업데이트
            if metadata is not None:
                # 메타데이터 필드 인덱스에서 제거
                for field, values in self.index['metadata_fields'].items():
                    for value, docs in values.items():
                        if doc_id in docs:
                            docs.remove(doc_id)
                
                # 메타데이터 업데이트
                doc['metadata'] = metadata
                
                # 메타데이터 필드 인덱스 업데이트
                for field, value in metadata.items():
                    # 숫자 및 문자열 값만 인덱싱
                    if isinstance(value, (int, float, str, bool)):
                        if field not in self.index['metadata_fields']:
                            self.index['metadata_fields'][field] = {}
                        
                        # 값을 문자열로 변환
                        str_value = str(value)
                        
                        if str_value not in self.index['metadata_fields'][field]:
                            self.index['metadata_fields'][field][str_value] = []
                        
                        self.index['metadata_fields'][field][str_value].append(doc_id)
                
                doc['indexed_at'] = datetime.datetime.now().isoformat()
                self.changed = True
            
            return True
        except Exception as e:
            self.logger.error(f"문서 업데이트 중 오류 발생: {str(e)}")
            return False
    
    def search(self, query, metadata_filters=None, limit=10):
        """
        문서 검색
        
        Args:
            query (str): 검색 쿼리
            metadata_filters (dict, optional): 메타데이터 필터
            limit (int, optional): 최대 결과 수
            
        Returns:
            list: 검색 결과 목록
        """
        try:
            # 결과 저장을 위한 딕셔너리
            results = {}
            
            # 쿼리가 있는 경우 텍스트 검색 수행
            if query:
                # 쿼리 토큰화
                query_tokens = self._tokenize(query)
                
                # 각 토큰에 대해 문서 찾기
                for token in query_tokens:
                    if token in self.index['terms']:
                        for doc_id in self.index['terms'][token]:
                            if doc_id not in results:
                                results[doc_id] = 1
                            else:
                                results[doc_id] += 1
            else:
                # 쿼리가 없는 경우 모든 문서를 기본 점수 1로 추가
                for doc_id in self.index['documents']:
                    results[doc_id] = 1
            
            # 메타데이터 필터 적용
            if metadata_filters:
                filtered_results = {}
                
                for doc_id, score in results.items():
                    doc = self.index['documents'][doc_id]
                    
                    # 모든 필터 조건 검사
                    match = True
                    for field, value in metadata_filters.items():
                        if field not in doc['metadata'] or doc['metadata'][field] != value:
                            match = False
                            break
                    
                    # 필터 조건 충족 시 결과에 추가
                    if match:
                        filtered_results[doc_id] = score
                
                results = filtered_results
            
            # 점수별 정렬
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            
            # 결과 형식 변환
            formatted_results = []
            for doc_id, score in sorted_results[:limit]:
                doc = self.index['documents'][doc_id]
                formatted_results.append({
                    'id': doc_id,
                    'score': score,
                    'preview': doc['content'],
                    'metadata': doc['metadata'],
                    'indexed_at': doc['indexed_at']
                })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"검색 중 오류 발생: {str(e)}")
            return []
    
    def get_metadata_facets(self, field, prefix=None, limit=10):
        """
        메타데이터 필드의 패싯 정보 조회
        
        Args:
            field (str): 메타데이터 필드명
            prefix (str, optional): 값 접두어 필터
            limit (int, optional): 최대 결과 수
            
        Returns:
            list: 패싯 정보 목록
        """
        try:
            # 필드가 인덱스에 없는 경우
            if field not in self.index['metadata_fields']:
                return []
            
            # 패싯 정보 계산
            facets = []
            
            for value, docs in self.index['metadata_fields'][field].items():
                # 접두어 필터 적용
                if prefix and not value.startswith(prefix):
                    continue
                
                facets.append({
                    'value': value,
                    'count': len(docs)
                })
            
            # 문서 수별 정렬
            facets.sort(key=lambda x: x['count'], reverse=True)
            
            return facets[:limit]
        except Exception as e:
            self.logger.error(f"패싯 정보 조회 중 오류 발생: {str(e)}")
            return []
    
    def get_document(self, doc_id):
        """
        문서 정보 조회
        
        Args:
            doc_id (str): 문서 ID
            
        Returns:
            dict: 문서 정보
        """
        try:
            if doc_id in self.index['documents']:
                doc = self.index['documents'][doc_id]
                return {
                    'id': doc_id,
                    'preview': doc['content'],
                    'metadata': doc['metadata'],
                    'indexed_at': doc['indexed_at']
                }
            return None
        except Exception as e:
            self.logger.error(f"문서 정보 조회 중 오류 발생: {str(e)}")
            return None
    
    def get_document_count(self):
        """
        인덱스된 문서 수 조회
        
        Returns:
            int: 문서 수
        """
        return len(self.index['documents'])
    
    def get_term_count(self):
        """
        인덱스된 용어 수 조회
        
        Returns:
            int: 용어 수
        """
        return len(self.index['terms'])
    
    def get_metadata_fields(self):
        """
        인덱스된 메타데이터 필드 목록 조회
        
        Returns:
            list: 메타데이터 필드 목록
        """
        return list(self.index['metadata_fields'].keys())
    
    def clear_index(self):
        """
        인덱스 초기화
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.index = {
                'metadata': {
                    'created': datetime.datetime.now().isoformat(),
                    'last_modified': datetime.datetime.now().isoformat(),
                    'document_count': 0
                },
                'documents': {},
                'terms': defaultdict(list),
                'metadata_fields': defaultdict(dict)
            }
            
            self.changed = True
            return True
        except Exception as e:
            self.logger.error(f"인덱스 초기화 중 오류 발생: {str(e)}")
            return False

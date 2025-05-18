"""
데이터 커넥터 클래스 - 다양한 데이터 소스 연결을 위한 모듈

이 모듈은 CSV, Excel, 데이터베이스 등 다양한 데이터 소스에 연결하여 데이터를 로드하는 기능을 제공합니다.
표준화된 인터페이스를 통해 데이터 소스 변경에 따른 코드 수정을 최소화합니다.
"""

import os
import json
import logging
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import pyodbc

# 로깅 설정
logger = logging.getLogger(__name__)

class DataConnector:
    """
    다양한 데이터 소스에 연결하여 데이터를 로드하는 클래스
    """
    
    def __init__(self, config=None):
        """
        DataConnector 초기화
        
        Args:
            config (dict, optional): 데이터 소스 구성
        """
        self.logger = logger
        self.config = config or {}
        
        # 캐싱 디렉토리
        self.cache_dir = self.config.get('cache_dir', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache', 'data'))
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_csv(self, file_path, **kwargs):
        """
        CSV 파일 로드
        
        Args:
            file_path (str): CSV 파일 경로
            **kwargs: pandas.read_csv에 전달할 추가 인수
            
        Returns:
            pandas.DataFrame: 로드된 데이터
        """
        try:
            # 기본 옵션
            encoding = kwargs.pop('encoding', 'utf-8')
            
            # CSV 파일 읽기
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            self.logger.info(f"CSV 파일 로드 완료: {file_path}, {len(df)} 행")
            return df
        except Exception as e:
            self.logger.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
            raise
    
    def load_excel(self, file_path, sheet_name=0, **kwargs):
        """
        Excel 파일 로드
        
        Args:
            file_path (str): Excel 파일 경로
            sheet_name (str or int, optional): 시트 이름 또는 인덱스
            **kwargs: pandas.read_excel에 전달할 추가 인수
            
        Returns:
            pandas.DataFrame: 로드된 데이터
        """
        try:
            # Excel 파일 읽기
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.logger.info(f"Excel 파일 로드 완료: {file_path}, 시트: {sheet_name}, {len(df)} 행")
            return df
        except Exception as e:
            self.logger.error(f"Excel 파일 로드 중 오류 발생: {str(e)}")
            raise
    
    def load_json(self, file_path, normalize=False, **kwargs):
        """
        JSON 파일 로드
        
        Args:
            file_path (str): JSON 파일 경로
            normalize (bool): JSON 데이터를 정규화할지 여부
            **kwargs: pandas.json_normalize에 전달할 추가 인수 (normalize=True인 경우)
            
        Returns:
            pandas.DataFrame or dict: 로드된 데이터
        """
        try:
            # JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터프레임으로 변환 (옵션)
            if normalize:
                df = pd.json_normalize(data, **kwargs)
                self.logger.info(f"JSON 파일 로드 및 정규화 완료: {file_path}, {len(df)} 행")
                return df
            else:
                self.logger.info(f"JSON 파일 로드 완료: {file_path}")
                return data
        except Exception as e:
            self.logger.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
            raise
    
    def load_sqlite(self, db_path, query):
        """
        SQLite 데이터베이스에서 데이터 로드
        
        Args:
            db_path (str): SQLite 데이터베이스 파일 경로
            query (str): SQL 쿼리
            
        Returns:
            pandas.DataFrame: 쿼리 결과
        """
        try:
            # SQLite 연결
            conn = sqlite3.connect(db_path)
            
            # 쿼리 실행
            df = pd.read_sql_query(query, conn)
            
            # 연결 종료
            conn.close()
            
            self.logger.info(f"SQLite 쿼리 실행 완료: {db_path}, {len(df)} 행")
            return df
        except Exception as e:
            self.logger.error(f"SQLite 쿼리 실행 중 오류 발생: {str(e)}")
            raise
    
    def load_sqlalchemy(self, connection_string, query):
        """
        SQLAlchemy를 통해 데이터베이스에서 데이터 로드
        
        Args:
            connection_string (str): SQLAlchemy 연결 문자열
            query (str): SQL 쿼리
            
        Returns:
            pandas.DataFrame: 쿼리 결과
        """
        try:
            # SQLAlchemy 엔진 생성
            engine = create_engine(connection_string)
            
            # 쿼리 실행
            df = pd.read_sql_query(query, engine)
            
            self.logger.info(f"SQLAlchemy 쿼리 실행 완료: {len(df)} 행")
            return df
        except Exception as e:
            self.logger.error(f"SQLAlchemy 쿼리 실행 중 오류 발생: {str(e)}")
            raise
    
    def load_odbc(self, connection_string, query):
        """
        ODBC를 통해 데이터베이스에서 데이터 로드
        
        Args:
            connection_string (str): ODBC 연결 문자열
            query (str): SQL 쿼리
            
        Returns:
            pandas.DataFrame: 쿼리 결과
        """
        try:
            # ODBC 연결
            conn = pyodbc.connect(connection_string)
            
            # 쿼리 실행
            df = pd.read_sql_query(query, conn)
            
            # 연결 종료
            conn.close()
            
            self.logger.info(f"ODBC 쿼리 실행 완료: {len(df)} 행")
            return df
        except Exception as e:
            self.logger.error(f"ODBC 쿼리 실행 중 오류 발생: {str(e)}")
            raise
    
    def save_to_cache(self, data, cache_name, format='csv'):
        """
        데이터를 캐시에 저장
        
        Args:
            data (pandas.DataFrame): 저장할 데이터
            cache_name (str): 캐시 이름
            format (str): 저장 형식 ('csv', 'excel', 'json', 'pickle')
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 캐시 파일 경로
            if format == 'csv':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.csv")
                data.to_csv(file_path, index=False, encoding='utf-8')
            elif format == 'excel':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.xlsx")
                data.to_excel(file_path, index=False)
            elif format == 'json':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.json")
                data.to_json(file_path, orient='records', force_ascii=False, indent=2)
            elif format == 'pickle':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
                data.to_pickle(file_path)
            else:
                raise ValueError(f"지원되지 않는 형식: {format}")
            
            self.logger.info(f"데이터를 캐시에 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"캐시 저장 중 오류 발생: {str(e)}")
            raise
    
    def load_from_cache(self, cache_name, format='csv', **kwargs):
        """
        캐시에서 데이터 로드
        
        Args:
            cache_name (str): 캐시 이름
            format (str): 저장 형식 ('csv', 'excel', 'json', 'pickle')
            **kwargs: 로드 함수에 전달할 추가 인수
            
        Returns:
            pandas.DataFrame: 로드된 데이터
            
        Raises:
            FileNotFoundError: 캐시 파일이 없는 경우
        """
        try:
            # 캐시 파일 경로
            if format == 'csv':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.csv")
                data = pd.read_csv(file_path, **kwargs)
            elif format == 'excel':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.xlsx")
                data = pd.read_excel(file_path, **kwargs)
            elif format == 'json':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.json")
                data = pd.read_json(file_path, **kwargs)
            elif format == 'pickle':
                file_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
                data = pd.read_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"지원되지 않는 형식: {format}")
            
            self.logger.info(f"캐시에서 데이터 로드 완료: {file_path}, {len(data)} 행")
            return data
        except FileNotFoundError:
            self.logger.warning(f"캐시 파일을 찾을 수 없음: {cache_name}.{format}")
            raise
        except Exception as e:
            self.logger.error(f"캐시 로드 중 오류 발생: {str(e)}")
            raise
    
    def cache_exists(self, cache_name, format='csv'):
        """
        캐시 파일 존재 여부 확인
        
        Args:
            cache_name (str): 캐시 이름
            format (str): 저장 형식
            
        Returns:
            bool: 캐시 파일 존재 여부
        """
        file_path = os.path.join(self.cache_dir, f"{cache_name}.{format}")
        return os.path.isfile(file_path)

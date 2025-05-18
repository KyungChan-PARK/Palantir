import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DuckDB 경로 설정
DUCKDB_PATH = r"C:\Users\packr\OneDrive\palantir\data\duckdb\palantir.db"

# API 설정 (예시: JSONPlaceholder API)
API_BASE_URL = "https://jsonplaceholder.typicode.com"
ENDPOINTS = {
    "posts": "/posts",
    "users": "/users",
    "comments": "/comments"
}

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.retry_count = 3
        self.timeout = httpx.Timeout(10.0)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def fetch_data(self, endpoint: str) -> List[Dict[str, Any]]:
        """지정된 엔드포인트에서 데이터를 비동기적으로 가져옵니다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{self.base_url}{endpoint}"
            logger.info(f"Fetching data from {url}")
            
            response = await client.get(url)
            response.raise_for_status()
            
            return response.json()

class DuckDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
    def init_tables(self):
        """필요한 테이블을 초기화합니다."""
        with duckdb.connect(self.db_path) as conn:
            # posts 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    userId INTEGER,
                    id INTEGER PRIMARY KEY,
                    title VARCHAR,
                    body TEXT,
                    created_at TIMESTAMP
                )
            """)
            
            # users 테이블 - website는 VARCHAR로 유지, address/company만 JSON으로 저장
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR,
                    username VARCHAR,
                    email VARCHAR,
                    phone VARCHAR,
                    website VARCHAR,  -- 일반 문자열로 저장
                    address VARCHAR,  -- JSON 문자열로 저장
                    company VARCHAR,  -- JSON 문자열로 저장
                    created_at TIMESTAMP
                )
            """)
            
            # comments 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    postId INTEGER,
                    id INTEGER PRIMARY KEY,
                    name VARCHAR,
                    email VARCHAR,
                    body TEXT,
                    created_at TIMESTAMP
                )
            """)
    
    def _preprocess_data(self, table_name: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터를 전처리합니다."""
        processed_data = []
        for item in data:
            processed_item = item.copy()  # 원본 데이터 보존을 위해 복사
            
            # 타임스탬프 추가
            processed_item['created_at'] = datetime.now()
            
            # users 테이블의 JSON 필드 처리
            if table_name == 'users':
                try:
                    # website는 그대로 VARCHAR로 유지
                    # address와 company만 JSON 문자열로 변환
                    if 'address' in processed_item:
                        processed_item['address'] = json.dumps(processed_item['address'])
                    if 'company' in processed_item:
                        processed_item['company'] = json.dumps(processed_item['company'])
                except Exception as e:
                    logger.error(f"JSON 처리 중 오류 발생: {str(e)}")
                    logger.error(f"문제가 된 데이터: {processed_item}")
                    continue
            
            processed_data.append(processed_item)
        
        if not processed_data:
            logger.warning(f"{table_name} 테이블의 처리된 데이터가 없습니다.")
        
        return processed_data

    def save_data(self, table_name: str, data: List[Dict[str, Any]]):
        """데이터를 지정된 테이블에 저장합니다."""
        if not data:
            logger.warning(f"No data to save for table {table_name}")
            return
        
        try:
            # 데이터 전처리
            processed_data = self._preprocess_data(table_name, data)
            if not processed_data:
                return
            
            # 판다스 DataFrame으로 변환
            df = pd.DataFrame(processed_data)
            
            with duckdb.connect(self.db_path) as conn:
                # website 컬럼이 있는 경우 명시적으로 VARCHAR로 캐스팅
                if table_name == 'users' and 'website' in df.columns:
                    conn.execute("CREATE TEMPORARY TABLE temp_users AS SELECT * FROM df")
                    conn.execute("""
                        INSERT INTO users
                        SELECT 
                            id,
                            name,
                            username,
                            email,
                            phone,
                            website::VARCHAR,  -- 명시적 VARCHAR 캐스팅
                            address,
                            company,
                            created_at
                        FROM temp_users
                        WHERE NOT EXISTS (
                            SELECT 1 FROM users
                            WHERE users.id = temp_users.id
                        )
                    """)
                    conn.execute("DROP TABLE temp_users")
                else:
                    # 다른 테이블은 기존 로직 유지
                    conn.execute(f"CREATE TEMPORARY TABLE temp_{table_name} AS SELECT * FROM df")
                    conn.execute(f"""
                        INSERT INTO {table_name}
                        SELECT * FROM temp_{table_name}
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {table_name}
                            WHERE {table_name}.id = temp_{table_name}.id
                        )
                    """)
                    conn.execute(f"DROP TABLE temp_{table_name}")
                
                logger.info(f"Saved {len(processed_data)} records to {table_name}")
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생 ({table_name}): {str(e)}")
            logger.error(f"문제가 된 데이터: {data[:1]}")  # 첫 번째 레코드만 로깅

async def main():
    """메인 실행 함수"""
    try:
        # DB 매니저 초기화
        db_manager = DuckDBManager(DUCKDB_PATH)
        db_manager.init_tables()
        
        # API 클라이언트 초기화
        api_client = APIClient(API_BASE_URL)
        
        # 각 엔드포인트에서 데이터 수집
        for table_name, endpoint in ENDPOINTS.items():
            try:
                data = await api_client.fetch_data(endpoint)
                db_manager.save_data(table_name, data)
            except Exception as e:
                logger.error(f"Error processing {endpoint}: {str(e)}")
                continue
        
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 
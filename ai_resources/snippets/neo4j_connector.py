"""
제목: Neo4j 연결 및 세션 관리 스니펫
설명: Neo4j 데이터베이스 연결 및 세션 관리를 위한 재사용 가능한 코드
사용법: 직접 import 하거나 코드를 복사하여 사용
의존성: neo4j, pyyaml
작성자: 팔란티어 파운드리 팀
버전: 1.0
업데이트: 2025-05-17
"""

import yaml
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from contextlib import contextmanager

class Neo4jConnector:
    """Neo4j 데이터베이스 연결 및 세션 관리 클래스.
    
    이 클래스는 Neo4j 데이터베이스에 연결하고 세션을 관리하는 기능을 제공합니다.
    YAML 구성 파일에서 연결 정보를 로드하고, 연결 풀 관리, 트랜잭션 지원,
    오류 처리 및 로깅 기능을 포함합니다.
    
    Attributes:
        uri (str): Neo4j 데이터베이스 URI
        user (str): 데이터베이스 사용자 이름
        password (str): 데이터베이스 비밀번호
        driver (neo4j.Driver): Neo4j 데이터베이스 드라이버
        logger (logging.Logger): 로거 인스턴스
    """
    
    def __init__(self, config_path):
        """Neo4jConnector 초기화.
        
        Args:
            config_path (str): Neo4j 구성 파일 경로 (YAML)
        
        Raises:
            FileNotFoundError: 구성 파일을 찾을 수 없는 경우
            ValueError: 구성 파일에 필수 설정이 없는 경우
            AuthError: 인증 정보가 잘못된 경우
            ServiceUnavailable: Neo4j 서비스에 연결할 수 없는 경우
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._validate_config()
        
        self.uri = self.config['neo4j']['uri']
        self.user = self.config['neo4j']['user']
        self.password = self.config['neo4j']['password']
        self.driver = None
        
        self._connect()
    
    def _load_config(self, config_path):
        """YAML 구성 파일에서 Neo4j 설정을 로드합니다.
        
        Args:
            config_path (str): YAML 구성 파일 경로
        
        Returns:
            dict: 로드된 구성 설정
        
        Raises:
            FileNotFoundError: 구성 파일을 찾을 수 없는 경우
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """구성에 필수 설정이 있는지 확인합니다.
        
        Raises:
            ValueError: 필수 설정이 누락된 경우
        """
        required_settings = ['neo4j']
        required_neo4j_settings = ['uri', 'user', 'password']
        
        if not all(setting in self.config for setting in required_settings):
            missing = [s for s in required_settings if s not in self.config]
            self.logger.error(f"Missing required configuration sections: {missing}")
            raise ValueError(f"Missing required configuration sections: {missing}")
        
        if not all(setting in self.config['neo4j'] for setting in required_neo4j_settings):
            missing = [s for s in required_neo4j_settings if s not in self.config['neo4j']]
            self.logger.error(f"Missing required Neo4j settings: {missing}")
            raise ValueError(f"Missing required Neo4j settings: {missing}")
    
    def _connect(self):
        """Neo4j 데이터베이스에 연결합니다.
        
        Raises:
            AuthError: 인증 정보가 잘못된 경우
            ServiceUnavailable: Neo4j 서비스에 연결할 수 없는 경우
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except AuthError as e:
            self.logger.error(f"Authentication error: {e}")
            raise
        except ServiceUnavailable as e:
            self.logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error connecting to Neo4j: {e}")
            raise
    
    def close(self):
        """드라이버 연결을 닫습니다."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
            self.driver = None
    
    @contextmanager
    def session(self):
        """Neo4j 세션을 컨텍스트 관리자로 생성합니다.
        
        Yields:
            neo4j.Session: Neo4j 세션 객체
        
        Example:
            ```python
            with connector.session() as session:
                result = session.run("MATCH (n) RETURN count(n)")
                count = result.single()[0]
            ```
        """
        if not self.driver:
            self.logger.error("Neo4j connection is not established")
            raise RuntimeError("Neo4j connection is not established")
        
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    @contextmanager
    def transaction(self, session=None):
        """Neo4j 트랜잭션을 컨텍스트 관리자로 생성합니다.
        
        Args:
            session (neo4j.Session, optional): 기존 세션. 기본값은 None.
                None인 경우 새로운 세션을 생성합니다.
        
        Yields:
            neo4j.Transaction: Neo4j 트랜잭션 객체
        
        Example:
            ```python
            # 새로운 세션에서 트랜잭션 생성
            with connector.transaction() as tx:
                tx.run("CREATE (n:Person {name: $name})", name="Alice")
                
            # 기존 세션에서 트랜잭션 생성
            with connector.session() as session:
                with connector.transaction(session) as tx:
                    tx.run("CREATE (n:Person {name: $name})", name="Bob")
            ```
        """
        close_session = False
        if session is None:
            session = self.driver.session()
            close_session = True
        
        tx = session.begin_transaction()
        try:
            yield tx
            tx.commit()
        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            tx.rollback()
            raise
        finally:
            if close_session:
                session.close()
    
    def execute_query(self, query, parameters=None):
        """읽기 쿼리를 실행하고 결과를 반환합니다.
        
        Args:
            query (str): 실행할 Cypher 쿼리
            parameters (dict, optional): 쿼리 매개변수
        
        Returns:
            list: 쿼리 결과 레코드 리스트
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(self, query, parameters=None):
        """쓰기 쿼리를 실행하고 결과를 반환합니다.
        
        Args:
            query (str): 실행할 Cypher 쿼리
            parameters (dict, optional): 쿼리 매개변수
        
        Returns:
            list: 쿼리 결과 레코드 리스트
        """
        with self.session() as session:
            with self.transaction(session) as tx:
                result = tx.run(query, parameters or {})
                return [record.data() for record in result]
    
    def __enter__(self):
        """컨텍스트 관리자 진입."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료."""
        self.close()

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Neo4j 연결
        connector = Neo4jConnector("config/neo4j.yaml")
        
        # 쿼리 실행
        with connector.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"Database contains {count} nodes")
        
        # 트랜잭션 사용 예시
        with connector.transaction() as tx:
            tx.run("""
                CREATE (p:Person {name: $name, age: $age})
                RETURN p
            """, name="Example User", age=30)
        
        print("Transaction completed successfully")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # 연결 닫기
        if 'connector' in locals():
            connector.close()

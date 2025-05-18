"""
Neo4j 데이터베이스 연결 및 기본 CRUD 작업을 위한 모듈
"""

import logging
import yaml
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

class Neo4jConnector:
    """Neo4j 데이터베이스 연결 및 기본 CRUD 작업을 처리하는 클래스"""
    
    def __init__(self, config_path=None, uri=None, user=None, password=None):
        """
        Neo4j 연결 초기화
        
        Args:
            config_path (str, optional): 구성 파일 경로
            uri (str, optional): Neo4j 서버 URI
            user (str, optional): 사용자 이름
            password (str, optional): 비밀번호
        
        직접 매개변수가 우선순위가 높으며, 제공되지 않으면 구성 파일에서 값을 로드합니다.
        """
        self.driver = None
        
        # 직접 제공된, 아니면 구성 파일에서 로드된 연결 정보
        if uri and user and password:
            self.uri = uri
            self.user = user
            self.password = password
        else:
            if config_path:
                self._load_config(config_path)
            else:
                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "neo4j.yaml"
                )
                self._load_config(default_config_path)
        
        self._connect()
        
        logger.info("Neo4j 연결기 초기화 완료")
    
    def _load_config(self, config_path):
        """
        구성 파일에서 Neo4j 연결 정보 로드
        
        Args:
            config_path (str): 구성 파일 경로
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Neo4j 연결 정보 추출
            self.uri = config['neo4j']['uri']
            self.user = config['neo4j']['user']
            self.password = config['neo4j']['password']
            
            logger.info(f"구성 파일 {config_path}에서 Neo4j 연결 정보 로드됨")
        except Exception as e:
            logger.error(f"구성 파일 {config_path} 로드 중 오류: {str(e)}")
            raise
    
    def _connect(self):
        """Neo4j 데이터베이스에 연결"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 연결 테스트
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise Exception("Neo4j 연결 테스트에 실패했습니다")
            
            logger.info(f"Neo4j 서버 {self.uri}에 성공적으로 연결됨")
        except ServiceUnavailable:
            logger.error(f"Neo4j 서버 {self.uri}에 연결할 수 없음")
            raise
        except AuthError:
            logger.error(f"Neo4j 인증 실패: 사용자={self.user}")
            raise
        except Exception as e:
            logger.error(f"Neo4j 연결 중 오류: {str(e)}")
            raise
    
    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 연결 종료됨")
    
    def execute_query(self, query, parameters=None):
        """
        Neo4j 쿼리 실행
        
        Args:
            query (str): 실행할 Cypher 쿼리
            parameters (dict, optional): 쿼리 매개변수
        
        Returns:
            list: 쿼리 결과 레코드 목록
        """
        if parameters is None:
            parameters = {}
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"쿼리 실행 중 오류: {str(e)}, 쿼리: {query}, 매개변수: {parameters}")
            raise
    
    def execute_write_transaction(self, transaction_function, *args, **kwargs):
        """
        쓰기 트랜잭션 실행
        
        Args:
            transaction_function (function): 트랜잭션 내에서 실행할 함수
            *args, **kwargs: 트랜잭션 함수에 전달할 인수
        
        Returns:
            object: 트랜잭션 함수의 반환값
        """
        try:
            with self.driver.session() as session:
                return session.write_transaction(transaction_function, *args, **kwargs)
        except Exception as e:
            logger.error(f"쓰기 트랜잭션 실행 중 오류: {str(e)}")
            raise
    
    def execute_read_transaction(self, transaction_function, *args, **kwargs):
        """
        읽기 트랜잭션 실행
        
        Args:
            transaction_function (function): 트랜잭션 내에서 실행할 함수
            *args, **kwargs: 트랜잭션 함수에 전달할 인수
        
        Returns:
            object: 트랜잭션 함수의 반환값
        """
        try:
            with self.driver.session() as session:
                return session.read_transaction(transaction_function, *args, **kwargs)
        except Exception as e:
            logger.error(f"읽기 트랜잭션 실행 중 오류: {str(e)}")
            raise
    
    def create_constraint(self, label, property_name, constraint_name=None):
        """
        노드 레이블과 속성에 대한 고유성 제약 조건 생성
        
        Args:
            label (str): 노드 레이블
            property_name (str): 속성 이름
            constraint_name (str, optional): 제약 조건 이름
        
        Returns:
            bool: 성공 여부
        """
        try:
            # Neo4j 버전에 따라 구문이 다를 수 있음
            constraint_name = constraint_name or f"{label}_{property_name}_unique"
            
            # Neo4j 4.x 이상 구문
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label})
            REQUIRE n.{property_name} IS UNIQUE
            """
            
            self.execute_query(query)
            logger.info(f"제약 조건 생성됨: {constraint_name}")
            return True
        except Exception as e:
            logger.error(f"제약 조건 생성 중 오류: {str(e)}")
            
            # 대체 구문 시도 (Neo4j 3.x)
            try:
                query = f"""
                CREATE CONSTRAINT ON (n:{label})
                ASSERT n.{property_name} IS UNIQUE
                """
                self.execute_query(query)
                logger.info(f"대체 구문으로 제약 조건 생성됨: ON (n:{label}) ASSERT n.{property_name} IS UNIQUE")
                return True
            except Exception as nested_e:
                logger.error(f"대체 구문으로 제약 조건 생성 중 오류: {str(nested_e)}")
                return False
    
    def create_index(self, label, property_name, index_name=None):
        """
        노드 레이블과 속성에 인덱스 생성
        
        Args:
            label (str): 노드 레이블
            property_name (str): 속성 이름
            index_name (str, optional): 인덱스 이름
        
        Returns:
            bool: 성공 여부
        """
        try:
            # Neo4j 버전에 따라 구문이 다를 수 있음
            index_name = index_name or f"{label}_{property_name}_index"
            
            # Neo4j 4.x 이상 구문
            query = f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON (n.{property_name})
            """
            
            self.execute_query(query)
            logger.info(f"인덱스 생성됨: {index_name}")
            return True
        except Exception as e:
            logger.error(f"인덱스 생성 중 오류: {str(e)}")
            
            # 대체 구문 시도 (Neo4j 3.x)
            try:
                query = f"""
                CREATE INDEX ON :{label}({property_name})
                """
                self.execute_query(query)
                logger.info(f"대체 구문으로 인덱스 생성됨: ON :{label}({property_name})")
                return True
            except Exception as nested_e:
                logger.error(f"대체 구문으로 인덱스 생성 중 오류: {str(nested_e)}")
                return False
    
    def get_node_by_id(self, node_id):
        """
        ID로 노드 조회
        
        Args:
            node_id (int): Neo4j 내부 노드 ID
        
        Returns:
            dict: 노드 데이터, 노드가 없으면 None
        """
        query = "MATCH (n) WHERE id(n) = $node_id RETURN n"
        parameters = {"node_id": node_id}
        
        results = self.execute_query(query, parameters)
        if results and len(results) > 0:
            return results[0]["n"]
        return None
    
    def get_nodes_by_label(self, label, limit=100):
        """
        레이블로 노드 조회
        
        Args:
            label (str): 노드 레이블
            limit (int, optional): 결과 제한 개수
        
        Returns:
            list: 노드 데이터 목록
        """
        query = f"MATCH (n:{label}) RETURN n LIMIT $limit"
        parameters = {"limit": limit}
        
        return self.execute_query(query, parameters)
    
    def get_node_by_property(self, label, property_name, property_value):
        """
        속성 값으로 노드 조회
        
        Args:
            label (str): 노드 레이블
            property_name (str): 속성 이름
            property_value (object): 속성 값
        
        Returns:
            dict: 노드 데이터, 노드가 없으면 None
        """
        query = f"MATCH (n:{label}) WHERE n.{property_name} = $value RETURN n"
        parameters = {"value": property_value}
        
        results = self.execute_query(query, parameters)
        if results and len(results) > 0:
            return results[0]["n"]
        return None
    
    def create_node(self, label, properties):
        """
        노드 생성
        
        Args:
            label (str): 노드 레이블
            properties (dict): 노드 속성
        
        Returns:
            dict: 생성된 노드 데이터
        """
        property_string = ", ".join([f"{key}: ${key}" for key in properties.keys()])
        query = f"CREATE (n:{label} {{{property_string}}}) RETURN n"
        
        results = self.execute_query(query, properties)
        if results and len(results) > 0:
            return results[0]["n"]
        return None
    
    def update_node(self, label, identification_property, identification_value, update_properties):
        """
        노드 업데이트
        
        Args:
            label (str): 노드 레이블
            identification_property (str): 식별 속성 이름
            identification_value (object): 식별 속성 값
            update_properties (dict): 업데이트할 속성
        
        Returns:
            dict: 업데이트된 노드 데이터
        """
        # 업데이트할 속성이 없으면 조기 반환
        if not update_properties:
            return self.get_node_by_property(label, identification_property, identification_value)
        
        # 업데이트 쿼리 생성
        set_clauses = []
        parameters = {"id_value": identification_value}
        
        for key, value in update_properties.items():
            set_clauses.append(f"n.{key} = ${key}")
            parameters[key] = value
        
        set_string = ", ".join(set_clauses)
        query = f"""
        MATCH (n:{label})
        WHERE n.{identification_property} = $id_value
        SET {set_string}
        RETURN n
        """
        
        results = self.execute_query(query, parameters)
        if results and len(results) > 0:
            return results[0]["n"]
        return None
    
    def delete_node(self, label, property_name, property_value):
        """
        노드 삭제
        
        Args:
            label (str): 노드 레이블
            property_name (str): 속성 이름
            property_value (object): 속성 값
        
        Returns:
            bool: 성공 여부
        """
        query = f"""
        MATCH (n:{label})
        WHERE n.{property_name} = $value
        DETACH DELETE n
        """
        parameters = {"value": property_value}
        
        try:
            self.execute_query(query, parameters)
            return True
        except Exception as e:
            logger.error(f"노드 삭제 중 오류: {str(e)}")
            return False
    
    def create_relationship(self, from_label, from_property, from_value, 
                          to_label, to_property, to_value, 
                          relationship_type, properties=None):
        """
        관계 생성
        
        Args:
            from_label (str): 시작 노드 레이블
            from_property (str): 시작 노드 식별 속성 이름
            from_value (object): 시작 노드 식별 속성 값
            to_label (str): 종료 노드 레이블
            to_property (str): 종료 노드 식별 속성 이름
            to_value (object): 종료 노드 식별 속성 값
            relationship_type (str): 관계 타입
            properties (dict, optional): 관계 속성
        
        Returns:
            dict: 생성된 관계 데이터
        """
        if properties is None:
            properties = {}
        
        property_string = ""
        if properties:
            property_string = " {" + ", ".join([f"{key}: ${key}" for key in properties.keys()]) + "}"
        
        query = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE a.{from_property} = $from_value AND b.{to_property} = $to_value
        CREATE (a)-[r:{relationship_type}{property_string}]->(b)
        RETURN r
        """
        
        parameters = {
            "from_value": from_value,
            "to_value": to_value,
            **properties
        }
        
        results = self.execute_query(query, parameters)
        if results and len(results) > 0:
            return results[0]["r"]
        return None
    
    def get_relationship(self, from_label, from_property, from_value, 
                        to_label, to_property, to_value, 
                        relationship_type=None):
        """
        관계 조회
        
        Args:
            from_label (str): 시작 노드 레이블
            from_property (str): 시작 노드 식별 속성 이름
            from_value (object): 시작 노드 식별 속성 값
            to_label (str): 종료 노드 레이블
            to_property (str): 종료 노드 식별 속성 이름
            to_value (object): 종료 노드 식별 속성 값
            relationship_type (str, optional): 관계 타입 
                               (None이면 모든 관계 타입)
        
        Returns:
            list: 관계 데이터 목록
        """
        rel_type = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
        MATCH (a:{from_label})-[r{rel_type}]->(b:{to_label})
        WHERE a.{from_property} = $from_value AND b.{to_property} = $to_value
        RETURN r
        """
        
        parameters = {
            "from_value": from_value,
            "to_value": to_value
        }
        
        return self.execute_query(query, parameters)
    
    def delete_relationship(self, from_label, from_property, from_value, 
                          to_label, to_property, to_value, 
                          relationship_type=None):
        """
        관계 삭제
        
        Args:
            from_label (str): 시작 노드 레이블
            from_property (str): 시작 노드 식별 속성 이름
            from_value (object): 시작 노드 식별 속성 값
            to_label (str): 종료 노드 레이블
            to_property (str): 종료 노드 식별 속성 이름
            to_value (object): 종료 노드 식별 속성 값
            relationship_type (str, optional): 관계 타입 
                               (None이면 모든 관계 타입)
        
        Returns:
            bool: 성공 여부
        """
        rel_type = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
        MATCH (a:{from_label})-[r{rel_type}]->(b:{to_label})
        WHERE a.{from_property} = $from_value AND b.{to_property} = $to_value
        DELETE r
        """
        
        parameters = {
            "from_value": from_value,
            "to_value": to_value
        }
        
        try:
            self.execute_query(query, parameters)
            return True
        except Exception as e:
            logger.error(f"관계 삭제 중 오류: {str(e)}")
            return False
    
    def __enter__(self):
        """컨텍스트 관리자 진입 메서드"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료 메서드"""
        self.close()

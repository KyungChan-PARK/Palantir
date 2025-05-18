"""
Neo4j 온톨로지 도구 모듈

Neo4j 데이터베이스와 통신하여 온톨로지를 관리하는 기본 도구 함수들을 제공합니다.
"""

import json
import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Union

import neo4j
from neo4j import GraphDatabase

# 로깅 설정
logger = logging.getLogger("neo4j_tools")

class Neo4jConnection:
    """Neo4j 데이터베이스 연결 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Neo4j 구성 파일 경로
        """
        self.config = self._load_config(config_path)
        self.driver = None
        self.connect()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Neo4j 구성 파일 로드
        
        Args:
            config_path: 구성 파일 경로
            
        Returns:
            구성 정보가 담긴 딕셔너리
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"구성 파일 로드 오류: {e}")
            raise
    
    def connect(self) -> None:
        """Neo4j 데이터베이스에 연결"""
        try:
            self.driver = GraphDatabase.driver(
                self.config['neo4j']['uri'],
                auth=(self.config['neo4j']['user'], self.config['neo4j']['password'])
            )
            logger.info("Neo4j 데이터베이스에 연결되었습니다.")
        except Exception as e:
            logger.error(f"Neo4j 연결 오류: {e}")
            raise
    
    def close(self) -> None:
        """연결 종료"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 연결이 종료되었습니다.")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Cypher 쿼리 실행
        
        Args:
            query: 실행할 Cypher 쿼리
            parameters: 쿼리 매개변수
            
        Returns:
            쿼리 결과 리스트
        """
        assert self.driver is not None, "Neo4j 드라이버가 초기화되지 않았습니다."
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"쿼리 실행 오류: {e}")
            raise

async def create_node(connection: Neo4jConnection, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    """Neo4j 그래프 데이터베이스에 노드 생성
    
    Args:
        connection: Neo4j 연결 객체
        label: 노드 레이블
        properties: 노드 속성
        
    Returns:
        생성된 노드 정보
    """
    query = f"""
    CREATE (n:{label} $properties)
    RETURN n
    """
    
    try:
        result = connection.execute_query(query, {"properties": properties})
        logger.info(f"{label} 노드가 생성되었습니다. ID: {result[0]['n'].id}")
        return result[0]
    except Exception as e:
        logger.error(f"노드 생성 오류: {e}")
        raise

async def create_relationship(connection: Neo4jConnection, 
                        source_id: int, target_id: int, 
                        relationship_type: str, 
                        properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """노드 간의 관계 생성
    
    Args:
        connection: Neo4j 연결 객체
        source_id: 소스 노드 ID
        target_id: 타겟 노드 ID
        relationship_type: 관계 유형
        properties: 관계 속성
        
    Returns:
        생성된 관계 정보
    """
    properties = properties or {}
    
    query = f"""
    MATCH (source), (target)
    WHERE ID(source) = $source_id AND ID(target) = $target_id
    CREATE (source)-[r:{relationship_type} $properties]->(target)
    RETURN source, r, target
    """
    
    try:
        result = connection.execute_query(
            query, 
            {
                "source_id": source_id,
                "target_id": target_id,
                "properties": properties
            }
        )
        logger.info(f"관계가 생성되었습니다: ({source_id})-[{relationship_type}]->({target_id})")
        return result[0]
    except Exception as e:
        logger.error(f"관계 생성 오류: {e}")
        raise

async def get_node_by_id(connection: Neo4jConnection, node_id: int) -> Dict[str, Any]:
    """ID로 노드 조회
    
    Args:
        connection: Neo4j 연결 객체
        node_id: 노드 ID
        
    Returns:
        노드 정보
    """
    query = """
    MATCH (n)
    WHERE ID(n) = $node_id
    RETURN n
    """
    
    try:
        result = connection.execute_query(query, {"node_id": node_id})
        if not result:
            logger.warning(f"ID {node_id}에 해당하는 노드를 찾을 수 없습니다.")
            return None
        return result[0]
    except Exception as e:
        logger.error(f"노드 조회 오류: {e}")
        raise

async def find_nodes_by_label_and_property(connection: Neo4jConnection, 
                                    label: str, 
                                    property_name: str, 
                                    property_value: Any) -> List[Dict[str, Any]]:
    """레이블과 속성으로 노드 검색
    
    Args:
        connection: Neo4j 연결 객체
        label: 노드 레이블
        property_name: 속성 이름
        property_value: 속성 값
        
    Returns:
        검색된 노드 목록
    """
    query = f"""
    MATCH (n:{label})
    WHERE n.{property_name} = $property_value
    RETURN n
    """
    
    try:
        result = connection.execute_query(query, {"property_value": property_value})
        return result
    except Exception as e:
        logger.error(f"노드 검색 오류: {e}")
        raise

async def update_node_properties(connection: Neo4jConnection, 
                          node_id: int, 
                          properties: Dict[str, Any]) -> Dict[str, Any]:
    """노드 속성 업데이트
    
    Args:
        connection: Neo4j 연결 객체
        node_id: 노드 ID
        properties: 업데이트할 속성
        
    Returns:
        업데이트된 노드 정보
    """
    query = """
    MATCH (n)
    WHERE ID(n) = $node_id
    SET n += $properties
    RETURN n
    """
    
    try:
        result = connection.execute_query(
            query, 
            {
                "node_id": node_id,
                "properties": properties
            }
        )
        logger.info(f"노드 속성이 업데이트되었습니다. ID: {node_id}")
        return result[0]
    except Exception as e:
        logger.error(f"노드 속성 업데이트 오류: {e}")
        raise

async def get_connected_nodes(connection: Neo4jConnection, 
                       node_id: int, 
                       relationship_type: str = None, 
                       direction: str = "OUTGOING") -> List[Dict[str, Any]]:
    """노드와 연결된 다른 노드 조회
    
    Args:
        connection: Neo4j 연결 객체
        node_id: 노드 ID
        relationship_type: 관계 유형 (None이면 모든 관계)
        direction: 관계 방향 ("OUTGOING", "INCOMING", "BOTH" 중 하나)
        
    Returns:
        연결된 노드 목록
    """
    if direction == "OUTGOING":
        direction_pattern = "-[r]->"
    elif direction == "INCOMING":
        direction_pattern = "<-[r]-"
    else:  # "BOTH"
        direction_pattern = "-[r]-"
    
    relationship_clause = f":{relationship_type}" if relationship_type else ""
    
    query = f"""
    MATCH (n)-[r{relationship_clause}]-(m)
    WHERE ID(n) = $node_id
    RETURN m, r, n
    """
    
    try:
        result = connection.execute_query(query, {"node_id": node_id})
        return result
    except Exception as e:
        logger.error(f"연결 노드 조회 오류: {e}")
        raise

async def import_ontology_from_json(connection: Neo4jConnection, json_path: str) -> None:
    """JSON 파일에서 온톨로지 가져오기
    
    Args:
        connection: Neo4j 연결 객체
        json_path: 온톨로지 JSON 파일 경로
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            ontology_data = json.load(file)
        
        # 노드 생성
        node_id_map = {}
        for node in ontology_data.get("nodes", []):
            result = await create_node(
                connection,
                node["label"],
                node["properties"]
            )
            node_id_map[node["temp_id"]] = result["n"].id
        
        # 관계 생성
        for relationship in ontology_data.get("relationships", []):
            await create_relationship(
                connection,
                node_id_map[relationship["source_id"]],
                node_id_map[relationship["target_id"]],
                relationship["type"],
                relationship.get("properties", {})
            )
        
        logger.info(f"온톨로지가 성공적으로 가져와졌습니다: {json_path}")
    except Exception as e:
        logger.error(f"온톨로지 가져오기 오류: {e}")
        raise

async def export_ontology_to_json(connection: Neo4jConnection, json_path: str) -> None:
    """온톨로지를 JSON 파일로 내보내기
    
    Args:
        connection: Neo4j 연결 객체
        json_path: 저장할 JSON 파일 경로
    """
    try:
        # 모든 노드 가져오기
        nodes_query = """
        MATCH (n)
        RETURN n, ID(n) AS id, labels(n) AS labels
        """
        nodes_result = connection.execute_query(nodes_query)
        
        # 모든 관계 가져오기
        relationships_query = """
        MATCH (source)-[r]->(target)
        RETURN ID(source) AS source_id, ID(target) AS target_id, 
               type(r) AS relationship_type, r AS properties,
               ID(r) AS relationship_id
        """
        relationships_result = connection.execute_query(relationships_query)
        
        # 결과 변환
        nodes = []
        for record in nodes_result:
            node = record["n"]
            node_data = {
                "id": record["id"],
                "label": record["labels"][0],  # 첫 번째 레이블 사용
                "properties": dict(node)
            }
            nodes.append(node_data)
        
        relationships = []
        for record in relationships_result:
            relationship_data = {
                "id": record["relationship_id"],
                "source_id": record["source_id"],
                "target_id": record["target_id"],
                "type": record["relationship_type"],
                "properties": dict(record["properties"])
            }
            relationships.append(relationship_data)
        
        # JSON으로 저장
        ontology_data = {
            "nodes": nodes,
            "relationships": relationships
        }
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(ontology_data, file, indent=2, ensure_ascii=False)
        
        logger.info(f"온톨로지가 성공적으로 내보내졌습니다: {json_path}")
    except Exception as e:
        logger.error(f"온톨로지 내보내기 오류: {e}")
        raise

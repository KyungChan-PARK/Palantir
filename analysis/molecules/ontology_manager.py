"""
Neo4j 기반 온톨로지 관리 시스템
"""

import logging
import os
import json
import uuid
from datetime import datetime
from analysis.atoms.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)

class OntologyManager:
    """Neo4j 기반 온톨로지 관리 시스템"""
    
    def __init__(self, config_path=None, connector=None):
        """
        온톨로지 관리자 초기화
        
        Args:
            config_path (str, optional): Neo4j 구성 파일 경로
            connector (Neo4jConnector, optional): 기존 Neo4j 연결기
        """
        if connector:
            self.neo4j = connector
        else:
            if config_path:
                self.neo4j = Neo4jConnector(config_path=config_path)
            else:
                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "neo4j.yaml"
                )
                self.neo4j = Neo4jConnector(config_path=default_config_path)
        
        logger.info("온톨로지 관리자 초기화 완료")
    
    def close(self):
        """연결 종료"""
        if self.neo4j:
            self.neo4j.close()
    
    def initialize_ontology_schema(self):
        """
        온톨로지 스키마 초기화 및 기본 제약 조건/인덱스 생성
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 기본 제약 조건 생성
            self.neo4j.create_constraint("ObjectType", "name")
            self.neo4j.create_constraint("Property", "name")
            self.neo4j.create_constraint("LinkType", "name")
            self.neo4j.create_constraint("Document", "id")
            
            # 유용한 인덱스 생성
            self.neo4j.create_index("Document", "created_at")
            self.neo4j.create_index("Document", "status")
            self.neo4j.create_index("Document", "title")
            
            logger.info("온톨로지 스키마 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"온톨로지 스키마 초기화 중 오류: {str(e)}")
            return False
    
    def create_object_type(self, name, description=None, display_name=None):
        """
        객체 타입 생성
        
        Args:
            name (str): 객체 타입 이름 (고유 식별자)
            description (str, optional): 객체 타입 설명
            display_name (str, optional): 화면에 표시할 이름
        
        Returns:
            dict: 생성된 객체 타입 데이터
        """
        # 기존 객체 타입 확인
        existing = self.neo4j.get_node_by_property("ObjectType", "name", name)
        if existing:
            logger.warning(f"객체 타입 '{name}'이(가) 이미 존재합니다")
            return existing
        
        properties = {
            "name": name,
            "description": description or "",
            "display_name": display_name or name,
            "created_at": datetime.now().isoformat()
        }
        
        try:
            result = self.neo4j.create_node("ObjectType", properties)
            logger.info(f"객체 타입 '{name}' 생성됨")
            return result
        except Exception as e:
            logger.error(f"객체 타입 '{name}' 생성 중 오류: {str(e)}")
            raise
    
    def create_property(self, name, data_type, description=None, display_name=None, required=False):
        """
        속성 정의 생성
        
        Args:
            name (str): 속성 이름 (고유 식별자)
            data_type (str): 데이터 타입 (string, number, boolean, date, array, object)
            description (str, optional): 속성 설명
            display_name (str, optional): 화면에 표시할 이름
            required (bool, optional): 필수 여부
        
        Returns:
            dict: 생성된 속성 데이터
        """
        # 기존 속성 확인
        existing = self.neo4j.get_node_by_property("Property", "name", name)
        if existing:
            logger.warning(f"속성 '{name}'이(가) 이미 존재합니다")
            return existing
        
        valid_types = ["string", "number", "boolean", "date", "array", "object"]
        if data_type.lower() not in valid_types:
            raise ValueError(f"유효하지 않은 데이터 타입: {data_type}. 유효한 타입: {', '.join(valid_types)}")
        
        properties = {
            "name": name,
            "data_type": data_type.lower(),
            "description": description or "",
            "display_name": display_name or name,
            "required": required,
            "created_at": datetime.now().isoformat()
        }
        
        try:
            result = self.neo4j.create_node("Property", properties)
            logger.info(f"속성 '{name}' 생성됨")
            return result
        except Exception as e:
            logger.error(f"속성 '{name}' 생성 중 오류: {str(e)}")
            raise
    
    def create_link_type(self, name, source_type, target_type, description=None, display_name=None, bidirectional=False):
        """
        링크 타입(관계 타입) 생성
        
        Args:
            name (str): 링크 타입 이름 (고유 식별자)
            source_type (str): 소스 객체 타입 이름
            target_type (str): 대상 객체 타입 이름
            description (str, optional): 링크 타입 설명
            display_name (str, optional): 화면에 표시할 이름
            bidirectional (bool, optional): 양방향 관계 여부
        
        Returns:
            dict: 생성된 링크 타입 데이터
        """
        # 기존 링크 타입 확인
        existing = self.neo4j.get_node_by_property("LinkType", "name", name)
        if existing:
            logger.warning(f"링크 타입 '{name}'이(가) 이미 존재합니다")
            return existing
        
        # 소스 및 대상 객체 타입 확인
        source = self.neo4j.get_node_by_property("ObjectType", "name", source_type)
        if not source:
            raise ValueError(f"소스 객체 타입 '{source_type}'이(가) 존재하지 않습니다")
        
        target = self.neo4j.get_node_by_property("ObjectType", "name", target_type)
        if not target:
            raise ValueError(f"대상 객체 타입 '{target_type}'이(가) 존재하지 않습니다")
        
        properties = {
            "name": name,
            "description": description or "",
            "display_name": display_name or name,
            "bidirectional": bidirectional,
            "created_at": datetime.now().isoformat()
        }
        
        try:
            # 링크 타입 노드 생성
            link_type = self.neo4j.create_node("LinkType", properties)
            
            # 소스 및 대상 객체 타입과의 관계 생성
            self.neo4j.create_relationship(
                "LinkType", "name", name,
                "ObjectType", "name", source_type,
                "SOURCE_TYPE"
            )
            
            self.neo4j.create_relationship(
                "LinkType", "name", name,
                "ObjectType", "name", target_type,
                "TARGET_TYPE"
            )
            
            logger.info(f"링크 타입 '{name}' 생성됨: {source_type} -> {target_type}")
            return link_type
        except Exception as e:
            logger.error(f"링크 타입 '{name}' 생성 중 오류: {str(e)}")
            raise
    
    def add_property_to_object_type(self, object_type_name, property_name):
        """
        객체 타입에 속성 추가
        
        Args:
            object_type_name (str): 객체 타입 이름
            property_name (str): 속성 이름
        
        Returns:
            dict: 생성된 관계 데이터
        """
        try:
            # 이미 관계가 존재하는지 확인
            existing = self.neo4j.get_relationship(
                "ObjectType", "name", object_type_name,
                "Property", "name", property_name,
                "HAS_PROPERTY"
            )
            
            if existing and len(existing) > 0:
                logger.warning(f"객체 타입 '{object_type_name}'에 속성 '{property_name}'이(가) 이미 존재합니다")
                return existing[0]
            
            # 관계 생성
            result = self.neo4j.create_relationship(
                "ObjectType", "name", object_type_name,
                "Property", "name", property_name,
                "HAS_PROPERTY"
            )
            
            logger.info(f"객체 타입 '{object_type_name}'에 속성 '{property_name}' 추가됨")
            return result
        except Exception as e:
            logger.error(f"객체 타입 '{object_type_name}'에 속성 '{property_name}' 추가 중 오류: {str(e)}")
            raise
    
    def get_object_type(self, name):
        """
        객체 타입 조회
        
        Args:
            name (str): 객체 타입 이름
        
        Returns:
            dict: 객체 타입 데이터, 없으면 None
        """
        return self.neo4j.get_node_by_property("ObjectType", "name", name)
    
    def get_property(self, name):
        """
        속성 조회
        
        Args:
            name (str): 속성 이름
        
        Returns:
            dict: 속성 데이터, 없으면 None
        """
        return self.neo4j.get_node_by_property("Property", "name", name)
    
    def get_link_type(self, name):
        """
        링크 타입 조회
        
        Args:
            name (str): 링크 타입 이름
        
        Returns:
            dict: 링크 타입 데이터, 없으면 None
        """
        return self.neo4j.get_node_by_property("LinkType", "name", name)
    
    def get_all_object_types(self):
        """
        모든 객체 타입 조회
        
        Returns:
            list: 객체 타입 데이터 목록
        """
        return self.neo4j.get_nodes_by_label("ObjectType")
    
    def get_all_properties(self):
        """
        모든 속성 조회
        
        Returns:
            list: 속성 데이터 목록
        """
        return self.neo4j.get_nodes_by_label("Property")
    
    def get_all_link_types(self):
        """
        모든 링크 타입 조회
        
        Returns:
            list: 링크 타입 데이터 목록
        """
        return self.neo4j.get_nodes_by_label("LinkType")
    
    def get_object_type_properties(self, object_type_name):
        """
        객체 타입에 연결된 속성 조회
        
        Args:
            object_type_name (str): 객체 타입 이름
        
        Returns:
            list: 속성 데이터 목록
        """
        query = """
        MATCH (o:ObjectType {name: $object_type_name})-[:HAS_PROPERTY]->(p:Property)
        RETURN p
        """
        parameters = {"object_type_name": object_type_name}
        
        try:
            results = self.neo4j.execute_query(query, parameters)
            return [record["p"] for record in results]
        except Exception as e:
            logger.error(f"객체 타입 '{object_type_name}'의 속성 조회 중 오류: {str(e)}")
            raise
    
    def create_document(self, doc_id=None, title=None, object_type=None, content=None, metadata=None, status="draft"):
        """
        문서 객체 생성
        
        Args:
            doc_id (str, optional): 문서 ID (제공되지 않으면 UUID 생성)
            title (str, optional): 문서 제목
            object_type (str, optional): 문서 객체 타입
            content (str, optional): 문서 내용
            metadata (dict, optional): 문서 메타데이터
            status (str, optional): 문서 상태 (draft, published, archived)
        
        Returns:
            dict: 생성된 문서 데이터
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        properties = {
            "id": doc_id,
            "title": title or f"Document-{doc_id[:8]}",
            "content": content or "",
            "status": status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **metadata
        }
        
        try:
            # 문서 노드 생성
            document = self.neo4j.create_node("Document", properties)
            
            # 객체 타입이 제공되면 관계 생성
            if object_type:
                self.neo4j.create_relationship(
                    "Document", "id", doc_id,
                    "ObjectType", "name", object_type,
                    "HAS_TYPE"
                )
            
            logger.info(f"문서 '{doc_id}' 생성됨")
            return document
        except Exception as e:
            logger.error(f"문서 '{doc_id}' 생성 중 오류: {str(e)}")
            raise
    
    def get_document(self, doc_id):
        """
        문서 조회
        
        Args:
            doc_id (str): 문서 ID
        
        Returns:
            dict: 문서 데이터, 없으면 None
        """
        return self.neo4j.get_node_by_property("Document", "id", doc_id)
    
    def update_document(self, doc_id, update_data):
        """
        문서 업데이트
        
        Args:
            doc_id (str): 문서 ID
            update_data (dict): 업데이트할 데이터
        
        Returns:
            dict: 업데이트된 문서 데이터
        """
        # 업데이트 데이터에 업데이트 시간 추가
        update_data["updated_at"] = datetime.now().isoformat()
        
        try:
            result = self.neo4j.update_node("Document", "id", doc_id, update_data)
            logger.info(f"문서 '{doc_id}' 업데이트됨")
            return result
        except Exception as e:
            logger.error(f"문서 '{doc_id}' 업데이트 중 오류: {str(e)}")
            raise
    
    def delete_document(self, doc_id):
        """
        문서 삭제
        
        Args:
            doc_id (str): 문서 ID
        
        Returns:
            bool: 성공 여부
        """
        try:
            result = self.neo4j.delete_node("Document", "id", doc_id)
            if result:
                logger.info(f"문서 '{doc_id}' 삭제됨")
            else:
                logger.warning(f"문서 '{doc_id}'을(를) 찾을 수 없음")
            return result
        except Exception as e:
            logger.error(f"문서 '{doc_id}' 삭제 중 오류: {str(e)}")
            raise
    
    def create_document_link(self, source_doc_id, target_doc_id, link_type, properties=None):
        """
        문서 간 링크 생성
        
        Args:
            source_doc_id (str): 소스 문서 ID
            target_doc_id (str): 대상 문서 ID
            link_type (str): 링크 타입 이름
            properties (dict, optional): 링크 속성
        
        Returns:
            dict: 생성된 링크 데이터
        """
        if properties is None:
            properties = {}
        
        # 링크 타입 존재 확인
        link_type_node = self.neo4j.get_node_by_property("LinkType", "name", link_type)
        if not link_type_node:
            raise ValueError(f"링크 타입 '{link_type}'이(가) 존재하지 않습니다")
        
        try:
            # 관계 생성
            result = self.neo4j.create_relationship(
                "Document", "id", source_doc_id,
                "Document", "id", target_doc_id,
                link_type.upper(),
                properties
            )
            
            logger.info(f"문서 링크 생성됨: {source_doc_id} -[{link_type}]-> {target_doc_id}")
            return result
        except Exception as e:
            logger.error(f"문서 링크 생성 중 오류: {str(e)}")
            raise
    
    def get_document_links(self, doc_id, direction="outgoing", link_type=None):
        """
        문서 링크 조회
        
        Args:
            doc_id (str): 문서 ID
            direction (str, optional): 링크 방향 (outgoing, incoming, both)
            link_type (str, optional): 링크 타입 이름 (None이면 모든 타입)
        
        Returns:
            list: 링크 데이터 목록
        """
        link_type_clause = f":{link_type.upper()}" if link_type else ""
        
        if direction.lower() == "outgoing":
            query = f"""
            MATCH (d:Document {{id: $doc_id}})-[r{link_type_clause}]->(target:Document)
            RETURN r, target
            """
        elif direction.lower() == "incoming":
            query = f"""
            MATCH (source:Document)-[r{link_type_clause}]->(d:Document {{id: $doc_id}})
            RETURN r, source as target
            """
        else:  # both
            query = f"""
            MATCH (d:Document {{id: $doc_id}})-[r{link_type_clause}]->(target:Document)
            RETURN r, target, 'outgoing' as direction
            UNION
            MATCH (source:Document)-[r{link_type_clause}]->(d:Document {{id: $doc_id}})
            RETURN r, source as target, 'incoming' as direction
            """
        
        parameters = {"doc_id": doc_id}
        
        try:
            results = self.neo4j.execute_query(query, parameters)
            return results
        except Exception as e:
            logger.error(f"문서 '{doc_id}'의 링크 조회 중 오류: {str(e)}")
            raise
    
    def search_documents(self, query_string, object_type=None, status=None, limit=100):
        """
        문서 검색
        
        Args:
            query_string (str): 검색 쿼리 문자열 (제목 또는 내용)
            object_type (str, optional): 객체 타입으로 필터링
            status (str, optional): 상태로 필터링
            limit (int, optional): 결과 제한 개수
        
        Returns:
            list: 문서 데이터 목록
        """
        where_clauses = []
        parameters = {
            "query": f"(?i).*{query_string}.*",
            "limit": limit
        }
        
        # 기본 검색 조건 (제목 또는 내용에 쿼리 문자열 포함)
        where_clauses.append("(d.title =~ $query OR d.content =~ $query)")
        
        # 객체 타입으로 필터링
        if object_type:
            where_clauses.append("exists((d)-[:HAS_TYPE]->(:ObjectType {name: $object_type}))")
            parameters["object_type"] = object_type
        
        # 상태로 필터링
        if status:
            where_clauses.append("d.status = $status")
            parameters["status"] = status
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (d:Document)
        WHERE {where_clause}
        RETURN d
        ORDER BY d.created_at DESC
        LIMIT $limit
        """
        
        try:
            results = self.neo4j.execute_query(query, parameters)
            return [record["d"] for record in results]
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}")
            raise
    
    def get_documents_by_status(self, status, object_type=None, limit=100):
        """
        상태별 문서 조회
        
        Args:
            status (str): 문서 상태
            object_type (str, optional): 객체 타입으로 필터링
            limit (int, optional): 결과 제한 개수
        
        Returns:
            list: 문서 데이터 목록
        """
        if object_type:
            query = """
            MATCH (d:Document)-[:HAS_TYPE]->(:ObjectType {name: $object_type})
            WHERE d.status = $status
            RETURN d
            ORDER BY d.created_at DESC
            LIMIT $limit
            """
            parameters = {
                "status": status,
                "object_type": object_type,
                "limit": limit
            }
        else:
            query = """
            MATCH (d:Document)
            WHERE d.status = $status
            RETURN d
            ORDER BY d.created_at DESC
            LIMIT $limit
            """
            parameters = {
                "status": status,
                "limit": limit
            }
        
        try:
            results = self.neo4j.execute_query(query, parameters)
            return [record["d"] for record in results]
        except Exception as e:
            logger.error(f"상태 '{status}'의 문서 조회 중 오류: {str(e)}")
            raise
    
    def get_document_type_count(self):
        """
        객체 타입별 문서 수 조회
        
        Returns:
            list: 객체 타입 및 문서 수 목록
        """
        query = """
        MATCH (d:Document)-[:HAS_TYPE]->(t:ObjectType)
        RETURN t.name as type_name, t.display_name as display_name, count(d) as document_count
        ORDER BY document_count DESC
        """
        
        try:
            return self.neo4j.execute_query(query)
        except Exception as e:
            logger.error(f"객체 타입별 문서 수 조회 중 오류: {str(e)}")
            raise
    
    def get_document_status_count(self):
        """
        상태별 문서 수 조회
        
        Returns:
            list: 상태 및 문서 수 목록
        """
        query = """
        MATCH (d:Document)
        RETURN d.status as status, count(d) as document_count
        ORDER BY document_count DESC
        """
        
        try:
            return self.neo4j.execute_query(query)
        except Exception as e:
            logger.error(f"상태별 문서 수 조회 중 오류: {str(e)}")
            raise
    
    def import_ontology_from_json(self, json_file_path):
        """
        JSON 파일에서 온톨로지 가져오기
        
        Args:
            json_file_path (str): JSON 파일 경로
        
        Returns:
            dict: 가져오기 결과 통계
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                ontology_data = json.load(file)
            
            stats = {
                "object_types": 0,
                "properties": 0,
                "link_types": 0,
                "object_type_properties": 0
            }
            
            # 객체 타입 생성
            for obj_type in ontology_data.get("object_types", []):
                self.create_object_type(
                    name=obj_type["name"],
                    description=obj_type.get("description"),
                    display_name=obj_type.get("display_name")
                )
                stats["object_types"] += 1
            
            # 속성 생성
            for prop in ontology_data.get("properties", []):
                self.create_property(
                    name=prop["name"],
                    data_type=prop["data_type"],
                    description=prop.get("description"),
                    display_name=prop.get("display_name"),
                    required=prop.get("required", False)
                )
                stats["properties"] += 1
            
            # 객체 타입에 속성 연결
            for obj_type_prop in ontology_data.get("object_type_properties", []):
                self.add_property_to_object_type(
                    object_type_name=obj_type_prop["object_type"],
                    property_name=obj_type_prop["property"]
                )
                stats["object_type_properties"] += 1
            
            # 링크 타입 생성
            for link_type in ontology_data.get("link_types", []):
                self.create_link_type(
                    name=link_type["name"],
                    source_type=link_type["source_type"],
                    target_type=link_type["target_type"],
                    description=link_type.get("description"),
                    display_name=link_type.get("display_name"),
                    bidirectional=link_type.get("bidirectional", False)
                )
                stats["link_types"] += 1
            
            logger.info(f"JSON 파일에서 온톨로지 가져오기 완료: {stats}")
            return stats
        except Exception as e:
            logger.error(f"JSON 파일에서 온톨로지 가져오기 중 오류: {str(e)}")
            raise
    
    def export_ontology_to_json(self, json_file_path):
        """
        온톨로지를 JSON 파일로 내보내기
        
        Args:
            json_file_path (str): JSON 파일 경로
        
        Returns:
            dict: 내보내기 결과 통계
        """
        try:
            # 객체 타입, 속성, 링크 타입 조회
            object_types = self.get_all_object_types()
            properties = self.get_all_properties()
            link_types = self.get_all_link_types()
            
            # 객체 타입-속성 관계 조회
            query = """
            MATCH (o:ObjectType)-[:HAS_PROPERTY]->(p:Property)
            RETURN o.name as object_type, p.name as property
            """
            object_type_properties = self.neo4j.execute_query(query)
            
            # JSON 데이터 구성
            ontology_data = {
                "object_types": [
                    {
                        "name": obj["name"],
                        "display_name": obj.get("display_name", obj["name"]),
                        "description": obj.get("description", "")
                    }
                    for obj in object_types
                ],
                "properties": [
                    {
                        "name": prop["name"],
                        "data_type": prop["data_type"],
                        "display_name": prop.get("display_name", prop["name"]),
                        "description": prop.get("description", ""),
                        "required": prop.get("required", False)
                    }
                    for prop in properties
                ],
                "link_types": [
                    {
                        "name": link["name"],
                        "display_name": link.get("display_name", link["name"]),
                        "description": link.get("description", ""),
                        "bidirectional": link.get("bidirectional", False),
                        "source_type": self._get_link_type_source(link["name"]),
                        "target_type": self._get_link_type_target(link["name"])
                    }
                    for link in link_types
                ],
                "object_type_properties": [
                    {
                        "object_type": relation["object_type"],
                        "property": relation["property"]
                    }
                    for relation in object_type_properties
                ]
            }
            
            # JSON 파일 저장
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(ontology_data, file, indent=2, ensure_ascii=False)
            
            stats = {
                "object_types": len(ontology_data["object_types"]),
                "properties": len(ontology_data["properties"]),
                "link_types": len(ontology_data["link_types"]),
                "object_type_properties": len(ontology_data["object_type_properties"])
            }
            
            logger.info(f"온톨로지를 JSON 파일로 내보내기 완료: {stats}")
            return stats
        except Exception as e:
            logger.error(f"온톨로지를 JSON 파일로 내보내기 중 오류: {str(e)}")
            raise
    
    def _get_link_type_source(self, link_type_name):
        """
        링크 타입의 소스 객체 타입 조회
        
        Args:
            link_type_name (str): 링크 타입 이름
        
        Returns:
            str: 소스 객체 타입 이름
        """
        query = """
        MATCH (l:LinkType {name: $link_type_name})-[:SOURCE_TYPE]->(o:ObjectType)
        RETURN o.name as source_type
        """
        parameters = {"link_type_name": link_type_name}
        
        results = self.neo4j.execute_query(query, parameters)
        if results and len(results) > 0:
            return results[0]["source_type"]
        return None
    
    def _get_link_type_target(self, link_type_name):
        """
        링크 타입의 대상 객체 타입 조회
        
        Args:
            link_type_name (str): 링크 타입 이름
        
        Returns:
            str: 대상 객체 타입 이름
        """
        query = """
        MATCH (l:LinkType {name: $link_type_name})-[:TARGET_TYPE]->(o:ObjectType)
        RETURN o.name as target_type
        """
        parameters = {"link_type_name": link_type_name}
        
        results = self.neo4j.execute_query(query, parameters)
        if results and len(results) > 0:
            return results[0]["target_type"]
        return None
    
    def visualize_ontology(self):
        """
        온톨로지 데이터를 시각화용으로 조회
        
        Returns:
            dict: 시각화용 온톨로지 데이터
                - nodes: 노드 데이터 목록 (id, label, group)
                - edges: 엣지 데이터 목록 (from, to, label)
        """
        # 노드 조회 쿼리
        nodes_query = """
        MATCH (n)
        WHERE n:ObjectType OR n:Property OR n:LinkType
        RETURN id(n) as id, labels(n)[0] as type, n.name as name, n.display_name as display_name
        """
        
        # 엣지 조회 쿼리
        edges_query = """
        MATCH (n)-[r]->(m)
        WHERE (n:ObjectType OR n:Property OR n:LinkType) AND (m:ObjectType OR m:Property OR m:LinkType)
        RETURN id(n) as source, id(m) as target, type(r) as type
        """
        
        try:
            # 노드 데이터 조회 및 변환
            nodes_result = self.neo4j.execute_query(nodes_query)
            nodes = [
                {
                    "id": record["id"],
                    "label": record.get("display_name", record["name"]),
                    "title": record["name"],
                    "group": record["type"]
                }
                for record in nodes_result
            ]
            
            # 엣지 데이터 조회 및 변환
            edges_result = self.neo4j.execute_query(edges_query)
            edges = [
                {
                    "from": record["source"],
                    "to": record["target"],
                    "label": record["type"].lower().replace("_", " ")
                }
                for record in edges_result
            ]
            
            return {
                "nodes": nodes,
                "edges": edges
            }
        except Exception as e:
            logger.error(f"온톨로지 시각화 데이터 조회 중 오류: {str(e)}")
            raise
    
    def get_document_graph(self, doc_id=None, depth=1, max_nodes=100):
        """
        문서 그래프 데이터 조회
        
        Args:
            doc_id (str, optional): 중심 문서 ID (None이면 모든 문서)
            depth (int, optional): 관계 깊이
            max_nodes (int, optional): 최대 노드 수
        
        Returns:
            dict: 그래프 데이터
                - nodes: 노드 데이터 목록 (id, label, group)
                - edges: 엣지 데이터 목록 (from, to, label)
        """
        if doc_id:
            # 특정 문서를 중심으로 그래프 조회
            query = f"""
            MATCH path = (d:Document {{id: $doc_id}})-[*0..{depth}]-(related)
            WHERE related:Document
            WITH d, related, [rel in relationships(path) | type(rel)] as rel_types
            RETURN
                id(d) as source_id,
                d.id as source_external_id,
                d.title as source_title,
                id(related) as target_id,
                related.id as target_external_id,
                related.title as target_title,
                related.status as target_status,
                rel_types
            LIMIT $max_nodes
            """
            parameters = {
                "doc_id": doc_id,
                "max_nodes": max_nodes
            }
        else:
            # 모든 문서 그래프 조회
            query = """
            MATCH (d:Document)-[r]->(related:Document)
            RETURN
                id(d) as source_id,
                d.id as source_external_id,
                d.title as source_title,
                id(related) as target_id,
                related.id as target_external_id,
                related.title as target_title,
                related.status as target_status,
                [type(r)] as rel_types
            LIMIT $max_nodes
            """
            parameters = {
                "max_nodes": max_nodes
            }
        
        try:
            results = self.neo4j.execute_query(query, parameters)
            
            # 노드 및 엣지 데이터 구성
            nodes_map = {}  # 중복 노드 방지용 맵
            edges = []
            
            for record in results:
                # 소스 노드 추가
                if record["source_id"] not in nodes_map:
                    nodes_map[record["source_id"]] = {
                        "id": record["source_id"],
                        "external_id": record["source_external_id"],
                        "label": record["source_title"],
                        "group": "Document"
                    }
                
                # 대상 노드 추가
                if record["target_id"] not in nodes_map:
                    nodes_map[record["target_id"]] = {
                        "id": record["target_id"],
                        "external_id": record["target_external_id"],
                        "label": record["target_title"],
                        "group": record["target_status"] or "Document"
                    }
                
                # 관계가 있는 경우 엣지 추가
                if record["source_id"] != record["target_id"] and record["rel_types"]:
                    for rel_type in record["rel_types"]:
                        edges.append({
                            "from": record["source_id"],
                            "to": record["target_id"],
                            "label": rel_type.lower().replace("_", " ")
                        })
            
            return {
                "nodes": list(nodes_map.values()),
                "edges": edges
            }
        except Exception as e:
            logger.error(f"문서 그래프 데이터 조회 중 오류: {str(e)}")
            raise
    
    def __enter__(self):
        """컨텍스트 관리자 진입 메서드"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료 메서드"""
        self.close()

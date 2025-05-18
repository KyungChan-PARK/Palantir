"""
온톨로지 관리 시스템 모듈

Neo4j 기반 온톨로지 관리 시스템을 제공합니다. 이 시스템은 데이터의 관계와 속성을 정의하고 
관리하는 기능을 제공합니다.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from analysis.tools.atoms.neo4j_tools import (
    Neo4jConnection, create_node, create_relationship, export_ontology_to_json, 
    find_nodes_by_label_and_property, get_connected_nodes, get_node_by_id, 
    import_ontology_from_json, update_node_properties
)
from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("ontology_manager")

class OntologySystem:
    """온톨로지 관리 시스템 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Neo4j 구성 파일 경로
        """
        self.connection = Neo4jConnection(config_path)
        self.config_path = config_path
        logger.info("온톨로지 관리 시스템이 초기화되었습니다.")
    
    async def initialize_base_ontology(self) -> None:
        """기본 온톨로지 초기화"""
        try:
            # 문서 타입 노드 생성
            document_node = await create_node(
                self.connection,
                "NodeType",
                {"name": "Document", "description": "문서 객체"}
            )
            
            # 속성 타입 노드 생성
            title_prop = await create_node(
                self.connection,
                "PropertyType",
                {"name": "title", "data_type": "string", "description": "문서 제목"}
            )
            
            content_prop = await create_node(
                self.connection,
                "PropertyType",
                {"name": "content", "data_type": "text", "description": "문서 내용"}
            )
            
            status_prop = await create_node(
                self.connection,
                "PropertyType",
                {"name": "status", "data_type": "string", "description": "문서 상태"}
            )
            
            created_at_prop = await create_node(
                self.connection,
                "PropertyType",
                {"name": "created_at", "data_type": "datetime", "description": "생성 시간"}
            )
            
            # 관계 설정
            await create_relationship(
                self.connection,
                document_node["n"].id,
                title_prop["n"].id,
                "HAS_PROPERTY"
            )
            
            await create_relationship(
                self.connection,
                document_node["n"].id,
                content_prop["n"].id,
                "HAS_PROPERTY"
            )
            
            await create_relationship(
                self.connection,
                document_node["n"].id,
                status_prop["n"].id,
                "HAS_PROPERTY"
            )
            
            await create_relationship(
                self.connection,
                document_node["n"].id,
                created_at_prop["n"].id,
                "HAS_PROPERTY"
            )
            
            # 폴더 타입 노드 생성
            folder_node = await create_node(
                self.connection,
                "NodeType",
                {"name": "Folder", "description": "폴더 객체"}
            )
            
            # 폴더-문서 관계 타입 설정
            await create_node(
                self.connection,
                "RelationshipType",
                {"name": "CONTAINS", "description": "폴더가 문서를 포함함"}
            )
            
            logger.info("기본 온톨로지가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"기본 온톨로지 초기화 오류: {e}")
            raise
    
    async def create_document_node(self, title: str, content: str, 
                           status: str = "draft", 
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """문서 노드 생성
        
        Args:
            title: 문서 제목
            content: 문서 내용
            status: 문서 상태
            metadata: 추가 메타데이터
            
        Returns:
            생성된 문서 노드 정보
        """
        try:
            properties = {
                "title": title,
                "content": content,
                "status": status,
                **metadata
            } if metadata else {
                "title": title,
                "content": content,
                "status": status
            }
            
            document_node = await create_node(
                self.connection,
                "Document",
                properties
            )
            
            logger.info(f"문서 노드가 생성되었습니다: {title}")
            return document_node
        except Exception as e:
            logger.error(f"문서 노드 생성 오류: {e}")
            raise
    
    async def create_folder_node(self, name: str, 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """폴더 노드 생성
        
        Args:
            name: 폴더 이름
            metadata: 추가 메타데이터
            
        Returns:
            생성된 폴더 노드 정보
        """
        try:
            properties = {
                "name": name,
                **metadata
            } if metadata else {
                "name": name
            }
            
            folder_node = await create_node(
                self.connection,
                "Folder",
                properties
            )
            
            logger.info(f"폴더 노드가 생성되었습니다: {name}")
            return folder_node
        except Exception as e:
            logger.error(f"폴더 노드 생성 오류: {e}")
            raise
    
    async def add_document_to_folder(self, document_id: int, 
                             folder_id: int) -> Dict[str, Any]:
        """폴더에 문서 추가
        
        Args:
            document_id: 문서 노드 ID
            folder_id: 폴더 노드 ID
            
        Returns:
            생성된 관계 정보
        """
        try:
            relationship = await create_relationship(
                self.connection,
                folder_id,
                document_id,
                "CONTAINS"
            )
            
            logger.info(f"문서({document_id})가 폴더({folder_id})에 추가되었습니다.")
            return relationship
        except Exception as e:
            logger.error(f"폴더에 문서 추가 오류: {e}")
            raise
    
    async def update_document_status(self, document_id: int, 
                             status: str) -> Dict[str, Any]:
        """문서 상태 업데이트
        
        Args:
            document_id: 문서 노드 ID
            status: 새 상태
            
        Returns:
            업데이트된 문서 노드 정보
        """
        try:
            updated_node = await update_node_properties(
                self.connection,
                document_id,
                {"status": status}
            )
            
            logger.info(f"문서({document_id})의 상태가 '{status}'로 업데이트되었습니다.")
            return updated_node
        except Exception as e:
            logger.error(f"문서 상태 업데이트 오류: {e}")
            raise
    
    async def find_documents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """상태별 문서 검색
        
        Args:
            status: 검색할 문서 상태
            
        Returns:
            검색된 문서 목록
        """
        try:
            documents = await find_nodes_by_label_and_property(
                self.connection,
                "Document",
                "status",
                status
            )
            
            logger.info(f"'{status}' 상태의 문서 {len(documents)}개를 찾았습니다.")
            return documents
        except Exception as e:
            logger.error(f"상태별 문서 검색 오류: {e}")
            raise
    
    async def get_folder_contents(self, folder_id: int) -> List[Dict[str, Any]]:
        """폴더 내용 조회
        
        Args:
            folder_id: 폴더 노드 ID
            
        Returns:
            폴더에 포함된 문서 목록
        """
        try:
            contents = await get_connected_nodes(
                self.connection,
                folder_id,
                "CONTAINS"
            )
            
            logger.info(f"폴더({folder_id})의 내용 {len(contents)}개를 조회했습니다.")
            return contents
        except Exception as e:
            logger.error(f"폴더 내용 조회 오류: {e}")
            raise
    
    async def export_ontology(self, export_path: str) -> str:
        """온톨로지 내보내기
        
        Args:
            export_path: 내보낼 파일 경로
            
        Returns:
            내보내기 파일 경로
        """
        try:
            await export_ontology_to_json(
                self.connection,
                export_path
            )
            
            logger.info(f"온톨로지가 내보내졌습니다: {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"온톨로지 내보내기 오류: {e}")
            raise
    
    async def import_ontology(self, import_path: str) -> None:
        """온톨로지 가져오기
        
        Args:
            import_path: 가져올 파일 경로
        """
        try:
            await import_ontology_from_json(
                self.connection,
                import_path
            )
            
            logger.info(f"온톨로지가 가져와졌습니다: {import_path}")
        except Exception as e:
            logger.error(f"온톨로지 가져오기 오류: {e}")
            raise

@mcp.system(
    name="ontology_system",
    description="Neo4j 기반 온톨로지 관리 시스템"
)
async def initialize_ontology_system(config_path: str) -> OntologySystem:
    """온톨로지 관리 시스템 초기화
    
    Args:
        config_path: Neo4j 구성 파일 경로
        
    Returns:
        초기화된 온톨로지 관리 시스템 인스턴스
    """
    ontology_system = OntologySystem(config_path)
    await ontology_system.initialize_base_ontology()
    return ontology_system

@mcp.workflow(
    name="document_status_analyzer",
    description="문서 상태별 통계 분석 워크플로우"
)
async def analyze_document_status(ontology_system: OntologySystem) -> Dict[str, int]:
    """문서 상태별 통계 분석
    
    Args:
        ontology_system: 온톨로지 관리 시스템 인스턴스
        
    Returns:
        상태별 문서 수 딕셔너리
    """
    # 상태 목록
    statuses = ["draft", "review", "approved", "published", "archived"]
    
    # 각 상태별 문서 수 계산
    status_counts = {}
    for status in statuses:
        documents = await ontology_system.find_documents_by_status(status)
        status_counts[status] = len(documents)
    
    logger.info(f"문서 상태별 통계: {status_counts}")
    return status_counts

@mcp.workflow(
    name="folder_document_counter",
    description="폴더별 문서 수 계산 워크플로우"
)
async def count_documents_in_folders(ontology_system: OntologySystem) -> Dict[str, int]:
    """폴더별 문서 수 계산
    
    Args:
        ontology_system: 온톨로지 관리 시스템 인스턴스
        
    Returns:
        폴더별 문서 수 딕셔너리
    """
    # 모든 폴더 노드 조회
    folders = await find_nodes_by_label_and_property(
        ontology_system.connection,
        "Folder",
        "name",
        None  # 모든 폴더 조회
    )
    
    # 각 폴더별 문서 수 계산
    folder_counts = {}
    for folder in folders:
        folder_id = folder["n"].id
        folder_name = folder["n"].get("name", f"폴더 {folder_id}")
        
        contents = await ontology_system.get_folder_contents(folder_id)
        folder_counts[folder_name] = len(contents)
    
    logger.info(f"폴더별 문서 수: {folder_counts}")
    return folder_counts

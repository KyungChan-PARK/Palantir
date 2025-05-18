"""
테스트용 온톨로지 처리 모듈

이 모듈은 Neo4j 온톨로지를 처리하는 기본 기능을 제공합니다.
Codex-Claude 통합 테스트용으로 작성되었습니다.
"""

class OntologyProcessor:
    """기본 온톨로지 처리 클래스."""
    
    def __init__(self, connector):
        """
        온톨로지 프로세서 초기화
        
        Args:
            connector: Neo4j 커넥터 인스턴스
        """
        self.connector = connector
    
    def get_all_nodes(self):
        """
        모든 노드를 가져옵니다.
        
        Returns:
            list: 노드 목록
        """
        query = "MATCH (n) RETURN n"
        return self.connector.execute_query(query)
    
    def create_node(self, label, properties):
        """
        새 노드를 생성합니다.
        
        Args:
            label (str): 노드 레이블
            properties (dict): 노드 속성
            
        Returns:
            dict: 생성된 노드 정보
        """
        query = f"CREATE (n:{label} $props) RETURN n"
        return self.connector.execute_write(query, {"props": properties})

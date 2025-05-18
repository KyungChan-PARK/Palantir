"""Neo4j 라인리지 엣지 생성 검증 테스트"""
import uuid
from neo4j import GraphDatabase
from common.neo4j_utils import write_lineage_edge

NEO4J_URI = "bolt://localhost:7687"
AUTH = ("neo4j", "pass")

def test_write_lineage_edge():
    src = f"src_{uuid.uuid4().hex[:8]}"
    dst = f"dst_{uuid.uuid4().hex[:8]}"
    write_lineage_edge(src, dst, dag="test_dag")

    driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
    with driver.session() as s:
        result = s.run(
            "MATCH (s:Table {name:$src})-[r:TRANSFORMS]->(d:Table {name:$dst}) RETURN r", src=src, dst=dst
        ).single()
    assert result is not None, "TRANSFORMS edge not created"
    props = result["r"]._properties
    assert props.get("dag") == "test_dag" 
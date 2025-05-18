"""neo4j_utils.py
Neo4j helper 함수 모음
- load_csv_to_neo4j: CSV 파일을 지정된 라벨로 로드
- write_lineage_edge: 라인리지 엣지(:TRANSFORMS) 기록
"""
from pathlib import Path
from typing import Dict, Any
from neo4j import GraphDatabase

_DRIVER = GraphDatabase.driver(
    "bolt://localhost:7687", auth=("neo4j", "pass"), encrypted=False
)


def _ensure_file(path: str | Path):
    if not Path(path).exists():
        raise FileNotFoundError(path)


def load_csv_to_neo4j(csv_path: str | Path, label: str) -> None:
    """LOAD CSV를 이용해 파일을 Neo4j 노드로 삽입."""
    _ensure_file(csv_path)
    query = (
        f"LOAD CSV WITH HEADERS FROM 'file:///{Path(csv_path).as_posix()}' AS row "
        f"MERGE (n:{label} {{id: coalesce(row.id, row.student_id)}}) "
        f"SET n += row"
    )
    with _DRIVER.session() as s:
        s.run(query)


def write_lineage_edge(source: str, target: str, dag: str, extra_props: Dict[str, Any] | None = None) -> None:
    """src 테이블→dst 테이블 라인리지 엣지를 기록."""
    props = {"dag": dag}
    if extra_props:
        props.update(extra_props)
    prop_str = ", ".join([f"{k}: ${k}" for k in props])
    query = (
        "MERGE (src:Table {name:$source})\n"
        "MERGE (dst:Table {name:$target})\n"
        "MERGE (src)-[:TRANSFORMS {" + prop_str + "}]->(dst)"
    )
    with _DRIVER.session() as s:
        s.run(query, source=source, target=target, **props) 
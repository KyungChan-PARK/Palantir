"""
Init Neo4j:
1) Create constraints / indexes
2) (Optional) Import OWL
"""
from neo4j import GraphDatabase
from pathlib import Path

DRIVER = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pass"))

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT student_id IF NOT EXISTS ON (s:Student) ASSERT s.student_id IS UNIQUE",
    "CREATE CONSTRAINT course_code IF NOT EXISTS ON (c:Course) ASSERT c.course_code IS UNIQUE",
]

def run_schema(tx):
    for q in SCHEMA_QUERIES:
        tx.run(q)


def import_owl(tx, path: Path):
    tx.run("CALL n10s.graphconfig.init({})")
    tx.run("CALL n10s.onto.import.fetch($url,'RDF/XML')", url=path.as_uri())


with DRIVER.session() as session:
    session.write_transaction(run_schema)
    owl = Path("docs/edu_ontology.owl")
    if owl.exists():
        session.write_transaction(import_owl, owl)
print("✅ Neo4j init complete") 
from fastapi import FastAPI
from neo4j import GraphDatabase
from common.config_loader import load

cfg = load("neo4j")
app = FastAPI(title="Palantir API (offline)")

driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))

@app.get("/graph")
def get_graph(limit: int = 500):
    with driver.session() as s:
        q = (
            """
        MATCH p=(s:Student)-[:ENROLLED_IN]->(c:Course)-[:COVERS]->(t:Topic)
        RETURN s,c,t LIMIT $limit
        """
        )
        recs = s.run(q, limit=limit).data()
    return recs

"""
Dash app: 교육 지식 그래프 인터랙티브 맵
- Neo4j → pandas → dash-cytoscape 네트워크
- Plotly bar: Topic PageRank
"""
import os
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import pandas as pd
from neo4j import GraphDatabase
from dash import Dash, dcc, html, Output, Input

# Neo4j 연결
driver = GraphDatabase.driver(
    "bolt://localhost:7687", auth=("neo4j", os.getenv("NEO4J_PASSWORD", "pass"))
)


def fetch_graph(limit: int = 200):
    """학생-강좌-토픽 서브그래프 추출"""
    q = """
    MATCH p=(s:Student)-[:ENROLLED_IN]->(c:Course)-[:COVERS]->(t:Topic)
    RETURN nodes(p) AS n, relationships(p) AS r LIMIT $lim
    """
    with driver.session() as s:
        rec = s.run(q, lim=limit).single()
    nodes = [
        {
            "data": {
                "id": n.id,
                "label": list(n.labels)[0],
                **n._properties,
            }
        }
        for n in rec["n"]
    ]
    edges = [
        {
            "data": {
                "source": r.start_node.id,
                "target": r.end_node.id,
                "label": r.type,
            }
        }
        for r in rec["r"]
    ]
    return nodes + edges


app = Dash(__name__)
app.title = "Knowledge Map"

REFRESH_SEC = 300

app.layout = html.Div([
    dcc.Interval(id="refresh", interval=REFRESH_SEC*1000, n_intervals=0),
    dcc.Loading(type="circle", children=[
        cyto.Cytoscape(
            id="graph",
            elements=[],
            style={"height": "600px", "width": "100%"},
            layout={"name": "cose"},
        ),
        dcc.Graph(id="pagerank-bar")
    ])
])

@app.callback(
    Output("graph", "elements"),
    Output("pagerank-bar", "figure"),
    Input("refresh", "n_intervals"),
)
def update_elements(_: int):
    elements = fetch_graph()
    with driver.session() as s:
        df = pd.DataFrame(
            s.run(
                "CALL gds.pageRank.stream('topicGraph') "
                "YIELD nodeId, score "
                "RETURN gds.util.asNode(nodeId).name_kr AS topic, score"
            ).data()
        )
    fig = px.bar(df, x="topic", y="score").update_layout(xaxis_tickangle=45)
    return elements, fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8050) 
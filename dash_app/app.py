import os
import dash
from dash import dcc, html, Output, Input
import dash_cytoscape as cyto
import plotly.express as px
import pandas as pd
from neo4j import GraphDatabase

# Neo4j connection
driver = GraphDatabase.driver(
    "bolt://localhost:7687", auth=("neo4j", os.getenv("NEO4J_PASSWORD", "pass"))
)


def fetch_graph(limit: int = 200):
    """Fetch student-course-topic subgraph."""
    q = (
        "MATCH p=(s:Student)-[:ENROLLED_IN]->(c:Course)-[:COVERS]->(t:Topic) "
        "RETURN nodes(p) AS n, relationships(p) AS r LIMIT $lim"
    )
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


app = dash.Dash(__name__)
app.title = "Knowledge Map"

REFRESH_SEC = 300

app.layout = html.Div(
    [
        dcc.Interval(id="refresh", interval=REFRESH_SEC * 1000, n_intervals=0),
        dcc.Store(id="graph-data"),
        dcc.Store(id="pagerank-data"),
        dcc.Loading(
            type="circle",
            children=[
                cyto.Cytoscape(
                    id="graph",
                    elements=[],
                    style={"height": "600px", "width": "100%"},
                    layout={"name": "cose"},
                ),
                dcc.Graph(id="pagerank-bar"),
            ],
        ),
    ]
)


@app.callback(
    Output("graph-data", "data"),
    Output("pagerank-data", "data"),
    Input("refresh", "n_intervals"),
)
def load_data(_: int):
    """Load graph elements and pagerank data from Neo4j."""
    elements = fetch_graph()
    with driver.session() as s:
        df = pd.DataFrame(
            s.run(
                "CALL gds.pageRank.stream('topicGraph') "
                "YIELD nodeId, score "
                "RETURN gds.util.asNode(nodeId).name_kr AS topic, score"
            ).data()
        )
    return elements, df.to_dict("records")


@app.callback(Output("graph", "elements"), Input("graph-data", "data"))
def update_graph(data):
    return data or []


@app.callback(Output("pagerank-bar", "figure"), Input("pagerank-data", "data"))
def update_pagerank(data):
    df = pd.DataFrame(data or [])
    fig = px.bar(df, x="topic", y="score").update_layout(xaxis_tickangle=45)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)


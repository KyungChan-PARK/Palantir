import os
import dash
from dash import dcc, html, Output, Input
import dash_cytoscape as cyto
import plotly.express as px
import pandas as pd
import requests, json

BACKEND_URL = "http://localhost:8000/graph"


def fetch_graph(limit=500):
    """백엔드 FastAPI에서 그래프 데이터를 받아온다"""
    return requests.get(BACKEND_URL, params={"limit": limit}).json()


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


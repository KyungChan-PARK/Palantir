"""
팔란티어 파운드리 대시보드 애플리케이션

Dash 기반 웹 대시보드로 시스템 상태와 데이터를 시각화합니다.
"""

import os
import logging
from typing import Dict, List

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'dashboard.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("dashboard")

# 대시보드 앱 초기화
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

server = app.server
app.title = "팔란티어 파운드리 로컬 대시보드"

# 네비게이션 바
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("홈", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("문서 관리", href="/documents", active="exact")),
        dbc.NavItem(dbc.NavLink("온톨로지", href="/ontology", active="exact")),
        dbc.NavItem(dbc.NavLink("성능 분석", href="/performance", active="exact")),
        dbc.NavItem(dbc.NavLink("LLM 현황", href="/llm", active="exact")),
    ],
    brand="팔란티어 파운드리",
    brand_href="#",
    color="dark",
    dark=True,
)

# 콘텐츠 컨테이너
content = html.Div(id="page-content", className="p-4")

# 앱 레이아웃
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    content
])

# 홈 대시보드 레이아웃
def get_home_layout():
    # 가상의 시스템 상태 데이터
    system_status = {
        "온톨로지 관리 시스템": "활성",
        "데이터 파이프라인 시스템": "활성",
        "문서 관리 시스템": "활성",
        "데이터 품질 시스템": "활성",
        "웹 대시보드 인터페이스": "활성",
        "API 시스템": "비활성",
        "LLM 통합 시스템": "활성"
    }
    
    # 시스템 상태 카드 생성
    status_cards = []
    for system, status in system_status.items():
        color = "success" if status == "활성" else "danger"
        status_cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5(system, className="card-title"),
                        html.P(status, className=f"text-{color}")
                    ]),
                    className="mb-4"
                ),
                width=4
            )
        )
    
    # 가상의 문서 상태 데이터
    document_status = {
        "초안": 15,
        "검토": 8,
        "승인": 12,
        "출판": 20,
        "보관": 5
    }
    
    # 문서 상태 파이 차트
    document_status_fig = px.pie(
        names=list(document_status.keys()),
        values=list(document_status.values()),
        title="문서 상태별 분포"
    )
    
    # 가상의 시스템 활동 데이터
    activities = pd.DataFrame({
        "일자": pd.date_range(start="2025-05-01", periods=14, freq="D"),
        "활동 수": [25, 30, 15, 22, 18, 5, 8, 28, 35, 20, 15, 10, 5, 30]
    })
    
    # 시스템 활동 라인 차트
    activity_fig = px.line(
        activities, 
        x="일자", 
        y="활동 수", 
        title="최근 시스템 활동",
        markers=True
    )
    
    # 레이아웃 반환
    return dbc.Container([
        html.H2("시스템 대시보드", className="mt-4 mb-4"),
        
        html.H4("시스템 상태", className="mt-4 mb-3"),
        dbc.Row(status_cards),
        
        html.H4("문서 현황", className="mt-4 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=document_status_fig), width=6),
            dbc.Col(dcc.Graph(figure=activity_fig), width=6)
        ]),
        
        html.H4("최근 이벤트", className="mt-4 mb-3"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("시간"), html.Th("이벤트"), html.Th("시스템")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("2025-05-16 09:30"), 
                    html.Td("새 문서 처리 완료"), 
                    html.Td("문서 관리 시스템")
                ]),
                html.Tr([
                    html.Td("2025-05-16 09:15"), 
                    html.Td("온톨로지 업데이트"), 
                    html.Td("온톨로지 관리 시스템")
                ]),
                html.Tr([
                    html.Td("2025-05-16 09:00"), 
                    html.Td("LLM 코드 생성 요청"), 
                    html.Td("LLM 통합 시스템")
                ]),
                html.Tr([
                    html.Td("2025-05-16 08:45"), 
                    html.Td("데이터 품질 검증 완료"), 
                    html.Td("데이터 품질 시스템")
                ]),
                html.Tr([
                    html.Td("2025-05-16 08:30"), 
                    html.Td("시스템 시작"), 
                    html.Td("모든 시스템")
                ])
            ])
        ], bordered=True, hover=True, responsive=True, striped=True)
    ], fluid=True)

# 문서 관리 레이아웃
def get_documents_layout():
    # 가상의 문서 데이터
    documents = [
        {"id": 1, "제목": "시스템 아키텍처 문서", "상태": "승인", "생성일": "2025-05-01", "페이지": 15},
        {"id": 2, "제목": "온톨로지 설계 지침", "상태": "출판", "생성일": "2025-05-02", "페이지": 22},
        {"id": 3, "제목": "데이터 파이프라인 계획", "상태": "초안", "생성일": "2025-05-10", "페이지": 8},
        {"id": 4, "제목": "API 문서", "상태": "검토", "생성일": "2025-05-12", "페이지": 12},
        {"id": 5, "제목": "사용자 매뉴얼", "상태": "출판", "생성일": "2025-05-05", "페이지": 35},
        {"id": 6, "제목": "성능 테스트 결과", "상태": "초안", "생성일": "2025-05-15", "페이지": 7},
        {"id": 7, "제목": "LLM 통합 지침", "상태": "검토", "생성일": "2025-05-14", "페이지": 18},
        {"id": 8, "제목": "데이터 품질 보고서", "상태": "승인", "생성일": "2025-05-08", "페이지": 10}
    ]
    
    # 문서 테이블
    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("ID"), html.Th("제목"), html.Th("상태"), html.Th("생성일"), html.Th("페이지"), html.Th("작업")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(doc["id"]), 
                html.Td(doc["제목"]), 
                html.Td(html.Span(doc["상태"], className=f"badge bg-{'success' if doc['상태'] == '출판' else 'warning' if doc['상태'] == '검토' else 'info' if doc['상태'] == '승인' else 'secondary'}")), 
                html.Td(doc["생성일"]), 
                html.Td(doc["페이지"]),
                html.Td(
                    html.Div([
                        dbc.Button("보기", color="primary", size="sm", className="me-1"),
                        dbc.Button("편집", color="secondary", size="sm")
                    ], className="d-flex")
                )
            ]) for doc in documents
        ])
    ], bordered=True, hover=True, responsive=True, striped=True)
    
    # 문서 생성 폼
    form = dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("문서 제목"),
                dbc.Input(type="text", placeholder="문서 제목 입력")
            ], width=6),
            dbc.Col([
                dbc.Label("문서 상태"),
                dbc.Select(
                    options=[
                        {"label": "초안", "value": "초안"},
                        {"label": "검토", "value": "검토"},
                        {"label": "승인", "value": "승인"},
                        {"label": "출판", "value": "출판"},
                        {"label": "보관", "value": "보관"}
                    ],
                    value="초안"
                )
            ], width=3),
            dbc.Col([
                dbc.Label("페이지 수"),
                dbc.Input(type="number", min=1, step=1, value=1)
            ], width=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("문서 내용"),
                dbc.Textarea(placeholder="문서 내용 입력", style={"height": "150px"})
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Button("문서 생성", color="primary")
            ], className="d-flex justify-content-end")
        ])
    ])
    
    # 레이아웃 반환
    return dbc.Container([
        html.H2("문서 관리", className="mt-4 mb-4"),
        
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    dbc.Row([
                        dbc.Col(html.H4("문서 목록", className="mt-3 mb-3"), width="auto"),
                        dbc.Col(
                            dbc.Input(type="text", placeholder="문서 검색...", className="ms-auto"), 
                            className="ms-auto d-flex align-items-center"
                        )
                    ], className="mb-3"),
                    table
                ], className="mt-3")
            ], label="문서 목록"),
            dbc.Tab([
                html.Div([
                    html.H4("새 문서 생성", className="mt-3 mb-3"),
                    form
                ], className="mt-3")
            ], label="새 문서")
        ])
    ], fluid=True)

# 온톨로지 뷰 레이아웃
def get_ontology_layout():
    # 가상의 온톨로지 그래프 데이터
    nodes = [
        {"id": "1", "label": "Document", "group": "NodeType"},
        {"id": "2", "label": "Folder", "group": "NodeType"},
        {"id": "3", "label": "title", "group": "PropertyType"},
        {"id": "4", "label": "content", "group": "PropertyType"},
        {"id": "5", "label": "status", "group": "PropertyType"},
        {"id": "6", "label": "created_at", "group": "PropertyType"},
        {"id": "7", "label": "CONTAINS", "group": "RelationshipType"}
    ]
    
    edges = [
        {"from": "1", "to": "3", "label": "HAS_PROPERTY"},
        {"from": "1", "to": "4", "label": "HAS_PROPERTY"},
        {"from": "1", "to": "5", "label": "HAS_PROPERTY"},
        {"from": "1", "to": "6", "label": "HAS_PROPERTY"},
        {"from": "2", "to": "1", "label": "CONTAINS"}
    ]
    
    # 그래프 시각화 (Plotly)
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    fig = go.Figure()
    
    # 노드 추가
    for group in nodes_df["group"].unique():
        group_nodes = nodes_df[nodes_df["group"] == group]
        fig.add_trace(go.Scatter(
            x=[i for i in range(len(group_nodes))],
            y=[0 for _ in range(len(group_nodes))],
            mode="markers+text",
            marker=dict(
                size=30,
                color=["skyblue" if group == "NodeType" else "lightgreen" if group == "PropertyType" else "lightpink"]
            ),
            text=group_nodes["label"],
            textposition="bottom center",
            name=group,
            hoverinfo="text",
            hovertext=group_nodes["label"]
        ))
    
    fig.update_layout(
        title="온톨로지 그래프 시각화",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        height=500
    )
    
    # 레이아웃 반환
    return dbc.Container([
        html.H2("온톨로지 뷰", className="mt-4 mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("온톨로지 그래프"),
                    dbc.CardBody([
                        html.P("현재 온톨로지 구조의 그래프 시각화입니다.", className="card-text"),
                        dcc.Graph(figure=fig)
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("온톨로지 관리"),
                    dbc.CardBody([
                        dbc.Button("온톨로지 가져오기", color="primary", className="mb-2 w-100"),
                        dbc.Button("온톨로지 내보내기", color="secondary", className="mb-2 w-100"),
                        dbc.Button("기본 온톨로지 초기화", color="success", className="mb-2 w-100"),
                        html.Hr(),
                        html.H5("노드 타입", className="mb-2"),
                        dbc.ListGroup([
                            dbc.ListGroupItem("Document"),
                            dbc.ListGroupItem("Folder")
                        ], className="mb-3"),
                        html.H5("속성 타입", className="mb-2"),
                        dbc.ListGroup([
                            dbc.ListGroupItem("title"),
                            dbc.ListGroupItem("content"),
                            dbc.ListGroupItem("status"),
                            dbc.ListGroupItem("created_at")
                        ], className="mb-3"),
                        html.H5("관계 타입", className="mb-2"),
                        dbc.ListGroup([
                            dbc.ListGroupItem("HAS_PROPERTY"),
                            dbc.ListGroupItem("CONTAINS")
                        ])
                    ])
                ])
            ], width=4)
        ])
    ], fluid=True)

# 성능 분석 레이아웃
def get_performance_layout():
    # 가상의 성능 테스트 데이터
    perf_data = pd.DataFrame({
        "문서 수": [10, 50, 100, 500, 1000],
        "처리 시간(ms)": [50, 150, 280, 1200, 2500],
        "메모리 사용(MB)": [15, 25, 45, 180, 350]
    })
    
    # 성능 그래프
    time_fig = px.line(
        perf_data, 
        x="문서 수", 
        y="처리 시간(ms)", 
        title="문서 수에 따른 처리 시간",
        markers=True
    )
    
    memory_fig = px.line(
        perf_data, 
        x="문서 수", 
        y="메모리 사용(MB)", 
        title="문서 수에 따른 메모리 사용량",
        markers=True
    )
    
    # 컨텍스트 최적화 데이터
    context_data = pd.DataFrame({
        "최적화 방법": ["기본 컨텍스트", "온톨로지 기반", "키워드 기반", "벡터 임베딩 기반", "하이브리드"],
        "정확도(%)": [75, 82, 80, 88, 92],
        "응답 시간(ms)": [200, 250, 220, 300, 320]
    })
    
    # 컨텍스트 최적화 그래프
    context_fig = px.bar(
        context_data, 
        x="최적화 방법", 
        y="정확도(%)", 
        title="컨텍스트 최적화 방법별 정확도",
        color="최적화 방법"
    )
    
    response_fig = px.bar(
        context_data, 
        x="최적화 방법", 
        y="응답 시간(ms)", 
        title="컨텍스트 최적화 방법별 응답 시간",
        color="최적화 방법"
    )
    
    # 레이아웃 반환
    return dbc.Container([
        html.H2("성능 분석", className="mt-4 mb-4"),
        
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    html.H4("문서 처리 성능", className="mt-3 mb-3"),
                    
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=time_fig), width=6),
                        dbc.Col(dcc.Graph(figure=memory_fig), width=6)
                    ]),
                    
                    html.H5("성능 테스트 결과 요약", className="mt-4 mb-3"),
                    dbc.Table.from_dataframe(perf_data, striped=True, bordered=True, hover=True)
                ], className="mt-3")
            ], label="문서 처리 성능"),
            
            dbc.Tab([
                html.Div([
                    html.H4("컨텍스트 최적화 성능", className="mt-3 mb-3"),
                    
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=context_fig), width=6),
                        dbc.Col(dcc.Graph(figure=response_fig), width=6)
                    ]),
                    
                    html.H5("최적화 방법 비교", className="mt-4 mb-3"),
                    dbc.Table.from_dataframe(context_data, striped=True, bordered=True, hover=True)
                ], className="mt-3")
            ], label="컨텍스트 최적화")
        ])
    ], fluid=True)

# LLM 현황 레이아웃
def get_llm_layout():
    # 가상의 LLM 생성 코드 통계
    code_stats = pd.DataFrame({
        "날짜": pd.date_range(start="2025-05-01", periods=10, freq="D"),
        "생성된 코드": [5, 8, 4, 10, 12, 7, 3, 9, 15, 6],
        "자가 개선 코드": [3, 5, 2, 7, 8, 4, 2, 6, 10, 4],
        "성공률(%)": [80, 85, 75, 90, 92, 78, 83, 88, 95, 82]
    })
    
    # LLM 코드 생성 그래프
    code_fig = px.line(
        code_stats, 
        x="날짜", 
        y=["생성된 코드", "자가 개선 코드"], 
        title="LLM 코드 생성 및 자가 개선 추이",
        markers=True
    )
    
    success_fig = px.line(
        code_stats, 
        x="날짜", 
        y="성공률(%)", 
        title="LLM 코드 생성 성공률",
        markers=True
    )
    
    # 가상의 최근 생성 코드 목록
    recent_codes = [
        {"ID": 1, "파일명": "document_status_analyzer.py", "생성 날짜": "2025-05-16", "상태": "완료", "개선 횟수": 2},
        {"ID": 2, "파일명": "ontology_exporter.py", "생성 날짜": "2025-05-15", "상태": "완료", "개선 횟수": 1},
        {"ID": 3, "파일명": "performance_tester.py", "생성 날짜": "2025-05-14", "상태": "개선 중", "개선 횟수": 1},
        {"ID": 4, "파일명": "data_quality_checker.py", "생성 날짜": "2025-05-13", "상태": "완료", "개선 횟수": 3},
        {"ID": 5, "파일명": "context_optimizer.py", "생성 날짜": "2025-05-12", "상태": "개선 중", "개선 횟수": 2}
    ]
    
    # 최근 코드 테이블
    recent_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("ID"), html.Th("파일명"), html.Th("생성 날짜"), html.Th("상태"), html.Th("개선 횟수"), html.Th("작업")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(code["ID"]), 
                html.Td(code["파일명"]), 
                html.Td(code["생성 날짜"]), 
                html.Td(html.Span(code["상태"], className=f"badge bg-{'success' if code['상태'] == '완료' else 'warning'}")), 
                html.Td(code["개선 횟수"]),
                html.Td(
                    html.Div([
                        dbc.Button("보기", color="primary", size="sm", className="me-1"),
                        dbc.Button("개선", color="secondary", size="sm")
                    ], className="d-flex")
                )
            ]) for code in recent_codes
        ])
    ], bordered=True, hover=True, responsive=True, striped=True)
    
    # 코드 생성 폼
    form = dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("파일명"),
                dbc.Input(type="text", placeholder="파일명.py")
            ], width=6),
            dbc.Col([
                dbc.Label("언어"),
                dbc.Select(
                    options=[
                        {"label": "Python", "value": "python"},
                        {"label": "JavaScript", "value": "javascript"},
                        {"label": "SQL", "value": "sql"}
                    ],
                    value="python"
                )
            ], width=3),
            dbc.Col([
                dbc.Label("개선 반복 횟수"),
                dbc.Input(type="number", min=0, max=5, step=1, value=2)
            ], width=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("코드 생성 지시사항"),
                dbc.Textarea(placeholder="코드 생성 지시사항 입력...", style={"height": "150px"})
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Button("코드 생성", color="primary")
            ], className="d-flex justify-content-end")
        ])
    ])
    
    # 레이아웃 반환
    return dbc.Container([
        html.H2("LLM 현황", className="mt-4 mb-4"),
        
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    html.H4("LLM 코드 생성 통계", className="mt-3 mb-3"),
                    
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=code_fig), width=6),
                        dbc.Col(dcc.Graph(figure=success_fig), width=6)
                    ]),
                    
                    html.H4("최근 생성 코드", className="mt-4 mb-3"),
                    recent_table
                ], className="mt-3")
            ], label="LLM 모니터링"),
            
            dbc.Tab([
                html.Div([
                    html.H4("새 코드 생성 요청", className="mt-3 mb-3"),
                    form
                ], className="mt-3")
            ], label="코드 생성 요청")
        ])
    ], fluid=True)

# 페이지 라우팅 콜백
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/documents":
        return get_documents_layout()
    elif pathname == "/ontology":
        return get_ontology_layout()
    elif pathname == "/performance":
        return get_performance_layout()
    elif pathname == "/llm":
        return get_llm_layout()
    else:
        return get_home_layout()

# 앱 실행 함수
def run_dashboard(host="0.0.0.0", port=5000, debug=True):
    logger.info(f"대시보드 서버 시작: http://{host}:{port}")
    app.run_server(host=host, port=port, debug=debug)

# 직접 실행 시
if __name__ == "__main__":
    run_dashboard()

# decision_support_system.py
"""
의사결정 지원 시스템
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 데이터 다운로드 (필요 시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from analysis.mcp_init import mcp
from analysis.tools.atoms.data_reader import read_data
from analysis.tools.molecules.exploratory_analysis import exploratory_analysis
from analysis.tools.molecules.predictive_modeling import build_predictive_model

@mcp.system(
    name="decision_support",
    description="비즈니스 질문에 답하기 위한 종합적인 분석을 수행하는 의사결정 지원 시스템"
)
async def decision_support(
    question: str,
    data_sources: List[str],
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    비즈니스 질문에 답하기 위한 종합적인 분석을 수행합니다.
    
    Parameters:
    -----------
    question : str
        해결하려는 비즈니스 질문 또는 의사결정 문제
    data_sources : List[str]
        분석에 사용할 데이터 소스 파일 경로 목록
    output_dir : str, optional
        결과를 저장할 디렉토리 경로
    params : Dict[str, Any], optional
        시스템 매개변수를 포함하는 딕셔너리
        
    Returns:
    --------
    Dict[str, Any]
        의사결정 지원 결과와 메타데이터를 포함하는 딕셔너리
    """
    # 기본값 설정
    params = params or {}
    
    if output_dir is None:
        output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "decisions")
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 보고서 ID 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_id = f"decision_{timestamp}"
    
    try:
        # 1. 질문 해석 및 필요한 분석 유형 결정
        analysis_plan = parse_question(question)
        
        # 2. 데이터 소스 로드 및 통합
        data_results = {}
        integrated_data = None
        primary_data = None
        
        for i, data_source in enumerate(data_sources):
            # 데이터 로드
            load_result = await read_data(data_source)
            
            if not load_result.get("success", False):
                data_results[data_source] = {
                    "success": False,
                    "error": load_result.get("error", "알 수 없는 오류")
                }
                continue
            
            df = load_result["data"]
            data_results[data_source] = {
                "success": True,
                "shape": df.shape,
                "columns": list(df.columns)
            }
            
            # 첫 번째 데이터 소스를 기본 데이터로 설정
            if i == 0:
                primary_data = data_source
                integrated_data = df.copy()
            else:
                # 향후 구현: 데이터 통합 로직
                # 현재는 간단하게 첫 번째 데이터 소스만 사용
                pass
        
        if primary_data is None or integrated_data is None:
            return {
                "success": False,
                "error": "유효한 데이터 소스를 로드할 수 없습니다."
            }
        
        # 3. 분석 유형에 따른 워크플로우 실행
        analysis_results = {}
        
        # 탐색적 데이터 분석 실행
        if analysis_plan.get("run_eda", True):
            # Create subdirectory for EDA outputs
            eda_output_dir = os.path.join(output_dir, report_id, "eda")
            os.makedirs(eda_output_dir, exist_ok=True)
            
            eda_result = await exploratory_analysis(
                file_path=primary_data,
                columns=analysis_plan.get("target_columns"),
                analysis_types=analysis_plan.get("analysis_types", ["descriptive", "correlation"]),
                output_dir=eda_output_dir
            )
            
            analysis_results["eda"] = eda_result
        
        # 예측 모델링 실행
        if analysis_plan.get("run_predictive_modeling", False):
            target_column = analysis_plan.get("target_variable")
            
            # Create subdirectory for models outputs
            model_output_dir = os.path.join(output_dir, report_id, "models")
            os.makedirs(model_output_dir, exist_ok=True)
            
            if target_column and target_column in integrated_data.columns:
                model_result = await build_predictive_model(
                    file_path=primary_data,
                    target_column=target_column,
                    problem_type=analysis_plan.get("problem_type"),
                    features=analysis_plan.get("feature_columns"),
                    output_dir=model_output_dir
                )
                
                analysis_results["predictive_modeling"] = model_result
            else:
                analysis_results["predictive_modeling"] = {
                    "success": False,
                    "error": f"타겟 변수를 찾을 수 없습니다: {target_column}"
                }
        
        # 4. 분석 결과 해석 및 통찰 생성
        insights = generate_insights(question, analysis_plan, analysis_results)
        
        # 5. 의사결정 지원 정보 생성
        recommendations = generate_recommendations(question, analysis_plan, analysis_results, insights)
        
        # 6. 최종 보고서 생성
        report = {
            "question": question,
            "analysis_plan": analysis_plan,
            "data_sources": data_results,
            "insights": insights,
            "recommendations": recommendations
        }
        
        # 보고서 저장
        report_file = os.path.join(output_dir, f"{report_id}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 보고서 요약 Markdown 생성
        markdown_summary = generate_decision_markdown_summary(report, report_id)
        markdown_file = os.path.join(output_dir, f"{report_id}.md")
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        
        return {
            "success": True,
            "report_id": report_id,
            "report_file": report_file,
            "markdown_summary": markdown_file,
            "analysis_plan": analysis_plan,
            "insights": insights,
            "recommendations": recommendations
        }
    
    except Exception as e:
        import traceback
        error_message = f"의사결정 지원 시스템 실행 중 오류 발생: {str(e)}"
        error_traceback = traceback.format_exc()
        
        # Create error log file
        error_log_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "logs")
        os.makedirs(error_log_dir, exist_ok=True)
        
        error_log_file = os.path.join(error_log_dir, f"decision_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Error: {error_message}\n\n")
            f.write(f"Traceback:\n{error_traceback}")
        
        return {
            "success": False,
            "error": error_message,
            "error_details": error_traceback,
            "error_log": error_log_file
        }

def parse_question(question: str) -> Dict[str, Any]:
    """
    비즈니스 질문을 분석하여 필요한 분석 유형과 매개변수를 결정합니다.
    
    Parameters:
    -----------
    question : str
        해결하려는 비즈니스 질문 또는 의사결정 문제
        
    Returns:
    --------
    Dict[str, Any]
        분석 계획과 매개변수
    """
    # 기본 분석 계획
    analysis_plan = {
        "run_eda": True,
        "analysis_types": ["descriptive", "correlation"],
        "run_predictive_modeling": False,
        "problem_type": None,
        "target_variable": None,
        "target_columns": None,
        "feature_columns": None,
        "time_dimension": None,
        "group_dimension": None
    }
    
    # 토큰화 및 키워드 추출
    text = question.lower()
    
    # 간단한 토큰화 방법 사용 (NLTK word_tokenize 대신)
    tokens = []
    # 영문자, 숫자, 공백만 허용하고 기타 문자는 공백으로 변경
    clean_text = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in text])
    # 공백으로 분할
    for token in clean_text.split():
        if token:  # 빈 토큰 제외
            tokens.append(token)
    
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # stopwords를 사용할 수 없는 경우 기본 영어 불용어 목록 사용
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                         'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                         'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                         'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                         'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                         'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                         'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                         'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                         'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                         'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                         'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                         'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                         'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                         'just', 'don', 'should', 'now'])
    
    filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    
    # 키워드 사전
    prediction_keywords = ['predict', 'forecast', 'estimation', 'future', 'will', 'expect']
    classification_keywords = ['classify', 'categorize', 'group', 'category', 'type', 'class', 'segment']
    regression_keywords = ['how much', 'value', 'amount', 'quantity', 'number', 'predict value']
    time_keywords = ['time', 'when', 'period', 'date', 'year', 'month', 'day', 'trend', 'seasonal']
    comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'against', 'relative']
    
    # 예측 분석이 필요한지 확인
    prediction_matches = [word for word in filtered_tokens if word in prediction_keywords]
    
    if prediction_matches or any(keyword in text for keyword in prediction_keywords):
        analysis_plan["run_predictive_modeling"] = True
        
        # 분류 또는 회귀 문제인지 판단
        if any(keyword in text for keyword in classification_keywords):
            analysis_plan["problem_type"] = "classification"
        elif any(keyword in text for keyword in regression_keywords):
            analysis_plan["problem_type"] = "regression"
    
    # 시간 차원이 있는지 확인
    if any(keyword in text for keyword in time_keywords):
        analysis_plan["analysis_types"].append("time_series")
        analysis_plan["time_dimension"] = True
    
    # 비교 분석이 필요한지 확인
    if any(keyword in text for keyword in comparison_keywords):
        analysis_plan["analysis_types"].append("group")
        analysis_plan["group_dimension"] = True
    
    # 타겟 변수 추출 (단순 휴리스틱)
    target_pattern = r"predict\s+(\w+)|forecast\s+(\w+)|of\s+(\w+)|about\s+(\w+)"
    target_matches = re.findall(target_pattern, text)
    
    if target_matches:
        # 첫 번째 매치에서 비어 있지 않은 그룹 선택
        for match in target_matches:
            for group in match:
                if group:
                    analysis_plan["target_variable"] = group
                    break
            if analysis_plan["target_variable"]:
                break
    
    return analysis_plan

def generate_insights(
    question: str,
    analysis_plan: Dict[str, Any],
    analysis_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    분석 결과를 바탕으로 통찰을 생성합니다.
    
    Parameters:
    -----------
    question : str
        원래 질문
    analysis_plan : Dict[str, Any]
        분석 계획
    analysis_results : Dict[str, Any]
        분석 결과
        
    Returns:
    --------
    List[Dict[str, Any]]
        통찰 목록
    """
    insights = []
    
    # EDA 결과에서 통찰 추출
    if "eda" in analysis_results and analysis_results["eda"].get("success", False):
        eda_insights = analysis_results["eda"].get("insights", [])
        
        for insight in eda_insights:
            insights.append({
                "type": "eda",
                "description": insight,
                "confidence": "high",
                "source": "exploratory_analysis"
            })
        
        # 상관 관계 통찰
        if "analysis_results" in analysis_results["eda"]:
            corr_results = analysis_results["eda"]["analysis_results"].get("correlation", {})
            if corr_results.get("success", False):
                corr_insights = corr_results.get("results", {}).get("insights", [])
                
                for insight in corr_insights:
                    insights.append({
                        "type": "correlation",
                        "description": insight,
                        "confidence": "high",
                        "source": "correlation_analysis"
                    })
    
    # 예측 모델링 결과에서 통찰 추출
    if "predictive_modeling" in analysis_results and analysis_results["predictive_modeling"].get("success", False):
        model_results = analysis_results["predictive_modeling"]
        
        # 최적 모델 정보
        best_model = model_results.get("best_model", {})
        model_key = best_model.get("model_key", "")
        performance = best_model.get("performance", {})
        
        if model_key and performance:
            if analysis_plan.get("problem_type") == "classification":
                accuracy = performance.get("accuracy", 0)
                f1 = performance.get("f1_score", 0)
                
                insights.append({
                    "type": "predictive_model",
                    "description": f"최적 분류 모델은 {model_key}로, 정확도 {accuracy:.2f}와 F1 점수 {f1:.2f}를 보였습니다.",
                    "confidence": "high" if f1 > 0.8 else "medium",
                    "source": "predictive_modeling"
                })
            else:  # regression
                r2 = performance.get("r2_score", 0)
                rmse = performance.get("root_mean_squared_error", 0)
                
                insights.append({
                    "type": "predictive_model",
                    "description": f"최적 회귀 모델은 {model_key}로, R² 점수 {r2:.2f}와 RMSE {rmse:.2f}를 보였습니다.",
                    "confidence": "high" if r2 > 0.8 else "medium",
                    "source": "predictive_modeling"
                })
        
        # 특성 중요도
        for model_result in model_results.get("model_results", []):
            if model_result.get("model_key") == model_key:
                feature_importance = model_result.get("feature_importance", {})
                
                if feature_importance:
                    # 중요도 상위 3개 특성
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    feature_list = ", ".join([f"{feature} ({importance:.2f})" for feature, importance in top_features])
                    
                    insights.append({
                        "type": "feature_importance",
                        "description": f"가장 중요한 특성은 {feature_list}입니다.",
                        "confidence": "high",
                        "source": "predictive_modeling"
                    })
    
    # 일반적인 통찰 (예: 질문에 대한 직접적인 답변)
    if insights:
        # 질문에 직접 응답하는 통찰 생성
        target_var = analysis_plan.get("target_variable")
        
        if target_var and analysis_plan.get("run_predictive_modeling", False):
            insights.append({
                "type": "direct_answer",
                "description": f"{target_var}에 가장 큰 영향을 미치는 요인이 식별되었으며, 적절한 예측 모델이 구축되었습니다.",
                "confidence": "medium",
                "source": "combined_analysis"
            })
    
    return insights

def generate_recommendations(
    question: str,
    analysis_plan: Dict[str, Any],
    analysis_results: Dict[str, Any],
    insights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    통찰을 바탕으로 권장 사항을 생성합니다.
    
    Parameters:
    -----------
    question : str
        원래 질문
    analysis_plan : Dict[str, Any]
        분석 계획
    analysis_results : Dict[str, Any]
        분석 결과
    insights : List[Dict[str, Any]]
        통찰 목록
        
    Returns:
    --------
    List[Dict[str, Any]]
        권장 사항 목록
    """
    recommendations = []
    
    # 예측 모델링 권장 사항
    if "predictive_modeling" in analysis_results and analysis_results["predictive_modeling"].get("success", False):
        model_results = analysis_results["predictive_modeling"]
        best_model = model_results.get("best_model", {})
        
        if best_model:
            # 모델 사용 권장 사항
            recommendations.append({
                "type": "model_usage",
                "title": "예측 모델 사용",
                "description": "구축된 예측 모델을 사용하여 새로운 데이터에 대한 예측을 수행하십시오.",
                "steps": [
                    "새 데이터를 모델과 동일한 형식으로 준비",
                    "저장된 모델 파일을 로드",
                    "새 데이터에 대한 예측 실행",
                    "결과 검증 및 해석"
                ],
                "priority": "high"
            })
            
            # 모델 모니터링 권장 사항
            recommendations.append({
                "type": "model_monitoring",
                "title": "모델 성능 모니터링",
                "description": "시간이 지남에 따라 모델 성능을 모니터링하고 필요시 재학습하십시오.",
                "steps": [
                    "정기적인 성능 평가 일정 수립",
                    "데이터 드리프트 모니터링",
                    "성능 저하 시 모델 재학습",
                    "모델 버전 관리 시스템 구축"
                ],
                "priority": "medium"
            })
    
    # 데이터 품질 권장 사항
    if "eda" in analysis_results and analysis_results["eda"].get("success", False):
        eda_results = analysis_results["eda"]
        
        data_quality_issues = False
        null_issues = False
        outlier_issues = False
        
        for insight in insights:
            if "결측값" in insight.get("description", ""):
                null_issues = True
            if "이상치" in insight.get("description", ""):
                outlier_issues = True
        
        if null_issues or outlier_issues:
            data_quality_issues = True
        
        if data_quality_issues:
            recommendations.append({
                "type": "data_quality",
                "title": "데이터 품질 개선",
                "description": "분석의 정확도를 높이기 위해 데이터 품질 문제를 해결하십시오.",
                "steps": [
                    "결측값 처리 전략 개선" if null_issues else None,
                    "이상치 탐지 및 처리 메커니즘 구현" if outlier_issues else None,
                    "데이터 검증 프로세스 구축",
                    "정기적인 데이터 품질 감사 수행"
                ],
                "priority": "high" if data_quality_issues else "medium"
            })
    
    # 추가 분석 권장 사항
    additional_analysis_needed = False
    
    if not analysis_plan.get("run_predictive_modeling", False):
        additional_analysis_needed = True
        
        recommendations.append({
            "type": "additional_analysis",
            "title": "예측 모델링 수행",
            "description": "식별된 패턴을 기반으로 예측 모델을 구축하여 미래 값을 예측하십시오.",
            "steps": [
                "목표 변수 선정",
                "주요 특성 선택",
                "여러 모델링 알고리즘 테스트",
                "최적 모델 선택 및 세부 조정"
            ],
            "priority": "medium"
        })
    
    if "group" not in analysis_plan.get("analysis_types", []):
        recommendations.append({
            "type": "additional_analysis",
            "title": "세분화 분석 수행",
            "description": "데이터를 의미 있는 세그먼트로 나누어 패턴과 차이점을 식별하십시오.",
            "steps": [
                "세분화 기준 정의",
                "그룹별 분석 수행",
                "그룹 간 주요 차이점 식별",
                "타겟 세그먼트에 대한 맞춤형 전략 개발"
            ],
            "priority": "medium"
        })
    
    # 의사결정 지원 권장 사항
    recommendations.append({
        "type": "decision_support",
        "title": "데이터 기반 의사 결정 프로세스 구축",
        "description": "분석 결과를 바탕으로 체계적인 의사 결정 프로세스를 구축하십시오.",
        "steps": [
            "주요 의사 결정 지점 식별",
            "각 결정에 필요한 데이터 및 분석 정의",
            "의사 결정 기준 및 임계값 설정",
            "결정 결과 추적 및 피드백 루프 구현"
        ],
        "priority": "high"
    })
    
    # 시각화 및 대시보드 권장 사항
    recommendations.append({
        "type": "visualization",
        "title": "의사 결정 대시보드 개발",
        "description": "주요 지표와 통찰을 시각화하는 대시보드를 개발하여 의사 결정 과정을 지원하십시오.",
        "steps": [
            "핵심 성과 지표(KPI) 정의",
            "직관적인 시각화 설계",
            "실시간 업데이트 메커니즘 구현",
            "사용자 피드백을 반영한 대시보드 개선"
        ],
        "priority": "medium"
    })
    
    # None 값 제거
    for i in range(len(recommendations)):
        if "steps" in recommendations[i]:
            recommendations[i]["steps"] = [step for step in recommendations[i]["steps"] if step is not None]
    
    return recommendations

def generate_decision_markdown_summary(report: Dict[str, Any], report_id: str) -> str:
    """
    의사결정 지원 보고서의 Markdown 요약을 생성합니다.
    
    Parameters:
    -----------
    report : Dict[str, Any]
        의사결정 지원 보고서 데이터
    report_id : str
        보고서 ID
        
    Returns:
    --------
    str
        Markdown 형식의 요약
    """
    markdown = f"# 의사결정 지원 보고서: {report_id}\n\n"
    
    # 비즈니스 질문
    markdown += "## 1. 비즈니스 질문\n\n"
    markdown += f"> {report['question']}\n\n"
    
    # 분석 계획
    markdown += "## 2. 분석 계획\n\n"
    
    analysis_plan = report['analysis_plan']
    
    markdown += "### 2.1 분석 유형\n\n"
    analysis_types = []
    
    if analysis_plan.get("run_eda", False):
        analysis_types.append("탐색적 데이터 분석(EDA)")
    
    if analysis_plan.get("run_predictive_modeling", False):
        problem_type = "분류" if analysis_plan.get("problem_type") == "classification" else "회귀"
        analysis_types.append(f"예측 모델링 ({problem_type})")
    
    for analysis_type in analysis_plan.get("analysis_types", []):
        if analysis_type == "descriptive":
            analysis_types.append("기술 통계 분석")
        elif analysis_type == "correlation":
            analysis_types.append("상관 관계 분석")
        elif analysis_type == "time_series":
            analysis_types.append("시계열 분석")
        elif analysis_type == "group":
            analysis_types.append("그룹 분석")
    
    for analysis_type in analysis_types:
        markdown += f"- {analysis_type}\n"
    
    markdown += "\n"
    
    # 타겟 변수가 있는 경우
    if analysis_plan.get("target_variable"):
        markdown += f"### 2.2 타겟 변수\n\n"
        markdown += f"- {analysis_plan['target_variable']}\n\n"
    
    # 데이터 소스
    markdown += "## 3. 데이터 소스\n\n"
    markdown += "| 데이터 소스 | 상태 | 행 수 | 열 수 |\n"
    markdown += "|------------|------|-------|-------|\n"
    
    for source, info in report['data_sources'].items():
        status = "성공" if info.get("success", False) else "실패"
        rows = info.get("shape", [0, 0])[0] if info.get("success", False) else "N/A"
        cols = info.get("shape", [0, 0])[1] if info.get("success", False) else "N/A"
        
        markdown += f"| {source} | {status} | {rows} | {cols} |\n"
    
    markdown += "\n"
    
    # 주요 통찰
    markdown += "## 4. 주요 통찰\n\n"
    
    insights = report.get("insights", [])
    if insights:
        # 통찰 유형별로 구성
        insight_types = {}
        
        for insight in insights:
            insight_type = insight.get("type", "기타")
            
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            
            insight_types[insight_type].append(insight)
        
        # 유형별로 표시
        for insight_type, type_insights in insight_types.items():
            if insight_type == "eda":
                title = "탐색적 데이터 분석"
            elif insight_type == "correlation":
                title = "상관 관계 분석"
            elif insight_type == "predictive_model":
                title = "예측 모델링"
            elif insight_type == "feature_importance":
                title = "특성 중요도"
            elif insight_type == "direct_answer":
                title = "직접 응답"
            else:
                title = insight_type.capitalize()
            
            markdown += f"### 4.{title}\n\n"
            
            for insight in type_insights:
                confidence = insight.get("confidence", "").capitalize()
                markdown += f"- {insight['description']} _{confidence}_\n"
            
            markdown += "\n"
    else:
        markdown += "_통찰이 발견되지 않았습니다._\n\n"
    
    # 권장 사항
    markdown += "## 5. 권장 사항\n\n"
    
    recommendations = report.get("recommendations", [])
    if recommendations:
        # 우선순위별 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_recommendations = sorted(
            recommendations, 
            key=lambda x: priority_order.get(x.get("priority", "medium"), 1)
        )
        
        for i, recommendation in enumerate(sorted_recommendations):
            title = recommendation.get("title", f"권장 사항 {i+1}")
            description = recommendation.get("description", "")
            priority = recommendation.get("priority", "medium").capitalize()
            
            markdown += f"### 5.{i+1}. {title} ({priority})\n\n"
            markdown += f"{description}\n\n"
            
            steps = recommendation.get("steps", [])
            if steps:
                markdown += "**실행 단계:**\n\n"
                for j, step in enumerate(steps):
                    markdown += f"{j+1}. {step}\n"
                
                markdown += "\n"
    else:
        markdown += "_권장 사항이 발견되지 않았습니다._\n\n"
    
    # 결론
    markdown += "## 6. 결론\n\n"
    
    # 주요 통찰을 기반으로 결론 도출
    if insights:
        markdown += "본 분석을 통해 다음과 같은 결론을 도출할 수 있습니다:\n\n"
        
        # 고신뢰도 통찰만 선택
        high_confidence_insights = [insight for insight in insights if insight.get("confidence") == "high"]
        
        if high_confidence_insights:
            for insight in high_confidence_insights[:3]:  # 상위 3개만
                markdown += f"- {insight['description']}\n"
        else:
            # 고신뢰도 통찰이 없으면 모든 통찰에서 선택
            for insight in insights[:3]:
                markdown += f"- {insight['description']}\n"
    else:
        markdown += "분석 결과, 명확한 결론을 도출하기 위해서는 추가 데이터 수집 및 분석이 필요합니다.\n"
    
    markdown += "\n"
    
    # 다음 단계
    markdown += "### 다음 단계\n\n"
    
    # 권장 사항에서 우선순위가 높은 항목 선택
    high_priority_recommendations = [rec for rec in recommendations if rec.get("priority") == "high"]
    
    if high_priority_recommendations:
        markdown += "다음 우선 순위 작업을 진행하십시오:\n\n"
        
        for rec in high_priority_recommendations:
            markdown += f"1. **{rec.get('title')}**: {rec.get('description')}\n"
    else:
        # 우선 순위가 높은 항목이 없으면 모든 권장 사항에서 선택
        markdown += "다음 단계로 권장 사항을 검토하고 우선 순위에 따라 구현 계획을 수립하십시오.\n"
    
    return markdown

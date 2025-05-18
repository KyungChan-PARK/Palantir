# exploratory_analysis.py
"""
탐색적 데이터 분석(EDA) 워크플로우
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.mcp_init import mcp
from analysis.tools.atoms.data_reader import read_data
from analysis.tools.atoms.data_processor import preprocess_data
from analysis.tools.atoms.data_analyzer import analyze_data

@mcp.workflow(
    name="exploratory_analysis",
    description="파일을 읽고 기본적인 탐색적 데이터 분석을 수행합니다. 데이터 로드, 전처리, 분석 및 시각화를 자동으로 수행합니다."
)
async def exploratory_analysis(
    file_path: str,
    columns: Optional[List[str]] = None,
    preprocessing_operations: Optional[List[str]] = None,
    analysis_types: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    파일을 읽고 기본적인 탐색적 데이터 분석을 수행합니다.
    
    Parameters:
    -----------
    file_path : str
        읽을 파일의 경로 (상대 경로는 프로젝트 루트 기준)
    columns : List[str], optional
        분석할 열 목록 (지정하지 않으면 모든 열 사용)
    preprocessing_operations : List[str], optional
        수행할 전처리 작업 목록 (기본값: ["remove_nulls"])
    analysis_types : List[str], optional
        수행할 분석 유형 목록 (기본값: ["descriptive", "distribution", "correlation"])
    output_dir : str, optional
        결과를 저장할 디렉토리 경로
    params : Dict[str, Any], optional
        워크플로우 매개변수를 포함하는 딕셔너리
        
    Returns:
    --------
    Dict[str, Any]
        EDA 결과와 메타데이터를 포함하는 딕셔너리
    """
    # 기본값 설정
    params = params or {}
    preprocessing_operations = preprocessing_operations or ["remove_nulls"]
    analysis_types = analysis_types or ["descriptive", "distribution", "correlation"]
    
    if output_dir is None:
        output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "reports")
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 이름에서 보고서 이름 생성
    file_name = os.path.basename(file_path)
    report_name = f"eda_report_{os.path.splitext(file_name)[0]}"
    
    try:
        # 1. 데이터 로드
        load_result = await read_data(file_path)
        
        if not load_result.get("success", False):
            return {
                "success": False,
                "error": f"데이터 로드 실패: {load_result.get('error', '알 수 없는 오류')}"
            }
        
        # 데이터 정보 추출
        df = load_result["data"]
        data_info = {
            "shape": load_result.get("shape", df.shape),
            "columns": load_result.get("columns", list(df.columns)),
            "dtypes": load_result.get("dtypes", df.dtypes.astype(str).to_dict())
        }
        
        # 컬럼 필터링 (지정된 경우)
        if columns:
            columns = [col for col in columns if col in df.columns]
        else:
            columns = list(df.columns)
        
        # 2. 데이터 전처리
        preprocessing_results = {}
        processed_data = load_result.copy()  # 초기값은 로드된 데이터
        
        for operation in preprocessing_operations:
            process_result = await preprocess_data(
                processed_data, 
                operations=[operation],
                columns=columns,
                params=params.get("preprocessing_params", {}).get(operation, {})
            )
            
            if not process_result.get("success", False):
                preprocessing_results[operation] = {
                    "success": False,
                    "error": process_result.get("error", "알 수 없는 오류")
                }
                continue
            
            processed_data = process_result
            preprocessing_results[operation] = {
                "success": True,
                "operation_results": process_result.get("operation_results", {}).get(operation, {})
            }
        
        # 3. 데이터 분석
        analysis_results = {}
        analysis_insights = []
        all_visualizations = []
        
        for analysis_type in analysis_types:
            analysis_params = params.get("analysis_params", {}).get(analysis_type, {})
            
            if analysis_type == "correlation" and len([col for col in columns if pd.api.types.is_numeric_dtype(df[col])]) < 2:
                analysis_results[analysis_type] = {
                    "success": False,
                    "error": "상관 관계 분석을 위해서는 최소 2개의 수치형 열이 필요합니다."
                }
                continue
            
            if analysis_type == "group" and "group_by_column" not in analysis_params:
                # 그룹화 열이 지정되지 않은 경우, 범주형 열 중 첫 번째를 선택
                categorical_columns = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
                
                if categorical_columns and df[categorical_columns[0]].nunique() <= 20:
                    analysis_params["group_by_column"] = categorical_columns[0]
                else:
                    analysis_results[analysis_type] = {
                        "success": False,
                        "error": "그룹 분석을 위한 적절한 범주형 열을 찾을 수 없습니다."
                    }
                    continue
            
            viz_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "viz", report_name, analysis_type)
            os.makedirs(viz_dir, exist_ok=True)
            
            analysis_result = await analyze_data(
                processed_data,
                analysis_type=analysis_type,
                columns=columns,
                params=analysis_params,
                output_dir=viz_dir
            )
            
            if not analysis_result.get("success", False):
                analysis_results[analysis_type] = {
                    "success": False,
                    "error": analysis_result.get("error", "알 수 없는 오류")
                }
                continue
            
            analysis_results[analysis_type] = {
                "success": True,
                "results": analysis_result
            }
            
            # 인사이트와 시각화 수집
            insights = analysis_result.get("insights", [])
            if insights:
                analysis_insights.extend([f"[{analysis_type}] {insight}" for insight in insights])
            
            visualizations = analysis_result.get("visualizations", [])
            if visualizations:
                # 시각화 유형 태그 추가
                for viz in visualizations:
                    viz["analysis_type"] = analysis_type
                
                all_visualizations.extend(visualizations)
        
        # 4. 추천 시각화 도출
        recommended_visualizations = []
        
        # 수치형 열에 대한 시각화 추천
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_columns) >= 2:
            # 상관 관계가 높은 열 쌍 찾기
            if "correlation" in analysis_results and analysis_results["correlation"].get("success", False):
                corr_results = analysis_results["correlation"]["results"]
                strong_correlations = corr_results.get("strong_correlations", [])
                
                for corr_info in strong_correlations[:3]:  # 최대 3개
                    recommended_visualizations.append({
                        "type": "scatter",
                        "columns": [corr_info["column1"], corr_info["column2"]],
                        "title": f"{corr_info['column1']} vs {corr_info['column2']} (r={corr_info['correlation']:.2f})",
                        "reason": f"{corr_info['strength']} {corr_info['type']} 상관 관계가 있습니다."
                    })
        
        # 범주형 열과 수치형 열의 관계 시각화 추천
        categorical_columns = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10]
        
        for cat_col in categorical_columns[:2]:  # 최대 2개의 범주형 열
            for num_col in numeric_columns[:2]:  # 최대 2개의 수치형 열
                recommended_visualizations.append({
                    "type": "boxplot",
                    "columns": [cat_col, num_col],
                    "title": f"{cat_col}별 {num_col} 분포",
                    "reason": f"범주별 {num_col} 분포 차이를 보여줍니다."
                })
        
        # 5. 보고서 생성
        report = {
            "file_name": file_name,
            "file_path": file_path,
            "data_info": data_info,
            "preprocessing": preprocessing_results,
            "analysis": {
                name: results.get("results", {}) 
                for name, results in analysis_results.items() 
                if results.get("success", False)
            },
            "insights": analysis_insights,
            "visualizations": all_visualizations,
            "recommended_visualizations": recommended_visualizations
        }
        
        # 보고서 저장
        report_file = os.path.join(output_dir, f"{report_name}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 보고서 요약 Markdown 생성
        markdown_summary = generate_eda_markdown_summary(report, report_name)
        markdown_file = os.path.join(output_dir, f"{report_name}.md")
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        
        return {
            "success": True,
            "report_file": report_file,
            "markdown_summary": markdown_file,
            "data_info": data_info,
            "preprocessing_results": preprocessing_results,
            "analysis_results": analysis_results,
            "insights": analysis_insights,
            "visualizations": all_visualizations,
            "recommended_visualizations": recommended_visualizations
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"탐색적 데이터 분석 중 오류 발생: {str(e)}"
        }

def generate_eda_markdown_summary(report: Dict[str, Any], report_name: str) -> str:
    """
    EDA 보고서의 Markdown 요약을 생성합니다.
    
    Parameters:
    -----------
    report : Dict[str, Any]
        EDA 보고서 데이터
    report_name : str
        보고서 이름
        
    Returns:
    --------
    str
        Markdown 형식의 요약
    """
    markdown = f"# 탐색적 데이터 분석 보고서: {report_name}\n\n"
    
    # 데이터 정보
    markdown += "## 1. 데이터 개요\n\n"
    markdown += f"- **파일명**: {report['file_name']}\n"
    markdown += f"- **파일경로**: {report['file_path']}\n"
    markdown += f"- **데이터 크기**: {report['data_info']['shape'][0]}행 × {report['data_info']['shape'][1]}열\n"
    markdown += f"- **열 개수**: {len(report['data_info']['columns'])}\n\n"
    
    # 열 정보
    markdown += "### 1.1 열 정보\n\n"
    markdown += "| 열 이름 | 데이터 타입 |\n"
    markdown += "|---------|------------|\n"
    
    for col, dtype in report['data_info']['dtypes'].items():
        markdown += f"| {col} | {dtype} |\n"
    
    markdown += "\n"
    
    # 전처리 정보
    markdown += "## 2. 데이터 전처리\n\n"
    
    for op, result in report['preprocessing'].items():
        markdown += f"### 2.{op.capitalize()}\n\n"
        
        if result.get("success", False):
            markdown += f"- **상태**: 성공\n"
            
            op_results = result.get("operation_results", {})
            if op == "remove_nulls":
                if "removed_rows" in op_results:
                    markdown += f"- **제거된 행**: {op_results['removed_rows']}\n"
                elif "method" in op_results:
                    markdown += f"- **처리 방법**: {op_results['method']}\n"
                    markdown += f"- **전략**: {op_results.get('strategy', 'N/A')}\n"
                    markdown += f"- **영향 받은 열**: {', '.join(op_results.get('affected_columns', []))}\n"
            
            elif op == "normalize":
                markdown += f"- **정규화 방법**: {op_results.get('method', 'N/A')}\n"
                markdown += f"- **영향 받은 열**: {', '.join(op_results.get('affected_columns', []))}\n"
            
            elif op == "outlier_removal":
                markdown += f"- **이상치 처리 방법**: {op_results.get('method', 'N/A')}\n"
                markdown += f"- **제거된 행**: {op_results.get('removed_rows', 'N/A')}\n"
                markdown += f"- **영향 받은 열**: {', '.join(op_results.get('affected_columns', []))}\n"
        else:
            markdown += f"- **상태**: 실패\n"
            markdown += f"- **오류**: {result.get('error', '알 수 없는 오류')}\n"
        
        markdown += "\n"
    
    # 분석 결과
    markdown += "## 3. 분석 결과\n\n"
    
    analysis_order = ["descriptive", "distribution", "correlation", "group"]
    analysis_titles = {
        "descriptive": "기술 통계",
        "distribution": "분포 분석",
        "correlation": "상관 관계 분석",
        "group": "그룹 분석"
    }
    
    for analysis_type in analysis_order:
        if analysis_type in report.get('analysis', {}):
            analysis_result = report['analysis'][analysis_type]
            markdown += f"### 3.{analysis_titles[analysis_type]}\n\n"
            
            if analysis_type == "descriptive" and "numeric_statistics" in analysis_result:
                # 수치형 통계 요약
                markdown += "#### 수치형 열 통계 요약\n\n"
                markdown += "| 열 | 평균 | 중앙값 | 표준편차 | 최소값 | 최대값 |\n"
                markdown += "|-----|------|--------|---------|--------|--------|\n"
                
                for col, stats in analysis_result['numeric_statistics'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        markdown += f"| {col} | {stats.get('mean', 'N/A'):.2f} | {stats.get('50%', 'N/A'):.2f} | {stats.get('std', 'N/A'):.2f} | {stats.get('min', 'N/A'):.2f} | {stats.get('max', 'N/A'):.2f} |\n"
                
                markdown += "\n"
                
                # 결측값 정보
                if "null_information" in analysis_result:
                    markdown += "#### 결측값 정보\n\n"
                    markdown += "| 열 | 결측값 수 | 결측값 비율(%) |\n"
                    markdown += "|-----|-----------|---------------|\n"
                    
                    for col in report['data_info']['columns']:
                        if col in analysis_result['null_information']['counts']:
                            count = analysis_result['null_information']['counts'][col]
                            pct = analysis_result['null_information']['percentages'][col]
                            markdown += f"| {col} | {count} | {pct:.2f} |\n"
                    
                    markdown += "\n"
            
            elif analysis_type == "correlation" and "strong_correlations" in analysis_result:
                # 강한 상관 관계
                markdown += "#### 강한 상관 관계\n\n"
                
                if analysis_result['strong_correlations']:
                    markdown += "| 열1 | 열2 | 상관계수 | 상관관계 유형 |\n"
                    markdown += "|-----|-----|---------|------------|\n"
                    
                    for corr_info in analysis_result['strong_correlations']:
                        markdown += f"| {corr_info['column1']} | {corr_info['column2']} | {corr_info['correlation']:.3f} | {corr_info['type']} ({corr_info['strength']}) |\n"
                else:
                    markdown += "*강한 상관 관계가 발견되지 않았습니다.*\n"
                
                markdown += "\n"
            
            # 여기에 다른 분석 유형에 대한 요약 추가 가능
            
            # 시각화 참조
            visualizations = [viz for viz in report.get('visualizations', []) if viz.get('analysis_type') == analysis_type]
            
            if visualizations:
                markdown += f"#### {analysis_titles[analysis_type]} 시각화\n\n"
                
                for i, viz in enumerate(visualizations[:5]):  # 최대 5개만 표시
                    file_path = viz.get('file_path', '')
                    if file_path:
                        file_name = os.path.basename(file_path)
                        viz_type = viz.get('type', '시각화')
                        
                        # 상대 경로로 변환
                        rel_path = os.path.join("../..", "output", "viz", report_name, analysis_type, file_name)
                        markdown += f"- [{viz_type.capitalize()} - {i+1}]({rel_path})\n"
                
                markdown += "\n"
    
    # 주요 인사이트
    markdown += "## 4. 주요 인사이트\n\n"
    
    if report.get('insights'):
        for insight in report['insights']:
            markdown += f"- {insight}\n"
    else:
        markdown += "*인사이트가 발견되지 않았습니다.*\n"
    
    markdown += "\n"
    
    # 추천 시각화
    markdown += "## 5. 추천 시각화\n\n"
    
    if report.get('recommended_visualizations'):
        for i, viz in enumerate(report['recommended_visualizations']):
            markdown += f"### 5.{i+1}. {viz['title']}\n\n"
            markdown += f"- **유형**: {viz['type'].capitalize()}\n"
            markdown += f"- **열**: {', '.join(viz['columns'])}\n"
            markdown += f"- **추천 이유**: {viz['reason']}\n\n"
    else:
        markdown += "*추천할 시각화가 없습니다.*\n"
    
    return markdown

# data_analyzer.py
"""
데이터 분석을 위한 도구 모음
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

from analysis.mcp_init import mcp

@mcp.tool(
    name="analyze_data",
    description="데이터에 지정된 분석을 수행합니다. 기술 통계, 상관 관계, 그룹 분석 등을 지원합니다.",
    examples=[
        {"input": {"data": "DataFrame", "analysis_type": "descriptive"}, 
         "output": {"statistics": {"mean": {...}, "median": {...}}, "insights": ["Column X has significant outliers"]}},
        {"input": {"data": "DataFrame", "analysis_type": "correlation", "columns": ["price", "size", "age"]}, 
         "output": {"correlation_matrix": [[1.0, 0.8, -0.5], [0.8, 1.0, -0.3], [-0.5, -0.3, 1.0]], "insights": ["Strong positive correlation between price and size"]}}
    ],
    tags=["data", "analysis", "statistics", "patterns"],
    contexts=["exploratory analysis", "pattern detection", "hypothesis testing"]
)
async def analyze_data(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    analysis_type: str,
    columns: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    데이터에 지정된 분석을 수행합니다.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, Dict[str, Any]]
        분석할 데이터 (DataFrame 또는 preprocess_data 결과)
    analysis_type : str
        수행할 분석 유형 (descriptive, correlation, group, distribution 등)
    columns : List[str], optional
        분석할 열 목록 (지정하지 않으면 모든 적절한 열에 적용)
    params : Dict[str, Any], optional
        분석에 필요한 추가 매개변수
    output_dir : str, optional
        시각화 결과를 저장할 디렉토리 경로
        
    Returns:
    --------
    Dict[str, Any]
        분석 결과와 메타데이터를 포함하는 딕셔너리
    """
    # 매개변수 기본값 설정
    params = params or {}
    
    # 기본 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "viz")
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # data가 이전 결과인 경우 처리
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], pd.DataFrame):
            df = data['data'].copy()
        else:
            return {
                "success": False,
                "error": "입력 데이터가 DataFrame이 아닙니다."
            }
    else:
        df = data.copy()
    
    # 열 목록이 없으면 모든 열 사용
    if columns is None:
        columns = df.columns.tolist()
    else:
        # 존재하는 열만 선택
        columns = [col for col in columns if col in df.columns]
    
    try:
        # 분석 유형에 따라 분석 수행
        if analysis_type == "descriptive":
            # 수치형 열 선택
            numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_columns = [col for col in columns if col not in numeric_columns]
            
            # 기술 통계량 계산
            desc_stats = df[numeric_columns].describe().transpose().to_dict()
            
            # 결측값 정보
            null_counts = df[columns].isnull().sum().to_dict()
            null_percentage = (df[columns].isnull().mean() * 100).to_dict()
            
            # 범주형 열에 대한 통계
            categorical_stats = {}
            for col in categorical_columns:
                if df[col].nunique() < 20:  # 너무 많은 범주가 있는 경우 제외
                    categorical_stats[col] = {
                        "unique_values": df[col].nunique(),
                        "top_values": df[col].value_counts().head(5).to_dict(),
                        "top_percentages": (df[col].value_counts(normalize=True) * 100).head(5).to_dict()
                    }
            
            # 인사이트 도출
            insights = []
            
            # 결측값 관련 인사이트
            high_null_cols = [col for col, pct in null_percentage.items() if pct > 5]
            if high_null_cols:
                insights.append(f"다음 열에 5% 이상의 결측값이 있습니다: {', '.join(high_null_cols)}")
            
            # 이상치 관련 인사이트
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col].count()
                if outliers > 0:
                    pct_outliers = (outliers / df.shape[0]) * 100
                    if pct_outliers > 5:
                        insights.append(f"'{col}' 열에 이상치가 많습니다 ({pct_outliers:.1f}%)")
            
            # 분포 관련 인사이트
            for col in numeric_columns:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    direction = "오른쪽" if skewness > 0 else "왼쪽"
                    insights.append(f"'{col}' 열이 {direction}으로 치우친 분포를 보입니다 (왜도: {skewness:.2f})")
            
            # 시각화
            visualizations = []
            if params.get("create_visualizations", True):
                # 수치형 열의 분포 시각화
                for i, col in enumerate(numeric_columns[:5]):  # 처음 5개 열만 시각화
                    plt.figure(figsize=(12, 6))
                    
                    plt.subplot(1, 2, 1)
                    sns.histplot(df[col], kde=True)
                    plt.title(f"{col} 분포")
                    
                    plt.subplot(1, 2, 2)
                    sns.boxplot(x=df[col])
                    plt.title(f"{col} 박스플롯")
                    
                    file_path = os.path.join(output_dir, f"descriptive_{col}_distribution.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "distribution",
                        "column": col,
                        "file_path": file_path
                    })
                
                # 범주형 열의 빈도 시각화
                for i, col in enumerate(categorical_columns[:5]):  # 처음 5개 열만 시각화
                    if df[col].nunique() < 15:  # 너무 많은 범주가 있는 경우 제외
                        plt.figure(figsize=(10, 6))
                        df[col].value_counts().head(10).plot(kind='bar')
                        plt.title(f"{col} 빈도")
                        plt.xticks(rotation=45)
                        
                        file_path = os.path.join(output_dir, f"descriptive_{col}_frequency.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        
                        visualizations.append({
                            "type": "frequency",
                            "column": col,
                            "file_path": file_path
                        })
            
            return {
                "success": True,
                "analysis_type": "descriptive",
                "numeric_statistics": desc_stats,
                "categorical_statistics": categorical_stats,
                "null_information": {
                    "counts": null_counts,
                    "percentages": null_percentage
                },
                "insights": insights,
                "visualizations": visualizations
            }
        
        elif analysis_type == "correlation":
            # 수치형 열 선택
            numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_columns) < 2:
                return {
                    "success": False,
                    "error": "상관 관계 분석을 위해서는 최소 2개의 수치형 열이 필요합니다."
                }
            
            # 상관 관계 계산
            correlation_method = params.get("correlation_method", "pearson")
            corr_matrix = df[numeric_columns].corr(method=correlation_method).round(3)
            
            # 인사이트 도출
            insights = []
            
            # 강한 상관 관계 찾기
            corr_threshold = params.get("correlation_threshold", 0.7)
            strong_correlations = []
            
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    col1 = numeric_columns[i]
                    col2 = numeric_columns[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_value) >= corr_threshold:
                        correlation_type = "양의" if corr_value > 0 else "음의"
                        strength = "매우 강한" if abs(corr_value) > 0.9 else "강한"
                        
                        strong_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": corr_value,
                            "type": correlation_type,
                            "strength": strength
                        })
                        
                        insights.append(f"'{col1}'와 '{col2}' 사이에 {strength} {correlation_type} 상관 관계가 있습니다 (r={corr_value:.2f})")
            
            # 시각화
            visualizations = []
            if params.get("create_visualizations", True):
                # 히트맵 생성
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title("상관 관계 히트맵")
                
                file_path = os.path.join(output_dir, f"correlation_heatmap.png")
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                
                visualizations.append({
                    "type": "heatmap",
                    "file_path": file_path
                })
                
                # 강한 상관 관계가 있는 변수쌍의 산점도
                for i, corr_info in enumerate(strong_correlations[:5]):  # 최대 5개까지만
                    col1 = corr_info["column1"]
                    col2 = corr_info["column2"]
                    
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=df[col1], y=df[col2])
                    
                    # 추세선 추가
                    sns.regplot(x=df[col1], y=df[col2], scatter=False, color='red')
                    
                    plt.title(f"{col1} vs {col2} (r={corr_info['correlation']:.2f})")
                    
                    file_path = os.path.join(output_dir, f"correlation_scatter_{col1}_{col2}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "scatter",
                        "column1": col1,
                        "column2": col2,
                        "file_path": file_path
                    })
            
            return {
                "success": True,
                "analysis_type": "correlation",
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": strong_correlations,
                "insights": insights,
                "visualizations": visualizations
            }
        
        elif analysis_type == "group":
            # 그룹화 열과 집계 열 구분
            group_by_col = params.get("group_by_column")
            if group_by_col is None or group_by_col not in df.columns:
                return {
                    "success": False,
                    "error": "유효한 그룹화 열을 지정해야 합니다."
                }
            
            # 집계할 열 (수치형) 선택
            agg_columns = [col for col in columns if col != group_by_col and pd.api.types.is_numeric_dtype(df[col])]
            
            if not agg_columns:
                return {
                    "success": False,
                    "error": "집계할 수치형 열이 없습니다."
                }
            
            # 집계 함수 설정
            agg_funcs = params.get("agg_functions", ["mean", "median", "min", "max", "count"])
            
            # 그룹별 집계
            grouped = df.groupby(group_by_col)[agg_columns].agg(agg_funcs)
            
            # 결과가 복잡한 다중 인덱스를 가질 수 있으므로 평탄화
            grouped = grouped.reset_index()
            
            # 인사이트 도출
            insights = []
            
            # 각 집계 열에 대한 인사이트
            for col in agg_columns:
                # 평균 기준 정렬
                if f"{col}_mean" in grouped.columns:
                    top_group = grouped.sort_values(f"{col}_mean", ascending=False).iloc[0]
                    bottom_group = grouped.sort_values(f"{col}_mean", ascending=False).iloc[-1]
                    
                    insights.append(f"'{col}'의 평균이 가장 높은 그룹은 '{top_group[group_by_col]}'입니다 ({top_group[f'{col}_mean']:.2f}).")
                    insights.append(f"'{col}'의 평균이 가장 낮은 그룹은 '{bottom_group[group_by_col]}'입니다 ({bottom_group[f'{col}_mean']:.2f}).")
                    
                    # 그룹 간 차이 계산
                    max_mean = top_group[f"{col}_mean"]
                    min_mean = bottom_group[f"{col}_mean"]
                    if max_mean > 0 and min_mean > 0:
                        diff_pct = ((max_mean - min_mean) / min_mean) * 100
                        insights.append(f"'{col}'의 최고 그룹과 최저 그룹 간 평균 차이는 {diff_pct:.1f}%입니다.")
                
                # 분산 또는 표준편차 계산 및 인사이트
                grouped_std = df.groupby(group_by_col)[col].std()
                if not grouped_std.empty:
                    most_variable_group = grouped_std.idxmax()
                    least_variable_group = grouped_std.idxmin()
                    
                    insights.append(f"'{col}'의 변동성이 가장 큰 그룹은 '{most_variable_group}'입니다.")
                    insights.append(f"'{col}'의 변동성이 가장 작은 그룹은 '{least_variable_group}'입니다.")
            
            # 시각화
            visualizations = []
            if params.get("create_visualizations", True):
                for i, col in enumerate(agg_columns[:3]):  # 처음 3개 열만 시각화
                    if f"{col}_mean" in grouped.columns:
                        # 그룹별 평균 시각화
                        plt.figure(figsize=(10, 6))
                        sorted_data = grouped.sort_values(f"{col}_mean", ascending=False)
                        sns.barplot(x=group_by_col, y=f"{col}_mean", data=sorted_data)
                        plt.title(f"그룹별 {col} 평균")
                        plt.xticks(rotation=45)
                        
                        file_path = os.path.join(output_dir, f"group_{group_by_col}_{col}_mean.png")
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        
                        visualizations.append({
                            "type": "bar",
                            "group_by": group_by_col,
                            "column": col,
                            "metric": "mean",
                            "file_path": file_path
                        })
                    
                    # 박스플롯으로 분포 비교
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=group_by_col, y=col, data=df)
                    plt.title(f"그룹별 {col} 분포")
                    plt.xticks(rotation=45)
                    
                    file_path = os.path.join(output_dir, f"group_{group_by_col}_{col}_boxplot.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "boxplot",
                        "group_by": group_by_col,
                        "column": col,
                        "file_path": file_path
                    })
            
            return {
                "success": True,
                "analysis_type": "group",
                "group_by_column": group_by_col,
                "aggregate_columns": agg_columns,
                "aggregate_functions": agg_funcs,
                "grouped_data": grouped.to_dict(orient='records'),
                "insights": insights,
                "visualizations": visualizations
            }
        
        elif analysis_type == "distribution":
            # 분석할 열 선택
            if not columns:
                return {
                    "success": False,
                    "error": "분석할 열을 지정해야 합니다."
                }
            
            # 수치형 열과 범주형 열 분리
            numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_columns = [col for col in columns if col not in numeric_columns]
            
            results = {"numeric": {}, "categorical": {}}
            insights = []
            visualizations = []
            
            # 수치형 열 분석
            for col in numeric_columns:
                # 기본 통계량
                stats = df[col].describe().to_dict()
                
                # 정규성 검정
                shapiro_test = None
                if df[col].count() <= 5000:  # Shapiro-Wilk 테스트는 대형 데이터셋에서 느림
                    non_null_data = df[col].dropna()
                    if len(non_null_data) >= 3:  # 최소 3개 이상의 값이 필요
                        shapiro_stat, shapiro_p = stats.shapiro(non_null_data)
                        shapiro_test = {
                            "statistic": shapiro_stat,
                            "p_value": shapiro_p,
                            "is_normal": shapiro_p > 0.05
                        }
                
                # 분포 특성
                skewness = df[col].skew()
                kurtosis = df[col].kurtosis()
                
                results["numeric"][col] = {
                    "statistics": stats,
                    "normality_test": shapiro_test,
                    "skewness": skewness,
                    "kurtosis": kurtosis
                }
                
                # 인사이트 도출
                if shapiro_test and shapiro_test["is_normal"]:
                    insights.append(f"'{col}'은(는) 정규 분포를 따르는 것으로 보입니다 (p={shapiro_test['p_value']:.3f}).")
                elif shapiro_test:
                    insights.append(f"'{col}'은(는) 정규 분포를 따르지 않는 것으로 보입니다 (p={shapiro_test['p_value']:.3f}).")
                
                if abs(skewness) > 1:
                    direction = "오른쪽" if skewness > 0 else "왼쪽"
                    insights.append(f"'{col}'은(는) {direction}으로 치우친 분포를 보입니다 (왜도: {skewness:.2f}).")
                
                if kurtosis > 1:
                    insights.append(f"'{col}'은(는) 뾰족한 분포(첨도: {kurtosis:.2f})를 보여 극단값이 많을 수 있습니다.")
                elif kurtosis < -1:
                    insights.append(f"'{col}'은(는) 평평한 분포(첨도: {kurtosis:.2f})를 보입니다.")
                
                # 시각화
                if params.get("create_visualizations", True):
                    plt.figure(figsize=(12, 8))
                    
                    plt.subplot(2, 2, 1)
                    sns.histplot(df[col], kde=True)
                    plt.title(f"{col} 히스토그램")
                    
                    plt.subplot(2, 2, 2)
                    stats.probplot(df[col].dropna(), plot=plt)
                    plt.title("Q-Q Plot")
                    
                    plt.subplot(2, 2, 3)
                    sns.boxplot(x=df[col])
                    plt.title(f"{col} 박스플롯")
                    
                    plt.subplot(2, 2, 4)
                    sns.kdeplot(df[col].dropna())
                    plt.title(f"{col} 밀도 플롯")
                    
                    file_path = os.path.join(output_dir, f"distribution_{col}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "distribution",
                        "column": col,
                        "file_path": file_path
                    })
            
            # 범주형 열 분석
            for col in categorical_columns:
                if df[col].nunique() > 100:  # 범주가 너무 많으면 분석 제외
                    continue
                
                value_counts = df[col].value_counts()
                value_percentages = df[col].value_counts(normalize=True) * 100
                
                results["categorical"][col] = {
                    "unique_values": df[col].nunique(),
                    "value_counts": value_counts.to_dict(),
                    "value_percentages": value_percentages.to_dict()
                }
                
                # 인사이트 도출
                top_category = value_counts.index[0]
                top_percentage = value_percentages.iloc[0]
                
                insights.append(f"'{col}'에서 가장 빈번한 값은 '{top_category}'로, 전체의 {top_percentage:.1f}%를 차지합니다.")
                
                if value_counts.size > 1:
                    second_category = value_counts.index[1]
                    diff_percentage = top_percentage - value_percentages.iloc[1]
                    
                    if diff_percentage > 50:
                        insights.append(f"'{col}'에서 최빈값('{top_category}')과 두 번째 빈번한 값('{second_category}') 사이에 큰 격차({diff_percentage:.1f}%p)가 있습니다.")
                
                # 균형/불균형 확인
                if df[col].nunique() > 1:
                    entropy = stats.entropy(value_percentages/100)
                    max_entropy = np.log(df[col].nunique())
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    if normalized_entropy < 0.5:
                        insights.append(f"'{col}'의 값 분포가 불균형합니다 (정규화된 엔트로피: {normalized_entropy:.2f}).")
                    elif normalized_entropy > 0.8:
                        insights.append(f"'{col}'의 값 분포가 매우 균등합니다 (정규화된 엔트로피: {normalized_entropy:.2f}).")
                
                # 시각화
                if params.get("create_visualizations", True) and df[col].nunique() <= 20:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f"{col} 범주 빈도")
                    plt.xticks(rotation=45)
                    
                    file_path = os.path.join(output_dir, f"distribution_categorical_{col}.png")
                    plt.tight_layout()
                    plt.savefig(file_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "categorical_distribution",
                        "column": col,
                        "file_path": file_path
                    })
            
            return {
                "success": True,
                "analysis_type": "distribution",
                "results": results,
                "insights": insights,
                "visualizations": visualizations
            }
        
        else:
            return {
                "success": False,
                "error": f"지원하지 않는 분석 유형입니다: {analysis_type}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"데이터 분석 중 오류 발생: {str(e)}"
        }

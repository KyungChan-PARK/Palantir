"""
데이터 분석 및 의사결정 지원 시스템
원자-분자-유기체 패턴을 따르는 데이터 분석 시스템 구현
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple

# 모듈 로드 경로에 MCP 초기화 모듈 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from analysis.mcp_init import mcp

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_analysis_system')

# 시스템 설정
DEFAULT_CONFIG = {
    "input_dir": "data",
    "output_dir": {
        "reports": "output/reports",
        "viz": "output/viz",
        "models": "output/models",
        "decisions": "output/decisions"
    },
    "max_concurrent": 5,
    "cache_enabled": True,
    "cache_dir": "temp/analysis_cache"
}

# ===== 원자 수준 도구 (Atoms) =====

@mcp.tool(
    name="load_structured_data",
    description="Load structured data from various sources",
    tags=["data", "loading"]
)
async def load_structured_data(source_path: str, format: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    다양한 형식의 구조화된 데이터를 로드합니다.
    
    Parameters:
        source_path (str): 데이터 소스 경로 (파일 경로 또는 URL)
        format (str, optional): 데이터 형식 (csv, json, excel, sql, parquet)
        options (dict, optional): 추가 옵션
        
    Returns:
        pd.DataFrame: 로드된 데이터 프레임
    """
    logger.info(f"Loading data from {source_path}")
    
    # 옵션 기본값 설정
    if options is None:
        options = {}
    
    # 파일 확장자로부터 형식 추론
    if format is None:
        if source_path.endswith('.csv'):
            format = 'csv'
        elif source_path.endswith('.json'):
            format = 'json'
        elif source_path.endswith(('.xls', '.xlsx')):
            format = 'excel'
        elif source_path.endswith('.parquet'):
            format = 'parquet'
        elif source_path.endswith('.sql'):
            format = 'sql'
        else:
            raise ValueError(f"Could not infer format from file extension: {source_path}")
    
    # 데이터 로드
    try:
        if format == 'csv':
            encoding = options.get('encoding', 'utf-8')
            sep = options.get('sep', ',')
            df = pd.read_csv(source_path, encoding=encoding, sep=sep)
        elif format == 'json':
            encoding = options.get('encoding', 'utf-8')
            df = pd.read_json(source_path, encoding=encoding)
        elif format == 'excel':
            sheet_name = options.get('sheet_name', 0)
            df = pd.read_excel(source_path, sheet_name=sheet_name)
        elif format == 'parquet':
            df = pd.read_parquet(source_path)
        elif format == 'sql':
            # SQL 연결은 실제 구현 필요
            raise NotImplementedError("SQL loading not implemented yet")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {source_path}: {e}")
        raise

@mcp.tool(
    name="preprocess_data",
    description="Preprocess structured data",
    tags=["data", "preprocessing"]
)
async def preprocess_data(data: pd.DataFrame, operations: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    데이터 전처리 작업을 수행합니다.
    
    Parameters:
        data (pd.DataFrame): 처리할 데이터 프레임
        operations (list, optional): 수행할 전처리 작업 목록
        
    Returns:
        pd.DataFrame: 처리된 데이터 프레임
    """
    logger.info("Preprocessing data")
    
    # 작업이 없는 경우 데이터 그대로 반환
    if operations is None or len(operations) == 0:
        logger.info("No preprocessing operations specified")
        return data
    
    # 원본 데이터 복사
    df = data.copy()
    
    # 각 작업 수행
    for i, op in enumerate(operations):
        op_type = op.get('type')
        
        try:
            if op_type == 'drop_na':
                # 결측치 제거
                columns = op.get('columns')
                how = op.get('how', 'any')
                
                if columns:
                    df = df.dropna(subset=columns, how=how)
                else:
                    df = df.dropna(how=how)
                    
                logger.info(f"Operation {i+1}: Dropped NA values")
                
            elif op_type == 'fill_na':
                # 결측치 채우기
                columns = op.get('columns')
                value = op.get('value')
                method = op.get('method')
                
                if columns:
                    if method:
                        df[columns] = df[columns].fillna(method=method)
                    else:
                        df[columns] = df[columns].fillna(value)
                else:
                    if method:
                        df = df.fillna(method=method)
                    else:
                        df = df.fillna(value)
                        
                logger.info(f"Operation {i+1}: Filled NA values")
                
            elif op_type == 'drop_duplicates':
                # 중복 제거
                columns = op.get('columns')
                keep = op.get('keep', 'first')
                
                if columns:
                    df = df.drop_duplicates(subset=columns, keep=keep)
                else:
                    df = df.drop_duplicates(keep=keep)
                    
                logger.info(f"Operation {i+1}: Dropped duplicates")
                
            elif op_type == 'rename_columns':
                # 열 이름 변경
                columns = op.get('columns', {})
                df = df.rename(columns=columns)
                logger.info(f"Operation {i+1}: Renamed columns")
                
            elif op_type == 'select_columns':
                # 열 선택
                columns = op.get('columns', [])
                df = df[columns]
                logger.info(f"Operation {i+1}: Selected columns")
                
            elif op_type == 'create_column':
                # 새 열 생성
                name = op.get('name')
                expression = op.get('expression')
                
                if name and expression:
                    df[name] = eval(expression)
                    logger.info(f"Operation {i+1}: Created column '{name}'")
                    
            elif op_type == 'normalize':
                # 정규화
                columns = op.get('columns', [])
                method = op.get('method', 'minmax')
                
                for col in columns:
                    if method == 'minmax':
                        min_val = df[col].min()
                        max_val = df[col].max()
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    elif method == 'zscore':
                        mean = df[col].mean()
                        std = df[col].std()
                        df[col] = (df[col] - mean) / std
                        
                logger.info(f"Operation {i+1}: Normalized columns using {method}")
                
            elif op_type == 'encode_categorical':
                # 범주형 변수 인코딩
                columns = op.get('columns', [])
                method = op.get('method', 'onehot')
                
                if method == 'onehot':
                    df = pd.get_dummies(df, columns=columns)
                elif method == 'label':
                    for col in columns:
                        df[col] = df[col].astype('category').cat.codes
                        
                logger.info(f"Operation {i+1}: Encoded categorical columns using {method}")
                
            elif op_type == 'apply_function':
                # 함수 적용
                column = op.get('column')
                function = op.get('function')
                new_column = op.get('new_column')
                
                if column and function:
                    if new_column:
                        df[new_column] = df[column].apply(eval(function))
                    else:
                        df[column] = df[column].apply(eval(function))
                        
                logger.info(f"Operation {i+1}: Applied function to column")
                
            else:
                logger.warning(f"Unknown operation type: {op_type}")
                
        except Exception as e:
            logger.error(f"Error in preprocessing operation {i+1} ({op_type}): {e}")
            raise
    
    logger.info(f"Preprocessing completed. Data shape: {df.shape}")
    return df

@mcp.tool(
    name="calculate_statistics",
    description="Calculate statistics for numerical data",
    tags=["data", "statistics"]
)
async def calculate_statistics(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    수치형 데이터에 대한 통계 계산을 수행합니다.
    
    Parameters:
        data (pd.DataFrame): 통계를 계산할 데이터 프레임
        columns (list, optional): 통계를 계산할 열 목록
        
    Returns:
        dict: 계산된 통계 정보
    """
    logger.info("Calculating statistics")
    
    # 분석할 열 선택
    if columns is None:
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    else:
        numeric_columns = [col for col in columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
    
    if not numeric_columns:
        logger.warning("No numeric columns found for statistics calculation")
        return {}
    
    # 기본 통계 계산
    stats = {}
    
    # 전체 통계
    try:
        desc_stats = data[numeric_columns].describe().to_dict()
        stats['descriptive'] = desc_stats
        logger.info("Calculated descriptive statistics")
    except Exception as e:
        logger.error(f"Error calculating descriptive statistics: {e}")
        stats['descriptive'] = {}
    
    # 상관관계
    try:
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr().to_dict()
            stats['correlation'] = corr_matrix
            logger.info("Calculated correlation matrix")
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        stats['correlation'] = {}
    
    # 각 열별 통계
    column_stats = {}
    for col in numeric_columns:
        try:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                logger.warning(f"Column {col} has no non-NA values")
                continue
                
            col_stats = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'skew': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'count': int(col_data.count()),
                'missing': int(data[col].isna().sum()),
                'missing_pct': float(data[col].isna().mean() * 100)
            }
            
            # 이상치 탐지
            iqr = col_stats['iqr']
            q1 = col_stats['q1']
            q3 = col_stats['q3']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            col_stats['outliers_count'] = int(((col_data < lower_bound) | (col_data > upper_bound)).sum())
            col_stats['outliers_pct'] = float(((col_data < lower_bound) | (col_data > upper_bound)).mean() * 100)
            
            column_stats[col] = col_stats
            logger.info(f"Calculated statistics for column: {col}")
            
        except Exception as e:
            logger.error(f"Error calculating statistics for column {col}: {e}")
            column_stats[col] = {"error": str(e)}
    
    stats['columns'] = column_stats
    
    logger.info(f"Statistics calculation completed for {len(numeric_columns)} columns")
    return stats

@mcp.tool(
    name="generate_visualizations",
    description="Generate visualizations for data",
    tags=["data", "visualization"]
)
async def generate_visualizations(data: pd.DataFrame, statistics: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    데이터 시각화를 생성합니다.
    
    Parameters:
        data (pd.DataFrame): 시각화할 데이터 프레임
        statistics (dict, optional): 통계 정보
        options (dict, optional): 시각화 옵션
        
    Returns:
        dict: 생성된 시각화 파일 경로
    """
    logger.info("Generating visualizations")
    
    # 옵션 기본값 설정
    if options is None:
        options = {}
    
    output_dir = options.get('output_dir', 'output/viz')
    prefix = options.get('prefix', f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    format = options.get('format', 'png')
    dpi = options.get('dpi', 100)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 시각화 파일 경로 저장
    viz_paths = {}
    
    # 데이터 및 통계 확인
    if data.empty:
        logger.warning("Data is empty, cannot generate visualizations")
        return viz_paths
    
    # 수치형 열 선택
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        logger.warning("No numeric columns found for visualization")
        return viz_paths
    
    try:
        # 1. 상관관계 히트맵
        if len(numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = data[numeric_columns].corr()
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
            plt.title('Correlation Matrix Heatmap')
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, f"{prefix}_correlation_heatmap.{format}")
            plt.savefig(file_path, dpi=dpi)
            plt.close()
            
            viz_paths['correlation_heatmap'] = file_path
            logger.info(f"Generated correlation heatmap: {file_path}")
        
        # 2. 주요 수치형 열에 대한 히스토그램 및 박스플롯
        for i, col in enumerate(numeric_columns[:min(5, len(numeric_columns))]):
            # 히스토그램
            plt.figure(figsize=(10, 6))
            plt.hist(data[col].dropna(), bins=30, alpha=0.7, color='skyblue')
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, f"{prefix}_histogram_{col}.{format}")
            plt.savefig(file_path, dpi=dpi)
            plt.close()
            
            viz_paths[f'histogram_{col}'] = file_path
            logger.info(f"Generated histogram for {col}: {file_path}")
            
            # 박스플롯
            plt.figure(figsize=(10, 6))
            plt.boxplot(data[col].dropna(), vert=False)
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, f"{prefix}_boxplot_{col}.{format}")
            plt.savefig(file_path, dpi=dpi)
            plt.close()
            
            viz_paths[f'boxplot_{col}'] = file_path
            logger.info(f"Generated boxplot for {col}: {file_path}")
        
        # 3. 산점도 행렬 (처음 4개 열)
        if len(numeric_columns) > 1:
            cols_to_plot = numeric_columns[:min(4, len(numeric_columns))]
            n = len(cols_to_plot)
            
            fig, axes = plt.subplots(n, n, figsize=(12, 12))
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            
            for i in range(n):
                for j in range(n):
                    ax = axes[i, j]
                    
                    if i == j:  # 대각선: 히스토그램
                        ax.hist(data[cols_to_plot[i]].dropna(), bins=20, alpha=0.7, color='skyblue')
                        ax.set_title(cols_to_plot[i])
                    else:  # 비대각선: 산점도
                        ax.scatter(data[cols_to_plot[j]], data[cols_to_plot[i]], alpha=0.5, s=10)
                        
                    if i == n-1:  # 마지막 행: x축 레이블
                        ax.set_xlabel(cols_to_plot[j])
                    if j == 0:  # 첫 번째 열: y축 레이블
                        ax.set_ylabel(cols_to_plot[i])
            
            plt.suptitle('Scatter Plot Matrix', y=0.95, fontsize=16)
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, f"{prefix}_scatter_matrix.{format}")
            plt.savefig(file_path, dpi=dpi)
            plt.close()
            
            viz_paths['scatter_matrix'] = file_path
            logger.info(f"Generated scatter plot matrix: {file_path}")
        
        # 4. 결측치 시각화
        na_counts = data.isna().sum()
        if na_counts.sum() > 0:
            plt.figure(figsize=(12, 6))
            ax = na_counts.plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.xlabel('Column')
            plt.ylabel('Count of Missing Values')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, f"{prefix}_missing_values.{format}")
            plt.savefig(file_path, dpi=dpi)
            plt.close()
            
            viz_paths['missing_values'] = file_path
            logger.info(f"Generated missing values bar chart: {file_path}")
        
        logger.info(f"Visualization generation completed. Generated {len(viz_paths)} visualizations")
        return viz_paths
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise

@mcp.tool(
    name="generate_report",
    description="Generate EDA report",
    tags=["data", "reporting"]
)
async def generate_report(data: pd.DataFrame, statistics: Dict[str, Any], visualizations: Dict[str, str], options: Optional[Dict[str, Any]] = None) -> str:
    """
    탐색적 데이터 분석 보고서를 생성합니다.
    
    Parameters:
        data (pd.DataFrame): 분석된 데이터 프레임
        statistics (dict): 계산된 통계 정보
        visualizations (dict): 생성된 시각화 파일 경로
        options (dict, optional): 보고서 생성 옵션
        
    Returns:
        str: 생성된 보고서 파일 경로
    """
    logger.info("Generating EDA report")
    
    # 옵션 기본값 설정
    if options is None:
        options = {}
    
    output_dir = options.get('output_dir', 'output/reports')
    prefix = options.get('prefix', f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    format = options.get('format', 'html')
    include_code = options.get('include_code', False)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 보고서 파일 경로
    file_path = os.path.join(output_dir, f"{prefix}.{format}")
    
    try:
        # HTML 보고서 생성
        if format == 'html':
            with open(file_path, 'w', encoding='utf-8') as f:
                # HTML 헤더
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exploratory Data Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 12px; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .visualization {{ margin: 20px 0; text-align: center; }}
        .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .stat-value {{ font-weight: bold; }}
        .section {{ margin-bottom: 30px; }}
        .code-block {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Exploratory Data Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")
                
                # 데이터 개요 섹션
                f.write(f"""
        <div class="section">
            <h2>Data Overview</h2>
            <p>Total Rows: {data.shape[0]}</p>
            <p>Total Columns: {data.shape[1]}</p>
            <p>Memory Usage: {data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB</p>
            
            <h3>Column Information</h3>
            <table>
                <tr>
                    <th>Column Name</th>
                    <th>Data Type</th>
                    <th>Non-Null Count</th>
                    <th>Missing Count</th>
                    <th>Missing Percentage</th>
                </tr>
""")

                # 열 정보 테이블
                for col in data.columns:
                    dtype = str(data[col].dtype)
                    non_null = data[col].count()
                    missing = data[col].isna().sum()
                    missing_pct = missing / len(data) * 100 if len(data) > 0 else 0
                    
                    f.write(f"""
                <tr>
                    <td>{col}</td>
                    <td>{dtype}</td>
                    <td>{non_null}</td>
                    <td>{missing}</td>
                    <td>{missing_pct:.2f}%</td>
                </tr>""")
                
                f.write("""
            </table>
        </div>
""")
                
                # 통계 정보 섹션
                f.write("""
        <div class="section">
            <h2>Statistical Analysis</h2>
""")
                
                # 열별 통계
                if 'columns' in statistics:
                    f.write("""
            <h3>Column Statistics</h3>
""")
                    for col, stats in statistics['columns'].items():
                        f.write(f"""
            <h4>{col}</h4>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td class="stat-value">{stats.get('mean', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Median</td>
                    <td class="stat-value">{stats.get('median', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td class="stat-value">{stats.get('std', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Minimum</td>
                    <td class="stat-value">{stats.get('min', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Maximum</td>
                    <td class="stat-value">{stats.get('max', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Q1 (25th Percentile)</td>
                    <td class="stat-value">{stats.get('q1', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Q3 (75th Percentile)</td>
                    <td class="stat-value">{stats.get('q3', 'N/A')}</td>
                </tr>
                <tr>
                    <td>IQR</td>
                    <td class="stat-value">{stats.get('iqr', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Skewness</td>
                    <td class="stat-value">{stats.get('skew', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Kurtosis</td>
                    <td class="stat-value">{stats.get('kurtosis', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Count (Non-null)</td>
                    <td class="stat-value">{stats.get('count', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Missing Values</td>
                    <td class="stat-value">{stats.get('missing', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Missing Percentage</td>
                    <td class="stat-value">{stats.get('missing_pct', 'N/A'):.2f}%</td>
                </tr>
                <tr>
                    <td>Outliers Count</td>
                    <td class="stat-value">{stats.get('outliers_count', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Outliers Percentage</td>
                    <td class="stat-value">{stats.get('outliers_pct', 'N/A'):.2f}%</td>
                </tr>
            </table>
""")
                
                f.write("""
        </div>
""")
                
                # 시각화 섹션
                f.write("""
        <div class="section">
            <h2>Visualizations</h2>
""")
                
                # 각 시각화 이미지 포함
                for viz_name, viz_path in visualizations.items():
                    title = ' '.join(viz_name.split('_')).title()
                    f.write(f"""
            <div class="visualization">
                <h3>{title}</h3>
                <img src="../{viz_path}" alt="{title}">
            </div>
""")
                
                f.write("""
        </div>
""")
                
                # 결론 및 요약 섹션
                f.write("""
        <div class="section">
            <h2>Conclusions and Summary</h2>
            <p>This report presents a comprehensive exploratory data analysis of the provided dataset.</p>
            <p>Key observations:</p>
            <ul>
""")
                
                # 데이터 품질 관련 결론
                missing_cols = [col for col in data.columns if data[col].isna().sum() > 0]
                if missing_cols:
                    missing_pct = data[missing_cols].isna().mean().mean() * 100
                    f.write(f"""
                <li>{len(missing_cols)} columns have missing values, with an average of {missing_pct:.2f}% missing data.</li>
""")
                else:
                    f.write("""
                <li>The dataset has no missing values.</li>
""")
                
                # 수치형 열 관련 결론
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    f.write(f"""
                <li>The dataset contains {len(numeric_cols)} numeric columns suitable for statistical analysis.</li>
""")
                    
                    # 이상치 관련 결론
                    if 'columns' in statistics:
                        outlier_cols = [col for col, stats in statistics['columns'].items() 
                                       if 'outliers_pct' in stats and stats['outliers_pct'] > 5]
                        if outlier_cols:
                            f.write(f"""
                <li>{len(outlier_cols)} columns have significant outliers (>5% of data).</li>
""")
                
                # 상관관계 관련 결론
                if 'correlation' in statistics and numeric_cols and len(numeric_cols) > 1:
                    corr_matrix = pd.DataFrame(statistics['correlation'])
                    high_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                    
                    if high_corr:
                        f.write(f"""
                <li>Found {len(high_corr)} pairs of highly correlated features (|r| > 0.7).</li>
""")
                
                f.write("""
            </ul>
            <p>Recommendations for further analysis:</p>
            <ul>
                <li>Address missing values through imputation or removal based on analysis goals.</li>
                <li>Consider treating outliers depending on their nature (data errors vs. valid extreme values).</li>
                <li>For modeling, be aware of multicollinearity in highly correlated features.</li>
                <li>Feature engineering may help extract more value from the existing data.</li>
            </ul>
        </div>
""")
                
                # HTML 푸터
                f.write("""
    </div>
</body>
</html>
""")
            
            logger.info(f"Generated HTML report: {file_path}")
        
        # 다른 형식의 보고서는 필요에 따라 추가 구현
        else:
            logger.error(f"Unsupported report format: {format}")
            raise ValueError(f"Unsupported report format: {format}")
        
        return file_path
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise

@mcp.tool(
    name="train_model",
    description="Train a predictive model",
    tags=["data", "modeling"]
)
async def train_model(data: pd.DataFrame, target: str, features: Optional[List[str]] = None, model_type: str = 'linear_regression', options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    예측 모델을 학습합니다.
    
    Parameters:
        data (pd.DataFrame): 학습 데이터 프레임
        target (str): 예측 대상 열
        features (list, optional): 사용할 특성 열 목록
        model_type (str): 모델 유형 (linear_regression, random_forest, decision_tree)
        options (dict, optional): 추가 모델링 옵션
        
    Returns:
        dict: 학습된 모델 정보
    """
    logger.info(f"Training {model_type} model")
    
    try:
        # 필요한 패키지 가져오기
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import joblib
        
        # 옵션 기본값 설정
        if options is None:
            options = {}
        
        test_size = options.get('test_size', 0.2)
        random_state = options.get('random_state', 42)
        output_dir = options.get('output_dir', 'output/models')
        prefix = options.get('prefix', f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 대상 데이터 확인
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # 특성 선택
        if features is None:
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            features = [col for col in numeric_cols if col != target]
        else:
            # 데이터에 존재하는 특성만 선택
            features = [col for col in features if col in data.columns]
        
        if not features:
            raise ValueError("No valid feature columns found")
        
        # 누락된 데이터 처리
        data_clean = data[features + [target]].dropna()
        if len(data_clean) < len(data):
            logger.warning(f"Dropped {len(data) - len(data_clean)} rows with missing values")
        
        # 훈련/테스트 분할
        X = data_clean[features]
        y = data_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # 모델 선택 및 학습
        if model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            n_estimators = options.get('n_estimators', 100)
            max_depth = options.get('max_depth', None)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeRegressor
            max_depth = options.get('max_depth', None)
            
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=random_state
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 평가 지표 계산
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'mse': mean_squared_error(y_train, y_pred_train),
                'rmse': mean_squared_error(y_train, y_pred_train, squared=False),
                'mae': mean_absolute_error(y_train, y_pred_train)
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': mean_squared_error(y_test, y_pred_test, squared=False),
                'mae': mean_absolute_error(y_test, y_pred_test)
            }
        }
        
        # 모델 정보
        model_info = {
            'type': model_type,
            'features': features,
            'target': target,
            'metrics': metrics,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 특성 중요도 (해당하는 경우)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(features, importances))
            model_info['feature_importance'] = feature_importance
        
        # 계수 (선형 회귀인 경우)
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            intercept = model.intercept_
            
            coef_dict = dict(zip(features, coefficients))
            coef_dict['intercept'] = intercept
            
            model_info['coefficients'] = coef_dict
        
        # 모델 저장
        model_path = os.path.join(output_dir, f"{prefix}_model.joblib")
        joblib.dump(model, model_path)
        model_info['model_path'] = model_path
        
        # 모델 정보 저장
        info_path = os.path.join(output_dir, f"{prefix}_info.json")
        with open(info_path, 'w') as f:
            # NumPy 값을 float로 변환
            info_json = json.dumps(model_info, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x, indent=4)
            f.write(info_json)
            
        model_info['info_path'] = info_path
        
        logger.info(f"Model training completed. Model saved to {model_path}")
        logger.info(f"Test set metrics - R2: {metrics['test']['r2']:.4f}, RMSE: {metrics['test']['rmse']:.4f}")
        
        return model_info
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

@mcp.tool(
    name="generate_predictions",
    description="Generate predictions from a trained model",
    tags=["data", "prediction"]
)
async def generate_predictions(model_info: Dict[str, Any], data: pd.DataFrame, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    학습된 모델을 사용하여 예측을 생성합니다.
    
    Parameters:
        model_info (dict): 학습된 모델 정보
        data (pd.DataFrame): 예측할 데이터 프레임
        options (dict, optional): 추가 예측 옵션
        
    Returns:
        dict: 예측 결과
    """
    logger.info("Generating predictions")
    
    try:
        # 필요한 패키지 가져오기
        import joblib
        
        # 옵션 기본값 설정
        if options is None:
            options = {}
        
        output_dir = options.get('output_dir', 'output/predictions')
        prefix = options.get('prefix', f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 로드
        model_path = model_info.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
            
        model = joblib.load(model_path)
        
        # 특성 확인
        features = model_info.get('features', [])
        if not features:
            raise ValueError("No feature information found in model_info")
            
        # 데이터에 모든 특성이 있는지 확인
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Data is missing required features: {missing_features}")
            
        # 예측용 데이터 준비
        X = data[features].copy()
        
        # 결측치 처리
        missing_rows = X.isna().any(axis=1)
        if missing_rows.any():
            logger.warning(f"Data contains {missing_rows.sum()} rows with missing values")
            X = X.dropna()
            
        # 예측 수행
        predictions = model.predict(X)
        
        # 결과 데이터프레임 생성
        result_df = data.copy()
        target = model_info.get('target', 'prediction')
        result_df[f'{target}_pred'] = np.nan  # 초기화
        
        # 결측치가 없는 행에만 예측값 할당
        result_df.loc[~missing_rows, f'{target}_pred'] = predictions
        
        # 예측 결과 저장
        output_path = os.path.join(output_dir, f"{prefix}_predictions.csv")
        result_df.to_csv(output_path, index=False)
        
        # 결과 요약
        prediction_results = {
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'prediction_count': len(predictions),
            'output_path': output_path,
            'target_column': f'{target}_pred',
            'model_type': model_info.get('type', 'unknown'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 기본 통계 추가
        if len(predictions) > 0:
            prediction_results['stats'] = {
                'mean': float(np.mean(predictions)),
                'median': float(np.median(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        
        logger.info(f"Generated predictions for {len(predictions)} samples")
        logger.info(f"Predictions saved to {output_path}")
        
        return prediction_results
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise

@mcp.tool(
    name="generate_insights",
    description="Generate insights from analysis results",
    tags=["data", "insights"]
)
async def generate_insights(analysis_results: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None, query: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    분석 결과와 예측에서 인사이트를 생성합니다.
    
    Parameters:
        analysis_results (dict): 데이터 분석 결과
        predictions (dict, optional): 예측 결과
        query (str, optional): 사용자 쿼리 또는 질문
        context (dict, optional): 추가 컨텍스트 정보
        
    Returns:
        dict: 생성된 인사이트
    """
    logger.info("Generating insights from analysis results")
    
    insights = {
        'summary': [],
        'key_findings': [],
        'recommendations': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # 분석 결과에서 데이터 추출
        statistics = analysis_results.get('statistics', {})
        column_stats = statistics.get('columns', {})
        
        # 1. 데이터 품질 인사이트
        data_quality_issues = []
        
        # 결측치 확인
        columns_with_missing = []
        for col, stats in column_stats.items():
            missing_pct = stats.get('missing_pct', 0)
            if missing_pct > 0:
                columns_with_missing.append((col, missing_pct))
        
        if columns_with_missing:
            columns_with_missing.sort(key=lambda x: x[1], reverse=True)
            top_missing = columns_with_missing[:3]
            
            data_quality_issues.append(f"Missing values detected in {len(columns_with_missing)} columns. "
                                      f"Top columns: " + ", ".join([f"{col} ({pct:.1f}%)" for col, pct in top_missing]))
            
            # 결측치 처리 권장사항
            if any(pct > 20 for _, pct in columns_with_missing):
                insights['recommendations'].append("Consider addressing columns with >20% missing values - either through imputation or by dropping these columns based on their importance.")
        
        # 이상치 확인
        columns_with_outliers = []
        for col, stats in column_stats.items():
            outlier_pct = stats.get('outliers_pct', 0)
            if outlier_pct > 5:  # 5% 이상이면 중요한 이상치로 간주
                columns_with_outliers.append((col, outlier_pct))
        
        if columns_with_outliers:
            columns_with_outliers.sort(key=lambda x: x[1], reverse=True)
            top_outliers = columns_with_outliers[:3]
            
            data_quality_issues.append(f"Significant outliers detected in {len(columns_with_outliers)} columns. "
                                      f"Top columns: " + ", ".join([f"{col} ({pct:.1f}%)" for col, pct in top_outliers]))
            
            # 이상치 처리 권장사항
            insights['recommendations'].append("Investigate outliers in key variables to determine if they represent valid extreme values or potential data errors.")
        
        if data_quality_issues:
            insights['key_findings'].extend(data_quality_issues)
        else:
            insights['key_findings'].append("Data quality is good with minimal missing values and outliers.")
        
        # 2. 변수 분포 인사이트
        distribution_insights = []
        
        for col, stats in column_stats.items():
            # 비대칭성 확인
            skew = stats.get('skew', 0)
            if abs(skew) > 1:
                skew_direction = "right" if skew > 0 else "left"
                distribution_insights.append(f"{col} has a {skew_direction}-skewed distribution (skewness: {skew:.2f}).")
                
                # 심한 비대칭에 대한 권장사항
                if abs(skew) > 2:
                    insights['recommendations'].append(f"Consider applying a transformation (e.g., log, square root) to {col} to address strong skewness.")
            
            # 첨도 확인
            kurtosis = stats.get('kurtosis', 0)
            if abs(kurtosis) > 3:
                kurt_type = "heavy-tailed" if kurtosis > 0 else "light-tailed"
                distribution_insights.append(f"{col} has a {kurt_type} distribution (kurtosis: {kurtosis:.2f}).")
        
        if distribution_insights:
            insights['key_findings'].extend(distribution_insights[:3])  # 상위 3개만 포함
        
        # 3. 상관관계 인사이트
        if 'correlation' in statistics:
            corr_matrix = statistics['correlation']
            
            # 강한 상관관계 식별
            strong_correlations = []
            for col1 in corr_matrix:
                for col2 in corr_matrix[col1]:
                    if col1 != col2:
                        corr = corr_matrix[col1][col2]
                        if abs(corr) > 0.7:  # 강한 상관관계 임계값
                            strong_correlations.append((col1, col2, corr))
            
            if strong_correlations:
                # 중복 제거 (A-B와 B-A는 동일한 상관관계)
                unique_correlations = []
                pairs_seen = set()
                
                for col1, col2, corr in strong_correlations:
                    pair = tuple(sorted([col1, col2]))
                    if pair not in pairs_seen:
                        pairs_seen.add(pair)
                        unique_correlations.append((col1, col2, corr))
                
                # 상위 상관관계 보고
                top_correlations = sorted(unique_correlations, key=lambda x: abs(x[2]), reverse=True)[:3]
                
                corr_findings = [f"Strong {'' if c[2] > 0 else 'negative '}correlation detected between {c[0]} and {c[1]} (r = {c[2]:.2f})." 
                                for c in top_correlations]
                
                insights['key_findings'].extend(corr_findings)
                
                # 다중공선성 관련 권장사항
                if len(unique_correlations) > 2:
                    insights['recommendations'].append("Be aware of multicollinearity in modeling - consider dimension reduction techniques or careful feature selection.")
        
        # 4. 예측 결과 인사이트 (제공된 경우)
        if predictions:
            prediction_stats = predictions.get('stats', {})
            model_type = predictions.get('model_type', 'unknown')
            
            if prediction_stats:
                pred_mean = prediction_stats.get('mean')
                pred_min = prediction_stats.get('min')
                pred_max = prediction_stats.get('max')
                
                if all(x is not None for x in [pred_mean, pred_min, pred_max]):
                    insights['key_findings'].append(f"Predictions range from {pred_min:.2f} to {pred_max:.2f}, with an average of {pred_mean:.2f}.")
            
            # 모델 성능 인사이트
            if 'model_info' in analysis_results:
                model_info = analysis_results['model_info']
                metrics = model_info.get('metrics', {}).get('test', {})
                
                if metrics:
                    r2 = metrics.get('r2')
                    rmse = metrics.get('rmse')
                    
                    if r2 is not None:
                        performance_level = "strong" if r2 > 0.7 else "moderate" if r2 > 0.5 else "weak"
                        insights['key_findings'].append(f"The {model_type} model shows {performance_level} predictive performance with R² = {r2:.2f}.")
                    
                    if rmse is not None:
                        insights['key_findings'].append(f"Model prediction error (RMSE) is {rmse:.2f}.")
                
                # 특성 중요도 인사이트
                feature_importance = model_info.get('feature_importance', {})
                if feature_importance:
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    top_features = sorted_features[:3]
                    
                    features_str = ", ".join([f"{feature} ({importance:.3f})" for feature, importance in top_features])
                    insights['key_findings'].append(f"Top predictive features: {features_str}.")
                    
                    # 모델 개선 권장사항
                    insights['recommendations'].append("Focus on the top predictive features for future data collection and model refinement.")
        
        # 5. 요약 생성
        if insights['key_findings']:
            # 데이터 품질 요약
            quality_summary = "The dataset has good quality." if not data_quality_issues else "There are some data quality issues that should be addressed."
            
            # 관계 요약
            relationship_summary = ""
            if 'correlation' in statistics and strong_correlations:
                relationship_summary = " Several strong relationships were identified between variables."
            
            # 예측 요약
            prediction_summary = ""
            if predictions and 'model_info' in analysis_results:
                r2 = analysis_results['model_info'].get('metrics', {}).get('test', {}).get('r2')
                if r2 is not None:
                    prediction_summary = f" The predictive model demonstrates {'strong' if r2 > 0.7 else 'moderate' if r2 > 0.5 else 'limited'} ability to explain the target variable."
            
            # 종합 요약
            summary = f"{quality_summary}{relationship_summary}{prediction_summary}"
            insights['summary'].append(summary)
            
            # 추가 요약: 분석의 주요 결과
            key_insight = insights['key_findings'][0] if insights['key_findings'] else ""
            insights['summary'].append(f"Key insight: {key_insight}")
        
        # 6. 추가 맥락별 인사이트 (제공된 경우)
        if query:
            # 쿼리 기반 인사이트 생성 로직
            query_lower = query.lower()
            
            # 특정 변수에 관한 질문
            for col in column_stats:
                if col.lower() in query_lower:
                    stats = column_stats[col]
                    col_insight = f"Regarding {col}: "
                    
                    col_insight += f"average is {stats.get('mean', 'N/A'):.2f}, "
                    col_insight += f"ranges from {stats.get('min', 'N/A'):.2f} to {stats.get('max', 'N/A'):.2f}, "
                    
                    if stats.get('missing_pct', 0) > 0:
                        col_insight += f"has {stats.get('missing_pct', 0):.1f}% missing values, "
                    
                    if stats.get('outliers_pct', 0) > 0:
                        col_insight += f"contains {stats.get('outliers_pct', 0):.1f}% outliers."
                    else:
                        col_insight += "has no significant outliers."
                    
                    insights['key_findings'].append(col_insight)
                    break
            
            # 예측에 관한 질문
            prediction_terms = ['predict', 'forecast', 'future', 'expected', 'projection']
            if any(term in query_lower for term in prediction_terms) and predictions:
                prediction_insight = "Based on the predictions: "
                
                pred_stats = predictions.get('stats', {})
                if pred_stats:
                    prediction_insight += f"expected average is {pred_stats.get('mean', 'N/A'):.2f}, "
                    prediction_insight += f"with values likely ranging from {pred_stats.get('min', 'N/A'):.2f} to {pred_stats.get('max', 'N/A'):.2f}."
                    
                    insights['key_findings'].append(prediction_insight)
        
        # 최소 권장사항 보장
        if not insights['recommendations']:
            insights['recommendations'] = [
                "Consider further exploratory analysis to identify patterns and relationships in the data.",
                "Evaluate the need for feature engineering to improve model performance.",
                "Monitor data quality over time to ensure consistent analysis results."
            ]
        
        logger.info(f"Generated {len(insights['key_findings'])} key findings and {len(insights['recommendations'])} recommendations")
        return insights
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        # 오류 발생 시 기본 인사이트 반환
        return {
            'summary': ["Unable to generate complete insights due to an error."],
            'key_findings': ["Error encountered during insight generation."],
            'recommendations': ["Review the analysis process and data for potential issues."],
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ===== 분자 수준 워크플로우 (Molecules) =====

@mcp.workflow(
    name="exploratory_data_analysis",
    description="Perform exploratory data analysis"
)
async def exploratory_data_analysis(data_source: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    탐색적 데이터 분석을 수행하고 보고서를 생성합니다.
    
    Parameters:
        data_source (str): 분석할 데이터 소스 경로
        options (dict, optional): 분석 옵션
        
    Returns:
        dict: 분석 결과
    """
    logger.info(f"Starting exploratory data analysis for {data_source}")
    
    # 옵션 기본값 설정
    if options is None:
        options = {}
    
    try:
        # 1. 데이터 로드
        data_format = options.get('format')
        data_load_options = options.get('load_options', {})
        
        data = await load_structured_data(data_source, format=data_format, options=data_load_options)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # 2. 데이터 전처리
        preprocess_operations = options.get('preprocess_operations', [])
        processed_data = await preprocess_data(data, operations=preprocess_operations)
        logger.info(f"Preprocessed data. New shape: {processed_data.shape}")
        
        # 3. 통계 계산
        stat_columns = options.get('stat_columns')
        stats = await calculate_statistics(processed_data, columns=stat_columns)
        logger.info("Calculated statistics")
        
        # 4. 시각화 생성
        viz_options = options.get('visualization_options', {})
        visualizations = await generate_visualizations(processed_data, stats, options=viz_options)
        logger.info(f"Generated {len(visualizations)} visualizations")
        
        # 5. 보고서 생성
        report_options = options.get('report_options', {})
        report = await generate_report(processed_data, stats, visualizations, options=report_options)
        logger.info(f"Generated report: {report}")
        
        # 결과 반환
        return {
            "data": processed_data,
            "statistics": stats,
            "visualizations": visualizations,
            "report": report
        }
    except Exception as e:
        logger.error(f"Error in exploratory_data_analysis: {e}")
        raise

@mcp.workflow(
    name="predictive_modeling",
    description="Perform predictive modeling"
)
async def predictive_modeling(data_source: str, target: str, features: Optional[List[str]] = None, model_type: str = 'linear_regression', options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    예측 모델링을 수행합니다.
    
    Parameters:
        data_source (str): 분석할 데이터 소스 경로
        target (str): 예측 대상 열
        features (list, optional): 사용할 특성 열 목록
        model_type (str): 모델 유형
        options (dict, optional): 모델링 옵션
        
    Returns:
        dict: 모델링 결과
    """
    logger.info(f"Starting predictive modeling for {data_source}, target: {target}")
    
    # 옵션 기본값 설정
    if options is None:
        options = {}
    
    try:
        # 1. 데이터 로드 및 전처리
        data_load_options = options.get('load_options', {})
        data = await load_structured_data(data_source, options=data_load_options)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        preprocess_operations = options.get('preprocess_operations', [])
        processed_data = await preprocess_data(data, operations=preprocess_operations)
        logger.info(f"Preprocessed data. New shape: {processed_data.shape}")
        
        # 2. 모델 학습
        model_options = options.get('model_options', {})
        model_info = await train_model(processed_data, target, features, model_type, options=model_options)
        logger.info(f"Trained {model_type} model. Test R²: {model_info['metrics']['test']['r2']:.4f}")
        
        # 3. 예측 생성
        prediction_options = options.get('prediction_options', {})
        predictions = await generate_predictions(model_info, processed_data, options=prediction_options)
        logger.info(f"Generated predictions for {predictions['prediction_count']} samples")
        
        # 결과 반환
        return {
            "data": processed_data,
            "model_info": model_info,
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error in predictive_modeling: {e}")
        raise

# ===== 유기체 수준 시스템 (Organisms) =====

@mcp.system(
    name="decision_support_system",
    description="Generate insights and recommendations from data"
)
async def decision_support_system(data_source: str, query: Optional[str] = None, context: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    데이터 분석을 수행하고 의사결정을 지원하는 인사이트를 생성합니다.
    
    Parameters:
        data_source (str): 분석할 데이터 소스 경로
        query (str, optional): 사용자 쿼리 또는 질문
        context (dict, optional): 추가 컨텍스트 정보
        options (dict, optional): 분석 옵션
        
    Returns:
        dict: 의사결정 지원 결과
    """
    logger.info(f"Starting decision support system for {data_source}")
    
    # 옵션 기본값 설정
    if options is None:
        options = {}
    
    output_dir = options.get('output_dir', 'output/decisions')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 탐색적 데이터 분석 수행
        eda_options = options.get('eda_options', {})
        eda_results = await exploratory_data_analysis(data_source, options=eda_options)
        logger.info("Completed exploratory data analysis")
        
        # 2. 예측 모델링 수행 (필요한 경우)
        model_results = None
        if options.get('perform_modeling', True):
            target = options.get('target')
            features = options.get('features')
            model_type = options.get('model_type', 'linear_regression')
            
            if target:
                modeling_options = options.get('modeling_options', {})
                model_results = await predictive_modeling(data_source, target, features, model_type, options=modeling_options)
                logger.info(f"Completed predictive modeling with {model_type}")
                
                # 모델 결과를 EDA 결과에 추가
                eda_results['model_info'] = model_results['model_info']
                eda_results['predictions'] = model_results['predictions']
            else:
                logger.warning("No target specified for modeling, skipping")
        
        # 3. 인사이트 생성
        predictions = model_results['predictions'] if model_results else None
        insights = await generate_insights(eda_results, predictions, query, context)
        logger.info("Generated insights")
        
        # 4. 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f"decision_support_{timestamp}.json")
        
        # 결과 객체 생성
        result = {
            "query": query,
            "insights": insights,
            "report_path": eda_results.get('report'),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # JSON으로 저장 (DataFrame 제외)
        with open(result_file, 'w', encoding='utf-8') as f:
            # NumPy 값을 float로 변환
            result_json = json.dumps(result, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x, indent=4)
            f.write(result_json)
            
        logger.info(f"Decision support results saved to {result_file}")
        
        return result
    except Exception as e:
        logger.error(f"Error in decision_support_system: {e}")
        raise

# 시스템 테스트 함수
async def test_system():
    """
    데이터 분석 시스템 테스트
    """
    logger.info("Testing data analysis system")
    
    try:
        # 테스트 데이터 생성
        test_data_path = "temp/test_data.csv"
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        
        # 간단한 테스트 데이터셋 생성
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 2, 3, 4, 5],
            'target': [10, 8, 6, 4, 2]
        })
        
        data.to_csv(test_data_path, index=False)
        logger.info(f"Created test data at {test_data_path}")
        
        # 1. 데이터 로드 테스트
        loaded_data = await load_structured_data(test_data_path)
        logger.info("SUCCESS: load_structured_data test passed")
        
        # 2. 전처리 테스트
        operations = [
            {'type': 'create_column', 'name': 'D', 'expression': 'data["A"] + data["B"]'}
        ]
        processed_data = await preprocess_data(loaded_data, operations)
        logger.info("SUCCESS: preprocess_data test passed")
        
        # 3. 통계 계산 테스트
        stats = await calculate_statistics(processed_data)
        logger.info("SUCCESS: calculate_statistics test passed")
        
        # 4. 시각화 생성 테스트
        viz_options = {'output_dir': 'temp/test_viz', 'prefix': 'test'}
        os.makedirs(viz_options['output_dir'], exist_ok=True)
        viz = await generate_visualizations(processed_data, stats, viz_options)
        logger.info("SUCCESS: generate_visualizations test passed")
        
        # 5. 보고서 생성 테스트
        report_options = {'output_dir': 'temp/test_reports', 'prefix': 'test'}
        os.makedirs(report_options['output_dir'], exist_ok=True)
        report = await generate_report(processed_data, stats, viz, report_options)
        logger.info("SUCCESS: generate_report test passed")
        
        # 6. 모델 학습 테스트
        model_options = {'output_dir': 'temp/test_models', 'prefix': 'test'}
        os.makedirs(model_options['output_dir'], exist_ok=True)
        model_info = await train_model(processed_data, 'target', ['A', 'B', 'C'], 'linear_regression', model_options)
        logger.info("SUCCESS: train_model test passed")
        
        # 7. 예측 생성 테스트
        pred_options = {'output_dir': 'temp/test_predictions', 'prefix': 'test'}
        os.makedirs(pred_options['output_dir'], exist_ok=True)
        predictions = await generate_predictions(model_info, processed_data, pred_options)
        logger.info("SUCCESS: generate_predictions test passed")
        
        # 8. 인사이트 생성 테스트
        insights = await generate_insights({'statistics': stats}, predictions, "What insights can you provide?")
        logger.info("SUCCESS: generate_insights test passed")
        
        # 9. 워크플로우 테스트: 탐색적 데이터 분석
        eda_options = {
            'visualization_options': {'output_dir': 'temp/test_viz', 'prefix': 'eda_test'},
            'report_options': {'output_dir': 'temp/test_reports', 'prefix': 'eda_test'}
        }
        eda_results = await exploratory_data_analysis(test_data_path, eda_options)
        logger.info("SUCCESS: exploratory_data_analysis workflow test passed")
        
        # 10. 워크플로우 테스트: 예측 모델링
        modeling_options = {
            'model_options': {'output_dir': 'temp/test_models', 'prefix': 'modeling_test'},
            'prediction_options': {'output_dir': 'temp/test_predictions', 'prefix': 'modeling_test'}
        }
        model_results = await predictive_modeling(test_data_path, 'target', ['A', 'B', 'C'], 'linear_regression', modeling_options)
        logger.info("SUCCESS: predictive_modeling workflow test passed")
        
        # 11. 시스템 테스트: 의사결정 지원 시스템
        system_options = {
            'perform_modeling': True,
            'target': 'target',
            'model_type': 'linear_regression',
            'output_dir': 'temp/test_decisions'
        }
        os.makedirs(system_options['output_dir'], exist_ok=True)
        decision_results = await decision_support_system(test_data_path, "What insights can you provide?", None, system_options)
        logger.info("SUCCESS: decision_support_system test passed")
        
        logger.info("All data analysis system tests passed successfully")
        return True
    except Exception as e:
        logger.error(f"Data analysis system test failed: {e}")
        raise

# 테스트 실행
if __name__ == "__main__":
    asyncio.run(test_system())

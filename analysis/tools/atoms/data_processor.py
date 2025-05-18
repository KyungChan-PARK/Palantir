# data_processor.py
"""
데이터 전처리를 위한 도구 모음
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from analysis.mcp_init import mcp

@mcp.tool(
    name="preprocess_data",
    description="데이터에 지정된 전처리 작업을 수행합니다. 결측치 처리, 정규화, 원-핫 인코딩 등을 지원합니다.",
    examples=[
        {"input": {"data": "DataFrame", "operations": ["remove_nulls", "normalize"]}, 
         "output": {"data": "Preprocessed DataFrame", "applied_operations": ["remove_nulls", "normalize"]}},
        {"input": {"data": "DataFrame", "operations": ["one_hot_encode"], "columns": ["category"]}, 
         "output": {"data": "DataFrame with one-hot encoded columns", "new_columns": ["category_A", "category_B"]}}
    ],
    tags=["data", "preprocessing", "cleaning", "transformation"],
    contexts=["data preparation", "feature engineering", "modeling"]
)
async def preprocess_data(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    operations: List[str],
    columns: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    데이터에 지정된 전처리 작업을 수행합니다.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, Dict[str, Any]]
        전처리할 데이터 (DataFrame 또는 read_data 결과)
    operations : List[str]
        수행할 전처리 작업 목록
    columns : List[str], optional
        작업을 적용할 열 목록 (지정하지 않으면 모든 적절한 열에 적용)
    params : Dict[str, Any], optional
        전처리 작업에 필요한 추가 매개변수
    
    Returns:
    --------
    Dict[str, Any]
        전처리된 데이터와 메타데이터를 포함하는 딕셔너리
    """
    # 매개변수 기본값 설정
    params = params or {}
    
    # data가 read_data의 결과인 경우 처리
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
    
    # 각 작업 수행
    applied_operations = []
    operation_results = {}
    transformed_columns = []
    
    try:
        for operation in operations:
            if operation == "remove_nulls":
                method = params.get("null_method", "drop")
                if method == "drop":
                    old_shape = df.shape
                    df = df.dropna(subset=columns)
                    operation_results["remove_nulls"] = {
                        "old_shape": old_shape,
                        "new_shape": df.shape,
                        "removed_rows": old_shape[0] - df.shape[0]
                    }
                elif method == "fill":
                    strategy = params.get("fill_strategy", "mean")
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if strategy == "mean":
                                df[col] = df[col].fillna(df[col].mean())
                            elif strategy == "median":
                                df[col] = df[col].fillna(df[col].median())
                            elif strategy == "mode":
                                df[col] = df[col].fillna(df[col].mode()[0])
                            elif strategy == "constant":
                                fill_value = params.get("fill_value", 0)
                                df[col] = df[col].fillna(fill_value)
                        else:
                            # 비수치형 열은 최빈값 또는 지정된 값으로 채움
                            if strategy in ["mean", "median", "mode"]:
                                if not df[col].mode().empty:
                                    df[col] = df[col].fillna(df[col].mode()[0])
                            elif strategy == "constant":
                                fill_value = params.get("fill_value", "")
                                df[col] = df[col].fillna(fill_value)
                    
                    operation_results["remove_nulls"] = {
                        "method": "fill",
                        "strategy": strategy,
                        "affected_columns": columns
                    }
                
                applied_operations.append("remove_nulls")
                transformed_columns.extend(columns)
            
            elif operation == "normalize":
                method = params.get("normalize_method", "minmax")
                numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if method == "minmax":
                    scaler = MinMaxScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                elif method == "standard":
                    scaler = StandardScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                elif method == "robust":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                
                operation_results["normalize"] = {
                    "method": method,
                    "affected_columns": numeric_columns
                }
                applied_operations.append("normalize")
                transformed_columns.extend(numeric_columns)
            
            elif operation == "one_hot_encode":
                categorical_columns = columns
                if params.get("auto_detect", True):
                    categorical_columns = [col for col in columns 
                                          if not pd.api.types.is_numeric_dtype(df[col]) 
                                          or df[col].nunique() < 10]
                
                encoder = OneHotEncoder(sparse=False, drop=params.get("drop_first", False))
                
                new_columns = []
                for col in categorical_columns:
                    # 결측값 처리
                    df[col] = df[col].fillna('Unknown')
                    
                    # 인코딩
                    encoded = encoder.fit_transform(df[[col]])
                    
                    # 새 열 이름 생성
                    categories = encoder.categories_[0]
                    prefixed_categories = [f"{col}_{category}" for category in categories]
                    
                    # 인코딩된 열 추가
                    encoded_df = pd.DataFrame(encoded, columns=prefixed_categories, index=df.index)
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                    
                    # 새 열 이름 추가
                    new_columns.extend(prefixed_categories)
                
                operation_results["one_hot_encode"] = {
                    "original_columns": categorical_columns,
                    "new_columns": new_columns
                }
                applied_operations.append("one_hot_encode")
                transformed_columns.extend(categorical_columns)
            
            elif operation == "outlier_removal":
                method = params.get("outlier_method", "iqr")
                numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
                
                old_shape = df.shape
                
                if method == "iqr":
                    for col in numeric_columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                elif method == "zscore":
                    from scipy import stats
                    threshold = params.get("zscore_threshold", 3)
                    for col in numeric_columns:
                        z_scores = stats.zscore(df[col], nan_policy='omit')
                        abs_z_scores = np.abs(z_scores)
                        df = df[abs_z_scores <= threshold]
                
                operation_results["outlier_removal"] = {
                    "method": method,
                    "affected_columns": numeric_columns,
                    "old_shape": old_shape,
                    "new_shape": df.shape,
                    "removed_rows": old_shape[0] - df.shape[0]
                }
                applied_operations.append("outlier_removal")
                transformed_columns.extend(numeric_columns)
            
            elif operation == "feature_engineering":
                # 추가 기능 엔지니어링 로직
                pass
        
        # 중복 제거
        transformed_columns = list(set(transformed_columns))
        
        return {
            "success": True,
            "data": df,
            "applied_operations": applied_operations,
            "operation_results": operation_results,
            "transformed_columns": transformed_columns
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"데이터 전처리 중 오류 발생: {str(e)}"
        }

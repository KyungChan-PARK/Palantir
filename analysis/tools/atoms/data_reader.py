# data_reader.py
"""
다양한 형식의 데이터 파일을 읽어 처리 가능한 형태로 반환하는 도구
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional, Union
import io

from analysis.mcp_init import mcp

@mcp.tool(
    name="read_data",
    description="다양한 형식의 데이터 파일을 읽어 처리 가능한 형태로 반환합니다. CSV, JSON, Excel, Parquet 등의 형식을 지원합니다.",
    examples=[
        {"input": {"file_path": "data/sales.csv"}, 
         "output": {"data": "DataFrame", "rows": 1000, "columns": ["date", "product", "revenue"]}},
        {"input": {"file_path": "data/config.json"}, 
         "output": {"data": {"settings": {"max_items": 100}}, "format": "json"}}
    ],
    tags=["data", "input", "reader", "import"],
    contexts=["data loading", "preprocessing", "analysis preparation"]
)
async def read_data(file_path: str, format: Optional[str] = None) -> Dict[str, Any]:
    """
    다양한 형식의 데이터 파일을 읽어 처리 가능한 형태로 반환합니다.
    
    Parameters:
    -----------
    file_path : str
        읽을 파일의 경로 (상대 경로는 프로젝트 루트 기준)
    format : str, optional
        파일 형식 (지정하지 않으면 확장자로부터 추론)
    
    Returns:
    --------
    Dict[str, Any]
        데이터와 메타데이터를 포함하는 딕셔너리
    """
    # 프로젝트 루트 경로 처리
    base_dir = "C:\\Users\\packr\\OneDrive\\palantir"
    if not os.path.isabs(file_path):
        file_path = os.path.join(base_dir, file_path)
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"파일을 찾을 수 없습니다: {file_path}"
        }
    
    # 파일 형식 추론
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lower().lstrip('.')
    
    try:
        # 파일 형식에 따른 로드
        if format in ('csv', 'txt'):
            df = pd.read_csv(file_path)
            return {
                "success": True,
                "data": df,
                "format": "dataframe",
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head(5).to_dict(orient='records')
            }
        
        elif format in ('xls', 'xlsx', 'excel'):
            df = pd.read_excel(file_path)
            return {
                "success": True,
                "data": df,
                "format": "dataframe",
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head(5).to_dict(orient='records')
            }
        
        elif format == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON이 테이블 형식인지 확인
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                return {
                    "success": True,
                    "data": df,
                    "format": "dataframe",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "head": df.head(5).to_dict(orient='records'),
                    "original_format": "json"
                }
            else:
                return {
                    "success": True,
                    "data": data,
                    "format": "json"
                }
        
        elif format == 'parquet':
            df = pd.read_parquet(file_path)
            return {
                "success": True,
                "data": df,
                "format": "dataframe",
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head(5).to_dict(orient='records')
            }
        
        else:
            return {
                "success": False,
                "error": f"지원하지 않는 파일 형식입니다: {format}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"파일 로드 중 오류 발생: {str(e)}"
        }

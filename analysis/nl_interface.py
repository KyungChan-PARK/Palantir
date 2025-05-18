# nl_interface.py
"""
자연어 인터페이스 - 자연어 질의를 MCP 도구 호출로 변환
"""

import os
import json
import asyncio
import argparse
from typing import Dict, Any, List, Optional, Tuple
import re

# MCP 통합 모듈 임포트
from analysis.mcp_integration import process_mcp_request, get_registered_tools

# 의도 감지를 위한 패턴 및 키워드
INTENT_PATTERNS = {
    "read_data": [
        r"(read|load|import|open).*file",
        r"(read|load|import|open).*data",
        r"get.*data from"
    ],
    "preprocess_data": [
        r"(clean|process|preprocess|transform).*data",
        r"(remove|handle).*missing",
        r"(normalize|scale).*data",
        r"encode.*categorical"
    ],
    "analyze_data": [
        r"(analyze|analyse|examine).*data",
        r"(calculate|compute|find).*statistics",
        r"(correlation|relationship).*between",
        r"distribution of"
    ],
    "exploratory_analysis": [
        r"(explore|eda|exploratory).*data",
        r"(overview|summary).*data",
        r"what.*data.*look like",
        r"data.*exploration"
    ],
    "build_predictive_model": [
        r"(build|create|train).*model",
        r"(predict|forecast).*",
        r"(classification|regression).*model",
        r"machine learning"
    ],
    "decision_support": [
        r"(decision|decide|recommend).*",
        r"(insight|advice).*for.*decision",
        r"(analyze|analyse).*for.*decision",
        r"(help|support).*decision"
    ]
}

# 주요 매개변수 패턴
PARAM_PATTERNS = {
    "file_path": [
        r"file[:\s]+([^\s,]+)",
        r"from file[:\s]+([^\s,]+)",
        r"(read|load|open)[:\s]+([^\s,]+\.(csv|json|xlsx|txt|parquet))"
    ],
    "target_column": [
        r"target[:\s]+([^\s,]+)",
        r"predict[:\s]+([^\s,]+)",
        r"forecasting[:\s]+([^\s,]+)"
    ],
    "columns": [
        r"columns[:\s]+\[(.*?)\]",
        r"using columns[:\s]+\[(.*?)\]",
        r"features[:\s]+\[(.*?)\]"
    ],
    "operations": [
        r"operations[:\s]+\[(.*?)\]",
        r"apply[:\s]+\[(.*?)\]",
        r"preprocess with[:\s]+\[(.*?)\]"
    ],
    "analysis_type": [
        r"analysis[:\s]+([^\s,]+)",
        r"analyze using[:\s]+([^\s,]+)",
        r"perform[:\s]+([^\s,]+).*analysis"
    ]
}

def detect_intent(query: str) -> Tuple[str, float]:
    """
    자연어 질의에서 의도 감지
    
    Parameters:
    -----------
    query : str
        자연어 질의
        
    Returns:
    --------
    Tuple[str, float]
        감지된 의도와 신뢰도
    """
    query = query.lower()
    best_intent = None
    max_matches = 0
    
    for intent, patterns in INTENT_PATTERNS.items():
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query):
                matches += 1
        
        if matches > max_matches:
            max_matches = matches
            best_intent = intent
    
    # 신뢰도 계산 (패턴 일치 수 / 전체 패턴 수)
    confidence = max_matches / len(INTENT_PATTERNS[best_intent]) if best_intent else 0.0
    
    return best_intent, confidence

def extract_parameters(query: str, intent: str) -> Dict[str, Any]:
    """
    자연어 질의에서 매개변수 추출
    
    Parameters:
    -----------
    query : str
        자연어 질의
    intent : str
        감지된 의도
        
    Returns:
    --------
    Dict[str, Any]
        추출된 매개변수
    """
    params = {}
    
    # 파일 경로 추출
    for pattern in PARAM_PATTERNS["file_path"]:
        match = re.search(pattern, query)
        if match:
            file_path = match.group(1) if len(match.groups()) == 1 else match.group(2)
            
            # 경로가 따옴표로 감싸져 있는 경우 제거
            file_path = file_path.strip('"\'')
            
            # 상대 경로를 절대 경로로 변환
            if not os.path.isabs(file_path):
                file_path = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", file_path)
            
            params["file_path"] = file_path
            break
    
    # 타겟 열 추출 (예측 모델링 의도인 경우)
    if intent == "build_predictive_model":
        for pattern in PARAM_PATTERNS["target_column"]:
            match = re.search(pattern, query)
            if match:
                params["target_column"] = match.group(1).strip('"\'')
                break
    
    # 열 목록 추출
    for pattern in PARAM_PATTERNS["columns"]:
        match = re.search(pattern, query)
        if match:
            columns_str = match.group(1)
            columns = [col.strip('" \'') for col in columns_str.split(',')]
            params["columns"] = columns
            break
    
    # 전처리 작업 추출 (전처리 의도인 경우)
    if intent == "preprocess_data":
        for pattern in PARAM_PATTERNS["operations"]:
            match = re.search(pattern, query)
            if match:
                operations_str = match.group(1)
                operations = [op.strip('" \'') for op in operations_str.split(',')]
                params["operations"] = operations
                break
        
        # 작업이 명시적으로 지정되지 않았지만 키워드가 있는 경우
        if "operations" not in params:
            operations = []
            
            if re.search(r"missing|null|na|nan", query):
                operations.append("remove_nulls")
            
            if re.search(r"normal|scale|normaliz", query):
                operations.append("normalize")
            
            if re.search(r"outlier", query):
                operations.append("outlier_removal")
            
            if re.search(r"one[-\s]?hot|encod", query):
                operations.append("one_hot_encode")
            
            if operations:
                params["operations"] = operations
    
    # 분석 유형 추출 (분석 의도인 경우)
    if intent == "analyze_data":
        for pattern in PARAM_PATTERNS["analysis_type"]:
            match = re.search(pattern, query)
            if match:
                analysis_type = match.group(1).strip('"\'')
                params["analysis_type"] = analysis_type
                break
        
        # 분석 유형이 명시적으로 지정되지 않았지만 키워드가 있는 경우
        if "analysis_type" not in params:
            if re.search(r"correlat|relationship|association", query):
                params["analysis_type"] = "correlation"
            elif re.search(r"distribut|histogram|density", query):
                params["analysis_type"] = "distribution"
            elif re.search(r"group|segment|categor", query):
                params["analysis_type"] = "group"
            else:
                params["analysis_type"] = "descriptive"  # 기본값
    
    # 의사결정 지원 의도인 경우 질문 추가
    if intent == "decision_support":
        params["question"] = query
        
        # 데이터 소스가 명시적으로 지정되지 않았지만 파일 경로가 있는 경우
        if "file_path" in params:
            params["data_sources"] = [params["file_path"]]
            del params["file_path"]
    
    return params

async def process_natural_language(query: str) -> Dict[str, Any]:
    """
    자연어 질의 처리
    
    Parameters:
    -----------
    query : str
        자연어 질의
        
    Returns:
    --------
    Dict[str, Any]
        처리 결과
    """
    # 의도 감지
    intent, confidence = detect_intent(query)
    
    if not intent or confidence < 0.5:
        return {
            "success": False,
            "error": "의도를 감지할 수 없거나 신뢰도가 낮습니다.",
            "detected_intent": intent,
            "confidence": confidence
        }
    
    # 매개변수 추출
    params = extract_parameters(query, intent)
    
    # MCP 요청 생성
    mcp_request = {
        "tool": intent,
        "parameters": params
    }
    
    # 요청 처리
    result = await process_mcp_request(mcp_request)
    
    # 메타데이터 추가
    result["nl_processing"] = {
        "query": query,
        "detected_intent": intent,
        "confidence": confidence,
        "extracted_parameters": params
    }
    
    return result

async def main():
    parser = argparse.ArgumentParser(description='자연어 MCP 인터페이스')
    parser.add_argument('--query', type=str, help='자연어 질의')
    parser.add_argument('--query-file', type=str, help='자연어 질의가 포함된 파일')
    parser.add_argument('--output', type=str, help='결과 출력 JSON 파일 경로')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("=== 자연어 MCP 인터페이스 (대화형 모드) ===")
        print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
        
        while True:
            query = input("\n질의: ")
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query.strip():
                continue
            
            print("처리 중...")
            result = await process_natural_language(query)
            
            print("\n=== 결과 ===\n")
            
            if result.get("success", False):
                if "insights" in result:
                    print("주요 통찰:")
                    for insight in result["insights"]:
                        print(f"- {insight}")
                
                if "recommendations" in result:
                    print("\n권장 사항:")
                    for recommendation in result["recommendations"]:
                        print(f"- {recommendation.get('title')}: {recommendation.get('description')}")
                
                # 파일 위치 안내
                if "report_file" in result:
                    print(f"\n보고서 파일: {result['report_file']}")
                
                if "markdown_summary" in result:
                    print(f"Markdown 요약: {result['markdown_summary']}")
                
                if "visualizations" in result:
                    print(f"\n생성된 시각화: {len(result['visualizations'])}개")
            else:
                print(f"오류: {result.get('error', '알 수 없는 오류')}")
                
                if "detected_intent" in result:
                    print(f"감지된 의도: {result['detected_intent']} (신뢰도: {result['confidence']:.2f})")
    
    elif args.query:
        result = await process_natural_language(args.query)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"결과가 '{args.output}'에 저장되었습니다.")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.query_file:
        try:
            with open(args.query_file, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            
            result = await process_natural_language(query)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"결과가 '{args.output}'에 저장되었습니다.")
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        
        except Exception as e:
            print(f"오류 발생: {str(e)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())

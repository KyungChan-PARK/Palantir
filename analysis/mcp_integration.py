# mcp_integration.py
"""
MCP 도구 통합 및 실행 스크립트
"""

import os
import json
import asyncio
import argparse
from typing import Dict, Any, List, Optional

# MCP 및 도구 임포트
from analysis.mcp_init import mcp
from analysis.tools.atoms import read_data, preprocess_data, analyze_data
from analysis.tools.molecules import exploratory_analysis, build_predictive_model
from analysis.tools.organisms import decision_support

def get_registered_tools() -> Dict[str, Any]:
    """
    등록된 모든 MCP 도구 정보를 반환
    """
    return {
        "atoms": mcp.get_all_tools(),
        "molecules": mcp.get_all_workflows(),
        "organisms": mcp.get_all_systems()
    }

def print_available_tools() -> None:
    """
    사용 가능한 모든 도구 목록 출력
    """
    tools = get_registered_tools()
    
    print("\n=== 사용 가능한 MCP 도구 ===\n")
    
    print("원자 레벨 도구:")
    for name, info in tools["atoms"].items():
        print(f"  - {name}: {info['description']}")
    
    print("\n분자 레벨 워크플로우:")
    for name, description in tools["molecules"].items():
        print(f"  - {name}: {description}")
    
    print("\n유기체 레벨 시스템:")
    for name, description in tools["organisms"].items():
        print(f"  - {name}: {description}")

async def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    지정된 도구 실행
    
    Parameters:
    -----------
    tool_name : str
        실행할 도구 이름
    args : Dict[str, Any]
        도구 매개변수
        
    Returns:
    --------
    Dict[str, Any]
        도구 실행 결과
    """
    # 원자 레벨 도구
    if tool_name == "read_data":
        return await read_data(**args)
    elif tool_name == "preprocess_data":
        return await preprocess_data(**args)
    elif tool_name == "analyze_data":
        return await analyze_data(**args)
    
    # 분자 레벨 워크플로우
    elif tool_name == "exploratory_analysis":
        return await exploratory_analysis(**args)
    elif tool_name == "build_predictive_model":
        return await build_predictive_model(**args)
    
    # 유기체 레벨 시스템
    elif tool_name == "decision_support":
        return await decision_support(**args)
    
    else:
        return {
            "success": False,
            "error": f"알 수 없는 도구: {tool_name}"
        }

async def process_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP 요청 처리
    
    Parameters:
    -----------
    request : Dict[str, Any]
        MCP 요청 (도구 및 매개변수 포함)
        
    Returns:
    --------
    Dict[str, Any]
        실행 결과
    """
    if "tool" not in request:
        return {
            "success": False,
            "error": "도구가 지정되지 않았습니다."
        }
    
    tool_name = request["tool"]
    args = request.get("parameters", {})
    
    try:
        result = await execute_tool(tool_name, args)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"도구 실행 중 오류 발생: {str(e)}"
        }

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    실행 결과를 JSON 파일로 저장
    
    Parameters:
    -----------
    results : Dict[str, Any]
        실행 결과
    output_file : str
        출력 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 '{output_file}'에 저장되었습니다.")

async def main():
    parser = argparse.ArgumentParser(description='MCP 도구 실행')
    parser.add_argument('--request', type=str, help='MCP 요청 JSON 파일 경로')
    parser.add_argument('--output', type=str, help='결과 출력 JSON 파일 경로')
    parser.add_argument('--list-tools', action='store_true', help='사용 가능한 도구 목록 표시')
    
    args = parser.parse_args()
    
    if args.list_tools:
        print_available_tools()
        return
    
    if not args.request:
        print("오류: MCP 요청 파일이 지정되지 않았습니다.")
        parser.print_help()
        return
    
    try:
        with open(args.request, 'r', encoding='utf-8') as f:
            request = json.load(f)
        
        print(f"요청 처리 중: {args.request}")
        results = await process_mcp_request(request)
        
        if args.output:
            save_results(results, args.output)
        else:
            print("\n=== 실행 결과 ===\n")
            print(json.dumps(results, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

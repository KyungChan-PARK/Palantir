"""
MCP 시스템 통합 테스트 모듈
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_integration')

async def test_integration():
    """시스템 통합 테스트"""
    logger.info("Starting system integration test")
    
    # 테스트 결과
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "tests": []
    }
    
    # 테스트 도우미 함수
    async def run_test(name, test_func):
        results["total"] += 1
        logger.info(f"Running test: {name}")
        try:
            start_time = datetime.now()
            await test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results["passed"] += 1
            results["tests"].append({
                "name": name,
                "status": "PASS",
                "duration": duration
            })
            logger.info(f"SUCCESS: Test {name} passed in {duration:.2f} seconds")
            return True
        except Exception as e:
            results["failed"] += 1
            results["tests"].append({
                "name": name,
                "status": "FAIL",
                "error": str(e)
            })
            logger.error(f"ERROR: Test {name} failed: {e}")
            return False
    
    # 1. 시스템 구성 테스트
    async def test_system_configuration():
        # MCP 모듈 가져오기
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
        from analysis.mcp_init import mcp
        
        if not hasattr(mcp, 'tools') or not hasattr(mcp, 'workflows') or not hasattr(mcp, 'systems'):
            raise ValueError("MCP instance does not have required attributes")
        
        # 도구, 워크플로우, 시스템이 있는지 확인
        if not mcp.tools:
            logger.warning("No tools registered in MCP")
        if not mcp.workflows:
            logger.warning("No workflows registered in MCP")
        if not mcp.systems:
            logger.warning("No systems registered in MCP")
            
        logger.info(f"MCP configuration: {len(mcp.tools)} tools, {len(mcp.workflows)} workflows, {len(mcp.systems)} systems")
    
    # 2. 디렉토리 구조 테스트
    async def test_directory_structure():
        required_dirs = [
            "analysis",
            "analysis/tools",
            "analysis/tools/atoms",
            "analysis/tools/molecules",
            "analysis/tools/organisms",
            "config",
            "data",
            "docs",
            "output",
            "output/reports",
            "output/viz",
            "output/models",
            "output/decisions",
            "logs",
            "temp"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory {dir_path} does not exist")
    
    # 3. 파일 접근 권한 테스트
    async def test_file_permissions():
        test_file = "temp/test_permissions.txt"
        test_content = f"Test content created at {datetime.now()}"
        
        # 파일 쓰기 테스트
        try:
            with open(test_file, "w") as f:
                f.write(test_content)
        except Exception as e:
            raise ValueError(f"Failed to write to test file: {e}")
        
        # 파일 읽기 테스트
        try:
            with open(test_file, "r") as f:
                content = f.read()
            if content != test_content:
                raise ValueError(f"File content mismatch. Expected: {test_content}, Got: {content}")
        except Exception as e:
            raise ValueError(f"Failed to read test file: {e}")
        
        # 파일 삭제 테스트
        try:
            os.remove(test_file)
            if os.path.exists(test_file):
                raise ValueError(f"Failed to delete test file {test_file}")
        except Exception as e:
            raise ValueError(f"Error during file deletion: {e}")
    
    # 4. 모듈 가져오기 테스트
    async def test_module_imports():
        modules = [
            "analysis.mcp_init",
            "analysis.tools.atoms",
            "analysis.tools.molecules",
            "analysis.tools.organisms"
        ]
        
        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                raise ValueError(f"Failed to import module {module_name}: {e}")
    
    # 5. 시스템 간 통합 테스트
    async def test_systems_integration():
        # 데이터 분석 시스템이 존재하는지 확인
        try:
            from analysis.tools.organisms import data_analysis_system
            
            # 테스트 함수가 있는지 확인
            if not hasattr(data_analysis_system, 'test_system'):
                raise ValueError("data_analysis_system module does not have test_system function")
        except ImportError as e:
            raise ValueError(f"Failed to import data_analysis_system: {e}")
        
        # YouTube API 시스템이 존재하는지 확인
        try:
            from analysis.tools.organisms import youtube_api_system
            
            # 테스트 함수가 있는지 확인
            if hasattr(youtube_api_system, 'test_system'):
                logger.info("YouTube API system has test_system function")
            else:
                logger.warning("YouTube API system does not have test_system function")
        except ImportError:
            logger.warning("YouTube API system not found (may be optional)")
        
        # 대규모 문서 테스트 시스템이 존재하는지 확인
        try:
            from analysis.tools.organisms import document_test_system
            
            # 테스트 함수가 있는지 확인
            if hasattr(document_test_system, 'test_system'):
                logger.info("Document test system has test_system function")
            else:
                logger.warning("Document test system does not have test_system function")
        except ImportError:
            logger.warning("Document test system not found (may be optional)")
    
    # 테스트 실행
    await run_test("System Configuration", test_system_configuration)
    await run_test("Directory Structure", test_directory_structure)
    await run_test("File Permissions", test_file_permissions)
    await run_test("Module Imports", test_module_imports)
    await run_test("Systems Integration", test_systems_integration)
    
    # 테스트 결과 요약
    logger.info(f"Test Summary: {results['passed']}/{results['total']} tests passed, {results['failed']} failed")
    
    if results["failed"] > 0:
        logger.error("Integration test failed")
        return False
    else:
        logger.info("SUCCESS: All integration tests passed")
        return True

if __name__ == "__main__":
    asyncio.run(test_integration())

"""
OpenAI Codex CLI MCP 서버 통합 모듈

이 모듈은 OpenAI Codex CLI를 MCP(Model Context Protocol) 시스템과 통합하여
코드 생성, 설명, 디버깅 등의 기능을 제공합니다.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import yaml
from jsonrpcserver import method, async_dispatch

# MCP 모듈 임포트
from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("mcp.codex")

class OpenAICodexClient:
    """OpenAI Codex CLI와 통신하는 클라이언트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000/"):
        """
        OpenAI Codex 클라이언트 초기화
        
        Args:
            base_url: Codex RPC 서버 기본 URL
        """
        self.base_url = base_url
        self.session = None
        self.config = self._load_config()
        self.logger = logger
        
        self.logger.info("OpenAI Codex 클라이언트 초기화 완료")
    
    def _load_config(self) -> Dict[str, Any]:
        """구성 파일 로드"""
        config_path = Path("config/ai_tools.yaml")
        if not config_path.exists():
            return {
                "codex": {
                    "model": "o4-mini",
                    "approval_mode": "auto-edit",
                    "default_directory": str(Path.cwd())
                }
            }
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config.get("codex", {})
    
    async def _ensure_session(self):
        """HTTP 세션 확보"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def _call_rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        JSON-RPC 메서드 호출
        
        Args:
            method: RPC 메서드 이름
            params: 메서드 매개변수
            
        Returns:
            응답 결과
        """
        await self._ensure_session()
        
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params
        }
        
        self.logger.debug(f"RPC 요청: {json.dumps(payload)}")
        
        try:
            async with self.session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    self.logger.error(f"RPC 오류: 상태 코드 {response.status}")
                    return {"error": f"상태 코드 {response.status}"}
                
                result = await response.json()
                self.logger.debug(f"RPC 응답: {json.dumps(result)}")
                
                if "error" in result:
                    self.logger.error(f"RPC 오류: {result['error']}")
                    return {"error": result["error"]}
                
                return result.get("result", {})
        
        except aiohttp.ClientError as e:
            self.logger.error(f"RPC 호출 오류: {str(e)}")
            return {"error": str(e)}
    
    async def write_code(self, 
                         prompt: str, 
                         model: Optional[str] = None,
                         context: Optional[str] = None,
                         file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        코드 생성 요청
        
        Args:
            prompt: 코드 생성 프롬프트
            model: 사용할 모델 (기본값: 구성 파일의 모델)
            context: 추가 컨텍스트
            file_path: 코드를 생성할 파일 경로
            
        Returns:
            생성된 코드 및 메타데이터
        """
        params = {
            "prompt": prompt,
            "model": model or self.config.get("model", "o4-mini"),
            "approval_mode": self.config.get("approval_mode", "auto-edit")
        }
        
        if context:
            params["context"] = context
        
        if file_path:
            params["file_path"] = file_path
        
        result = await self._call_rpc("generate_code", params)
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "code": None
            }
        
        return {
            "success": True,
            "code": result.get("code", ""),
            "tokens_used": result.get("tokens_used", 0),
            "model": result.get("model", params["model"])
        }
    
    async def explain_code(self, 
                          code: str, 
                          model: Optional[str] = None,
                          detail_level: str = "medium") -> Dict[str, Any]:
        """
        코드 설명 요청
        
        Args:
            code: 설명할 코드
            model: 사용할 모델 (기본값: 구성 파일의 모델)
            detail_level: 설명 상세 수준 (low, medium, high)
            
        Returns:
            코드 설명 및 메타데이터
        """
        params = {
            "code": code,
            "model": model or self.config.get("model", "o4-mini"),
            "detail_level": detail_level
        }
        
        result = await self._call_rpc("explain_code", params)
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "explanation": None
            }
        
        return {
            "success": True,
            "explanation": result.get("explanation", ""),
            "tokens_used": result.get("tokens_used", 0),
            "model": result.get("model", params["model"])
        }
    
    async def debug_code(self, 
                        code: str, 
                        error_message: Optional[str] = None,
                        model: Optional[str] = None) -> Dict[str, Any]:
        """
        코드 디버깅 요청
        
        Args:
            code: 디버깅할 코드
            error_message: 발생한 오류 메시지 (선택 사항)
            model: 사용할 모델 (기본값: 구성 파일의 모델)
            
        Returns:
            디버깅 결과 및 수정된 코드
        """
        params = {
            "code": code,
            "model": model or self.config.get("model", "o4-mini")
        }
        
        if error_message:
            params["error_message"] = error_message
        
        result = await self._call_rpc("debug_code", params)
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "fixed_code": None,
                "explanation": None
            }
        
        return {
            "success": True,
            "fixed_code": result.get("fixed_code", ""),
            "explanation": result.get("explanation", ""),
            "issues_found": result.get("issues_found", []),
            "tokens_used": result.get("tokens_used", 0),
            "model": result.get("model", params["model"])
        }
    
    async def refactor_code(self, 
                           code: str, 
                           instructions: str,
                           model: Optional[str] = None) -> Dict[str, Any]:
        """
        코드 리팩토링 요청
        
        Args:
            code: 리팩토링할 코드
            instructions: 리팩토링 지침
            model: 사용할 모델 (기본값: 구성 파일의 모델)
            
        Returns:
            리팩토링된 코드 및 메타데이터
        """
        params = {
            "code": code,
            "instructions": instructions,
            "model": model or self.config.get("model", "o4-mini")
        }
        
        result = await self._call_rpc("refactor_code", params)
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "refactored_code": None
            }
        
        return {
            "success": True,
            "refactored_code": result.get("refactored_code", ""),
            "changes_made": result.get("changes_made", []),
            "tokens_used": result.get("tokens_used", 0),
            "model": result.get("model", params["model"])
        }
    
    async def generate_tests(self, 
                            code: str, 
                            test_framework: str = "pytest",
                            model: Optional[str] = None) -> Dict[str, Any]:
        """
        테스트 코드 생성 요청
        
        Args:
            code: 테스트할 코드
            test_framework: 테스트 프레임워크 (pytest, unittest 등)
            model: 사용할 모델 (기본값: 구성 파일의 모델)
            
        Returns:
            생성된 테스트 코드 및 메타데이터
        """
        params = {
            "code": code,
            "test_framework": test_framework,
            "model": model or self.config.get("model", "o4-mini")
        }
        
        result = await self._call_rpc("generate_tests", params)
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "test_code": None
            }
        
        return {
            "success": True,
            "test_code": result.get("test_code", ""),
            "test_cases": result.get("test_cases", []),
            "tokens_used": result.get("tokens_used", 0),
            "model": result.get("model", params["model"])
        }
    
    async def close(self):
        """클라이언트 세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("OpenAI Codex 클라이언트 세션 종료")

# OpenAI Codex 클라이언트 인스턴스
codex_client = OpenAICodexClient()

# MCP 도구 데코레이터 적용
@mcp.tool(
    name="openai_codex_write_code",
    description="OpenAI Codex를 사용하여 코드 생성",
    tags=["openai_codex", "code_generation"]
)
async def write_code(prompt: str, 
                    model: Optional[str] = None,
                    context: Optional[str] = None,
                    file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    OpenAI Codex를 사용하여 코드 생성
    
    Args:
        prompt: 코드 생성 프롬프트
        model: 사용할 모델 (o4-mini, o4-preview, o4-pro 등)
        context: 추가 컨텍스트
        file_path: 코드를 생성할 파일 경로
        
    Returns:
        생성된 코드 및 메타데이터를 포함한 딕셔너리
    """
    return await codex_client.write_code(prompt, model, context, file_path)

@mcp.tool(
    name="openai_codex_explain_code",
    description="OpenAI Codex를 사용하여 코드 설명",
    tags=["openai_codex", "code_explanation"]
)
async def explain_code(code: str, 
                      model: Optional[str] = None,
                      detail_level: str = "medium") -> Dict[str, Any]:
    """
    OpenAI Codex를 사용하여 코드 설명
    
    Args:
        code: 설명할 코드
        model: 사용할 모델 (o4-mini, o4-preview, o4-pro 등)
        detail_level: 설명 상세 수준 (low, medium, high)
        
    Returns:
        코드 설명 및 메타데이터를 포함한 딕셔너리
    """
    return await codex_client.explain_code(code, model, detail_level)

@mcp.tool(
    name="openai_codex_debug_code",
    description="OpenAI Codex를 사용하여 코드 디버깅",
    tags=["openai_codex", "code_debugging"]
)
async def debug_code(code: str, 
                    error_message: Optional[str] = None,
                    model: Optional[str] = None) -> Dict[str, Any]:
    """
    OpenAI Codex를 사용하여 코드 디버깅
    
    Args:
        code: 디버깅할 코드
        error_message: 발생한 오류 메시지 (선택 사항)
        model: 사용할 모델 (o4-mini, o4-preview, o4-pro 등)
        
    Returns:
        디버깅 결과 및 수정된 코드를 포함한 딕셔너리
    """
    return await codex_client.debug_code(code, error_message, model)

@mcp.tool(
    name="openai_codex_refactor_code",
    description="OpenAI Codex를 사용하여 코드 리팩토링",
    tags=["openai_codex", "code_refactoring"]
)
async def refactor_code(code: str, 
                       instructions: str,
                       model: Optional[str] = None) -> Dict[str, Any]:
    """
    OpenAI Codex를 사용하여 코드 리팩토링
    
    Args:
        code: 리팩토링할 코드
        instructions: 리팩토링 지침
        model: 사용할 모델 (o4-mini, o4-preview, o4-pro 등)
        
    Returns:
        리팩토링된 코드 및 메타데이터를 포함한 딕셔너리
    """
    return await codex_client.refactor_code(code, instructions, model)

@mcp.tool(
    name="openai_codex_generate_tests",
    description="OpenAI Codex를 사용하여 테스트 코드 생성",
    tags=["openai_codex", "test_generation"]
)
async def generate_tests(code: str, 
                        test_framework: str = "pytest",
                        model: Optional[str] = None) -> Dict[str, Any]:
    """
    OpenAI Codex를 사용하여 테스트 코드 생성
    
    Args:
        code: 테스트할 코드
        test_framework: 테스트 프레임워크 (pytest, unittest 등)
        model: 사용할 모델 (o4-mini, o4-preview, o4-pro 등)
        
    Returns:
        생성된 테스트 코드 및 메타데이터를 포함한 딕셔너리
    """
    return await codex_client.generate_tests(code, test_framework, model)

# RPC 서버 메서드
@method
async def generate_code(prompt: str, 
                       model: str = "o4-mini",
                       approval_mode: str = "auto-edit",
                       context: Optional[str] = None,
                       file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    코드 생성 RPC 메서드
    
    Args:
        prompt: 코드 생성 프롬프트
        model: 사용할 모델
        approval_mode: 승인 모드 (suggest, auto-edit, full-auto)
        context: 추가 컨텍스트
        file_path: 코드를 생성할 파일 경로
    
    Returns:
        생성된 코드 및 메타데이터
    """
    # 실제 OpenAI Codex CLI 명령 실행
    cmd = ["codex"]
    
    if model:
        cmd.extend(["--model", model])
    
    if approval_mode:
        cmd.extend(["--approval-mode", approval_mode])
    
    if context:
        cmd.extend(["--context", context])
    
    if file_path:
        cmd.extend(["--file", file_path])
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    cmd.extend(["--file-prompt", prompt_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.unlink(prompt_file)
        
        # 결과 파싱
        code_output = result.stdout.strip()
        
        # 토큰 사용량 추출 (실제 CLI 출력에서 파싱 필요)
        tokens_used = 0
        
        return {
            "code": code_output,
            "tokens_used": tokens_used,
            "model": model
        }
    
    except subprocess.CalledProcessError as e:
        os.unlink(prompt_file)
        logger.error(f"Codex CLI 오류: {e.stderr}")
        return {"error": e.stderr}
    
    except Exception as e:
        os.unlink(prompt_file)
        logger.error(f"처리 오류: {str(e)}")
        return {"error": str(e)}

@method
async def explain_code(code: str, 
                      model: str = "o4-mini",
                      detail_level: str = "medium") -> Dict[str, Any]:
    """
    코드 설명 RPC 메서드
    
    Args:
        code: 설명할 코드
        model: 사용할 모델
        detail_level: 설명 상세 수준 (low, medium, high)
        
    Returns:
        코드 설명 및 메타데이터
    """
    # 실제 OpenAI Codex CLI 명령 실행
    cmd = ["codex", "--model", model]
    
    # 프롬프트 준비
    detail_map = {
        "low": "간략하게",
        "medium": "적절한 상세도로",
        "high": "매우 자세하게"
    }
    
    detail_instruction = detail_map.get(detail_level, "적절한 상세도로")
    
    prompt = f"다음 코드를 {detail_instruction} 설명해주세요:\n\n```\n{code}\n```"
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    cmd.extend(["--file-prompt", prompt_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.unlink(prompt_file)
        
        # 결과 파싱
        explanation = result.stdout.strip()
        
        # 토큰 사용량 추출 (실제 CLI 출력에서 파싱 필요)
        tokens_used = 0
        
        return {
            "explanation": explanation,
            "tokens_used": tokens_used,
            "model": model
        }
    
    except subprocess.CalledProcessError as e:
        os.unlink(prompt_file)
        logger.error(f"Codex CLI 오류: {e.stderr}")
        return {"error": e.stderr}
    
    except Exception as e:
        os.unlink(prompt_file)
        logger.error(f"처리 오류: {str(e)}")
        return {"error": str(e)}

@method
async def debug_code(code: str, 
                    error_message: Optional[str] = None,
                    model: str = "o4-mini") -> Dict[str, Any]:
    """
    코드 디버깅 RPC 메서드
    
    Args:
        code: 디버깅할 코드
        error_message: 발생한 오류 메시지 (선택 사항)
        model: 사용할 모델
        
    Returns:
        디버깅 결과 및 수정된 코드
    """
    # 실제 OpenAI Codex CLI 명령 실행
    cmd = ["codex", "--model", model]
    
    # 프롬프트 준비
    prompt = f"다음 코드를 디버깅해주세요:\n\n```\n{code}\n```"
    
    if error_message:
        prompt += f"\n\n발생한 오류 메시지:\n```\n{error_message}\n```"
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    cmd.extend(["--file-prompt", prompt_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.unlink(prompt_file)
        
        # 결과 파싱 (실제로는 Codex의 출력 형식에 맞게 파싱 필요)
        output = result.stdout.strip()
        
        # 간단한 파싱 로직 (실제로는 더 정교하게 구현 필요)
        fixed_code_start = output.find("```")
        explanation = output[:fixed_code_start].strip() if fixed_code_start > 0 else ""
        
        fixed_code = ""
        if fixed_code_start > 0:
            code_block = output[fixed_code_start:]
            start = code_block.find("\n") + 1
            end = code_block.rfind("```")
            if start > 0 and end > start:
                fixed_code = code_block[start:end].strip()
        
        # 토큰 사용량 추출 (실제 CLI 출력에서 파싱 필요)
        tokens_used = 0
        
        return {
            "fixed_code": fixed_code,
            "explanation": explanation,
            "issues_found": [], # 실제로는 파싱 필요
            "tokens_used": tokens_used,
            "model": model
        }
    
    except subprocess.CalledProcessError as e:
        os.unlink(prompt_file)
        logger.error(f"Codex CLI 오류: {e.stderr}")
        return {"error": e.stderr}
    
    except Exception as e:
        os.unlink(prompt_file)
        logger.error(f"처리 오류: {str(e)}")
        return {"error": str(e)}

@method
async def refactor_code(code: str, 
                       instructions: str,
                       model: str = "o4-mini") -> Dict[str, Any]:
    """
    코드 리팩토링 RPC 메서드
    
    Args:
        code: 리팩토링할 코드
        instructions: 리팩토링 지침
        model: 사용할 모델
        
    Returns:
        리팩토링된 코드 및 메타데이터
    """
    # 실제 OpenAI Codex CLI 명령 실행
    cmd = ["codex", "--model", model]
    
    # 프롬프트 준비
    prompt = f"다음 코드를 리팩토링해주세요:\n\n```\n{code}\n```\n\n리팩토링 지침:\n{instructions}"
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    cmd.extend(["--file-prompt", prompt_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.unlink(prompt_file)
        
        # 결과 파싱 (실제로는 Codex의 출력 형식에 맞게 파싱 필요)
        output = result.stdout.strip()
        
        # 간단한 파싱 로직 (실제로는 더 정교하게 구현 필요)
        refactored_code_start = output.find("```")
        changes_explanation = output[:refactored_code_start].strip() if refactored_code_start > 0 else ""
        
        refactored_code = ""
        if refactored_code_start > 0:
            code_block = output[refactored_code_start:]
            start = code_block.find("\n") + 1
            end = code_block.rfind("```")
            if start > 0 and end > start:
                refactored_code = code_block[start:end].strip()
        
        # 토큰 사용량 추출 (실제 CLI 출력에서 파싱 필요)
        tokens_used = 0
        
        return {
            "refactored_code": refactored_code,
            "changes_made": [changes_explanation], # 실제로는 파싱 필요
            "tokens_used": tokens_used,
            "model": model
        }
    
    except subprocess.CalledProcessError as e:
        os.unlink(prompt_file)
        logger.error(f"Codex CLI 오류: {e.stderr}")
        return {"error": e.stderr}
    
    except Exception as e:
        os.unlink(prompt_file)
        logger.error(f"처리 오류: {str(e)}")
        return {"error": str(e)}

@method
async def generate_tests(code: str, 
                        test_framework: str = "pytest",
                        model: str = "o4-mini") -> Dict[str, Any]:
    """
    테스트 코드 생성 RPC 메서드
    
    Args:
        code: 테스트할 코드
        test_framework: 테스트 프레임워크 (pytest, unittest 등)
        model: 사용할 모델
        
    Returns:
        생성된 테스트 코드 및 메타데이터
    """
    # 실제 OpenAI Codex CLI 명령 실행
    cmd = ["codex", "--model", model]
    
    # 프롬프트 준비
    prompt = f"다음 코드에 대한 {test_framework} 테스트 코드를 작성해주세요:\n\n```\n{code}\n```"
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    cmd.extend(["--file-prompt", prompt_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.unlink(prompt_file)
        
        # 결과 파싱 (실제로는 Codex의 출력 형식에 맞게 파싱 필요)
        output = result.stdout.strip()
        
        # 간단한 파싱 로직 (실제로는 더 정교하게 구현 필요)
        test_code_start = output.find("```")
        explanation = output[:test_code_start].strip() if test_code_start > 0 else ""
        
        test_code = ""
        if test_code_start > 0:
            code_block = output[test_code_start:]
            start = code_block.find("\n") + 1
            end = code_block.rfind("```")
            if start > 0 and end > start:
                test_code = code_block[start:end].strip()
        
        # 토큰 사용량 추출 (실제 CLI 출력에서 파싱 필요)
        tokens_used = 0
        
        return {
            "test_code": test_code,
            "test_cases": [], # 실제로는 파싱 필요
            "tokens_used": tokens_used,
            "model": model
        }
    
    except subprocess.CalledProcessError as e:
        os.unlink(prompt_file)
        logger.error(f"Codex CLI 오류: {e.stderr}")
        return {"error": e.stderr}
    
    except Exception as e:
        os.unlink(prompt_file)
        logger.error(f"처리 오류: {str(e)}")
        return {"error": str(e)}

# RPC 서버 시작 함수
async def start_server(host: str = "localhost", port: int = 8000):
    """
    OpenAI Codex RPC 서버 시작
    
    Args:
        host: 서버 호스트
        port: 서버 포트
    """
    from aiohttp import web
    
    async def handle(request):
        request_text = await request.text()
        response = await async_dispatch(request_text)
        return web.Response(text=response, content_type="application/json")
    
    app = web.Application()
    app.router.add_post("/", handle)
    
    logger.info(f"OpenAI Codex RPC 서버 시작 (http://{host}:{port}/)")
    return await web._run_app(app, host=host, port=port)

# 서버 시작 유틸리티 함수
def run_server():
    """OpenAI Codex RPC 서버 실행 유틸리티"""
    logger.info("OpenAI Codex RPC 서버 시작 중...")
    asyncio.run(start_server())

if __name__ == "__main__":
    run_server()

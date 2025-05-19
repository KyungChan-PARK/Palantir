import subprocess
import sys
import logging
import os
import asyncio
import shlex
from typing import Optional, List, NoReturn, Dict, Protocol, Union
from datetime import datetime
from dataclasses import dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


# 설정 클래스
@dataclass
class CodexConfig:
    """Codex CLI 설정"""
    timeout: int = 300
    log_level: str = 'INFO'
    max_retries: int = 3
    wait_min: int = 1
    wait_max: int = 10


# 로깅 설정
def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """모듈별 로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger('codex_wrapper')


class CodexError(Exception):
    """Codex CLI 실행 중 발생하는 예외를 처리하기 위한 커스텀 예외 클래스"""
    pass


class CodexTimeoutError(CodexError):
    """Codex CLI 실행 시간 초과 예외"""
    pass


def setup_environment(env_vars: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Codex CLI 실행을 위한 환경 변수를 설정합니다.

    Args:
        env_vars (Optional[Dict[str, str]]): 설정할 환경 변수 딕셔너리

    Returns:
        Dict[str, str]: 업데이트된 환경 변수
    """
    current_env = os.environ.copy()
    
    # 기본 환경 변수 설정
    default_env = {
        'CODEX_TIMEOUT': '300',  # 기본 타임아웃 5분
        'CODEX_LOG_LEVEL': 'INFO'
    }
    
    current_env.update(default_env)
    if env_vars:
        current_env.update(env_vars)
        
    return current_env


def codex_cli(prompt: str,
              approval_mode: str = "suggest",
              extra_args: Optional[List[str]] = None,
              timeout: Optional[int] = None,
              env_vars: Optional[Dict[str, str]] = None) -> str:
    """
    Run the OpenAI Codex CLI with the given prompt and return its output.

    Args:
        prompt (str): 자연어로 된 Codex 지시문
        approval_mode (str): 실행 모드 설정
            - 'suggest': 변경 사항을 제안만 함
            - 'auto-edit': 사용자 승인 후 자동 적용
            - 'full-auto': 자동으로 모든 변경사항 적용
        extra_args (Optional[List[str]]): 추가 CLI 파라미터 리스트
        timeout (Optional[int]): 실행 타임아웃 (초)
        env_vars (Optional[Dict[str, str]]): 환경 변수 설정

    Returns:
        str: Codex CLI의 실행 결과 텍스트

    Raises:
        CodexError: Codex CLI 실행 실패 시
        CodexTimeoutError: 실행 시간 초과 시
        ValueError: 잘못된 approval_mode 값 입력 시
    """
    start_time = datetime.now()
    logger.info(f"Starting Codex CLI execution with approval_mode: {approval_mode}")

    # approval_mode 검증
    valid_modes = {"suggest", "auto-edit", "full-auto"}
    if approval_mode not in valid_modes:
        raise ValueError(f"Invalid approval_mode. Must be one of: {valid_modes}")

    # 환경 변수 설정
    env = setup_environment(env_vars)
    
    # 타임아웃 설정
    if timeout is None:
        timeout = int(env.get('CODEX_TIMEOUT', 300))

    cmd = ["codex", "--approval-mode", approval_mode]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(prompt)

    try:
        logger.debug(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env
        )

        if result.returncode != 0:
            error_msg = f"Codex CLI error ({result.returncode}):\n{result.stderr}"
            logger.error(error_msg)
            raise CodexError(error_msg)

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Codex CLI execution completed in {execution_time:.2f} seconds")
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        error_msg = f"Codex CLI execution timed out after {timeout} seconds"
        logger.error(error_msg)
        raise CodexTimeoutError(error_msg)
    except subprocess.SubprocessError as e:
        error_msg = f"Failed to execute Codex CLI: {e}"
        logger.error(error_msg)
        raise CodexError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error while running Codex CLI: {e}"
        logger.error(error_msg)
        raise CodexError(error_msg)


def codex_cli_stream(prompt: str,
                    approval_mode: str = "suggest",
                    timeout: Optional[int] = None,
                    env_vars: Optional[Dict[str, str]] = None) -> None:
    """
    실시간으로 Codex CLI 출력을 스트리밍하여 화면에 출력합니다.

    Args:
        prompt (str): 자연어로 된 Codex 지시문
        approval_mode (str): 실행 모드 ('suggest', 'auto-edit', 'full-auto')
        timeout (Optional[int]): 실행 타임아웃 (초)
        env_vars (Optional[Dict[str, str]]): 환경 변수 설정

    Raises:
        CodexError: Codex CLI 실행 실패 시
        CodexTimeoutError: 실행 시간 초과 시
        ValueError: 잘못된 approval_mode 값 입력 시
    """
    start_time = datetime.now()
    logger.info(f"Starting Codex CLI streaming with approval_mode: {approval_mode}")

    # approval_mode 검증
    valid_modes = {"suggest", "auto-edit", "full-auto"}
    if approval_mode not in valid_modes:
        raise ValueError(f"Invalid approval_mode. Must be one of: {valid_modes}")

    # 환경 변수 설정
    env = setup_environment(env_vars)
    
    # 타임아웃 설정
    if timeout is None:
        timeout = int(env.get('CODEX_TIMEOUT', 300))

    cmd = ["codex", "--approval-mode", approval_mode, prompt]
    
    try:
        logger.debug(f"Executing streaming command: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            error_msg = f"Codex CLI streaming timed out after {timeout} seconds"
            logger.error(error_msg)
            raise CodexTimeoutError(error_msg)

        if proc.returncode != 0:
            assert proc.stderr is not None
            error_msg = proc.stderr.read()
            logger.error(f"Codex CLI streaming failed ({proc.returncode}): {error_msg}")
            raise CodexError(f"Codex CLI streaming failed ({proc.returncode}): {error_msg}")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Codex CLI streaming completed in {execution_time:.2f} seconds")

    except subprocess.SubprocessError as e:
        error_msg = f"Failed to execute Codex CLI in streaming mode: {e}"
        logger.error(error_msg)
        raise CodexError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during Codex CLI streaming: {e}"
        logger.error(error_msg)
        raise CodexError(error_msg)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(CodexError)
)
async def _codex_cli_async_with_retry(
    prompt: str,
    approval_mode: str = "suggest",
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None
) -> str:
    """
    재시도 로직이 포함된 내부 함수입니다.
    """
    start_time = datetime.now()
    logger.info(f"Starting async Codex CLI execution with approval_mode: {approval_mode}")
    
    # approval_mode 검증
    valid_modes = {"suggest", "auto-edit", "full-auto"}
    if approval_mode not in valid_modes:
        raise ValueError(f"Invalid approval_mode. Must be one of: {valid_modes}")
    
    # 환경 변수 설정
    env = setup_environment(env_vars)
    
    # 타임아웃 설정
    if timeout is None:
        timeout = int(env.get('CODEX_TIMEOUT', 300))
    
    # 명령어 구성 (셸 이스케이프 적용)
    base_cmd = ["codex", "--approval-mode", approval_mode]
    if extra_args:
        base_cmd.extend(map(shlex.quote, extra_args))
    base_cmd.append(shlex.quote(prompt))
    
    try:
        # 비동기 서브프로세스 생성
        process = await asyncio.create_subprocess_exec(
            *base_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error while killing process: {e}")
            finally:
                error_msg = f"Async Codex CLI execution timed out after {timeout} seconds"
                logger.error(error_msg)
                raise CodexTimeoutError(error_msg) from None
            
        if process.returncode != 0:
            error_msg = f"Async Codex CLI error ({process.returncode}):\n{stderr.decode()}"
            logger.error(error_msg)
            raise CodexError(error_msg)
            
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Async Codex CLI execution completed in {execution_time:.2f} seconds")
        return stdout.decode().strip()
        
    except asyncio.CancelledError:
        logger.warning("Async Codex CLI execution was cancelled")
        raise
    except Exception as e:
        if isinstance(e, (CodexTimeoutError, asyncio.CancelledError)):
            raise
        error_msg = f"Unexpected error during async Codex CLI execution: {e}"
        logger.error(error_msg)
        raise CodexError(error_msg)


async def codex_cli_async(
    prompt: str,
    approval_mode: str = "suggest",
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None
) -> str:
    """
    비동기로 Codex CLI를 실행합니다.
    
    Args:
        prompt (str): 자연어로 된 Codex 지시문
        approval_mode (str): 실행 모드 설정
        extra_args (Optional[List[str]]): 추가 CLI 파라미터
        timeout (Optional[int]): 실행 타임아웃 (초)
        env_vars (Optional[Dict[str, str]]): 환경 변수 설정
        
    Returns:
        str: Codex CLI의 실행 결과
        
    Raises:
        CodexError: Codex CLI 실행 실패 시
        CodexTimeoutError: 실행 시간 초과 시
    """
    try:
        return await _codex_cli_async_with_retry(
            prompt,
            approval_mode=approval_mode,
            extra_args=extra_args,
            timeout=timeout,
            env_vars=env_vars
        )
    except Exception as e:
        if isinstance(e, (CodexTimeoutError, asyncio.CancelledError)):
            raise
        if isinstance(e, RetryError):
            cause = e.__cause__
            if isinstance(cause, CodexTimeoutError):
                raise cause
            raise CodexError(str(cause))
        raise CodexError(str(e))


def main() -> NoReturn:
    """CLI 진입점"""
    try:
        print("=== Codex CLI Wrapper Test ===")
        
        # 환경 변수 설정 예시
        test_env = {
            'CODEX_TIMEOUT': '60',
            'CODEX_LOG_LEVEL': 'DEBUG'
        }
        
        print("\nTesting basic execution:")
        out = codex_cli(
            "Generate a Python function that multiplies two numbers",
            env_vars=test_env
        )
        print("\nOutput:")
        print(out)
        
        print("\nTesting streaming execution:")
        codex_cli_stream(
            "Generate a Python class for handling user data",
            env_vars=test_env
        )
        
    except (CodexError, ValueError) as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}")
        print(f"Critical error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 
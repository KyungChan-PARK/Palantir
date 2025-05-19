import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from tenacity import RetryError
from codex_wrapper import (
    codex_cli_async,
    CodexError,
    CodexTimeoutError,
    CodexConfig
)


@pytest.fixture
def mock_process():
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"async test output", b"")
    return process


@pytest.mark.asyncio
async def test_async_success(mock_process):
    """비동기 실행 성공 케이스"""
    with patch('asyncio.create_subprocess_exec',
              return_value=mock_process) as mock_exec:
        result = await codex_cli_async("test prompt")
        assert result == "async test output"
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_async_retry_success(mock_process):
    """재시도 후 성공 케이스"""
    fail_count = 0
    
    async def failing_exec(*args, **kwargs):
        nonlocal fail_count
        process = AsyncMock()
        if fail_count < 2:  # 처음 2번은 실패
            fail_count += 1
            process.returncode = 1
            process.communicate.return_value = (b"", b"error")
        else:  # 3번째 시도에서 성공
            process.returncode = 0
            process.communicate.return_value = (b"success after retry", b"")
        return process
    
    with patch('asyncio.create_subprocess_exec', failing_exec):
        result = await codex_cli_async("test prompt")
        assert result == "success after retry"
        assert fail_count == 2


@pytest.mark.asyncio
async def test_async_retry_failure(mock_process):
    """모든 재시도 실패 케이스"""
    process = AsyncMock()
    process.returncode = 1
    process.communicate.return_value = (b"", b"persistent error")
    
    with patch('asyncio.create_subprocess_exec',
              return_value=process):
        with pytest.raises(RetryError):
            await codex_cli_async("test prompt")


@pytest.mark.asyncio
async def test_async_timeout_handling():
    """타임아웃 처리 테스트"""
    process = AsyncMock()
    
    async def mock_communicate():
        await asyncio.sleep(2)
        return b"", b""
    
    process.communicate = mock_communicate
    process.kill = AsyncMock()
    process.wait = AsyncMock()
    
    with patch('asyncio.create_subprocess_exec',
              return_value=process):
        with pytest.raises(CodexTimeoutError) as exc_info:
            await codex_cli_async("test prompt", timeout=1)
        
        assert str(exc_info.value) == "Async Codex CLI execution timed out after 1 seconds"
        process.kill.assert_called_once()
        process.wait.assert_called_once()


@pytest.mark.asyncio
async def test_async_cancel_handling(mock_process):
    """작업 취소 처리 테스트"""
    process = AsyncMock()
    
    async def cancel_communicate():
        await asyncio.sleep(0.1)
        raise asyncio.CancelledError()
    
    process.communicate = cancel_communicate
    
    with patch('asyncio.create_subprocess_exec',
              return_value=process):
        with pytest.raises(asyncio.CancelledError):
            await codex_cli_async("test prompt")


@pytest.mark.asyncio
async def test_async_custom_config():
    """커스텀 설정 테스트"""
    config = CodexConfig(
        timeout=30,
        max_retries=2,
        wait_min=2,
        wait_max=5
    )
    
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"custom config test", b"")
    
    with patch('asyncio.create_subprocess_exec',
              return_value=process) as mock_exec:
        result = await codex_cli_async(
            "test prompt",
            timeout=config.timeout,
            env_vars={'CODEX_MAX_RETRIES': str(config.max_retries)}
        )
        
        assert result == "custom config test"
        assert mock_exec.call_args.kwargs['env']['CODEX_MAX_RETRIES'] == '2' 
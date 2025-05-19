import pytest
import asyncio
from unittest.mock import patch, MagicMock
from codex_wrapper import (
    codex_cli,
    codex_cli_async,
    CodexError,
    CodexTimeoutError,
    CodexConfig,
    setup_environment
)


@pytest.fixture
def mock_subprocess_run():
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test output"
        yield mock_run


@pytest.fixture
def mock_subprocess_exec():
    async def mock_create_subprocess_exec(*args, **kwargs):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = asyncio.coroutine(
            lambda: (b"async test output", b"")
        )
        return mock_process
    
    with patch('asyncio.create_subprocess_exec', mock_create_subprocess_exec):
        yield


def test_codex_cli_success(mock_subprocess_run):
    """동기 실행 성공 테스트"""
    result = codex_cli("test prompt")
    assert result == "test output"
    mock_subprocess_run.assert_called_once()


def test_codex_cli_error(mock_subprocess_run):
    """동기 실행 실패 테스트"""
    mock_subprocess_run.return_value.returncode = 1
    mock_subprocess_run.return_value.stderr = "error message"
    
    with pytest.raises(CodexError):
        codex_cli("test prompt")


@pytest.mark.asyncio
async def test_codex_cli_async_success(mock_subprocess_exec):
    """비동기 실행 성공 테스트"""
    result = await codex_cli_async("test prompt")
    assert result == "async test output"


@pytest.mark.asyncio
async def test_codex_cli_async_timeout():
    """비동기 실행 타임아웃 테스트"""
    async def slow_subprocess(*args, **kwargs):
        mock_process = MagicMock()
        mock_process.communicate = asyncio.coroutine(
            lambda: asyncio.sleep(2)
        )
        return mock_process
    
    with patch('asyncio.create_subprocess_exec', slow_subprocess):
        with pytest.raises(CodexTimeoutError):
            await codex_cli_async("test prompt", timeout=1)


def test_setup_environment():
    """환경 변수 설정 테스트"""
    test_env = {'CODEX_TIMEOUT': '60', 'CUSTOM_VAR': 'value'}
    env = setup_environment(test_env)
    
    assert env['CODEX_TIMEOUT'] == '60'
    assert env['CUSTOM_VAR'] == 'value'
    assert env.get('CODEX_LOG_LEVEL') == 'INFO'  # 기본값


def test_codex_config():
    """설정 클래스 테스트"""
    config = CodexConfig(
        timeout=60,
        log_level='DEBUG',
        max_retries=5
    )
    
    assert config.timeout == 60
    assert config.log_level == 'DEBUG'
    assert config.max_retries == 5
    assert config.wait_min == 1  # 기본값
    assert config.wait_max == 10  # 기본값 
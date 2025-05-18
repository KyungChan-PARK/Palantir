import subprocess
from typing import Optional, List

def codex_cli(prompt: str,
              approval_mode: str = "suggest",
              extra_args: Optional[List[str]] = None) -> str:
    """
    Run the OpenAI Codex CLI with the given prompt and return its output.

    :param prompt: 자연어로 된 Codex 지시문
    :param approval_mode: 'suggest', 'auto-edit', 'full-auto' 중 선택
    :param extra_args: 추가로 넘기고 싶은 CLI 파라미터 리스트
    :return: Codex CLI가 출력한 텍스트
    :raises RuntimeError: CLI 실행 실패 시 에러 메시지 포함
    """
    cmd = ["codex", "--approval-mode", approval_mode]
    if extra_args:
        cmd.extend(extra_args)
    # 마지막에 prompt 추가
    cmd.append(prompt)

    # Codex CLI 실행
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI error ({result.returncode}):\n{result.stderr}")

    return result.stdout.strip()


def codex_cli_stream(prompt: str,
                     approval_mode: str = "suggest") -> None:
    """
    실시간으로 Codex CLI 출력을 스트리밍하여 화면에 출력합니다.
    """
    cmd = ["codex", "--approval-mode", approval_mode, prompt]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    # 한 줄씩 읽어들이며 실시간 출력
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Codex CLI 스트리밍 실패 ({proc.returncode})")


if __name__ == "__main__":
    # 간단한 예시 호출
    try:
        out = codex_cli("Generate a Python function that multiplies two numbers")
        print("=== Codex Output ===")
        print(out)
    except RuntimeError as e:
        print(f"에러 발생: {e}") 
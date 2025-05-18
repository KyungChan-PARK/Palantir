#!/usr/bin/env python
import os
import yaml
import subprocess
from datetime import datetime
from pathlib import Path

def load_config():
    """Codex 설정 파일을 로드합니다."""
    config_path = Path("config/codex_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("설정 파일을 찾을 수 없습니다: config/codex_config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_codex_path():
    """Codex CLI 실행 파일의 경로를 반환합니다."""
    # Windows 환경에서 Codex CLI 경로
    codex_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python313", "Scripts", "codex-cli.exe")
    if not os.path.exists(codex_path):
        raise FileNotFoundError(f"Codex CLI를 찾을 수 없습니다: {codex_path}")
    return codex_path

def generate_code(prompt, config):
    """Codex CLI를 사용하여 코드를 생성합니다."""
    output_dir = Path(config["output_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Codex CLI 명령어 구성
    command = [
        get_codex_path(),
        f"--model={config['model']}",
        f"--approval-mode={config['approval_mode']}",
        f"--max-tokens={config['max_tokens']}",
        f"--temperature={config['temperature']}",
        prompt
    ]

    # 실행 결과 저장
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    # 로그 저장
    log_path = output_path / "generation.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(command)}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write("\n=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Codex CLI 실행 실패: {result.stderr}")

    return output_path

def main():
    """메인 실행 함수"""
    try:
        config = load_config()
        prompt = input("생성할 코드에 대한 설명을 입력하세요: ")
        
        output_path = generate_code(prompt, config)
        print(f"\n코드 생성이 완료되었습니다. 결과는 다음 경로에서 확인할 수 있습니다:")
        print(f"경로: {output_path}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
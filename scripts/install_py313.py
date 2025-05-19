#!/usr/bin/env python
"""
Python 3.13 패키지 설치 스크립트
compatible.txt / incompatible.txt 기반으로 3단계 설치 전략 실행:
1. 호환 패키지: 일반 설치
2. 비호환 패키지: --pre 시도
3. 실패 시: --no-binary + --ignore-requires-python 시도
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

class InstallError(Exception):
    """패키지 설치 중 발생하는 예외"""
    pass

def run_pip_command(cmd: List[str], env: Optional[dict] = None) -> Tuple[int, str, str]:
    """pip 명령어 실행 및 결과 반환"""
    try:
        env_dict = os.environ.copy()
        if env:
            env_dict.update(env)
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_dict
        )
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        return 1, "", str(e)

def install_package(package: str, strategy: str = "normal",
                   env_vars: Optional[dict] = None) -> Tuple[bool, str]:
    """
    주어진 전략으로 패키지 설치 시도
    
    Args:
        package: 패키지 이름
        strategy: 설치 전략 ("normal", "pre", "source")
        env_vars: 추가 환경 변수
    
    Returns:
        (성공여부, 로그메시지)
    """
    base_cmd = [sys.executable, "-m", "pip", "install"]
    
    if strategy == "normal":
        cmd = base_cmd + [package]
    elif strategy == "pre":
        cmd = base_cmd + ["--pre", package]
    elif strategy == "source":
        cmd = base_cmd + [
            "--no-binary", ":all:",
            "--ignore-requires-python",
            package
        ]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    returncode, stdout, stderr = run_pip_command(cmd, env_vars)
    success = returncode == 0
    log = f"$ {' '.join(cmd)}\n"
    log += stdout if success else stderr
    
    return success, log

def write_markdown_log(content: str, file: Path):
    """Markdown 형식으로 로그 파일 작성"""
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Python 3.13 패키지 설치")
    parser.add_argument("--req", type=Path, default="requirements.txt",
                      help="requirements.txt 파일 경로")
    args = parser.parse_args()
    
    # 로그 파일 초기화
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    install_log = Path("report/install_log.md")
    failures_log = Path("report/build_failures.md")
    
    for log_file in [install_log, failures_log]:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# Python 3.13 패키지 설치 로그\n\n")
            f.write(f"실행 시간: {timestamp}\n\n")
    
    # 호환 패키지 설치
    print("\n1단계: 호환 패키지 설치")
    try:
        with open("compatible.txt") as f:
            compatible_pkgs = f.read().splitlines()
        
        write_markdown_log("## 1. 호환 패키지 설치\n\n```\n", install_log)
        for pkg in compatible_pkgs:
            print(f"  설치 중: {pkg}")
            success, log = install_package(pkg)
            write_markdown_log(f"{log}\n", install_log)
        write_markdown_log("```\n\n", install_log)
            
    except FileNotFoundError:
        print("Warning: compatible.txt 파일을 찾을 수 없습니다.")
        compatible_pkgs = []
    
    # 비호환 패키지 설치
    print("\n2단계: 비호환 패키지 설치")
    try:
        with open("incompatible.txt") as f:
            incompatible_pkgs = f.read().splitlines()
        
        write_markdown_log("## 2. 비호환 패키지 설치\n\n", install_log)
        failed_pkgs = []
        
        for pkg in incompatible_pkgs:
            if pkg == "markdownlint-cli2":
                msg = f"Note: {pkg}는 npm 패키지입니다. 다음 명령어로 설치하세요:\n"
                msg += "```\nnpm install -g markdownlint-cli2\n```\n"
                write_markdown_log(msg, install_log)
                continue
                
            print(f"\n  설치 중: {pkg}")
            env_vars = {"GX_PYTHON_EXPERIMENTAL": "1"} if "great_expectations" in pkg else None
            
            # Step 1: --pre 시도
            print("    --pre 옵션으로 시도 중...")
            write_markdown_log(f"\n### {pkg} (--pre)\n```\n", install_log)
            success, log = install_package(pkg, "pre", env_vars)
            write_markdown_log(f"{log}\n```\n", install_log)
            
            if not success:
                # Step 2: 소스 빌드 시도
                print("    소스 빌드 시도 중...")
                write_markdown_log(f"\n### {pkg} (source build)\n```\n", install_log)
                success, log = install_package(pkg, "source", env_vars)
                write_markdown_log(f"{log}\n```\n", install_log)
            
            if not success:
                failed_pkgs.append(pkg)
                
        # 실패한 패키지 기록
        if failed_pkgs:
            with open(failures_log, "a", encoding="utf-8") as f:
                f.write("\n## 설치 실패한 패키지\n\n")
                for pkg in failed_pkgs:
                    f.write(f"- {pkg}\n")
                f.write("\n### 권장 조치\n\n")
                f.write("1. 최신 프리릴리스 버전 확인\n")
                f.write("2. 소스 코드에서 직접 빌드\n")
                f.write("3. conda-forge 채널 사용 검토\n")
                
    except FileNotFoundError:
        print("Warning: incompatible.txt 파일을 찾을 수 없습니다.")
    
    print("\n설치 완료!")
    print(f"설치 로그: {install_log}")
    if Path(failures_log).exists():
        print(f"실패 보고서: {failures_log}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
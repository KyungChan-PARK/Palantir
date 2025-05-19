#!/usr/bin/env python
"""
Python 3.13 호환성 점검 스크립트
requirements.txt의 패키지들이 Python 3.13을 지원하는지 PyPI JSON API를 통해 확인하고
결과를 Markdown 표로 출력합니다.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
import requests
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from packaging.requirements import Requirement

# Python 3.13 버전 상수
PY_VERSION = Version("3.13")

def clean_package_name(pkg_spec: str) -> str:
    """
    패키지 스펙에서 순수 패키지 이름만 추출
    예: 'package>=1.0.0' -> 'package'
    """
    try:
        req = Requirement(pkg_spec)
        return req.name
    except Exception:
        # 기본적인 정규식으로 패키지명 추출 시도
        match = re.match(r'^([a-zA-Z0-9\-_\.]+)', pkg_spec)
        if match:
            return match.group(1)
        return pkg_spec

def parse_requirements(req_file: Path) -> list[str]:
    """requirements.txt 파일에서 패키지 이름 파싱"""
    packages = []
    try:
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                pkg_name = clean_package_name(line)
                if pkg_name:
                    packages.append(pkg_name)
    except Exception as e:
        print(f"Error reading requirements file: {e}", file=sys.stderr)
        sys.exit(1)
    return packages

def check_package_compatibility(package: str) -> tuple[str, bool, str]:
    """
    PyPI JSON API를 사용해 패키지의 Python 3.13 호환성 확인
    
    Returns:
        tuple[str, bool, str]: (패키지명, 호환성여부, requires_python 스펙)
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        # requires_python 스펙 확인
        requires_python = data["info"]["requires_python"]
        if not requires_python:
            # 명시된 제한이 없으면 호환되는 것으로 간주
            return package, True, "n/a"
            
        # packaging 라이브러리로 버전 호환성 체크
        spec_set = SpecifierSet(requires_python)
        is_compatible = PY_VERSION in spec_set
        
        return package, is_compatible, requires_python
        
    except requests.exceptions.RequestException as e:
        return package, False, f"error: {e}"
    except Exception as e:
        return package, False, f"error: unexpected - {e}"

def generate_markdown_report(results: list[tuple[str, bool, str]], output_file: Path):
    """호환성 결과를 Markdown 표로 저장"""
    compatible_pkgs = []
    incompatible_pkgs = []
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        # 헤더 작성
        f.write("# Python 3.13 패키지 호환성 보고서\n\n")
        f.write("| Package | Requires Python | Python 3.13 지원 |\n")
        f.write("|---------|----------------|------------------|\n")
        
        # 결과 행 작성
        for pkg, is_compatible, spec in sorted(results):
            status = "✔" if is_compatible else "❌"
            f.write(f"| {pkg} | {spec} | {status} |\n")
            
            # 호환성별 패키지 목록 분류
            if is_compatible:
                compatible_pkgs.append(pkg)
            else:
                incompatible_pkgs.append(pkg)
        
        # 통계 요약
        total = len(results)
        compatible_count = len(compatible_pkgs)
        f.write(f"\n## 요약\n\n")
        f.write(f"- 전체 패키지: {total}개\n")
        f.write(f"- Python 3.13 호환: {compatible_count}개 ({compatible_count/total*100:.1f}%)\n")
        f.write(f"- 호환되지 않음: {total - compatible_count}개\n")
    
    # 호환성별 패키지 목록 파일 생성
    with open("compatible.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(compatible_pkgs))
    
    with open("incompatible.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(incompatible_pkgs))

def main():
    parser = argparse.ArgumentParser(description="Python 3.13 호환성 점검")
    parser.add_argument("--req", type=Path, default="requirements.txt",
                      help="requirements.txt 파일 경로")
    parser.add_argument("--out", type=Path, default="report/py313_compat.md",
                      help="출력할 Markdown 파일 경로")
    args = parser.parse_args()

    # requirements.txt 파싱
    print("패키지 목록을 읽는 중...")
    packages = parse_requirements(args.req)
    
    # 각 패키지의 Python 3.13 호환성 확인
    print(f"\n{len(packages)}개 패키지의 호환성을 확인하는 중...")
    results = []
    for pkg in packages:
        print(f"  확인 중: {pkg}")
        result = check_package_compatibility(pkg)
        results.append(result)
    
    # 결과를 Markdown 파일로 저장
    print(f"\n결과를 {args.out}에 저장하는 중...")
    generate_markdown_report(results, args.out)
    
    # 호환성 통계 출력
    compatible = sum(1 for _, is_compatible, _ in results if is_compatible)
    total = len(results)
    print(f"\n호환성 통계:")
    print(f"- 전체 패키지: {total}개")
    print(f"- Python 3.13 호환: {compatible}개 ({compatible/total*100:.1f}%)")
    print(f"- 호환되지 않음: {total - compatible}개")
    print(f"\n상세 보고서: {args.out}")
    print("호환 패키지 목록: compatible.txt")
    print("비호환 패키지 목록: incompatible.txt")
    
    return 0 if compatible == total else 1

if __name__ == "__main__":
    sys.exit(main()) 
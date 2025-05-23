# 로컬 팔란티어 파운드리 구현 프로젝트

이 프로젝트는 1인 개발자 환경에서 팔란티어 파운드리의 핵심 기능을 구현한 시스템입니다.

## 프로젝트 개요

이 프로젝트는 다음 핵심 기능을 구현합니다:
- **온톨로지 관리**: Neo4j 기반 데이터 관계 및 메타데이터 관리
- **데이터 파이프라인**: Apache Airflow 기반 워크플로우 자동화
- **문서 관리**: 문서의 효율적 관리 및 최적화
- **데이터 품질**: Great Expectations 기반 데이터 검증
- **웹 대시보드**: Dash 기반 데이터 시각화 및 관리
- **LLM 통합**: Claude 3.7 Sonnet 기반 AI 페어 프로그래밍 및 자가 개선 워크플로

## ⚡ Quick Start

> 💡 **Windows:** 별도 설정 없이 `make env lint test` 명령을 바로 사용할 수 있습니다. (activate 불필요)

# Python 3.13 프로젝트 환경 설정

## 시스템 요구사항

- Python 3.13
- Node.js 14.x 이상 (markdownlint-cli2용)
- Docker Desktop (Airflow용)

## 설치 방법

### 1. Python 패키지 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. Airflow 설정 (Docker)

```bash
# 필요한 디렉토리 생성
mkdir -p dags logs plugins tests great_expectations

# Airflow 실행
docker compose up airflow

# 웹 UI 접속: http://localhost:8080
# 기본 계정: admin / {AIRFLOW_ADMIN_PASSWORD}
```

### 3. Markdownlint 설치 (Node.js)

```bash
# 전역 설치
npm install -g markdownlint-cli2

# VS Code 확장 설치 (선택사항)
code --install-extension DavidAnson.vscode-markdownlint
```

## 개발 환경 주의사항

### Python 3.13 호환성

- `great_expectations`: 실험적 지원 활성화됨 (`GX_PYTHON_EXPERIMENTAL=1`)
- `apache-airflow`: Docker 이미지로 실행 (Python 3.12 기반)
- 기타 패키지: 모두 Python 3.13 호환 확인됨

### 버전 관리

- Python 패키지: `requirements.txt`에 버전 고정
- Airflow: Docker 이미지 태그로 버전 관리 (`slim-3.0.1-python3.12`)
- Markdownlint: npm으로 별도 관리

## 테스트 실행

```bash
# 단위 테스트
pytest tests/

# DAG 파싱 테스트
docker compose exec airflow python -m pytest tests/dags/

# Markdownlint 검사
markdownlint-cli2 "**/*.md"
```

## CI/CD 파이프라인

GitHub Actions에서 다음 작업이 자동화됩니다:

1. Python 3.13 호환성 검사
2. 단위 테스트 실행
3. Airflow DAG 검증
4. Markdown 문법 검사
5. Docker 이미지 빌드 및 푸시

자세한 내용은 `.github/workflows/` 디렉토리의 워크플로우 파일을 참고하세요.

## 문제 해결

### Python 3.13 관련 이슈

1. 패키지 설치 실패 시:
   ```bash
   # 프리릴리스 버전 시도
   pip install --pre <package>
   
   # 소스 빌드 시도
   pip install --no-binary :all: --ignore-requires-python <package>
   ```

2. Great Expectations 오류 시:
   ```bash
   # 환경변수 설정 후 재시도
   export GX_PYTHON_EXPERIMENTAL=1
   pip install great_expectations
   ```

### Airflow 관련 이슈

1. Docker 컨테이너 접속:
   ```bash
   docker compose exec airflow bash
   ```

2. 로그 확인:
   ```bash
   docker compose logs -f airflow
   ```

## 유지보수

- Python 3.13 지원: 2029년 10월까지 보안 패치 제공 (PEP 719)
- 정기적인 의존성 업데이트 권장 (`pip-audit` 또는 `safety` 사용)
- Docker 이미지 태그는 명시적 버전 사용 (latest 태그 지양)
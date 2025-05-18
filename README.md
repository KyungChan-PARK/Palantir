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

```bash
# 1) 가상환경
poetry install

# 2) 컨테이너 스택 (Docker Compose)
docker compose up -d   # neo4j, postgres, airflow 세 서비스 기동

# 3) Ontology & Test Data
poetry run python scripts/init_neo4j.py

# 4) Dash 앱
poetry run python analysis/tools/dash_knowledge_map.py
```
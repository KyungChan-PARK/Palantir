# 📁 Directory Structure v2

> **핵심 컨셉** – **Ontology-First**:  
> 모든 데이터·파이프라인·앱은 `docs/ontology_guide.md`에 정의된 **공통 객체/링크 모델**을 참조한다.

| Path | Purpose |
|------|---------|
| `docs/` | 설계·가이드 문서. **Ontology**, **Lineage/ETL**, **DevOps** 각 섹션 구분. |
| `analysis/airflow/dags/` | **Apache Airflow** DAGs.<br>모든 ETL·검증·온톨로지 업데이트 파이프라인은 여기서 관리. |
| `analysis/tools/` | Dash 앱·그래프 알고리즘 스크립트 등 분석 도구. |
| `ai_resources/prompts/` | **Cursor AI / Codex CLI** 자동화용 프롬프트 세트. |
| `config/` | YAML/JSON 설정 파일. 🔑 보안 항목은 `.env`에 별도 저장. |
| `scripts/` | 원클릭 초기화 유틸리티(Neo4j 스키마·데모 데이터 로드 등). |
| `data/` | ✔ *Source* 원천<br>✔ *Interim* 중간 산출<br>✔ *Processed* 분석·ML 입력 |
| `output/` | ETL 결과 리포트, 대시보드 스냅샷 |
| `temp/` | 임시 작업 영역 (자동 git-ignore) |

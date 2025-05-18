# Data Lineage & ETL Guide

[ETL Quick Start](quick_start_etl.md)

- **Airflow DAG ID** : `ontology_lineage_pipeline`
- **Update Frequency** : `@hourly`
- **Lineage Edge 패턴**
  ```cypher
  MERGE (src:Table {name:$source})
  MERGE (dst:Table {name:$target})
  MERGE (src)-[:TRANSFORMS {dag:$dag, ts:timestamp()}]->(dst)
  ```

| Task                 | Operator                    | Output                     |
| -------------------- | --------------------------- | -------------------------- |
| `extract_lms_logs`   | `PythonOperator`            | raw CSV (S3)               |
| `clean_logs`         | `SparkSubmitOperator`       | parquet (`/data/interim/`) |
| `load_to_pg`         | `PostgresOperator`          | `public.raw_logs`          |
| `sync_neo4j`         | `PythonOperator`            | `(:LogEntry)` 그래프 노드       |
| `ge_expect_validate` | `GreatExpectationsOperator` | HTML 검증 리포트                |

> **GraphXR Tip** – `TRANSFORMS.duration` 색상 매핑으로 병목 태스크 시각 즉시 파악. 
"""
Airflow DAG: ontology_lineage_pipeline
Runs hourly.

1) Extract LMS logs
2) Transform (Spark)
3) Load Postgres
4) Update Neo4j ontology instances
5) Write lineage edges

Cursor AI & Codex CLI ready (PEP-8).
"""
from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from common.neo4j_utils import load_csv_to_neo4j, write_lineage_edge
from pathlib import Path
import os
import boto3
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import great_expectations as ge

EXPECT_PATH = Path("data/quality/expectations/clean.json")

def extract_logs(**_):
    """S3에서 LMS 로그 CSV를 다운로드하여 data/source/raw_logs.csv 로 저장"""
    bucket = os.getenv("LMS_LOG_BUCKET", "my-lms-logs")
    key = os.getenv("LMS_LOG_KEY", "raw_logs/raw_logs.csv")
    dest_path = Path("data/source/raw_logs.csv")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dest_path))


def clean_logs(**_):
    """Spark를 사용해 원시 로그를 정제하여 data/interim/clean.parquet 저장"""
    pass


def load_to_pg(**_):
    """data/interim/clean.csv → Postgres raw_logs 테이블로 적재"""
    hook = PostgresHook(postgres_conn_id="ontology_pg")
    engine = hook.get_sqlalchemy_engine()
    df = pd.read_csv("data/interim/clean.csv")
    df.to_sql("raw_logs", engine, if_exists="replace", index=False)


def validate_ge(**_):
    """Great Expectations를 사용해 clean.csv 데이터 품질을 검증한다."""
    df = pd.read_csv("data/interim/clean.csv")
    gdf = ge.from_pandas(df)
    # 기본 검증 규칙: null 없음, 필드 타입 확인
    results = [
        gdf.expect_column_values_to_not_be_null("student_id"),
        gdf.expect_column_values_to_not_be_null("course"),
        gdf.expect_column_values_to_not_be_null("timestamp"),
    ]
    if not all(r.success for r in results):
        raise ValueError("Great Expectations validation failed")

    if not EXPECT_PATH.exists():
        from scripts.generate_ge_suite import generate_suite
        generate_suite(Path("data/interim/clean.csv"))
    suite = ge.core.ExpectationSuite(expectation_suite_name="clean")
    suite = ge.core.ExpectationSuite(**(ge.data_context.types.base.generate_expectation_suite_from_json(str(EXPECT_PATH))))
    # run validation
    result = gdf.validate(expectation_suite=suite)
    if not result.success:
        raise ValueError("Great Expectations validation failed")


def sync_neo4j(**_):
    load_csv_to_neo4j("data/interim/clean.csv", label="LogEntry")
    write_lineage_edge("raw_logs.csv", "clean.csv", dag="ontology_lineage_pipeline")


with DAG(
    dag_id="ontology_lineage_pipeline",
    start_date=datetime(2025, 5, 17),
    schedule_interval="@hourly",
    catchup=False,
) as dag:
    t1 = PythonOperator(task_id="extract_lms_logs", python_callable=extract_logs)
    t2 = DockerOperator(
        task_id="clean_logs",
        image="apache/spark:latest",
        command="spark-submit analysis/airflow/jobs/clean_logs.py",
        auto_remove=True,
    )
    t3 = PythonOperator(task_id="load_to_pg", python_callable=load_to_pg)
    t4 = PythonOperator(task_id="sync_neo4j", python_callable=sync_neo4j)
    t5 = PythonOperator(task_id="ge_expect_validate", python_callable=validate_ge)

    t1 >> t2 >> t3 >> t4 >> t5 
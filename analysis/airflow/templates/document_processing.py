"""
문서 처리 파이프라인 DAG 템플릿
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# 필요한 모듈 임포트
import os
import sys
import logging
from pathlib import Path

# 프로젝트 모듈 로드
project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.atoms.document_processor import scan_documents, process_document_text
from analysis.atoms.document_processor import extract_document_metadata, save_processed_documents

# 기본 인수 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    '{{ dag_id }}',
    default_args=default_args,
    description='{{ description }}',
    schedule_interval='{{ schedule }}',
    catchup=False
)

# 전역 설정
INPUT_PATH = '{{ input_path }}'
OUTPUT_PATH = '{{ output_path }}'
PROCESS_TYPE = '{{ process_type }}'

# 태스크 1: 문서 스캔
scan_task = PythonOperator(
    task_id='scan_documents',
    python_callable=scan_documents,
    op_kwargs={'input_path': INPUT_PATH},
    dag=dag
)

# 태스크 2: 문서 텍스트 처리 (선택적)
if PROCESS_TYPE in ['all', 'text']:
    process_text_task = PythonOperator(
        task_id='process_document_text',
        python_callable=process_document_text,
        op_kwargs={'output_path': OUTPUT_PATH},
        dag=dag
    )
    scan_task >> process_text_task

# 태스크 3: 문서 메타데이터 추출 (선택적)
if PROCESS_TYPE in ['all', 'metadata']:
    extract_metadata_task = PythonOperator(
        task_id='extract_document_metadata',
        python_callable=extract_document_metadata,
        op_kwargs={'output_path': OUTPUT_PATH},
        dag=dag
    )
    scan_task >> extract_metadata_task

# 태스크 4: 처리된 문서 저장
save_task = PythonOperator(
    task_id='save_processed_documents',
    python_callable=save_processed_documents,
    op_kwargs={'output_path': OUTPUT_PATH},
    dag=dag
)

# 태스크 의존성 설정
if PROCESS_TYPE == 'all':
    [process_text_task, extract_metadata_task] >> save_task
elif PROCESS_TYPE == 'text':
    process_text_task >> save_task
elif PROCESS_TYPE == 'metadata':
    extract_metadata_task >> save_task
else:
    scan_task >> save_task

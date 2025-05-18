"""
온톨로지 처리 파이프라인 DAG 템플릿
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

from analysis.molecules.ontology_manager import OntologyManager

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
CONFIG_PATH = '{{ config_path }}'
BASE_ONTOLOGY_PATH = '{{ base_ontology_path }}'
OUTPUT_PATH = '{{ output_path }}'

# 온톨로지 관리자 초기화 함수
def init_ontology_manager():
    return OntologyManager(config_path=CONFIG_PATH)

# 태스크 1: 온톨로지 스키마 초기화
def initialize_ontology_schema(**kwargs):
    ontology = init_ontology_manager()
    try:
        result = ontology.initialize_ontology_schema()
        return result
    finally:
        ontology.close()

schema_task = PythonOperator(
    task_id='initialize_ontology_schema',
    python_callable=initialize_ontology_schema,
    dag=dag
)

# 태스크 2: 기본 온톨로지 가져오기
def import_base_ontology(**kwargs):
    ontology = init_ontology_manager()
    try:
        if os.path.exists(BASE_ONTOLOGY_PATH):
            result = ontology.import_ontology_from_json(BASE_ONTOLOGY_PATH)
            return result
        else:
            logging.warning(f"기본 온톨로지 파일이 존재하지 않음: {BASE_ONTOLOGY_PATH}")
            return {"status": "warning", "message": "기본 온톨로지 파일이 존재하지 않음"}
    finally:
        ontology.close()

import_task = PythonOperator(
    task_id='import_base_ontology',
    python_callable=import_base_ontology,
    dag=dag
)

# 태스크 3: 온톨로지 내보내기
def export_ontology(**kwargs):
    ontology = init_ontology_manager()
    try:
        # 출력 디렉토리가 존재하는지 확인
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        result = ontology.export_ontology_to_json(OUTPUT_PATH)
        return result
    finally:
        ontology.close()

export_task = PythonOperator(
    task_id='export_ontology',
    python_callable=export_ontology,
    dag=dag
)

# 태스크 4: 온톨로지 시각화
def visualize_ontology(**kwargs):
    ontology = init_ontology_manager()
    try:
        visualization_data = ontology.visualize_ontology()
        
        # 시각화 데이터 저장
        vis_path = os.path.join(os.path.dirname(OUTPUT_PATH), "ontology_visualization.json")
        
        import json
        with open(vis_path, 'w', encoding='utf-8') as f:
            json.dump(visualization_data, f, indent=2, ensure_ascii=False)
        
        return {"visualization_path": vis_path}
    finally:
        ontology.close()

visualize_task = PythonOperator(
    task_id='visualize_ontology',
    python_callable=visualize_ontology,
    dag=dag
)

# 태스크 의존성 설정
schema_task >> import_task >> export_task >> visualize_task

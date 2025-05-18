"""
데이터 파이프라인: 품질 모니터링 파이프라인

이 DAG는 데이터 품질을 모니터링하는 작업을 수행합니다:
1. 판매 데이터 로드: 데이터 소스에서 판매 데이터를 로드
2. 데이터 품질 검증: Great Expectations를 활용한 데이터 검증
3. 품질 보고서 생성: 검증 결과 보고서 생성
4. 알림 생성: 품질 이슈가 있는 경우 알림 생성
5. 검증 결과 저장: 검증 결과와 메트릭을 온톨로지에 저장
"""

from datetime import datetime, timedelta
import os
import sys
import json
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

# 프로젝트 루트 경로를 Python 경로에 추가
PALANTIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PALANTIR_ROOT)

# 필요한 모듈 임포트
from analysis.atoms.data_connector import DataConnector
from analysis.atoms.neo4j_connector import Neo4jConnector
from analysis.molecules.quality_validator import QualityValidator
from analysis.molecules.ontology_manager import OntologyManager
from analysis.molecules.notification_manager import NotificationManager
from airflow.hooks.base import BaseHook

# 기본 인수 정의
default_args = {
    'owner': 'palantir',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'data_quality_pipeline',
    default_args=default_args,
    description='데이터 품질 모니터링 파이프라인',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['data', 'quality', 'monitoring'],
)

# 공용 구성 로드
def load_config():
    with open(os.path.join(PALANTIR_ROOT, 'config', 'app_config.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

# Neo4j 커넥션 로드
def get_neo4j_connector():
    conn = BaseHook.get_connection('neo4j_default')
    uri = conn.host
    if conn.port:
        uri = f"{uri}:{conn.port}"
    if not uri.startswith('bolt://'):
        uri = f"bolt://{uri}"
    return Neo4jConnector(uri=uri, username=conn.login, password=conn.password)

# 품질 기대치 로드
def load_expectations(expectation_path):
    with open(expectation_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 1. 판매 데이터 로드
def load_sales_data(**kwargs):
    config = load_config()
    data_connector = DataConnector(config['data_sources'])
    
    # 데이터 소스에서 판매 데이터 로드
    sales_data_path = config['data_sources']['sales_data_path']
    sales_data = data_connector.load_csv(sales_data_path)
    
    # 임시 파일로 저장 (Airflow task 간 전달을 위해)
    temp_path = os.path.join(PALANTIR_ROOT, 'temp', 'sales_data_temp.csv')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    sales_data.to_csv(temp_path, index=False)
    
    print(f"로드된 판매 데이터: {len(sales_data)} 행")
    return temp_path

# 2. 데이터 품질 검증
def validate_data_quality(**kwargs):
    ti = kwargs['ti']
    sales_data_path = ti.xcom_pull(task_ids='load_sales_data')
    
    # 기대치 파일 로드
    expectations_path = os.path.join(PALANTIR_ROOT, 'data', 'quality', 'sales_expectations.json')
    expectations = load_expectations(expectations_path)
    
    # 검증 실행
    validator = QualityValidator()
    validation_results = validator.validate_pandas_df(
        pd.read_csv(sales_data_path),
        expectations
    )
    
    # 결과 임시 저장
    results_path = os.path.join(PALANTIR_ROOT, 'temp', 'validation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    return results_path

# 3. 품질 보고서 생성
def generate_quality_report(**kwargs):
    ti = kwargs['ti']
    validation_results_path = ti.xcom_pull(task_ids='validate_data_quality')
    
    with open(validation_results_path, 'r', encoding='utf-8') as f:
        validation_results = json.load(f)
    
    # 보고서 생성
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_expectations': validation_results['summary']['total_expectations'],
            'passed_expectations': validation_results['summary']['passed_expectations'],
            'failed_expectations': validation_results['summary']['failed_expectations'],
            'success_percent': validation_results['summary']['success_percent']
        },
        'column_results': validation_results['column_results'],
        'table_results': validation_results['table_results'],
        'details': validation_results['details']
    }
    
    # 보고서 저장
    report_path = os.path.join(PALANTIR_ROOT, 'reports', 'quality', 
                              f'quality_report_{datetime.now().strftime("%Y%m%d")}.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report_path

# 4. 알림 생성
def generate_alerts(**kwargs):
    ti = kwargs['ti']
    validation_results_path = ti.xcom_pull(task_ids='validate_data_quality')
    config = load_config()
    
    with open(validation_results_path, 'r', encoding='utf-8') as f:
        validation_results = json.load(f)
    
    # 실패한 검증이 있는 경우 알림 생성
    notification_manager = NotificationManager(config['notifications'])
    alerts = []
    
    if validation_results['summary']['failed_expectations'] > 0:
        # 실패한 기대치에 대한 알림 생성
        for detail in validation_results['details']:
            if not detail['success']:
                alert = {
                    'level': 'warning',
                    'title': f'데이터 품질 이슈 발견: {detail["expectation_type"]}',
                    'message': detail['message'],
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                
                # 알림 발송
                notification_manager.send_notification(alert)
    
    # 알림 저장
    if alerts:
        alerts_path = os.path.join(PALANTIR_ROOT, 'alerts', 
                                 f'quality_alerts_{datetime.now().strftime("%Y%m%d")}.json')
        os.makedirs(os.path.dirname(alerts_path), exist_ok=True)
        
        with open(alerts_path, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
        
        return alerts_path
    else:
        return "알림 없음"

# 5. 검증 결과 저장
def store_validation_results(**kwargs):
    ti = kwargs['ti']
    validation_results_path = ti.xcom_pull(task_ids='validate_data_quality')
    config = load_config()
    
    with open(validation_results_path, 'r', encoding='utf-8') as f:
        validation_results = json.load(f)
    
    # Neo4j에 결과 저장
    neo4j_connector = get_neo4j_connector()
    ontology_manager = OntologyManager(neo4j_connector)
    
    # 검증 실행 기록 생성
    validation_id = ontology_manager.create_validation_record(
        validation_type='SalesDataQuality',
        timestamp=datetime.now().isoformat(),
        success_rate=validation_results['summary']['success_percent'],
        total_expectations=validation_results['summary']['total_expectations'],
        passed_expectations=validation_results['summary']['passed_expectations'],
        failed_expectations=validation_results['summary']['failed_expectations']
    )
    
    # 실패한 항목들에 대한 상세 기록 생성
    for detail in validation_results['details']:
        if not detail['success']:
            issue_id = ontology_manager.create_quality_issue(
                validation_id=validation_id,
                issue_type=detail['expectation_type'],
                message=detail['message'],
                severity='warning'
            )
    
    return f"검증 결과 저장 완료: ID {validation_id}"

# 작업 정의
start = DummyOperator(
    task_id='start',
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_sales_data',
    python_callable=load_sales_data,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    provide_context=True,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_quality_report',
    python_callable=generate_quality_report,
    provide_context=True,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='generate_alerts',
    python_callable=generate_alerts,
    provide_context=True,
    dag=dag,
)

store_task = PythonOperator(
    task_id='store_validation_results',
    python_callable=store_validation_results,
    provide_context=True,
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag,
)

# 작업 의존성 설정
start >> load_task >> validate_task >> [report_task, alert_task, store_task] >> end

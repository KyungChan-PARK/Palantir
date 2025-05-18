"""
데이터 품질 검증 파이프라인

이 DAG는 다음과 같은 데이터 품질 검증 작업을 수행합니다:
1. 데이터 소스에서 데이터 로드
2. Great Expectations를 사용한 데이터 검증
3. 검증 결과 보고서 생성
4. 문제가 있는 경우 알림 발송
"""

from datetime import datetime, timedelta
import os
import json
import pandas as pd
import great_expectations as ge
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

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
    'data_quality_validation',
    default_args=default_args,
    description='데이터 품질 검증 파이프라인',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['data', 'quality', 'validation'],
)

def load_data(**context):
    """데이터 소스에서 데이터를 로드합니다."""
    # 데이터 소스 경로 설정
    data_path = os.path.join(os.getcwd(), 'data', 'source', 'raw_data.csv')
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    
    # XCom에 데이터 경로 저장
    context['ti'].xcom_push(key='data_path', value=data_path)
    
    return f"데이터 로드 완료: {data_path}"

def validate_data(**context):
    """Great Expectations를 사용하여 데이터를 검증합니다."""
    # 데이터 경로 가져오기
    data_path = context['ti'].xcom_pull(key='data_path')
    
    # Great Expectations 컨텍스트 생성
    context = ge.get_context()
    
    # 데이터 소스 생성
    datasource = context.sources.add_pandas(name="my_pandas_source")
    
    # 데이터 에셋 생성
    asset = datasource.add_pandas_dataframe_asset(
        name="my_dataframe",
        dataframe=pd.read_csv(data_path)
    )
    
    # 검증 스위트 생성
    suite = context.add_or_update_expectation_suite("my_suite")
    
    # 검증 규칙 정의
    validator = context.get_validator(
        batch_request=asset.build_batch_request(),
        expectation_suite=suite
    )
    
    # 검증 규칙 추가
    validator.expect_column_values_to_not_be_null("id")
    validator.expect_column_values_to_be_unique("id")
    validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
    
    # 검증 실행
    results = validator.validate()
    
    # 결과 저장
    output_path = os.path.join(os.getcwd(), 'output', 'validation_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results.to_json_dict(), f, indent=2)
    
    # XCom에 결과 경로 저장
    context['ti'].xcom_push(key='validation_results_path', value=output_path)
    
    return f"데이터 검증 완료: {output_path}"

def generate_report(**context):
    """검증 결과 보고서를 생성합니다."""
    # 검증 결과 경로 가져오기
    results_path = context['ti'].xcom_pull(key='validation_results_path')
    
    # 결과 로드
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # 보고서 생성
    report_path = os.path.join(os.getcwd(), 'output', 'validation_report.html')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # HTML 보고서 생성
    html_content = f"""
    <html>
    <head>
        <title>데이터 품질 검증 보고서</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>데이터 품질 검증 보고서</h1>
        <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>검증 결과</h2>
        <ul>
    """
    
    for result in results['results']:
        status = "success" if result['success'] else "failure"
        html_content += f"""
            <li class="{status}">
                {result['expectation_config']['expectation_type']}: 
                {result['expectation_config']['kwargs']}
                - {'성공' if result['success'] else '실패'}
            </li>
        """
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # XCom에 보고서 경로 저장
    context['ti'].xcom_push(key='report_path', value=report_path)
    
    return f"보고서 생성 완료: {report_path}"

def send_notification(**context):
    """검증 결과에 따라 알림을 발송합니다."""
    # 검증 결과 경로 가져오기
    results_path = context['ti'].xcom_pull(key='validation_results_path')
    
    # 결과 로드
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # 실패한 검증 확인
    failures = [r for r in results['results'] if not r['success']]
    
    if failures:
        # 실패가 있는 경우 알림 발송
        print(f"경고: {len(failures)}개의 검증 실패가 발견되었습니다.")
        for failure in failures:
            print(f"- {failure['expectation_config']['expectation_type']}: {failure['expectation_config']['kwargs']}")
    else:
        print("모든 검증이 성공적으로 완료되었습니다.")
    
    return "알림 발송 완료"

# 태스크 정의
start = DummyOperator(task_id='start', dag=dag)
load = PythonOperator(task_id='load_data', python_callable=load_data, dag=dag)
validate = PythonOperator(task_id='validate_data', python_callable=validate_data, dag=dag)
report = PythonOperator(task_id='generate_report', python_callable=generate_report, dag=dag)
notify = PythonOperator(task_id='send_notification', python_callable=send_notification, dag=dag)
end = DummyOperator(task_id='end', dag=dag)

# 태스크 의존성 설정
start >> load >> validate >> report >> notify >> end 
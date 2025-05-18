"""
데이터 품질 검사 파이프라인 DAG 템플릿
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# 필요한 모듈 임포트
import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 모듈 로드
project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

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
DATA_PATH = '{{ data_path }}'
EXPECTATIONS_PATH = '{{ expectations_path }}'
OUTPUT_PATH = '{{ output_path }}'
DATA_TYPE = '{{ data_type }}'  # csv, excel, json, sql

# 데이터 로드 함수
def load_data(**kwargs):
    """데이터 로드 및 기본 통계 생성"""
    try:
        # 데이터 타입에 따라 다른 로드 방법 사용
        if DATA_TYPE.lower() == 'csv':
            df = pd.read_csv(DATA_PATH)
        elif DATA_TYPE.lower() == 'excel':
            df = pd.read_excel(DATA_PATH)
        elif DATA_TYPE.lower() == 'json':
            df = pd.read_json(DATA_PATH)
        elif DATA_TYPE.lower() == 'sql':
            # SQL 쿼리 실행 (구성 파일에서 연결 정보 로드 필요)
            import sqlalchemy
            # 연결 문자열 및 쿼리 로드
            with open(DATA_PATH, 'r') as f:
                query_info = json.load(f)
            
            engine = sqlalchemy.create_engine(query_info['connection_string'])
            df = pd.read_sql(query_info['query'], engine)
        else:
            raise ValueError(f"지원되지 않는 데이터 타입: {DATA_TYPE}")
        
        # 기본 통계 계산
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        # 데이터프레임 및 통계 반환 (XCom에 저장)
        # 주의: 데이터프레임은 직렬화하여 저장
        kwargs['ti'].xcom_push(key='dataframe', value=df.to_json(orient='split'))
        kwargs['ti'].xcom_push(key='stats', value=stats)
        
        return stats
    except Exception as e:
        logging.error(f"데이터 로드 중 오류: {str(e)}")
        raise

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

# 데이터 검증 함수
def validate_data(**kwargs):
    """기대치에 따른 데이터 검증"""
    try:
        # 이전 태스크에서 데이터프레임 가져오기
        df_json = kwargs['ti'].xcom_pull(task_ids='load_data', key='dataframe')
        df = pd.read_json(df_json, orient='split')
        
        # 기대치 로드
        if os.path.exists(EXPECTATIONS_PATH):
            with open(EXPECTATIONS_PATH, 'r') as f:
                expectations = json.load(f)
        else:
            # 기본 기대치 생성
            columns = df.columns.tolist()
            expectations = {
                "column_expectations": {col: {"not_null": True} for col in columns},
                "table_expectations": {
                    "row_count_min": 1
                }
            }
        
        # 검증 결과 초기화
        validation_results = {
            "passed": True,
            "column_results": {},
            "table_results": {},
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0
            }
        }
        
        # 열 기대치 검증
        for col, checks in expectations.get("column_expectations", {}).items():
            if col in df.columns:
                validation_results["column_results"][col] = {}
                
                # 각 검사 실행
                for check_name, check_value in checks.items():
                    validation_results["summary"]["total_checks"] += 1
                    
                    if check_name == "not_null" and check_value:
                        # Null 값 검사
                        null_count = df[col].isnull().sum()
                        check_result = null_count == 0
                        validation_results["column_results"][col][check_name] = {
                            "passed": check_result,
                            "details": {
                                "null_count": null_count
                            }
                        }
                    elif check_name == "unique" and check_value:
                        # 고유값 검사
                        unique_count = df[col].nunique()
                        duplicates = len(df) - unique_count
                        check_result = duplicates == 0
                        validation_results["column_results"][col][check_name] = {
                            "passed": check_result,
                            "details": {
                                "unique_count": unique_count,
                                "duplicates": duplicates
                            }
                        }
                    elif check_name == "min_value":
                        # 최소값 검사
                        min_value = df[col].min()
                        check_result = min_value >= check_value
                        validation_results["column_results"][col][check_name] = {
                            "passed": check_result,
                            "details": {
                                "actual_min": min_value,
                                "expected_min": check_value
                            }
                        }
                    elif check_name == "max_value":
                        # 최대값 검사
                        max_value = df[col].max()
                        check_result = max_value <= check_value
                        validation_results["column_results"][col][check_name] = {
                            "passed": check_result,
                            "details": {
                                "actual_max": max_value,
                                "expected_max": check_value
                            }
                        }
                    elif check_name == "allowed_values":
                        # 허용 값 검사
                        invalid_values = df[~df[col].isin(check_value)][col].unique()
                        check_result = len(invalid_values) == 0
                        validation_results["column_results"][col][check_name] = {
                            "passed": check_result,
                            "details": {
                                "invalid_values": invalid_values.tolist() if len(invalid_values) > 0 else []
                            }
                        }
                    
                    # 검사 결과 업데이트
                    if check_name in validation_results["column_results"][col]:
                        if validation_results["column_results"][col][check_name]["passed"]:
                            validation_results["summary"]["passed_checks"] += 1
                        else:
                            validation_results["summary"]["failed_checks"] += 1
                            validation_results["passed"] = False
            else:
                # 열이 존재하지 않는 경우
                validation_results["column_results"][col] = {
                    "error": f"열 '{col}'이(가) 데이터프레임에 존재하지 않음"
                }
                validation_results["summary"]["total_checks"] += 1
                validation_results["summary"]["failed_checks"] += 1
                validation_results["passed"] = False
        
        # 테이블 수준 기대치 검증
        table_expectations = expectations.get("table_expectations", {})
        validation_results["table_results"] = {}
        
        # 행 수 최소 검사
        if "row_count_min" in table_expectations:
            validation_results["summary"]["total_checks"] += 1
            min_rows = table_expectations["row_count_min"]
            check_result = len(df) >= min_rows
            validation_results["table_results"]["row_count_min"] = {
                "passed": check_result,
                "details": {
                    "actual": len(df),
                    "expected_min": min_rows
                }
            }
            
            if check_result:
                validation_results["summary"]["passed_checks"] += 1
            else:
                validation_results["summary"]["failed_checks"] += 1
                validation_results["passed"] = False
        
        # 행 수 최대 검사
        if "row_count_max" in table_expectations:
            validation_results["summary"]["total_checks"] += 1
            max_rows = table_expectations["row_count_max"]
            check_result = len(df) <= max_rows
            validation_results["table_results"]["row_count_max"] = {
                "passed": check_result,
                "details": {
                    "actual": len(df),
                    "expected_max": max_rows
                }
            }
            
            if check_result:
                validation_results["summary"]["passed_checks"] += 1
            else:
                validation_results["summary"]["failed_checks"] += 1
                validation_results["passed"] = False
        
        # 필수 열 검사
        if "required_columns" in table_expectations:
            validation_results["summary"]["total_checks"] += 1
            required_cols = table_expectations["required_columns"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            check_result = len(missing_cols) == 0
            validation_results["table_results"]["required_columns"] = {
                "passed": check_result,
                "details": {
                    "missing_columns": missing_cols
                }
            }
            
            if check_result:
                validation_results["summary"]["passed_checks"] += 1
            else:
                validation_results["summary"]["failed_checks"] += 1
                validation_results["passed"] = False
        
        # 결과 반환 (XCom에 저장)
        kwargs['ti'].xcom_push(key='validation_results', value=validation_results)
        
        return validation_results
    except Exception as e:
        logging.error(f"데이터 검증 중 오류: {str(e)}")
        raise

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

# 결과 보고서 생성 함수
def generate_report(**kwargs):
    """데이터 품질 검사 결과 보고서 생성"""
    try:
        # 이전 태스크에서 결과 가져오기
        stats = kwargs['ti'].xcom_pull(task_ids='load_data', key='stats')
        validation_results = kwargs['ti'].xcom_pull(task_ids='validate_data', key='validation_results')
        
        # 보고서 생성
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_path": DATA_PATH,
            "data_type": DATA_TYPE,
            "stats": stats,
            "validation_results": validation_results
        }
        
        # 출력 디렉토리가 존재하는지 확인
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        # JSON 보고서 저장
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Markdown 보고서 생성
        md_path = os.path.splitext(OUTPUT_PATH)[0] + '.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 데이터 품질 검사 보고서\n\n")
            f.write(f"- **검사 시간**: {report['timestamp']}\n")
            f.write(f"- **데이터 경로**: {report['data_path']}\n")
            f.write(f"- **데이터 유형**: {report['data_type']}\n\n")
            
            f.write(f"## 데이터 통계\n\n")
            f.write(f"- **행 수**: {stats['row_count']}\n")
            f.write(f"- **열 수**: {stats['column_count']}\n")
            f.write(f"- **열 목록**: {', '.join(stats['columns'])}\n\n")
            
            f.write(f"## 검증 결과 요약\n\n")
            summary = validation_results['summary']
            overall_status = "✅ 성공" if validation_results['passed'] else "❌ 실패"
            f.write(f"- **전체 결과**: {overall_status}\n")
            f.write(f"- **총 검사 수**: {summary['total_checks']}\n")
            f.write(f"- **성공한 검사 수**: {summary['passed_checks']}\n")
            f.write(f"- **실패한 검사 수**: {summary['failed_checks']}\n\n")
            
            if summary['failed_checks'] > 0:
                f.write(f"## 실패한 검사 상세\n\n")
                
                # 열 검사 실패
                for col, checks in validation_results['column_results'].items():
                    for check_name, result in checks.items():
                        if isinstance(result, dict) and 'passed' in result and not result['passed']:
                            f.write(f"### 열: {col}, 검사: {check_name}\n\n")
                            f.write(f"- **결과**: ❌ 실패\n")
                            
                            for key, value in result['details'].items():
                                f.write(f"- **{key}**: {value}\n")
                            
                            f.write("\n")
                
                # 테이블 검사 실패
                for check_name, result in validation_results['table_results'].items():
                    if isinstance(result, dict) and 'passed' in result and not result['passed']:
                        f.write(f"### 테이블 검사: {check_name}\n\n")
                        f.write(f"- **결과**: ❌ 실패\n")
                        
                        for key, value in result['details'].items():
                            f.write(f"- **{key}**: {value}\n")
                        
                        f.write("\n")
        
        return {
            "report_path": OUTPUT_PATH,
            "markdown_path": md_path,
            "passed": validation_results['passed']
        }
    except Exception as e:
        logging.error(f"보고서 생성 중 오류: {str(e)}")
        raise

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag
)

# 태스크 의존성 설정
load_task >> validate_task >> report_task

"""
데이터 파이프라인: 온톨로지 관리 파이프라인

이 DAG는 온톨로지 관리 관련 작업을 수행합니다:
1. 온톨로지 상태 확인: Neo4j 온톨로지 구조와 데이터의 일관성 확인
2. 온톨로지 스키마 동기화: 정의된 스키마와 실제 Neo4j 구조 동기화
3. 온톨로지 데이터 정리: 오래된/중복 데이터 정리
4. 온톨로지 관계 검증: 객체 간 관계의 유효성 검증
5. 온톨로지 백업: 현재 온톨로지 상태 백업
"""

from datetime import datetime, timedelta
import os
import sys
import json
import shutil

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

# 프로젝트 루트 경로를 Python 경로에 추가
PALANTIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PALANTIR_ROOT)

# 필요한 모듈 임포트
from analysis.atoms.neo4j_connector import Neo4jConnector
from analysis.molecules.ontology_manager import OntologyManager
from analysis.molecules.notification_manager import NotificationManager

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
    'ontology_management_pipeline',
    default_args=default_args,
    description='온톨로지 관리 파이프라인',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['ontology', 'management', 'neo4j'],
)

# 공용 구성 로드
def load_config():
    with open(os.path.join(PALANTIR_ROOT, 'config', 'app_config.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

# 1. 온톨로지 상태 확인
def check_ontology_status(**kwargs):
    config = load_config()
    
    neo4j_connector = Neo4jConnector(
        uri=config['neo4j']['uri'],
        username=config['neo4j']['username'],
        password=config['neo4j']['password']
    )
    ontology_manager = OntologyManager(neo4j_connector)
    
    # 온톨로지 상태 확인
    status = ontology_manager.check_status()
    
    # 결과 저장
    status_path = os.path.join(PALANTIR_ROOT, 'temp', 'ontology_status.json')
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    
    return status_path

# 2. 온톨로지 스키마 동기화
def sync_ontology_schema(**kwargs):
    ti = kwargs['ti']
    status_path = ti.xcom_pull(task_ids='check_ontology_status')
    config = load_config()
    
    with open(status_path, 'r', encoding='utf-8') as f:
        status = json.load(f)
    
    neo4j_connector = Neo4jConnector(
        uri=config['neo4j']['uri'],
        username=config['neo4j']['username'],
        password=config['neo4j']['password']
    )
    ontology_manager = OntologyManager(neo4j_connector)
    
    # 온톨로지 정의 파일 로드
    ontology_def_path = os.path.join(PALANTIR_ROOT, 'config', 'ontology_definition.json')
    with open(ontology_def_path, 'r', encoding='utf-8') as f:
        ontology_def = json.load(f)
    
    # 스키마 동기화
    sync_results = ontology_manager.sync_schema(ontology_def)
    
    # 결과 저장
    sync_path = os.path.join(PALANTIR_ROOT, 'temp', 'ontology_sync_results.json')
    with open(sync_path, 'w', encoding='utf-8') as f:
        json.dump(sync_results, f, ensure_ascii=False, indent=2)
    
    return sync_path

# 3. 온톨로지 데이터 정리
def clean_ontology_data(**kwargs):
    config = load_config()
    
    neo4j_connector = Neo4jConnector(
        uri=config['neo4j']['uri'],
        username=config['neo4j']['username'],
        password=config['neo4j']['password']
    )
    ontology_manager = OntologyManager(neo4j_connector)
    
    # 데이터 정리 (오래된 데이터, 중복 등)
    clean_results = {
        'orphan_nodes_removed': ontology_manager.remove_orphan_nodes(),
        'duplicate_relationships_removed': ontology_manager.remove_duplicate_relationships(),
        'invalid_properties_fixed': ontology_manager.fix_invalid_properties()
    }
    
    # 결과 저장
    clean_path = os.path.join(PALANTIR_ROOT, 'temp', 'ontology_clean_results.json')
    with open(clean_path, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # 주요 이슈가 있으면 알림 생성
    if (clean_results['orphan_nodes_removed'] > 10 or 
        clean_results['duplicate_relationships_removed'] > 10 or
        clean_results['invalid_properties_fixed'] > 10):
        
        notification_manager = NotificationManager(config['notifications'])
        notification_manager.send_notification({
            'level': 'warning',
            'title': '온톨로지 데이터 정리 알림',
            'message': f"다수의 이슈가 해결됨: {clean_results['orphan_nodes_removed']} 고아 노드, " +
                      f"{clean_results['duplicate_relationships_removed']} 중복 관계, " +
                      f"{clean_results['invalid_properties_fixed']} 잘못된 속성",
            'timestamp': datetime.now().isoformat()
        })
    
    return clean_path

# 4. 온톨로지 관계 검증
def validate_ontology_relationships(**kwargs):
    config = load_config()
    
    neo4j_connector = Neo4jConnector(
        uri=config['neo4j']['uri'],
        username=config['neo4j']['username'],
        password=config['neo4j']['password']
    )
    ontology_manager = OntologyManager(neo4j_connector)
    
    # 관계 검증
    validation_results = ontology_manager.validate_relationships()
    
    # 결과 저장
    validation_path = os.path.join(PALANTIR_ROOT, 'temp', 'ontology_validation_results.json')
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    # 검증 실패가 있는 경우 알림 생성
    if validation_results['invalid_relationships'] > 0:
        notification_manager = NotificationManager(config['notifications'])
        notification_manager.send_notification({
            'level': 'warning',
            'title': '온톨로지 관계 검증 알림',
            'message': f"{validation_results['invalid_relationships']} 개의 잘못된 관계가 발견되었습니다.",
            'timestamp': datetime.now().isoformat()
        })
    
    return validation_path

# 5. 온톨로지 백업
def backup_ontology(**kwargs):
    config = load_config()
    
    neo4j_connector = Neo4jConnector(
        uri=config['neo4j']['uri'],
        username=config['neo4j']['username'],
        password=config['neo4j']['password']
    )
    ontology_manager = OntologyManager(neo4j_connector)
    
    # 현재 날짜로 백업 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(PALANTIR_ROOT, 'backups', 'ontology')
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_file = os.path.join(backup_dir, f"ontology_backup_{timestamp}.json")
    
    # 온톨로지 내보내기
    ontology_data = ontology_manager.export_ontology()
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(ontology_data, f, ensure_ascii=False, indent=2)
    
    # 백업 보관 정책 (30일 이상 된 백업 삭제)
    retention_days = 30
    for file in os.listdir(backup_dir):
        file_path = os.path.join(backup_dir, file)
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        if (datetime.now() - file_mtime).days > retention_days:
            os.remove(file_path)
            print(f"오래된 백업 파일 삭제: {file}")
    
    return backup_file

# 작업 정의
start = DummyOperator(
    task_id='start',
    dag=dag,
)

check_task = PythonOperator(
    task_id='check_ontology_status',
    python_callable=check_ontology_status,
    provide_context=True,
    dag=dag,
)

sync_task = PythonOperator(
    task_id='sync_ontology_schema',
    python_callable=sync_ontology_schema,
    provide_context=True,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_ontology_data',
    python_callable=clean_ontology_data,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_ontology_relationships',
    python_callable=validate_ontology_relationships,
    provide_context=True,
    dag=dag,
)

backup_task = PythonOperator(
    task_id='backup_ontology',
    python_callable=backup_ontology,
    provide_context=True,
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag,
)

# 작업 의존성 설정
start >> check_task >> sync_task >> clean_task >> validate_task >> backup_task >> end

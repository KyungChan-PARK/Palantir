"""
데이터 파이프라인: 문서 처리 파이프라인

이 DAG는 다음 작업을 수행합니다:
1. 새 문서 감지: OneDrive에서 새로운 문서 파일 스캔
2. 문서 메타데이터 추출: 문서에서 메타데이터 추출
3. 텍스트 추출: 문서에서 텍스트 내용 추출
4. 온톨로지 업데이트: Neo4j 온톨로지에 문서 정보 추가
5. 문서 인덱싱: 검색을 위한 인덱싱
"""

from datetime import datetime, timedelta
import os
import sys
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

# 프로젝트 루트 경로를 Python 경로에 추가
PALANTIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PALANTIR_ROOT)

# 필요한 모듈 임포트
from analysis.atoms.onedrive_connector import OneDriveConnector
from analysis.atoms.neo4j_connector import Neo4jConnector
from analysis.molecules.ontology_manager import OntologyManager
from analysis.molecules.document_processor import DocumentProcessor
from analysis.molecules.indexer import Indexer
from airflow.hooks.base import BaseHook

# 기본 인수 정의
default_args = {
    'owner': 'palantir',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'document_processing_pipeline',
    default_args=default_args,
    description='문서 처리 파이프라인',
    schedule_interval=timedelta(hours=1),
    start_date=days_ago(1),
    tags=['document', 'processing'],
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

# 1. 새 문서 감지
def detect_new_documents(**kwargs):
    config = load_config()
    onedrive = OneDriveConnector(config['onedrive'])
    documents_dir = config['documents']['source_dir']
    
    # 새 문서 확인 (마지막 실행 이후 변경된 파일)
    new_docs = onedrive.list_new_files(documents_dir)
    
    print(f"감지된 새 문서: {len(new_docs)} 개")
    return new_docs

# 2. 문서 메타데이터 추출
def extract_metadata(**kwargs):
    ti = kwargs['ti']
    new_docs = ti.xcom_pull(task_ids='detect_new_documents')
    config = load_config()
    
    doc_processor = DocumentProcessor(config)
    results = []
    
    for doc_path in new_docs:
        metadata = doc_processor.extract_metadata(doc_path)
        results.append({
            'path': doc_path,
            'metadata': metadata
        })
    
    return results

# 3. 텍스트 추출
def extract_text(**kwargs):
    ti = kwargs['ti']
    docs_with_metadata = ti.xcom_pull(task_ids='extract_metadata')
    config = load_config()
    
    doc_processor = DocumentProcessor(config)
    results = []
    
    for doc in docs_with_metadata:
        text_content = doc_processor.extract_text(doc['path'])
        doc['text_content'] = text_content
        results.append(doc)
    
    return results

# 4. 온톨로지 업데이트
def update_ontology(**kwargs):
    ti = kwargs['ti']
    processed_docs = ti.xcom_pull(task_ids='extract_text')
    config = load_config()
    
    neo4j_connector = get_neo4j_connector()
    ontology_manager = OntologyManager(neo4j_connector)
    
    for doc in processed_docs:
        # 문서 객체 생성 또는 업데이트
        doc_id = ontology_manager.create_document(
            title=doc['metadata'].get('title', os.path.basename(doc['path'])),
            doc_type=doc['metadata'].get('doc_type', 'Unknown'),
            created_date=doc['metadata'].get('created_date'),
            modified_date=doc['metadata'].get('modified_date'),
            author=doc['metadata'].get('author', 'Unknown'),
            path=doc['path'],
            properties=doc['metadata']
        )
        
        # 문서 관계 설정 (예: 문서가 언급하는 다른 엔티티들)
        if 'entities' in doc['metadata'] and doc['metadata']['entities']:
            for entity in doc['metadata']['entities']:
                ontology_manager.create_relationship(
                    source_id=doc_id,
                    target_type=entity['type'],
                    target_name=entity['name'],
                    relationship_type='MENTIONS'
                )
    
    return [doc['path'] for doc in processed_docs]

# 5. 문서 인덱싱
def index_documents(**kwargs):
    ti = kwargs['ti']
    processed_docs = ti.xcom_pull(task_ids='extract_text')
    doc_paths = ti.xcom_pull(task_ids='update_ontology')
    config = load_config()
    
    indexer = Indexer(config['indexer'])
    
    for doc in processed_docs:
        indexer.add_document(
            doc_id=os.path.basename(doc['path']),
            content=doc['text_content'],
            metadata=doc['metadata']
        )
    
    indexer.commit()
    return f"인덱싱 완료: {len(processed_docs)} 문서"

# 작업 정의
start = DummyOperator(
    task_id='start',
    dag=dag,
)

detect_task = PythonOperator(
    task_id='detect_new_documents',
    python_callable=detect_new_documents,
    provide_context=True,
    dag=dag,
)

metadata_task = PythonOperator(
    task_id='extract_metadata',
    python_callable=extract_metadata,
    provide_context=True,
    dag=dag,
)

text_task = PythonOperator(
    task_id='extract_text',
    python_callable=extract_text,
    provide_context=True,
    dag=dag,
)

ontology_task = PythonOperator(
    task_id='update_ontology',
    python_callable=update_ontology,
    provide_context=True,
    dag=dag,
)

index_task = PythonOperator(
    task_id='index_documents',
    python_callable=index_documents,
    provide_context=True,
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag,
)

# 작업 의존성 설정
start >> detect_task >> metadata_task >> text_task >> ontology_task >> index_task >> end

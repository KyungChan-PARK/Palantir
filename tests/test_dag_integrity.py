"""Airflow DAG 파싱 테스트
DAG 파일이 문법 오류 없이 로드되는지 확인한다."""
from airflow.models import DagBag

 
def test_dagbag_import():
    dagbag = DagBag(dag_folder="analysis/airflow/dags", include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}" 
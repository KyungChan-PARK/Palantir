"""
Apache Airflow 기반 데이터 파이프라인 관리 시스템
"""

import os
import yaml
import logging
import json
import time
from datetime import datetime, timedelta
import importlib.util
import tempfile
import shutil
import re

from analysis.atoms.airflow_connector import AirflowConnector

logger = logging.getLogger(__name__)

class DataPipelineManager:
    """Apache Airflow 기반 데이터 파이프라인 관리 클래스"""
    
    def __init__(self, config_path=None, connector=None):
        """
        데이터 파이프라인 관리자 초기화
        
        Args:
            config_path (str, optional): 구성 파일 경로
            connector (AirflowConnector, optional): 기존 Airflow 연결기
        """
        # Airflow 연결기 초기화
        if connector:
            self.airflow = connector
        else:
            if config_path:
                self.airflow = AirflowConnector(config_path=config_path)
            else:
                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "airflow.yaml"
                )
                self.airflow = AirflowConnector(config_path=default_config_path)
        
        # 구성 파일 경로 저장
        self.config_path = config_path
        
        # 템플릿 디렉토리 경로
        self.template_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "analysis", "airflow", "templates"
        )
        
        # 템플릿 디렉토리가 존재하는지 확인
        os.makedirs(self.template_dir, exist_ok=True)
        
        logger.info("데이터 파이프라인 관리자 초기화 완료")
    
    def initialize_airflow(self, reset=False):
        """
        Airflow 초기화
        
        Args:
            reset (bool, optional): 기존 DB 초기화 여부
        
        Returns:
            bool: 성공 여부
        """
        return self.airflow.initialize_airflow(reset=reset)
    
    def start_airflow(self, wait_for_webserver=True, max_retries=10):
        """
        Airflow 시작
        
        Args:
            wait_for_webserver (bool, optional): 웹서버 시작 대기 여부
            max_retries (int, optional): 웹서버 시작 대기 최대 재시도 횟수
        
        Returns:
            dict: 프로세스 정보
        """
        # Airflow 프로세스 시작
        processes = self.airflow.start_airflow()
        
        # 웹서버 시작 대기
        if wait_for_webserver:
            self.airflow.wait_for_webserver(max_retries=max_retries)
        
        return processes
    
    def stop_airflow(self, processes=None):
        """
        Airflow 중지
        
        Args:
            processes (dict, optional): 프로세스 정보
        
        Returns:
            bool: 성공 여부
        """
        return self.airflow.stop_airflow(processes=processes)
    
    def create_dag_from_template(self, template_name, dag_id, description, schedule=None, params=None):
        """
        템플릿에서 DAG 생성
        
        Args:
            template_name (str): 템플릿 이름
            dag_id (str): 생성할 DAG ID
            description (str): DAG 설명
            schedule (str, optional): 스케줄 표현식
            params (dict, optional): 템플릿 매개변수
        
        Returns:
            str: 생성된 DAG 파일 경로
        """
        try:
            # 템플릿 파일 경로
            template_path = os.path.join(self.template_dir, f"{template_name}.py")
            
            # 템플릿 파일이 존재하는지 확인
            if not os.path.exists(template_path):
                logger.error(f"템플릿 파일 '{template_name}.py'을(를) 찾을 수 없음")
                return None
            
            # 템플릿 파일 읽기
            with open(template_path, 'r', encoding='utf-8') as file:
                template_content = file.read()
            
            # 기본 매개변수 설정
            if params is None:
                params = {}
            
            # DAG ID 및 설명 추가
            params["dag_id"] = dag_id
            params["description"] = description
            params["schedule"] = schedule or '@daily'  # 기본값: 매일
            
            # 템플릿에 매개변수 적용
            dag_content = template_content
            
            for key, value in params.items():
                if isinstance(value, str):
                    dag_content = dag_content.replace(f"{{{{ {key} }}}}", value)
                else:
                    dag_content = dag_content.replace(f"{{{{ {key} }}}}", str(value))
            
            # 임시 DAG 파일 생성
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.py')
            temp_file.close()
            
            with open(temp_file.name, 'w', encoding='utf-8') as file:
                file.write(dag_content)
            
            # DAG 파일 배포
            dag_file_name = f"{dag_id}.py"
            dest_path = os.path.join(self.airflow.dags_folder, dag_file_name)
            
            shutil.copy2(temp_file.name, dest_path)
            os.unlink(temp_file.name)
            
            logger.info(f"DAG '{dag_id}'이(가) 템플릿 '{template_name}'에서 생성됨")
            
            # DAG 목록 갱신
            self.airflow.refresh_dags()
            
            return dest_path
        except Exception as e:
            logger.error(f"템플릿에서 DAG 생성 중 오류: {str(e)}")
            return None
    
    def create_custom_dag(self, dag_id, description, schedule, tasks, dependencies=None):
        """
        사용자 정의 DAG 생성 (Python 코드 생성)
        
        Args:
            dag_id (str): DAG ID
            description (str): DAG 설명
            schedule (str): 스케줄 표현식
            tasks (list): 태스크 목록 [{'id': 'task_id', 'callable': 'module.function', 'params': {...}}]
            dependencies (list, optional): 태스크 의존성 목록 [('upstream_task_id', 'downstream_task_id')]
        
        Returns:
            str: 생성된 DAG 파일 경로
        """
        try:
            if dependencies is None:
                dependencies = []
            
            # DAG 파일 내용 생성
            dag_code = [
                '"""',
                f'데이터 파이프라인: {dag_id}',
                f'설명: {description}',
                '"""',
                '',
                'from datetime import datetime, timedelta',
                'from airflow import DAG',
                'from airflow.operators.python import PythonOperator',
                '',
                '# 가져오기 및 함수 정의',
            ]
            
            # 함수 가져오기 추가
            imports = set()
            for task in tasks:
                if 'callable' in task:
                    callable_path = task['callable']
                    module_path = callable_path.rsplit('.', 1)[0]
                    imports.add(f"from {module_path} import {callable_path.rsplit('.', 1)[1]}")
            
            for import_stmt in sorted(imports):
                dag_code.append(import_stmt)
            
            dag_code.extend([
                '',
                '# 기본 인수 설정',
                'default_args = {',
                "    'owner': 'airflow',",
                "    'depends_on_past': False,",
                f"    'start_date': datetime({datetime.now().year}, {datetime.now().month}, {datetime.now().day}),",
                "    'email_on_failure': False,",
                "    'email_on_retry': False,",
                "    'retries': 1,",
                "    'retry_delay': timedelta(minutes=5),",
                '}',
                '',
                '# DAG 정의',
                f"dag = DAG(",
                f"    '{dag_id}',",
                f"    default_args=default_args,",
                f"    description='{description}',",
                f"    schedule_interval='{schedule}',",
                f"    catchup=False",
                f")",
                '',
                '# 태스크 정의',
            ])
            
            # 태스크 정의 추가
            for task in tasks:
                task_id = task['id']
                callable_name = task['callable'].rsplit('.', 1)[1] if 'callable' in task else None
                params = task.get('params', {})
                
                if callable_name:
                    params_str = ', '.join([f"{k}={repr(v)}" for k, v in params.items()])
                    
                    dag_code.extend([
                        f"{task_id} = PythonOperator(",
                        f"    task_id='{task_id}',",
                        f"    python_callable={callable_name},",
                        f"    op_kwargs={{{params_str}}},",
                        f"    dag=dag",
                        f")",
                        ''
                    ])
            
            # 태스크 의존성 추가
            if dependencies:
                dag_code.append('# 태스크 의존성 설정')
                
                for upstream, downstream in dependencies:
                    dag_code.append(f"{upstream} >> {downstream}")
            
            # DAG 파일 저장
            dag_file_name = f"{dag_id}.py"
            dest_path = os.path.join(self.airflow.dags_folder, dag_file_name)
            
            with open(dest_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(dag_code))
            
            logger.info(f"사용자 정의 DAG '{dag_id}'이(가) 생성됨")
            
            # DAG 목록 갱신
            self.airflow.refresh_dags()
            
            return dest_path
        except Exception as e:
            logger.error(f"사용자 정의 DAG 생성 중 오류: {str(e)}")
            return None
    
    def deploy_dag_file(self, file_path, overwrite=True):
        """
        DAG 파일 배포
        
        Args:
            file_path (str): DAG 파일 경로
            overwrite (bool, optional): 기존 파일 덮어쓰기 여부
        
        Returns:
            bool: 성공 여부
        """
        return self.airflow.deploy_dag(file_path, overwrite=overwrite)
    
    def get_dag_id_from_file(self, file_path):
        """
        파일에서 DAG ID 추출
        
        Args:
            file_path (str): DAG 파일 경로
        
        Returns:
            str: DAG ID, 찾을 수 없으면 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # DAG ID 추출 패턴
            pattern = r"dag\s*=\s*DAG\s*\(\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, content)
            
            if match:
                return match.group(1)
            else:
                logger.warning(f"파일 '{file_path}'에서 DAG ID를 찾을 수 없음")
                return None
        except Exception as e:
            logger.error(f"파일에서 DAG ID 추출 중 오류: {str(e)}")
            return None
    
    def trigger_dag_run(self, dag_id, conf=None, wait_for_completion=False, check_interval=5, timeout=300):
        """
        DAG 실행 트리거 및 완료 대기
        
        Args:
            dag_id (str): DAG ID
            conf (dict, optional): 실행 구성
            wait_for_completion (bool, optional): 완료 대기 여부
            check_interval (int, optional): 상태 확인 간격 (초)
            timeout (int, optional): 타임아웃 (초)
        
        Returns:
            dict: 실행 결과
        """
        try:
            # DAG 실행 트리거
            run_id = self.airflow.trigger_dag(dag_id, conf=conf)
            
            if not run_id:
                logger.error(f"DAG '{dag_id}' 트리거 실패")
                return {
                    "status": "failed",
                    "message": "DAG 트리거 실패",
                    "run_id": None
                }
            
            # 완료 대기가 활성화되지 않았으면 즉시 반환
            if not wait_for_completion:
                return {
                    "status": "triggered",
                    "message": "DAG 트리거됨",
                    "run_id": run_id
                }
            
            # DAG 실행 완료 대기
            start_time = time.time()
            last_state = None
            
            while time.time() - start_time < timeout:
                # 실행 상태 확인
                dag_runs = self.airflow.get_dag_runs(dag_id, limit=10)
                
                if dag_runs:
                    for run in dag_runs:
                        if run.get("run_id") == run_id:
                            current_state = run.get("state")
                            
                            if current_state != last_state:
                                logger.info(f"DAG '{dag_id}' 실행 상태: {current_state}")
                                last_state = current_state
                            
                            # 종료 상태 확인
                            if current_state in ["success", "failed"]:
                                return {
                                    "status": current_state,
                                    "message": f"DAG 실행 {current_state}",
                                    "run_id": run_id,
                                    "dag_run": run
                                }
                
                # 잠시 대기
                time.sleep(check_interval)
            
            # 타임아웃
            logger.warning(f"DAG '{dag_id}' 실행 타임아웃")
            return {
                "status": "timeout",
                "message": "DAG 실행 타임아웃",
                "run_id": run_id
            }
        except Exception as e:
            logger.error(f"DAG '{dag_id}' 트리거 및 대기 중 오류: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "run_id": None
            }
    
    def get_pipeline_status(self, dag_id=None, limit=10):
        """
        파이프라인 상태 조회
        
        Args:
            dag_id (str, optional): DAG ID (None이면 모든 DAG)
            limit (int, optional): 최대 결과 수
        
        Returns:
            dict: 파이프라인 상태 정보
        """
        try:
            if dag_id:
                # 특정 DAG 상태 조회
                dag_runs = self.airflow.get_dag_runs(dag_id, limit=limit)
                dag_details = self.airflow.get_dag_details(dag_id)
                
                return {
                    "dag_id": dag_id,
                    "details": dag_details,
                    "runs": dag_runs
                }
            else:
                # 모든 DAG 목록 및 개별 상태 조회
                dags = self.airflow.list_dags()
                result = {
                    "dags": dags,
                    "count": len(dags),
                    "details": {}
                }
                
                for dag in dags:
                    dag_id = dag.get("dag_id")
                    if dag_id:
                        dag_runs = self.airflow.get_dag_runs(dag_id, limit=1)
                        result["details"][dag_id] = {
                            "latest_run": dag_runs[0] if dag_runs else None,
                            "schedule": dag.get("schedule_interval"),
                            "is_active": dag.get("is_active", False)
                        }
                
                return result
        except Exception as e:
            logger.error(f"파이프라인 상태 조회 중 오류: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def register_pipeline_template(self, template_name, template_content, overwrite=False):
        """
        파이프라인 템플릿 등록
        
        Args:
            template_name (str): 템플릿 이름
            template_content (str): 템플릿 내용
            overwrite (bool, optional): 기존 템플릿 덮어쓰기 여부
        
        Returns:
            str: 템플릿 파일 경로
        """
        try:
            # 템플릿 디렉토리가 존재하는지 확인
            os.makedirs(self.template_dir, exist_ok=True)
            
            # 템플릿 파일 경로
            template_path = os.path.join(self.template_dir, f"{template_name}.py")
            
            # 템플릿 파일이 이미 존재하는지 확인
            if os.path.exists(template_path) and not overwrite:
                logger.warning(f"템플릿 '{template_name}'이(가) 이미 존재하며 덮어쓰기가 비활성화되어 있습니다")
                return None
            
            # 템플릿 파일 저장
            with open(template_path, 'w', encoding='utf-8') as file:
                file.write(template_content)
            
            logger.info(f"템플릿 '{template_name}'이(가) 등록됨 (경로: {template_path})")
            return template_path
        except Exception as e:
            logger.error(f"템플릿 등록 중 오류: {str(e)}")
            return None
    
    def list_pipeline_templates(self):
        """
        등록된 파이프라인 템플릿 목록 조회
        
        Returns:
            list: 템플릿 이름 목록
        """
        try:
            # 템플릿 디렉토리가 존재하는지 확인
            if not os.path.exists(self.template_dir):
                return []
            
            # 템플릿 파일 목록 조회
            templates = []
            
            for file_name in os.listdir(self.template_dir):
                if file_name.endswith('.py'):
                    template_name = file_name[:-3]  # .py 확장자 제거
                    templates.append(template_name)
            
            return templates
        except Exception as e:
            logger.error(f"템플릿 목록 조회 중 오류: {str(e)}")
            return []
    
    def get_pipeline_template(self, template_name):
        """
        파이프라인 템플릿 내용 조회
        
        Args:
            template_name (str): 템플릿 이름
        
        Returns:
            str: 템플릿 내용
        """
        try:
            # 템플릿 파일 경로
            template_path = os.path.join(self.template_dir, f"{template_name}.py")
            
            # 템플릿 파일이 존재하는지 확인
            if not os.path.exists(template_path):
                logger.error(f"템플릿 '{template_name}'을(를) 찾을 수 없음")
                return None
            
            # 템플릿 파일 읽기
            with open(template_path, 'r', encoding='utf-8') as file:
                template_content = file.read()
            
            return template_content
        except Exception as e:
            logger.error(f"템플릿 내용 조회 중 오류: {str(e)}")
            return None
    
    def delete_pipeline_template(self, template_name):
        """
        파이프라인 템플릿 삭제
        
        Args:
            template_name (str): 템플릿 이름
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 템플릿 파일 경로
            template_path = os.path.join(self.template_dir, f"{template_name}.py")
            
            # 템플릿 파일이 존재하는지 확인
            if not os.path.exists(template_path):
                logger.error(f"템플릿 '{template_name}'을(를) 찾을 수 없음")
                return False
            
            # 템플릿 파일 삭제
            os.remove(template_path)
            
            logger.info(f"템플릿 '{template_name}'이(가) 삭제됨")
            return True
        except Exception as e:
            logger.error(f"템플릿 삭제 중 오류: {str(e)}")
            return False
    
    def generate_ontology_dag(self, dag_id, description, schedule, output_path):
        """
        온톨로지 처리 DAG 생성
        
        Args:
            dag_id (str): DAG ID
            description (str): DAG 설명
            schedule (str): 스케줄 표현식
            output_path (str): 온톨로지 출력 경로
        
        Returns:
            str: 생성된 DAG 파일 경로
        """
        try:
            # 태스크 정의
            tasks = [
                {
                    "id": "check_ontology_schema",
                    "callable": "analysis.molecules.ontology_manager.OntologyManager.initialize_ontology_schema",
                    "params": {}
                },
                {
                    "id": "import_base_ontology",
                    "callable": "analysis.molecules.ontology_manager.OntologyManager.import_ontology_from_json",
                    "params": {
                        "json_file_path": "{{ base_ontology_path }}"
                    }
                },
                {
                    "id": "export_ontology",
                    "callable": "analysis.molecules.ontology_manager.OntologyManager.export_ontology_to_json",
                    "params": {
                        "json_file_path": output_path
                    }
                }
            ]
            
            # 태스크 의존성 정의
            dependencies = [
                ("check_ontology_schema", "import_base_ontology"),
                ("import_base_ontology", "export_ontology")
            ]
            
            # DAG 생성
            return self.create_custom_dag(
                dag_id=dag_id,
                description=description,
                schedule=schedule,
                tasks=tasks,
                dependencies=dependencies
            )
        except Exception as e:
            logger.error(f"온톨로지 처리 DAG 생성 중 오류: {str(e)}")
            return None
    
    def generate_document_processing_dag(self, dag_id, description, schedule, 
                                       input_path, output_path, process_type="all"):
        """
        문서 처리 DAG 생성
        
        Args:
            dag_id (str): DAG ID
            description (str): DAG 설명
            schedule (str): 스케줄 표현식
            input_path (str): 입력 문서 경로
            output_path (str): 출력 문서 경로
            process_type (str, optional): 처리 유형 (all, text, metadata)
        
        Returns:
            str: 생성된 DAG 파일 경로
        """
        try:
            # 태스크 정의
            tasks = [
                {
                    "id": "scan_documents",
                    "callable": "analysis.atoms.document_processor.scan_documents",
                    "params": {
                        "input_path": input_path
                    }
                }
            ]
            
            # 처리 유형에 따라 태스크 추가
            if process_type in ["all", "text"]:
                tasks.append({
                    "id": "process_document_text",
                    "callable": "analysis.atoms.document_processor.process_document_text",
                    "params": {
                        "output_path": output_path
                    }
                })
            
            if process_type in ["all", "metadata"]:
                tasks.append({
                    "id": "extract_document_metadata",
                    "callable": "analysis.atoms.document_processor.extract_document_metadata",
                    "params": {
                        "output_path": output_path
                    }
                })
            
            # 최종 태스크 추가
            tasks.append({
                "id": "save_processed_documents",
                "callable": "analysis.atoms.document_processor.save_processed_documents",
                "params": {
                    "output_path": output_path
                }
            })
            
            # 태스크 의존성 정의
            dependencies = [
                ("scan_documents", "process_document_text" if process_type in ["all", "text"] else "save_processed_documents"),
                ("scan_documents", "extract_document_metadata" if process_type in ["all", "metadata"] else None)
            ]
            
            # 의존성에서 None 항목 제거
            dependencies = [d for d in dependencies if d[1] is not None]
            
            # 마지막 태스크 연결
            if process_type == "all":
                dependencies.extend([
                    ("process_document_text", "save_processed_documents"),
                    ("extract_document_metadata", "save_processed_documents")
                ])
            elif process_type == "text":
                dependencies.append(("process_document_text", "save_processed_documents"))
            elif process_type == "metadata":
                dependencies.append(("extract_document_metadata", "save_processed_documents"))
            
            # DAG 생성
            return self.create_custom_dag(
                dag_id=dag_id,
                description=description,
                schedule=schedule,
                tasks=tasks,
                dependencies=dependencies
            )
        except Exception as e:
            logger.error(f"문서 처리 DAG 생성 중 오류: {str(e)}")
            return None

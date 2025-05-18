"""
데이터 파이프라인 관리 모듈

Apache Airflow 기반 데이터 파이프라인 관리 시스템을 제공합니다.
이 시스템은 워크플로우를 자동화하고 모니터링합니다.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("data_pipeline")

class DataPipelineSystem:
    """데이터 파이프라인 관리 시스템 클래스"""
    
    def __init__(self, airflow_config_dir: str):
        """
        Args:
            airflow_config_dir: Airflow 구성 디렉토리 경로
        """
        self.airflow_config_dir = airflow_config_dir
        self.template_dir = os.path.join(airflow_config_dir, "templates")
        self.dags_dir = os.path.join(airflow_config_dir, "dags")
        
        # DAG 템플릿 디렉토리와 DAG 디렉토리 생성
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.dags_dir, exist_ok=True)
        
        logger.info(f"데이터 파이프라인 관리 시스템 초기화: airflow_config_dir={airflow_config_dir}")
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """등록된 파이프라인 목록 조회
        
        Returns:
            파이프라인 정보 목록
        """
        try:
            logger.info("파이프라인 목록 조회")
            
            pipelines = []
            
            # DAG 디렉토리의 파이썬 파일 탐색
            for file_name in os.listdir(self.dags_dir):
                if file_name.endswith(".py"):
                    file_path = os.path.join(self.dags_dir, file_name)
                    
                    # 파일 생성 및 수정 시간
                    stat = os.stat(file_path)
                    created_time = datetime.fromtimestamp(stat.st_ctime)
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    # 파일 내용에서 DAG ID 추출
                    dag_id = self._extract_dag_id(file_path)
                    
                    pipelines.append({
                        "file_name": file_name,
                        "file_path": file_path,
                        "dag_id": dag_id,
                        "created_at": created_time.isoformat(),
                        "modified_at": modified_time.isoformat()
                    })
            
            return pipelines
        except Exception as e:
            logger.error(f"파이프라인 목록 조회 오류: {e}")
            raise
    
    async def create_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """새 파이프라인 생성
        
        Args:
            pipeline_config: 파이프라인 구성 정보
                - name: 파이프라인 이름
                - description: 파이프라인 설명
                - schedule: 스케줄 간격 (예: "@daily", "0 0 * * *")
                - tasks: 태스크 목록
                - dependencies: 태스크 의존성 목록
                
        Returns:
            생성된 파이프라인 정보
        """
        try:
            name = pipeline_config.get("name")
            description = pipeline_config.get("description", "")
            schedule = pipeline_config.get("schedule", "@daily")
            tasks = pipeline_config.get("tasks", [])
            dependencies = pipeline_config.get("dependencies", [])
            
            if not name:
                raise ValueError("파이프라인 이름은 필수입니다.")
            
            if not tasks:
                raise ValueError("최소한 하나의 태스크가 필요합니다.")
            
            # DAG ID 생성 (공백 및 특수문자 제거)
            dag_id = name.lower().replace(" ", "_").replace("-", "_")
            
            # DAG 파일 경로
            dag_file = os.path.join(self.dags_dir, f"{dag_id}.py")
            
            # DAG 코드 생성
            dag_code = self._generate_dag_code(dag_id, name, description, schedule, tasks, dependencies)
            
            # DAG 파일 저장
            with open(dag_file, "w", encoding="utf-8") as file:
                file.write(dag_code)
            
            logger.info(f"파이프라인 생성 완료: {name} (dag_id: {dag_id})")
            
            return {
                "name": name,
                "description": description,
                "dag_id": dag_id,
                "schedule": schedule,
                "file_path": dag_file,
                "task_count": len(tasks),
                "created_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"파이프라인 생성 오류: {e}")
            raise
    
    async def delete_pipeline(self, dag_id: str) -> Dict[str, Any]:
        """파이프라인 삭제
        
        Args:
            dag_id: 삭제할 DAG ID
            
        Returns:
            삭제 결과 정보
        """
        try:
            # DAG 파일 경로
            dag_file = os.path.join(self.dags_dir, f"{dag_id}.py")
            
            # 파일 존재 여부 확인
            if not os.path.exists(dag_file):
                raise FileNotFoundError(f"DAG 파일을 찾을 수 없습니다: {dag_id}")
            
            # 파일 삭제
            os.remove(dag_file)
            
            logger.info(f"파이프라인 삭제 완료: {dag_id}")
            
            return {
                "dag_id": dag_id,
                "file_path": dag_file,
                "deleted_at": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"파이프라인 삭제 오류: {e}")
            raise
    
    async def get_pipeline_details(self, dag_id: str) -> Dict[str, Any]:
        """파이프라인 상세 정보 조회
        
        Args:
            dag_id: 조회할 DAG ID
            
        Returns:
            파이프라인 상세 정보
        """
        try:
            # DAG 파일 경로
            dag_file = os.path.join(self.dags_dir, f"{dag_id}.py")
            
            # 파일 존재 여부 확인
            if not os.path.exists(dag_file):
                raise FileNotFoundError(f"DAG 파일을 찾을 수 없습니다: {dag_id}")
            
            # 파일 내용 읽기
            with open(dag_file, "r", encoding="utf-8") as file:
                content = file.read()
            
            # 파일 생성 및 수정 시간
            stat = os.stat(dag_file)
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # 파이프라인 설정 추출
            name = self._extract_pipeline_name(content)
            description = self._extract_pipeline_description(content)
            schedule = self._extract_pipeline_schedule(content)
            tasks = self._extract_pipeline_tasks(content)
            dependencies = self._extract_pipeline_dependencies(content)
            
            return {
                "dag_id": dag_id,
                "name": name,
                "description": description,
                "schedule": schedule,
                "file_path": dag_file,
                "created_at": created_time.isoformat(),
                "modified_at": modified_time.isoformat(),
                "tasks": tasks,
                "dependencies": dependencies,
                "content": content
            }
        except Exception as e:
            logger.error(f"파이프라인 상세 정보 조회 오류: {e}")
            raise
    
    async def create_document_processing_pipeline(self, name: str, schedule: str = "@daily") -> Dict[str, Any]:
        """문서 처리 파이프라인 생성
        
        Args:
            name: 파이프라인 이름
            schedule: 스케줄 간격
            
        Returns:
            생성된 파이프라인 정보
        """
        try:
            # 파이프라인 구성
            pipeline_config = {
                "name": name,
                "description": "문서 생성, 테스트 및 최적화를 위한 파이프라인",
                "schedule": schedule,
                "tasks": [
                    {
                        "id": "generate_documents",
                        "name": "문서 생성",
                        "python_callable": "generate_documents",
                        "function_def": """
def generate_documents(**kwargs):
    from analysis.tools.atoms.generate_test_documents import create_test_document_set
    import asyncio
    
    result = asyncio.run(create_test_document_set(
        output_dir="C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\temp\\\\test_documents\\\\docs_100",
        count=100,
        distribution={'report': 0.5, 'analysis': 0.3, 'memo': 0.2},
        metadata_file="C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\temp\\\\test_documents\\\\metadata_100.json"
    ))
    
    return result
"""
                    },
                    {
                        "id": "test_document_performance",
                        "name": "문서 성능 테스트",
                        "python_callable": "test_document_performance",
                        "function_def": """
def test_document_performance(**kwargs):
    from analysis.tools.atoms.test_document_performance import DocumentPerformanceTest
    import asyncio
    
    tester = DocumentPerformanceTest(
        test_dir="C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\temp\\\\test_documents",
        results_dir="C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\output\\\\reports\\\\performance"
    )
    
    results = asyncio.run(tester.run_all_tests(document_counts=[10, 50, 100]))
    return results
"""
                    },
                    {
                        "id": "optimize_context",
                        "name": "컨텍스트 최적화",
                        "python_callable": "optimize_context",
                        "function_def": """
def optimize_context(**kwargs):
    # 최적화 작업 수행
    # (실제 구현할 때는 advanced_context.py 모듈의 함수 호출)
    return "Context optimization completed"
"""
                    },
                    {
                        "id": "analyze_results",
                        "name": "결과 분석",
                        "python_callable": "analyze_results",
                        "function_def": """
def analyze_results(**kwargs):
    test_results = kwargs['ti'].xcom_pull(task_ids='test_document_performance')
    
    # 결과 분석 및 보고서 생성
    report = {
        "timestamp": f"{datetime.now().isoformat()}",
        "summary": "문서 처리 성능 테스트 결과 분석",
        "results": test_results
    }
    
    # 보고서 저장
    import json
    import os
    
    report_file = os.path.join(
        "C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\output\\\\reports",
        f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report_file
"""
                    }
                ],
                "dependencies": [
                    ["generate_documents", "test_document_performance"],
                    ["test_document_performance", "optimize_context"],
                    ["optimize_context", "analyze_results"]
                ]
            }
            
            # 파이프라인 생성
            result = await self.create_pipeline(pipeline_config)
            
            logger.info(f"문서 처리 파이프라인 생성 완료: {name}")
            
            return result
        except Exception as e:
            logger.error(f"문서 처리 파이프라인 생성 오류: {e}")
            raise
    
    async def create_ontology_pipeline(self, name: str, schedule: str = "@daily") -> Dict[str, Any]:
        """온톨로지 관리 파이프라인 생성
        
        Args:
            name: 파이프라인 이름
            schedule: 스케줄 간격
            
        Returns:
            생성된 파이프라인 정보
        """
        try:
            # 파이프라인 구성
            pipeline_config = {
                "name": name,
                "description": "온톨로지 데이터 업데이트 및 관리를 위한 파이프라인",
                "schedule": schedule,
                "tasks": [
                    {
                        "id": "initialize_ontology",
                        "name": "온톨로지 초기화",
                        "python_callable": "initialize_ontology",
                        "function_def": """
def initialize_ontology(**kwargs):
    from analysis.tools.molecules.ontology_manager import OntologySystem
    import asyncio
    
    ontology_system = OntologySystem("C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\config\\\\neo4j.yaml")
    asyncio.run(ontology_system.initialize_base_ontology())
    
    return "온톨로지 초기화 완료"
"""
                    },
                    {
                        "id": "analyze_document_status",
                        "name": "문서 상태 분석",
                        "python_callable": "analyze_document_status",
                        "function_def": """
def analyze_document_status(**kwargs):
    from analysis.tools.molecules.ontology_manager import OntologySystem
    import asyncio
    
    ontology_system = OntologySystem("C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\config\\\\neo4j.yaml")
    
    # 상태 목록
    statuses = ["draft", "review", "approved", "published", "archived"]
    
    # 각 상태별 문서 수 계산
    status_counts = {}
    
    for status in statuses:
        documents = asyncio.run(ontology_system.find_documents_by_status(status))
        status_counts[status] = len(documents)
    
    return status_counts
"""
                    },
                    {
                        "id": "export_ontology",
                        "name": "온톨로지 내보내기",
                        "python_callable": "export_ontology",
                        "function_def": """
def export_ontology(**kwargs):
    from analysis.tools.molecules.ontology_manager import OntologySystem
    import asyncio
    from datetime import datetime
    
    ontology_system = OntologySystem("C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\config\\\\neo4j.yaml")
    
    # 현재 시간을 포함한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = f"C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\output\\\\ontology\\\\ontology_export_{timestamp}.json"
    
    asyncio.run(ontology_system.export_ontology(export_path))
    
    return export_path
"""
                    },
                    {
                        "id": "generate_ontology_report",
                        "name": "온톨로지 보고서 생성",
                        "python_callable": "generate_ontology_report",
                        "function_def": """
def generate_ontology_report(**kwargs):
    status_counts = kwargs['ti'].xcom_pull(task_ids='analyze_document_status')
    export_path = kwargs['ti'].xcom_pull(task_ids='export_ontology')
    
    # 보고서 생성
    from datetime import datetime
    import json
    import os
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "document_status_counts": status_counts,
        "ontology_export_path": export_path,
        "summary": "온톨로지 관리 파이프라인 실행 결과"
    }
    
    # 보고서 저장
    report_file = os.path.join(
        "C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\output\\\\reports",
        f"ontology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report_file
"""
                    }
                ],
                "dependencies": [
                    ["initialize_ontology", "analyze_document_status"],
                    ["initialize_ontology", "export_ontology"],
                    ["analyze_document_status", "generate_ontology_report"],
                    ["export_ontology", "generate_ontology_report"]
                ]
            }
            
            # 파이프라인 생성
            result = await self.create_pipeline(pipeline_config)
            
            logger.info(f"온톨로지 관리 파이프라인 생성 완료: {name}")
            
            return result
        except Exception as e:
            logger.error(f"온톨로지 관리 파이프라인 생성 오류: {e}")
            raise
    
    async def create_llm_code_pipeline(self, name: str, schedule: str = "@daily") -> Dict[str, Any]:
        """LLM 코드 생성 파이프라인 생성
        
        Args:
            name: 파이프라인 이름
            schedule: 스케줄 간격
            
        Returns:
            생성된 파이프라인 정보
        """
        try:
            # 파이프라인 구성
            pipeline_config = {
                "name": name,
                "description": "LLM을 이용한 코드 생성, 개선 및 검증 파이프라인",
                "schedule": schedule,
                "tasks": [
                    {
                        "id": "generate_code",
                        "name": "코드 생성",
                        "python_callable": "generate_code",
                        "function_def": """
def generate_code(**kwargs):
    from analysis.tools.molecules.llm_integration import ClaudeAIPair
    import asyncio
    
    claude_pair = ClaudeAIPair("C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\config\\\\llm.yaml")
    
    # 코드 생성 요청
    prompt = "Neo4j 데이터베이스에서 온톨로지 그래프를 시각화하는 Python 함수를 작성해주세요."
    
    # API 키가 설정된 경우에만 실행
    try:
        # 코드 생성 (비동기 함수 호출)
        generated_code = asyncio.run(claude_pair.ask_for_code(prompt))
        
        # 파일 저장
        file_path = asyncio.run(claude_pair.save_code_generation(
            code=generated_code,
            filename="ontology_visualizer.py",
            is_improved=False
        ))
        
        return {
            "status": "success",
            "file_path": file_path,
            "prompt": prompt
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "prompt": prompt
        }
"""
                    },
                    {
                        "id": "review_code",
                        "name": "코드 검토",
                        "python_callable": "review_code",
                        "function_def": """
def review_code(**kwargs):
    from analysis.tools.molecules.llm_integration import ClaudeAIPair
    import asyncio
    import os
    
    # 이전 태스크의 결과 가져오기
    result = kwargs['ti'].xcom_pull(task_ids='generate_code')
    
    if result.get("status") != "success":
        return {"status": "error", "message": "코드 생성 단계에서 오류가 발생했습니다."}
    
    file_path = result.get("file_path")
    
    # 파일 읽기
    with open(file_path, "r", encoding="utf-8") as file:
        code = file.read()
    
    # LLM 클라이언트 초기화
    claude_pair = ClaudeAIPair("C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\config\\\\llm.yaml")
    
    try:
        # 코드 검토 (비동기 함수 호출)
        review_result = asyncio.run(claude_pair.review_generated_code(code))
        
        # 검토 결과 저장
        review_file = os.path.join(
            os.path.dirname(file_path),
            f"{os.path.splitext(os.path.basename(file_path))[0]}_review.md"
        )
        
        with open(review_file, "w", encoding="utf-8") as file:
            file.write(review_result)
        
        return {
            "status": "success",
            "review_file": review_file,
            "original_file": file_path
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
"""
                    },
                    {
                        "id": "improve_code",
                        "name": "코드 개선",
                        "python_callable": "improve_code",
                        "function_def": """
def improve_code(**kwargs):
    from analysis.tools.molecules.llm_integration import ClaudeAIPair
    import asyncio
    import os
    
    # 이전 태스크의 결과 가져오기
    result = kwargs['ti'].xcom_pull(task_ids='review_code')
    
    if result.get("status") != "success":
        return {"status": "error", "message": "코드 검토 단계에서 오류가 발생했습니다."}
    
    original_file = result.get("original_file")
    review_file = result.get("review_file")
    
    # 파일 읽기
    with open(original_file, "r", encoding="utf-8") as file:
        code = file.read()
    
    with open(review_file, "r", encoding="utf-8") as file:
        review = file.read()
    
    # LLM 클라이언트 초기화
    claude_pair = ClaudeAIPair("C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\config\\\\llm.yaml")
    
    try:
        # 코드 개선 결과
        improved_code = asyncio.run(claude_pair.self_refine_code(code, iterations=1))
        
        # 개선된 코드 저장
        improved_file = os.path.join(
            "C:\\\\Users\\\\packr\\\\OneDrive\\\\palantir\\\\output\\\\llm_generated\\\\improved",
            os.path.basename(original_file)
        )
        
        os.makedirs(os.path.dirname(improved_file), exist_ok=True)
        
        asyncio.run(claude_pair.save_code_generation(
            code=improved_code["refined_code"],
            filename=os.path.basename(original_file),
            is_improved=True
        ))
        
        return {
            "status": "success",
            "original_file": original_file,
            "improved_file": improved_file
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
"""
                    }
                ],
                "dependencies": [
                    ["generate_code", "review_code"],
                    ["review_code", "improve_code"]
                ]
            }
            
            # 파이프라인 생성
            result = await self.create_pipeline(pipeline_config)
            
            logger.info(f"LLM 코드 생성 파이프라인 생성 완료: {name}")
            
            return result
        except Exception as e:
            logger.error(f"LLM 코드 생성 파이프라인 생성 오류: {e}")
            raise
    
    def _generate_dag_code(self, dag_id: str, name: str, description: str, 
                          schedule: str, tasks: List[Dict[str, Any]], 
                          dependencies: List[List[str]]) -> str:
        """DAG 코드 생성
        
        Args:
            dag_id: DAG ID
            name: 파이프라인 이름
            description: 파이프라인 설명
            schedule: 스케줄 간격
            tasks: 태스크 목록
            dependencies: 태스크 의존성 목록
            
        Returns:
            생성된 DAG 코드
        """
        # 코드 헤더
        code = f'''"""
{name}

{description}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# 기본 인수 정의
default_args = {{
    'owner': 'foundry',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}}

# DAG 정의
dag = DAG(
    '{dag_id}',
    default_args=default_args,
    description='{description}',
    schedule_interval='{schedule}',
    catchup=False,
)

'''
        
        # 태스크 함수 정의
        for task in tasks:
            code += f"{task['function_def']}\n\n"
        
        # 태스크 정의
        code += "# 태스크 정의\n"
        for task in tasks:
            code += f'''{task['id']}_task = PythonOperator(
    task_id='{task['id']}',
    python_callable={task['python_callable']},
    dag=dag,
)

'''
        
        # 태스크 의존성 설정
        if dependencies:
            code += "# 태스크 의존성 설정\n"
            for dep in dependencies:
                if len(dep) == 2:
                    code += f"{dep[0]}_task >> {dep[1]}_task\n"
        
        return code
    
    def _extract_dag_id(self, file_path: str) -> str:
        """파일 내용에서 DAG ID 추출
        
        Args:
            file_path: DAG 파일 경로
            
        Returns:
            추출된 DAG ID
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                
                # DAG ID 추출
                import re
                match = re.search(r"dag\s*=\s*DAG\s*\(\s*['\"]([^'\"]+)['\"]", content)
                
                if match:
                    return match.group(1)
                else:
                    # 파일 이름에서 확장자 제거하여 반환
                    return os.path.splitext(os.path.basename(file_path))[0]
        except Exception as e:
            logger.error(f"DAG ID 추출 오류: {e}")
            return os.path.splitext(os.path.basename(file_path))[0]
    
    def _extract_pipeline_name(self, content: str) -> str:
        """파일 내용에서 파이프라인 이름 추출
        
        Args:
            content: DAG 파일 내용
            
        Returns:
            추출된 파이프라인 이름
        """
        import re
        docstring = re.search(r'"""(.+?)"""', content, re.DOTALL)
        
        if docstring:
            lines = docstring.group(1).strip().split('\n')
            if lines:
                return lines[0].strip()
        
        # 이름을 찾지 못한 경우 DAG ID 검색
        dag_id = re.search(r"dag\s*=\s*DAG\s*\(\s*['\"]([^'\"]+)['\"]", content)
        if dag_id:
            return dag_id.group(1).replace('_', ' ').title()
        
        return "Unknown Pipeline"
    
    def _extract_pipeline_description(self, content: str) -> str:
        """파일 내용에서 파이프라인 설명 추출
        
        Args:
            content: DAG 파일 내용
            
        Returns:
            추출된 파이프라인 설명
        """
        import re
        docstring = re.search(r'"""(.+?)"""', content, re.DOTALL)
        
        if docstring:
            lines = docstring.group(1).strip().split('\n')
            if len(lines) > 1:
                return '\n'.join(lines[1:]).strip()
        
        # 설명을 찾지 못한 경우 DAG 설명 검색
        description = re.search(r"description\s*=\s*['\"]([^'\"]+)['\"]", content)
        if description:
            return description.group(1)
        
        return ""
    
    def _extract_pipeline_schedule(self, content: str) -> str:
        """파일 내용에서 파이프라인 스케줄 추출
        
        Args:
            content: DAG 파일 내용
            
        Returns:
            추출된 파이프라인 스케줄
        """
        import re
        schedule = re.search(r"schedule_interval\s*=\s*['\"]([^'\"]+)['\"]", content)
        
        if schedule:
            return schedule.group(1)
        
        return "@daily"
    
    def _extract_pipeline_tasks(self, content: str) -> List[Dict[str, Any]]:
        """파일 내용에서 파이프라인 태스크 추출
        
        Args:
            content: DAG 파일 내용
            
        Returns:
            추출된 파이프라인 태스크 목록
        """
        import re
        tasks = []
        
        # 태스크 정의 추출
        task_definitions = re.findall(r"(\w+)_task\s*=\s*PythonOperator\s*\((.+?)\)", content, re.DOTALL)
        
        for task_id, task_def in task_definitions:
            # 태스크 이름 추출
            task_name = task_id.replace('_', ' ').title()
            
            # callable 함수 이름 추출
            callable_match = re.search(r"python_callable\s*=\s*(\w+)", task_def)
            python_callable = callable_match.group(1) if callable_match else task_id
            
            tasks.append({
                "id": task_id,
                "name": task_name,
                "python_callable": python_callable
            })
        
        return tasks
    
    def _extract_pipeline_dependencies(self, content: str) -> List[List[str]]:
        """파일 내용에서 파이프라인 의존성 추출
        
        Args:
            content: DAG 파일 내용
            
        Returns:
            추출된 파이프라인 의존성 목록
        """
        import re
        dependencies = []
        
        # 의존성 정의 추출
        dep_definitions = re.findall(r"(\w+)_task\s*>>\s*(\w+)_task", content)
        
        for source, target in dep_definitions:
            dependencies.append([source, target])
        
        return dependencies

@mcp.workflow(
    name="create_document_pipeline",
    description="문서 처리 파이프라인 생성 워크플로우"
)
async def create_document_pipeline(name: str, schedule: str = "@daily") -> Dict[str, Any]:
    """문서 처리 파이프라인 생성 워크플로우
    
    Args:
        name: 파이프라인 이름
        schedule: 스케줄 간격
        
    Returns:
        생성된 파이프라인 정보
    """
    system = DataPipelineSystem("C:\\Users\\packr\\OneDrive\\palantir\\analysis\\airflow")
    return await system.create_document_processing_pipeline(name, schedule)

@mcp.workflow(
    name="create_ontology_pipeline",
    description="온톨로지 관리 파이프라인 생성 워크플로우"
)
async def create_ontology_pipeline(name: str, schedule: str = "@daily") -> Dict[str, Any]:
    """온톨로지 관리 파이프라인 생성 워크플로우
    
    Args:
        name: 파이프라인 이름
        schedule: 스케줄 간격
        
    Returns:
        생성된 파이프라인 정보
    """
    system = DataPipelineSystem("C:\\Users\\packr\\OneDrive\\palantir\\analysis\\airflow")
    return await system.create_ontology_pipeline(name, schedule)

@mcp.workflow(
    name="create_llm_pipeline",
    description="LLM 코드 생성 파이프라인 생성 워크플로우"
)
async def create_llm_pipeline(name: str, schedule: str = "@daily") -> Dict[str, Any]:
    """LLM 코드 생성 파이프라인 생성 워크플로우
    
    Args:
        name: 파이프라인 이름
        schedule: 스케줄 간격
        
    Returns:
        생성된 파이프라인 정보
    """
    system = DataPipelineSystem("C:\\Users\\packr\\OneDrive\\palantir\\analysis\\airflow")
    return await system.create_llm_code_pipeline(name, schedule)

# 직접 실행 시
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    import asyncio
    
    async def main():
        system = DataPipelineSystem("C:\\Users\\packr\\OneDrive\\palantir\\analysis\\airflow")
        
        # 예시: 문서 처리 파이프라인 생성
        doc_pipeline = await system.create_document_processing_pipeline("문서 처리 파이프라인")
        print(f"문서 처리 파이프라인 생성 결과: {doc_pipeline}")
        
        # 예시: 온톨로지 관리 파이프라인 생성
        onto_pipeline = await system.create_ontology_pipeline("온톨로지 관리 파이프라인")
        print(f"온톨로지 관리 파이프라인 생성 결과: {onto_pipeline}")
        
        # 예시: LLM 코드 생성 파이프라인 생성
        llm_pipeline = await system.create_llm_code_pipeline("LLM 코드 생성 파이프라인")
        print(f"LLM 코드 생성 파이프라인 생성 결과: {llm_pipeline}")
    
    asyncio.run(main())

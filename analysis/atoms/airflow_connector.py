"""
Apache Airflow 연결 및 기본 작업을 위한 모듈
"""

import os
import yaml
import logging
import subprocess
import shutil
import time
from datetime import datetime, timedelta
import requests
from urllib.parse import urljoin
import json

logger = logging.getLogger(__name__)

class AirflowConnector:
    """Apache Airflow 연결 및 기본 작업 처리 클래스"""
    
    def __init__(self, config_path=None, webserver_url=None, airflow_home=None):
        """
        Airflow 연결 초기화
        
        Args:
            config_path (str, optional): 구성 파일 경로
            webserver_url (str, optional): Airflow 웹서버 URL
            airflow_home (str, optional): AIRFLOW_HOME 환경 변수 값
        
        직접 매개변수가 우선순위가 높으며, 제공되지 않으면 구성 파일에서 값을 로드합니다.
        """
        # 직접 제공된, 아니면 구성 파일에서 로드된 연결 정보
        if webserver_url and airflow_home:
            self.webserver_url = webserver_url
            self.airflow_home = airflow_home
        else:
            if config_path:
                self._load_config(config_path)
            else:
                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "airflow.yaml"
                )
                self._load_config(default_config_path)
        
        # Airflow CLI 명령을 실행하기 위한 환경 변수 설정
        self.env = os.environ.copy()
        self.env["AIRFLOW_HOME"] = self.airflow_home
        
        logger.info("Airflow 연결기 초기화 완료")
    
    def _load_config(self, config_path):
        """
        구성 파일에서 Airflow 연결 정보 로드
        
        Args:
            config_path (str): 구성 파일 경로
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Airflow 연결 정보 추출
            airflow_config = config.get('airflow', {})
            self.webserver_url = airflow_config.get('webserver_url', 'http://localhost:8080')
            self.airflow_home = airflow_config.get('airflow_home', os.getenv('AIRFLOW_HOME', './airflow'))
            self.api_base_path = airflow_config.get('api_base_path', '/api/v1')
            self.username = airflow_config.get('username', 'admin')
            self.password = airflow_config.get('password', 'admin')
            self.dags_folder = airflow_config.get('dags_folder')
            
            if not self.dags_folder:
                self.dags_folder = os.path.join(self.airflow_home, 'dags')
            
            logger.info(f"구성 파일 {config_path}에서 Airflow 연결 정보 로드됨")
        except Exception as e:
            logger.error(f"구성 파일 {config_path} 로드 중 오류: {str(e)}")
            
            # 기본값 설정
            self.webserver_url = 'http://localhost:8080'
            self.airflow_home = os.getenv('AIRFLOW_HOME', './airflow')
            self.api_base_path = '/api/v1'
            self.username = 'admin'
            self.password = 'admin'
            self.dags_folder = os.path.join(self.airflow_home, 'dags')
            
            logger.warning("기본 Airflow 연결 정보를 사용합니다.")
    
    def initialize_airflow(self, reset=False):
        """
        Airflow 초기화 (db init, 사용자 생성)
        
        Args:
            reset (bool, optional): 기존 DB 초기화 여부
        
        Returns:
            bool: 성공 여부
        """
        try:
            # Airflow 홈 디렉토리가 존재하는지 확인
            os.makedirs(self.airflow_home, exist_ok=True)
            os.makedirs(self.dags_folder, exist_ok=True)
            
            # DB 초기화 명령 실행
            db_init_cmd = ["airflow", "db", "init" if not reset else "reset", "-y"]
            result = subprocess.run(db_init_cmd, env=self.env, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Airflow DB 초기화 실패: {result.stderr}")
                return False
            
            logger.info("Airflow DB 초기화 성공")
            
            # 관리자 사용자 생성
            create_user_cmd = [
                "airflow", "users", "create",
                "--username", self.username,
                "--password", self.password,
                "--firstname", "Admin",
                "--lastname", "User",
                "--role", "Admin",
                "--email", "admin@example.com"
            ]
            
            result = subprocess.run(create_user_cmd, env=self.env, capture_output=True, text=True)
            
            if result.returncode != 0:
                # 사용자가 이미 존재하면 무시
                if "already exists" in result.stderr:
                    logger.info(f"관리자 사용자 '{self.username}'가 이미 존재합니다")
                else:
                    logger.error(f"관리자 사용자 생성 실패: {result.stderr}")
                    return False
            else:
                logger.info(f"관리자 사용자 '{self.username}' 생성 성공")
            
            return True
        except Exception as e:
            logger.error(f"Airflow 초기화 중 오류: {str(e)}")
            return False
    
    def start_airflow(self, background=True):
        """
        Airflow 웹서버 및 스케줄러 시작
        
        Args:
            background (bool, optional): 백그라운드 실행 여부
        
        Returns:
            dict: 프로세스 정보
        """
        try:
            # 웹서버 시작
            webserver_cmd = ["airflow", "webserver"]
            
            if background:
                webserver_process = subprocess.Popen(
                    webserver_cmd, 
                    env=self.env, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                logger.info(f"Airflow 웹서버 시작됨 (PID: {webserver_process.pid})")
            else:
                webserver_process = None
                logger.info("백그라운드 실행이 비활성화되어 웹서버를 시작하지 않았습니다")
            
            # 스케줄러 시작
            scheduler_cmd = ["airflow", "scheduler"]
            
            if background:
                scheduler_process = subprocess.Popen(
                    scheduler_cmd, 
                    env=self.env, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                logger.info(f"Airflow 스케줄러 시작됨 (PID: {scheduler_process.pid})")
            else:
                scheduler_process = None
                logger.info("백그라운드 실행이 비활성화되어 스케줄러를 시작하지 않았습니다")
            
            return {
                "webserver": webserver_process,
                "scheduler": scheduler_process
            }
        except Exception as e:
            logger.error(f"Airflow 시작 중 오류: {str(e)}")
            return {
                "webserver": None,
                "scheduler": None
            }
    
    def stop_airflow(self, processes=None):
        """
        Airflow 웹서버 및 스케줄러 중지
        
        Args:
            processes (dict, optional): 프로세스 정보
        
        Returns:
            bool: 성공 여부
        """
        try:
            if processes:
                # 명시적으로 제공된 프로세스 종료
                if processes.get('webserver'):
                    processes['webserver'].terminate()
                    logger.info("Airflow 웹서버 종료됨")
                
                if processes.get('scheduler'):
                    processes['scheduler'].terminate()
                    logger.info("Airflow 스케줄러 종료됨")
            else:
                # 실행 중인 모든 Airflow 프로세스 검색 및 종료
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/F', '/IM', 'airflow.exe'], capture_output=True)
                else:  # Linux/Mac
                    subprocess.run(['pkill', '-f', 'airflow'], capture_output=True)
                
                logger.info("모든 Airflow 프로세스 종료 요청됨")
            
            return True
        except Exception as e:
            logger.error(f"Airflow 중지 중 오류: {str(e)}")
            return False
    
    def list_dags(self):
        """
        등록된 DAG 목록 조회
        
        Returns:
            list: DAG 정보 목록
        """
        try:
            result = subprocess.run(
                ["airflow", "dags", "list", "--output", "json"], 
                env=self.env, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"DAG 목록 조회 실패: {result.stderr}")
                return []
            
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"DAG 목록 조회 중 오류: {str(e)}")
            return []
    
    def get_dag_details(self, dag_id):
        """
        특정 DAG 상세 정보 조회
        
        Args:
            dag_id (str): DAG ID
        
        Returns:
            dict: DAG 상세 정보
        """
        try:
            result = subprocess.run(
                ["airflow", "dags", "show", dag_id, "--output", "json"], 
                env=self.env, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"DAG '{dag_id}' 상세 정보 조회 실패: {result.stderr}")
                return None
            
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"DAG '{dag_id}' 상세 정보 조회 중 오류: {str(e)}")
            return None
    
    def trigger_dag(self, dag_id, conf=None):
        """
        DAG 실행 트리거
        
        Args:
            dag_id (str): DAG ID
            conf (dict, optional): 실행 구성
        
        Returns:
            str: 실행 ID
        """
        try:
            cmd = ["airflow", "dags", "trigger", dag_id]
            
            if conf:
                conf_str = json.dumps(conf)
                cmd.extend(["--conf", conf_str])
            
            result = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"DAG '{dag_id}' 트리거 실패: {result.stderr}")
                return None
            
            # 실행 ID 추출 (출력 형식에 따라 다를 수 있음)
            for line in result.stdout.splitlines():
                if "Triggered" in line and "run_id" in line:
                    run_id = line.split("run_id=")[1].strip()
                    logger.info(f"DAG '{dag_id}' 트리거됨 (run_id: {run_id})")
                    return run_id
            
            logger.warning(f"DAG '{dag_id}' 트리거되었지만 run_id를 찾을 수 없음")
            return "unknown"
        except Exception as e:
            logger.error(f"DAG '{dag_id}' 트리거 중 오류: {str(e)}")
            return None
    
    def get_dag_runs(self, dag_id, limit=10):
        """
        DAG 실행 기록 조회
        
        Args:
            dag_id (str): DAG ID
            limit (int, optional): 최대 결과 수
        
        Returns:
            list: 실행 기록 목록
        """
        try:
            result = subprocess.run(
                ["airflow", "dags", "list-runs", "--dag-id", dag_id, "--output", "json", "--limit", str(limit)], 
                env=self.env, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"DAG '{dag_id}' 실행 기록 조회 실패: {result.stderr}")
                return []
            
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"DAG '{dag_id}' 실행 기록 조회 중 오류: {str(e)}")
            return []
    
    def deploy_dag(self, dag_file_path, overwrite=True):
        """
        DAG 파일 배포
        
        Args:
            dag_file_path (str): DAG 파일 경로
            overwrite (bool, optional): 기존 파일 덮어쓰기 여부
        
        Returns:
            bool: 성공 여부
        """
        try:
            # DAG 파일 이름 추출
            dag_file_name = os.path.basename(dag_file_path)
            destination = os.path.join(self.dags_folder, dag_file_name)
            
            # 대상 파일이 이미 존재하는지 확인
            if os.path.exists(destination) and not overwrite:
                logger.warning(f"DAG 파일 '{dag_file_name}'이(가) 이미 존재하며 덮어쓰기가 비활성화되어 있습니다")
                return False
            
            # DAG 파일 복사
            shutil.copy2(dag_file_path, destination)
            logger.info(f"DAG 파일 '{dag_file_name}'이(가) 성공적으로 배포됨")
            
            # DAG 갱신
            self.refresh_dags()
            
            return True
        except Exception as e:
            logger.error(f"DAG 파일 '{dag_file_path}' 배포 중 오류: {str(e)}")
            return False
    
    def refresh_dags(self):
        """
        DAG 목록 갱신
        
        Returns:
            bool: 성공 여부
        """
        try:
            result = subprocess.run(
                ["airflow", "dags", "reserialize"], 
                env=self.env, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"DAG 갱신 실패: {result.stderr}")
                return False
            
            logger.info("DAG 목록 갱신됨")
            return True
        except Exception as e:
            logger.error(f"DAG 갱신 중 오류: {str(e)}")
            return False
    
    def check_webserver_status(self, timeout=5):
        """
        Airflow 웹서버 상태 확인
        
        Args:
            timeout (int, optional): 요청 타임아웃 (초)
        
        Returns:
            bool: 웹서버 실행 중 여부
        """
        try:
            response = requests.get(self.webserver_url, timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    def wait_for_webserver(self, max_retries=10, retry_interval=2):
        """
        Airflow 웹서버가 시작될 때까지 대기
        
        Args:
            max_retries (int, optional): 최대 재시도 횟수
            retry_interval (int, optional): 재시도 간격 (초)
        
        Returns:
            bool: 웹서버 시작 여부
        """
        for i in range(max_retries):
            if self.check_webserver_status():
                logger.info(f"Airflow 웹서버가 실행 중입니다 (URL: {self.webserver_url})")
                return True
            
            logger.info(f"Airflow 웹서버 시작 대기 중... ({i+1}/{max_retries})")
            time.sleep(retry_interval)
        
        logger.error(f"Airflow 웹서버 시작 타임아웃 (URL: {self.webserver_url})")
        return False
    
    def get_api_token(self):
        """
        Airflow REST API 토큰 요청
        
        Returns:
            str: API 토큰, 실패 시 None
        """
        try:
            auth_url = urljoin(self.webserver_url, "/api/v1/security/login")
            response = requests.post(
                auth_url,
                json={"username": self.username, "password": self.password}
            )
            
            if response.status_code == 200:
                return response.json().get("access_token")
            else:
                logger.error(f"API 토큰 요청 실패 (상태 코드: {response.status_code})")
                return None
        except Exception as e:
            logger.error(f"API 토큰 요청 중 오류: {str(e)}")
            return None
    
    def api_request(self, endpoint, method="GET", data=None):
        """
        Airflow REST API 요청
        
        Args:
            endpoint (str): API 엔드포인트
            method (str, optional): HTTP 메서드
            data (dict, optional): 요청 데이터
        
        Returns:
            dict: API 응답 데이터
        """
        try:
            # API 토큰 가져오기
            token = self.get_api_token()
            if not token:
                logger.error("API 토큰을 가져올 수 없음")
                return None
            
            # API 요청 URL 생성
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"
            
            request_url = urljoin(self.webserver_url, f"{self.api_base_path}{endpoint}")
            
            # 헤더 설정
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # API 요청 실행
            if method.upper() == "GET":
                response = requests.get(request_url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(request_url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(request_url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(request_url, headers=headers)
            else:
                logger.error(f"지원되지 않는 HTTP 메서드: {method}")
                return None
            
            # 응답 처리
            if response.status_code >= 200 and response.status_code < 300:
                return response.json()
            else:
                logger.error(f"API 요청 실패 (URL: {request_url}, 상태 코드: {response.status_code})")
                return None
        except Exception as e:
            logger.error(f"API 요청 중 오류 (엔드포인트: {endpoint}): {str(e)}")
            return None

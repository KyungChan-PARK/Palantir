# OneDrive 통합 가이드
> ⚠️ 본 문서에서 언급된 Cursor AI 및 Codex CLI 관련 내용은 Cursor AI 및 Codex CLI가 담당하는 것으로 변경되었습니다.

## 1. 개요

OneDrive 통합 모듈은 팔란티어 파운드리 프로젝트의 문서 관리 시스템의 핵심 구성요소입니다. 이 모듈은 OneDrive에 저장된 문서를 감지하고, 메타데이터를 추출하며, 문서 내용을 처리하여 온톨로지와 RAG 시스템에 통합합니다. 이 가이드는 OneDrive 통합 모듈의 설정, 개발, 사용 방법을 상세히 설명합니다.

## 2. OneDrive 통합 아키텍처

### 2.1 주요 구성요소
- **OneDrive 커넥터**: OneDrive API와의 통신을 담당
- **문서 처리기**: 다양한 문서 형식(Word, Excel, PowerPoint, PDF 등)의 내용 추출
- **메타데이터 추출기**: 문서의 메타데이터 추출 및 정리
- **변경 감지기**: OneDrive의 문서 변경 사항 모니터링
- **동기화 관리자**: 로컬 캐시와 OneDrive 간의 동기화 관리

### 2.2 데이터 흐름
```
[OneDrive] → [OneDrive 커넥터] → [문서 처리기] → [메타데이터 추출기]
                                       ↓
                                [변경 감지기] → [동기화 관리자]
                                       ↓
                 [온톨로지 관리 시스템] ← [처리된 데이터] → [RAG 시스템]
```

## 3. 설정 및 구성

### 3.1 필수 요구사항
- Microsoft 365 계정 또는 개인 OneDrive 계정
- Microsoft Graph API 액세스 권한
- Azure 애플리케이션 등록 (API 접근용)

### 3.2 OneDrive API 설정
1. Azure 포털(portal.azure.com)에 로그인
2. Azure Active Directory → 앱 등록 → 새 등록
3. 애플리케이션 이름 입력 (예: "팔란티어 파운드리")
4. 리디렉션 URI 설정 (웹 애플리케이션의 경우)
5. 등록 완료 후 클라이언트 ID 및 테넌트 ID 기록
6. 인증서 및 비밀 → 새 클라이언트 비밀 생성
7. API 권한 → Microsoft Graph → 위임된 권한 추가:
   - Files.Read
   - Files.ReadWrite
   - Files.Read.All
   - 필요에 따라 추가 권한

### 3.3 구성 파일 설정
`config/onedrive.yaml` 파일을 다음과 같이 구성합니다:

```yaml
onedrive:
  client_id: "your_client_id"          # Azure 앱 등록에서 얻은 클라이언트 ID
  tenant_id: "your_tenant_id"          # Azure 테넌트 ID
  client_secret: "your_client_secret"  # 클라이언트 비밀 (안전하게 관리)
  redirect_uri: "http://localhost:8000/callback"  # 개발 환경 리디렉션 URI
  scopes:
    - "Files.Read"
    - "Files.Read.All"
    - "Files.ReadWrite"
  
  # 문서 처리 설정
  document_types:
    - "docx"
    - "xlsx"
    - "pptx"
    - "pdf"
    - "txt"
  
  # 캐시 및 동기화 설정
  cache_location: "data/cache/onedrive"
  refresh_interval: 300  # 5분마다 새 문서 확인
  max_file_size: 52428800  # 50MB 최대 파일 크기
  
  # 모니터링 폴더 설정
  monitored_folders:
    - "/Documents/팔란티어"
    - "/Shared Documents/프로젝트 파일"
```

## 4. 핵심 모듈 개발 가이드

### 4.1 OneDrive 커넥터 (atoms/onedrive_connector.py)

OneDrive API와의 연결 및 기본 작업을 담당하는 모듈입니다.

#### 핵심 기능
- OneDrive API 인증 및 연결
- 파일 및 폴더 목록 조회
- 파일 다운로드 및 업로드
- 변경 사항 추적

#### AI 개발 접근 방식
```python
# Cursor AI 프롬프트 예시
"""
O365 라이브러리를 사용하여 OneDrive 연결 및 인증을 처리하는 OneDriveConnector 클래스를 구현해줘.
다음 기능을 포함해야 함:
1. YAML 구성 파일에서 설정 로드
2. OAuth 인증 처리
3. 파일 및 폴더 목록 조회 메서드
4. 파일 다운로드 메서드
5. 변경 사항 추적 메서드
6. 오류 처리 및 로깅
"""
```

#### 코드 구조
```python
import yaml
import logging
from O365 import Account
from pathlib import Path

class OneDriveConnector:
    def __init__(self, config_path):
        """OneDrive 커넥터 초기화"""
        self.config = self._load_config(config_path)
        self.account = None
        self.drive = None
        self.logger = logging.getLogger(__name__)
        self._authenticate()
    
    def _load_config(self, config_path):
        """구성 파일 로드"""
        # 구성 파일 로드 로직
        
    def _authenticate(self):
        """OneDrive API 인증"""
        # 인증 로직
        
    def list_files(self, folder_path=None):
        """폴더 내 파일 목록 조회"""
        # 파일 목록 조회 로직
        
    def download_file(self, file_id, destination):
        """파일 다운로드"""
        # 파일 다운로드 로직
        
    def track_changes(self, folder_path=None, delta_token=None):
        """변경 사항 추적"""
        # 변경 사항 추적 로직
        
    def get_file_metadata(self, file_id):
        """파일 메타데이터 조회"""
        # 메타데이터 조회 로직
```

### 4.2 문서 처리기 (molecules/document_processor.py)

다양한 형식의 문서를 처리하고 내용을 추출하는 모듈입니다.

#### 핵심 기능
- 다양한 문서 형식 처리 (Word, Excel, PowerPoint, PDF)
- 텍스트 내용 추출
- 메타데이터 추출
- 문서 구조 분석

#### AI 개발 접근 방식
```python
# Codex 프롬프트 예시
"""
다양한 문서 형식을 처리하는 DocumentProcessor 클래스를 구현해줘.
다음 처리기를 포함해야 함:
1. Word 문서 처리기 (python-docx 사용)
2. Excel 문서 처리기 (pandas, openpyxl 사용)
3. PowerPoint 문서 처리기 (python-pptx 사용)
4. PDF 문서 처리기 (PyPDF2 사용)
5. 텍스트 문서 처리기

각 처리기는 텍스트 내용 추출, 메타데이터 추출, 문서 구조 분석 기능을 제공해야 함.
전략 패턴을 사용하여 각 문서 유형에 맞는 처리기를 선택하도록 구현해줘.
"""
```

#### 코드 구조
```python
import logging
from pathlib import Path
from abc import ABC, abstractmethod

# 문서 처리기 인터페이스
class DocumentHandler(ABC):
    @abstractmethod
    def extract_text(self, file_path):
        """텍스트 추출"""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path):
        """메타데이터 추출"""
        pass
    
    @abstractmethod
    def analyze_structure(self, file_path):
        """문서 구조 분석"""
        pass

# Word 문서 처리기
class WordHandler(DocumentHandler):
    # Word 문서 처리 로직

# Excel 문서 처리기
class ExcelHandler(DocumentHandler):
    # Excel 문서 처리 로직

# PowerPoint 문서 처리기
class PowerPointHandler(DocumentHandler):
    # PowerPoint 문서 처리 로직

# 메인 처리기 클래스
class DocumentProcessor:
    def __init__(self):
        """문서 처리기 초기화"""
        self.handlers = {
            '.docx': WordHandler(),
            '.xlsx': ExcelHandler(),
            '.pptx': PowerPointHandler(),
            # 추가 핸들러
        }
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, file_path):
        """문서 처리"""
        # 문서 유형에 맞는 핸들러 선택 및 처리
        
    def get_handler(self, file_extension):
        """파일 확장자에 맞는 핸들러 반환"""
        # 핸들러 선택 로직
```

### 4.3 Excel 문서 처리기 (tools/molecules/excel_processor.py)

Excel 문서를 처리하고 데이터를 추출하는 특화된 모듈입니다.

#### 핵심 기능
- Excel 시트 데이터 추출
- 데이터 구조 분석
- 차트 및 그래프 분석
- 메타데이터 추출

#### AI 개발 접근 방식
```python
# 복합 AI 접근 방식 (Cursor AI + Codex)
"""
pandas와 openpyxl을 사용하여 Excel 파일을 처리하는 ExcelProcessor 클래스를 구현해줘.
다음 기능을 포함해야 함:
1. 시트 목록 조회
2. 시트 데이터를 pandas DataFrame으로 변환
3. 헤더 및 데이터 타입 자동 감지
4. 차트 및 그래프 정보 추출
5. 대용량 파일 처리를 위한 최적화 (메모리 효율적)
6. 오류 처리 (손상된 파일, 비밀번호 보호 등)

추가로 다음 작업도 처리할 수 있어야 함:
- 피벗 테이블 데이터 추출
- 수식 및 계산 결과 처리
- 메타데이터 추출 (작성자, 수정 일자 등)
"""
```

#### 코드 구조
```python
import pandas as pd
import openpyxl
from openpyxl.chart import BarChart, PieChart, LineChart
import logging
from pathlib import Path

class ExcelProcessor:
    def __init__(self):
        """Excel 처리기 초기화"""
        self.logger = logging.getLogger(__name__)
    
    def get_sheet_names(self, file_path):
        """시트 이름 목록 조회"""
        # 시트 이름 추출 로직
        
    def read_sheet_to_dataframe(self, file_path, sheet_name=0, header_row=0, chunk_size=None):
        """시트 데이터를 DataFrame으로 변환"""
        # DataFrame 변환 로직
        
    def detect_headers(self, file_path, sheet_name=0, max_rows=10):
        """헤더 자동 감지"""
        # 헤더 감지 로직
        
    def extract_charts(self, file_path, sheet_name=0):
        """차트 정보 추출"""
        # 차트 추출 로직
        
    def extract_pivot_tables(self, file_path, sheet_name=0):
        """피벗 테이블 데이터 추출"""
        # 피벗 테이블 추출 로직
        
    def extract_metadata(self, file_path):
        """문서 메타데이터 추출"""
        # 메타데이터 추출 로직
        
    def process_large_file(self, file_path, sheet_name=0, chunk_size=10000):
        """대용량 파일 청크 단위 처리"""
        # 대용량 파일 처리 로직
```

### 4.4 PowerPoint 문서 처리기 (tools/molecules/ppt_processor.py)

PowerPoint 문서를 처리하고 내용을 추출하는 특화된 모듈입니다.

#### 핵심 기능
- 슬라이드 텍스트 추출
- 슬라이드 구조 분석
- 이미지 및 도형 정보 추출
- 메타데이터 추출

#### AI 개발 접근 방식
```python
# Cursor AI 및 Codex CLI 프롬프트 예시
"""
python-pptx 라이브러리를 사용하여 PowerPoint 파일을 처리하는 PowerPointProcessor 클래스를 구현해줘.
다음 기능을 포함해야 함:
1. 모든 슬라이드의 텍스트 내용 추출
2. 슬라이드 제목 및 구조 분석
3. 이미지, 차트, 도형 정보 추출
4. 노트 텍스트 추출
5. 메타데이터 추출 (작성자, 수정 일자, 테마 등)
6. 슬라이드 간의 계층 구조 분석

각 기능은 메모리 효율적으로 구현하고, 대용량 프레젠테이션도 처리할 수 있어야 함.
오류 처리 및 로깅도 포함해줘.
"""
```

#### 코드 구조
```python
from pptx import Presentation
import logging
from pathlib import Path
from io import BytesIO

class PowerPointProcessor:
    def __init__(self):
        """PowerPoint 처리기 초기화"""
        self.logger = logging.getLogger(__name__)
    
    def extract_all_text(self, file_path):
        """모든 슬라이드 텍스트 추출"""
        # 텍스트 추출 로직
        
    def extract_slide_titles(self, file_path):
        """슬라이드 제목 추출"""
        # 제목 추출 로직
        
    def analyze_structure(self, file_path):
        """슬라이드 구조 분석"""
        # 구조 분석 로직
        
    def extract_notes(self, file_path):
        """슬라이드 노트 추출"""
        # 노트 추출 로직
        
    def extract_media_info(self, file_path):
        """이미지 및 미디어 정보 추출"""
        # 미디어 정보 추출 로직
        
    def extract_metadata(self, file_path):
        """프레젠테이션 메타데이터 추출"""
        # 메타데이터 추출 로직
        
    def process_slide(self, slide):
        """개별 슬라이드 처리"""
        # 슬라이드 처리 로직
```

## 5. OneDrive 통합 파이프라인

### 5.1 문서 처리 파이프라인 (airflow/dags/document_processing_pipeline.py)

OneDrive 문서를 자동으로 처리하는 Airflow 파이프라인입니다.

#### 핵심 단계
1. OneDrive에서 새로운/변경된 문서 감지
2. 문서 다운로드 및 캐싱
3. 문서 유형별 처리 및 텍스트 추출
4. 메타데이터 추출 및 저장
5. 온톨로지 업데이트
6. 임베딩 생성 및 벡터 저장소 업데이트

#### AI 개발 접근 방식
```python
# Codex 프롬프트 예시
"""
Airflow를 사용하여 OneDrive 문서 처리 파이프라인을 구현해줘.
다음 작업을 포함해야 함:
1. 변경된 문서 감지 작업 (OneDriveChangesTask)
2. 문서 다운로드 작업 (DownloadDocumentsTask)
3. 문서 처리 작업 (ProcessDocumentsTask)
4. 메타데이터 추출 작업 (ExtractMetadataTask)
5. 온톨로지 업데이트 작업 (UpdateOntologyTask)
6. 임베딩 생성 작업 (GenerateEmbeddingsTask)

DAG는 하루에 한 번 실행되고, 각 작업은 이전 작업이 성공한 경우에만 실행됨.
실패한 작업에 대한 재시도 및 알림 메커니즘도 포함해줘.
"""
```

#### 코드 구조
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email

# 기본 인수 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# 작업 함수들
def detect_changed_documents(**context):
    """변경된 문서 감지"""
    # 변경 감지 로직
    
def download_documents(**context):
    """문서 다운로드"""
    # 다운로드 로직
    
def process_documents(**context):
    """문서 처리"""
    # 처리 로직
    
def extract_metadata(**context):
    """메타데이터 추출"""
    # 메타데이터 추출 로직
    
def update_ontology(**context):
    """온톨로지 업데이트"""
    # 온톨로지 업데이트 로직
    
def generate_embeddings(**context):
    """임베딩 생성"""
    # 임베딩 생성 로직

# DAG 정의
with DAG(
    'document_processing_pipeline',
    default_args=default_args,
    description='OneDrive 문서 처리 파이프라인',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 5, 1),
    catchup=False,
) as dag:
    # 작업 정의
    detect_task = PythonOperator(
        task_id='detect_changed_documents',
        python_callable=detect_changed_documents,
    )
    
    download_task = PythonOperator(
        task_id='download_documents',
        python_callable=download_documents,
    )
    
    process_task = PythonOperator(
        task_id='process_documents',
        python_callable=process_documents,
    )
    
    metadata_task = PythonOperator(
        task_id='extract_metadata',
        python_callable=extract_metadata,
    )
    
    ontology_task = PythonOperator(
        task_id='update_ontology',
        python_callable=update_ontology,
    )
    
    embedding_task = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
    )
    
    # 작업 의존성 설정
    detect_task >> download_task >> process_task >> metadata_task >> ontology_task >> embedding_task
```

## 6. SharePoint 통합 확장

### 6.1 SharePoint 통합 개요
SharePoint는 OneDrive와 유사한 API를 제공하므로, OneDrive 통합 모듈을 확장하여 SharePoint 문서 라이브러리도 처리할 수 있습니다.

### 6.2 SharePoint 통합 설정
SharePoint 통합을 위해 추가적인 구성이 필요합니다:

```yaml
# config/onedrive.yaml에 추가
sharepoint:
  enabled: true
  site_url: "https://your-tenant.sharepoint.com/sites/your-site"
  document_libraries:
    - "Shared Documents"
    - "Project Files"
  
  # 접근 설정
  use_same_credentials: true  # OneDrive와 동일한 인증 정보 사용
```

### 6.3 SharePoint 커넥터 구현
OneDrive 커넥터를 확장하여 SharePoint 연결 기능을 추가합니다:

```python
# Cursor AI 프롬프트 예시
"""
OneDriveConnector 클래스를 확장하여 SharePoint 연결 기능을 추가한 
SharePointConnector 클래스를 구현해줘.
동일한 인증 메커니즘을 사용하되, SharePoint 사이트 및 문서 라이브러리에 
접근할 수 있는 메서드를 추가해야 함.
"""
```

#### 코드 구조
```python
from analysis.atoms.onedrive_connector import OneDriveConnector

class SharePointConnector(OneDriveConnector):
    def __init__(self, config_path):
        """SharePoint 커넥터 초기화"""
        super().__init__(config_path)
        self.sharepoint_config = self.config.get('sharepoint', {})
        self.site_url = self.sharepoint_config.get('site_url')
        self.site = None
        if self.sharepoint_config.get('enabled', False):
            self._connect_to_sharepoint()
    
    def _connect_to_sharepoint(self):
        """SharePoint 사이트 연결"""
        # SharePoint 연결 로직
        
    def list_document_libraries(self):
        """문서 라이브러리 목록 조회"""
        # 문서 라이브러리 목록 조회 로직
        
    def list_files_in_library(self, library_name):
        """문서 라이브러리의 파일 목록 조회"""
        # 파일 목록 조회 로직
        
    def download_file_from_library(self, library_name, file_path, destination):
        """문서 라이브러리에서 파일 다운로드"""
        # 파일 다운로드 로직
        
    def track_library_changes(self, library_name, delta_token=None):
        """문서 라이브러리 변경 사항 추적"""
        # 변경 사항 추적 로직
```

## 7. 문제 해결 및 최적화

### 7.1 일반적인 문제 및 해결책

#### API 제한 문제
OneDrive API는 요청 수에 제한이 있습니다. 이를 해결하기 위한 전략:
- 지수 백오프 재시도 메커니즘 구현
- 배치 처리로 요청 수 최소화
- 델타 쿼리를 사용하여 변경된 파일만 처리

```python
# 지수 백오프 재시도 예시
def exponential_backoff_retry(function, max_retries=5, initial_delay=1):
    """지수 백오프 재시도 메커니즘"""
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            return function()
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise
            
            # 지수 백오프 계산
            delay *= 2
            logging.warning(f"API 요청 실패, {delay}초 후 재시도 ({retries}/{max_retries}): {str(e)}")
            time.sleep(delay)
```

#### 대용량 파일 처리
대용량 Excel 또는 PowerPoint 파일 처리 시 메모리 문제가 발생할 수 있습니다:
- 청크 단위로 파일 처리
- 스트림 기반 처리로 전환
- 메모리 사용량 모니터링 및 최적화

```python
# 대용량 Excel 파일 청크 단위 처리 예시
def process_large_excel(file_path, chunk_size=10000):
    """대용량 Excel 파일을 청크 단위로 처리"""
    chunks = pd.read_excel(file_path, chunksize=chunk_size)
    results = []
    
    for i, chunk in enumerate(chunks):
        # 청크 처리 로직
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
        
    return pd.concat(results)
```

#### 인증 문제
OAuth 인증 관련 문제 해결:
- 토큰 만료 자동 감지 및 갱신
- 인증 오류 상세 로깅
- 백업 인증 메커니즘 준비

### 7.2 성능 최적화

#### 병렬 처리
다수의 문서를 효율적으로 처리하기 위한 병렬 처리 전략:

```python
# 병렬 문서 처리 예시
import concurrent.futures

def process_documents_parallel(file_paths, max_workers=4):
    """문서 병렬 처리"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_document, file_path): file_path for file_path in file_paths}
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results[file_path] = result
            except Exception as e:
                logging.error(f"파일 처리 오류 {file_path}: {str(e)}")
                results[file_path] = None
    
    return results
```

#### 캐싱 전략
반복 작업에 대한 효율성 향상을 위한 캐싱:
- 로컬 파일 캐시로 다운로드 최소화
- 처리 결과 캐싱으로 반복 계산 방지
- 메타데이터 캐싱으로 API 호출 최소화

```python
# 파일 캐싱 예시
import hashlib
import os

def get_cache_path(file_id, cache_dir="data/cache/onedrive"):
    """파일 ID에 대한 캐시 경로 계산"""
    file_hash = hashlib.md5(file_id.encode()).hexdigest()
    return os.path.join(cache_dir, file_hash)

def get_cached_file(file_id, cache_dir="data/cache/onedrive"):
    """캐시된 파일 조회"""
    cache_path = get_cache_path(file_id, cache_dir)
    if os.path.exists(cache_path):
        return cache_path
    return None

def cache_file(file_id, file_content, cache_dir="data/cache/onedrive"):
    """파일 내용을 캐시에 저장"""
    cache_path = get_cache_path(file_id, cache_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        f.write(file_content)
    
    return cache_path
```

## 8. OneDrive 통합 테스트

### 8.1 단위 테스트
각 모듈의 기능을 검증하는 단위 테스트:

```python
# OneDrive 커넥터 단위 테스트 예시
import unittest
from unittest.mock import MagicMock, patch
from analysis.atoms.onedrive_connector import OneDriveConnector

class TestOneDriveConnector(unittest.TestCase):
    @patch('analysis.atoms.onedrive_connector.Account')
    def test_authentication(self, mock_account):
        """인증 기능 테스트"""
        # 테스트 설정
        mock_account_instance = MagicMock()
        mock_account.return_value = mock_account_instance
        mock_account_instance.authenticate.return_value = True
        
        # 테스트 실행
        connector = OneDriveConnector('tests/test_config.yaml')
        
        # 검증
        mock_account_instance.authenticate.assert_called_once()
        self.assertTrue(connector.authenticated)
    
    # 추가 테스트 케이스
```

### 8.2 통합 테스트
여러 모듈의 상호작용을 검증하는 통합 테스트:

```python
# 문서 처리 통합 테스트 예시
import unittest
from analysis.atoms.onedrive_connector import OneDriveConnector
from analysis.molecules.document_processor import DocumentProcessor

class TestDocumentProcessingIntegration(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.connector = OneDriveConnector('tests/test_config.yaml')
        self.processor = DocumentProcessor()
    
    def test_download_and_process_document(self):
        """문서 다운로드 및 처리 통합 테스트"""
        # 테스트 파일 다운로드
        file_id = 'test_file_id'
        local_path = self.connector.download_file(file_id, 'tests/temp')
        
        # 문서 처리
        result = self.processor.process_document(local_path)
        
        # 검증
        self.assertIsNotNone(result)
        self.assertIn('text_content', result)
        self.assertIn('metadata', result)
    
    # 추가 통합 테스트 케이스
```

## 9. AI 개발 지원 및 최적화

### 9.1 OneDrive 통합 관련 효과적인 프롬프트

Cursor AI, Codex, Cursor AI 및 Codex CLI를 활용한 OneDrive 통합 개발을 위한 효과적인 프롬프트 예시:

#### Cursor AI 프롬프트
```
# 파일 메타데이터 매핑
"/generate 다음 OneDrive 파일 메타데이터를 Neo4j 노드 속성으로 매핑하는 함수를 작성해줘.
메타데이터 필드는 다음과 같음:
- id: 파일 고유 ID
- name: 파일 이름
- created_datetime: 생성 날짜/시간
- last_modified_datetime: 마지막 수정 날짜/시간
- size: 파일 크기
- web_url: 파일 웹 URL
- created_by: 생성자 정보
- file_type: 파일 유형

Neo4j Document 노드에는 다음 속성이 포함되어야 함:
- id: Neo4j 노드 ID
- file_id: OneDrive 파일 ID
- name: 파일 이름
- created_date: 생성 날짜(ISO 형식)
- modified_date: 수정 날짜(ISO 형식)
- size_bytes: 파일 크기
- url: 웹 URL
- author: 생성자 이름
- file_type: 파일 유형
- import_date: 현재 날짜/시간(ISO 형식)"
```

#### Codex 프롬프트
```
"OneDrive 델타 API를 사용하여 변경된 파일만 효율적으로 추적하는 함수를 구현해줘.
델타 토큰을 저장하고 관리하는 메커니즘이 포함되어야 하고, 
변경 유형(추가, 수정, 삭제)에 따라 다른 처리 로직을 적용해야 함.
또한 재시도 메커니즘과 오류 처리도 구현해줘."
```

#### Cursor AI 및 Codex CLI 프롬프트
```
"OneDrive 통합 시스템과 RAG 시스템을 연결하는 아키텍처를 설계해줘.
OneDrive에서 문서를 가져와 처리하고, 텍스트를 추출한 후, 청크로 분할하고,
임베딩을 생성하여 벡터 저장소에 저장하는 전체 흐름을 설명해줘.
각 단계에서 사용되는 주요 클래스, 메서드, 데이터 구조를 포함해야 함.
또한 시스템의 확장성, 견고성, 성능을 고려한 설계 결정도 설명해줘."
```

### 9.2 코드 생성 전략

대규모 OneDrive 통합 모듈 개발을 위한 AI 코드 생성 전략:

1. **계층적 접근**: 원자 모듈부터 시작하여 분자, 유기체 모듈 순으로 개발
2. **테스트 주도 개발**: 테스트 케이스를 먼저 작성하고 AI에게 구현 요청
3. **단계적 확장**: 기본 기능을 먼저 구현하고 점진적으로 기능 추가
4. **하이브리드 접근**: 복잡한 알고리즘은 Codex로, 구조 설계는 Cursor AI 및 Codex CLI로, 코드 작성은 Cursor AI로

## 10. OneDrive 통합 확장 계획

### 10.1 추가 문서 형식 지원
향후 지원할 추가 문서 형식:
- OneNote 노트북
- Visio 다이어그램
- 프로젝트 파일
- 대화형 양식

### 10.2 고급 처리 기능
개발 예정인 고급 기능:
- OCR을 통한 이미지 내 텍스트 추출
- 다국어 문서 처리 및 번역
- 문서 요약 및 핵심 내용 추출
- 문서 분류 및 자동 태깅

### 10.3 실시간 변경 감지
실시간 변경 감지 및 처리를 위한 계획:
- Microsoft Graph 웹훅 구현
- 이벤트 기반 처리 파이프라인
- 실시간 알림 시스템

---

문서 버전: 2.0  
최종 업데이트: 2025-05-17

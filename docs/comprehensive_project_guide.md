# 1인 개발자를 위한 팔란티어 파운드리 종합 가이드 (2025 개정판)
> ⚠️ 본 문서에서 언급된 Cursor AI 및 Codex CLI 관련 내용은 Cursor AI 및 Codex CLI가 담당하는 것으로 변경되었습니다.

모든 문서를 철저히 검토한 후에만 작업을 시작하십시오. 문서 내용을 완전히 이해하지 못했다면 작업을 시작하지 말고 대기하세요.

## 1. 프로젝트 개요 및 비전

### 1.1 프로젝트 비전
팔란티어 파운드리 프로젝트는 1인 개발자 환경에서 AI 협업 도구를 활용하여 엔터프라이즈급 데이터 분석 및 관리 플랫폼을 구축하는 것을 목표로 합니다. 최신 AI 기술인 Cursor AI, OpenAI Codex, Cursor AI 및 Codex CLI를 통합적으로 활용하여 개발 효율성을 극대화하고, 코드 품질을 높은 수준으로 유지합니다.

### 1.2 프로젝트 구성요소
이 프로젝트는 다음과 같은 주요 시스템으로 구성됩니다:
- **온톨로지 관리 시스템**: Neo4j 기반 데이터 관계 및 메타데이터 관리
- **데이터 파이프라인 시스템**: Apache Airflow 기반 워크플로우 자동화
- **문서 관리 시스템**: OneDrive 통합을 통한 문서의 효율적 관리 및 최적화
- **데이터 품질 시스템**: Great Expectations 기반 데이터 검증
- **웹 대시보드 인터페이스**: Dash 기반 데이터 시각화 및 관리
- **API 시스템**: FastAPI 기반 REST API
- **RAG 시스템**: 검색 증강 생성(Retrieval Augmented Generation) 시스템
- **AI 개발 지원 시스템**: Cursor AI 및 Codex CLI 기반 AI 페어 프로그래밍 및 자가 개선 워크플로

모든 시스템은 원자-분자-유기체 패턴을 따르며 독립적으로 운영되지만 통합될 수 있습니다.

### 1.3 AI 개발 환경
- **Cursor AI**: 주요 코드 편집 및 AI 코드 생성 도구
- **OpenAI Codex**: 복잡한 알고리즘 및 로직 구현을 위한 코드 생성 엔진
- **Cursor AI 및 Codex CLI**: 시스템 설계, 아키텍처 결정, 코드 검토 및 문서화 지원

### 1.4 필수 참조 문서
새 대화나 작업 세션을 시작할 때는 다음 문서들을 순서대로 검토하세요:

1. **`project_plan.md`**: 전체 프로젝트 계획 및 구현 상태
2. **`docs/status_report.md`**: 현재 프로젝트 상태 및 진행 사항
3. **`docs/system_architecture.md`**: 시스템 구조 및 모듈 구성
4. **`docs/directory_structure.md`**: 프로젝트 디렉토리 구조 및 설명
5. **`docs/document_integration_guide.md`**: OneDrive 통합 관련 지침
6. **`docs/ai_integration_guide.md`**: AI 도구 통합 가이드
7. **`docs/cursor_prompts.md`**: Cursor AI 효과적인 프롬프트 모음
8. **`docs/development_workflow.md`**: AI 기반 개발 워크플로우

## 2. 디렉토리 구조 및 설정

### 2.1 디렉토리 구조
프로젝트 구조가 정리되어 다음과 같이 구성되었습니다:

```
C:\Users\packr\OneDrive\palantir\
│
├── analysis\                          # 핵심 분석 시스템 디렉토리
│   ├── airflow\                       # Airflow 파이프라인
│   │   ├── dags\                      # DAG 파일들
│   │   └── templates\                 # 파이프라인 템플릿
│   ├── atoms\                         # 기본 기능 단위 모듈
│   ├── molecules\                     # 복합 기능 단위 모듈
│   ├── tools\                         # 도구 및 유틸리티
│   │   ├── atoms\                     # 기본 도구
│   │   ├── molecules\                 # 복합 도구
│   │   └── organisms\                 # 통합 시스템
│   └── mcp_init.py                    # MCP 초기화 스크립트
│
├── config\                            # 구성 파일 디렉토리
│   ├── app_config.json                # 애플리케이션 전체 설정
│   ├── neo4j.yaml                     # Neo4j 설정
│   ├── airflow.yaml                   # Airflow 설정 
│   ├── onedrive.yaml                  # OneDrive 설정
│   ├── llm.yaml                       # LLM 설정
│   ├── rag.yaml                       # RAG 시스템 설정
│   └── ai_tools.yaml                  # AI 도구 설정 및 프롬프트 템플릿
│
├── data\                              # 데이터 디렉토리
│   ├── ontology\                      # 온톨로지 관련 데이터
│   ├── quality\                       # 데이터 품질 관련 파일
│   ├── samples\                       # 샘플 데이터
│   └── embeddings\                    # 벡터 임베딩 저장소
│
├── docs\                              # 프로젝트 문서 디렉토리
│   ├── status_report.md               # 상태 보고서
│   ├── system_architecture.md         # 시스템 아키텍처
│   ├── directory_structure.md         # 디렉토리 구조 설명
│   ├── document_integration_guide.md  # OneDrive 통합 관련 지침
│   ├── ai_integration_guide.md        # AI 도구 통합 가이드
│   ├── cursor_prompts.md              # Cursor AI 효과적인 프롬프트 모음
│   ├── codex_examples.md              # Codex 활용 예제 모음
│   └── development_workflow.md        # AI 기반 개발 워크플로우
│
├── logs\                              # 로그 디렉토리
│
├── output\                            # 결과물 디렉토리
│   ├── reports\                       # 보고서
│   ├── viz\                           # 시각화 결과
│   └── generated_code\                # AI 생성 코드 보관
│
├── ai_resources\                      # AI 관련 리소스
│   ├── prompts\                       # 효과적인 프롬프트 템플릿
│   ├── snippets\                      # 재사용 가능한 코드 스니펫
│   └── patterns\                      # 설계 패턴 및 코딩 패턴
│
├── project_plan.md                    # 프로젝트 계획서
└── comprehensive_project_guide.md     # 종합 프로젝트 가이드
```

### 2.2 구성 파일
각 핵심 구성 요소의 설정은 config 디렉토리에 있는 JSON/YAML 파일에 저장됩니다.

1. Neo4j 구성: `config/neo4j.yaml`
   ```yaml
   neo4j:
     uri: "neo4j://localhost:7687"
     user: "neo4j"
     password: "password"
   ontology:
     directory: "C:\Users\packr\OneDrive\palantir\data\ontology"
   ```

2. Airflow 구성: `config/airflow.yaml`
   ```yaml
   airflow:
     dags_folder: "C:\Users\packr\OneDrive\palantir\analysis\airflow\dags"
     logs_folder: "C:\Users\packr\OneDrive\palantir\logs\airflow"
     executor: "SequentialExecutor"
     sql_alchemy_conn: "sqlite:///C:\Users\packr\OneDrive\palantir\data\airflow.db"
   ```

3. OneDrive 구성: `config/onedrive.yaml`
   ```yaml
   onedrive:
     client_id: "your_client_id"
     tenant_id: "your_tenant_id"
     scopes:
       - "Files.Read"
       - "Files.Read.All"
     cache_location: "data/cache/onedrive"
     refresh_interval: 300  # 5분마다 새 문서 확인
   ```

4. 애플리케이션 설정: `config/app_config.json`
   ```json
   {
     "document_processing": {
       "supported_formats": ["pdf", "docx", "xlsx", "pptx", "txt"],
       "chunk_size": 1000,
       "overlap": 200,
       "max_documents_per_run": 50,
       "extract_metadata": true,
       "extract_text": true,
       "create_embeddings": true
     },
     "data_quality": {
       "expectations_path": "C:\\Users\\packr\\OneDrive\\palantir\\data\\quality",
       "validation_results_path": "C:\\Users\\packr\\OneDrive\\palantir\\output\\reports",
       "notification_enabled": true
     },
     "ontology": {
       "base_ontology_path": "C:\\Users\\packr\\OneDrive\\palantir\\data\\ontology\\base_ontology.json",
       "auto_update": true
     }
   }
   ```

5. LLM 통합 구성: `config/llm.yaml`
   ```yaml
   claude:
     api_key: "your_claude_api_key"
     model: "claude-3-7-sonnet-20250219"
     max_tokens: 4000
   
   prompts:
     template_directory: "C:\Users\packr\OneDrive\palantir\ai_resources\prompts"
   
   output:
     generated_code_dir: "C:\Users\packr\OneDrive\palantir\output\generated_code"
   ```

6. AI 도구 설정: `config/ai_tools.yaml`
   ```yaml
   cursor_ai:
     preferred_model: "claude-3-7"
     auto_complete: true
     context_length: "maximum"
     suggestions: true
   
   codex:
     model: "o4-mini"
     approval_mode: "auto-edit"
     default_directory: "C:\Users\packr\OneDrive\palantir"
   
   integration:
     prompt_library: "C:\Users\packr\OneDrive\palantir\ai_resources\prompts"
     snippet_library: "C:\Users\packr\OneDrive\palantir\ai_resources\snippets"
   ```

### 2.3 AI 개발 도구 설정

#### Cursor AI 설정
- GitHub에서 Cursor AI 다운로드 및 설치
- Cursor AI 및 Codex CLI 모델 또는 GPT-4 모델 선택
- 프로젝트 폴더 열기 및 초기화
- 추천 설정:
  - 자동 포맷팅 활성화
  - 실시간 코드 제안 활성화
  - 컨텍스트 길이 최대화
  - 파일 인덱싱 허용

#### Codex CLI 설정
```bash
# Codex CLI 설치
npm install -g @openai/codex

# 기본 설정
codex config set approval-mode auto-edit
codex config set model o4-mini

# 프로젝트 폴더 설정
codex config set default-directory "C:\Users\packr\OneDrive\palantir"
```

#### 의존성 설치
필요한 Python 패키지를 설치하기 위한 스크립트를 작성하여 사용하세요:

```python
# scripts/install_dependencies.py
import subprocess
import sys

def install_dependencies():
    packages = [
        "apache-airflow==2.5.1",
        "neo4j==4.4.10",
        "dash==2.9.3",
        "dash-bootstrap-components==1.4.1",
        "great-expectations==0.15.50",
        "fastapi==0.95.1",
        "uvicorn==0.22.0",
        "O365==2.0.26",
        "pandas==1.5.3",
        "plotly==5.14.1",
        "anthropic==0.7.4",
        "sentence-transformers==2.2.2",
        "chromadb==0.4.18",
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All dependencies installed successfully.")

if __name__ == "__main__":
    install_dependencies()
```

## 3. AI 기반 개발 방법론

### 3.1 개발 원칙
- **원자-분자-유기체 패턴**: 작은 기능 단위에서 복합 시스템으로 확장
- **점진적 개발**: 작은 기능 단위로 개발하고 지속적으로 통합
- **AI 주도 개발**: AI 도구를 주요 개발 파트너로 활용
- **지속적 품질 관리**: 자동화된 테스트와 리팩토링으로 코드 품질 유지
- **문서 주도 개발**: 설계 문서와 코드 문서화를 개발과 동시에 진행

### 3.2 AI 도구별 역할

#### Cursor AI
- 주 코드 편집기로 활용
- 실시간 코드 완성 및 제안
- 코드 설명 및 리팩토링
- 간단한 기능 및 알고리즘 구현
- 주요 명령어:
  - `/explain` - 선택한 코드 설명 받기
  - `/edit` - 코드 수정 요청
  - `/generate` - 새 코드 생성
  - `/chat` - AI와 대화

#### OpenAI Codex
- 복잡한 알고리즘 및 로직 구현
- 보일러플레이트 코드 자동 생성
- 특정 패턴 기반 코드 생성
- 대용량 코드 처리 및 최적화

#### Cursor AI 및 Codex CLI
- 시스템 설계 및 아키텍처 결정
- 고수준 개발 계획 수립
- 코드 검토 및 품질 평가
- 문서화 및 지식 관리
- 문제 해결 및 디버깅 지원

### 3.3 개발 워크플로우

1. **계획 단계**
   - Cursor AI 및 Codex CLI를 통한 구성요소 설계 및 아키텍처 정의
   - 필요한 모듈 및 함수 식별
   - 테스트 전략 수립

2. **구현 단계**
   - Cursor AI로 기본 코드 구조 생성
   - Codex를 활용한 복잡한 로직 구현
   - 단위별 개발 및 테스트

3. **검토 단계**
   - Cursor AI 및 Codex CLI를 통한 코드 리뷰
   - 성능 및 품질 평가
   - 개선 사항 식별

4. **최적화 단계**
   - 리팩토링 및 코드 품질 개선
   - 성능 최적화
   - 문서화 완성

5. **통합 단계**
   - 구성요소 통합
   - 통합 테스트
   - 최종 검토 및 배포

## 4. 구성요소별 개발 가이드

### 4.1 온톨로지 관리 시스템 (Neo4j)

#### 핵심 기능
- 객체 타입 및 속성 정의
- 객체 간 관계 정의
- 온톨로지 가져오기/내보내기
- 온톨로지 시각화

#### AI 개발 접근 방식
```
// Cursor AI 프롬프트 예시
"Neo4j 데이터베이스에 연결하고 'Document' 타입 노드와 'Topic' 타입 노드 간의 
'RELATES_TO' 관계를 생성하는 유틸리티 함수를 만들어줘. 관계 속성으로 
'confidence_score'를 포함해야 함."
```

#### 주요 파일
- `analysis/atoms/neo4j_connector.py`: 데이터베이스 연결 및 기본 CRUD 작업
- `analysis/molecules/ontology_manager.py`: 온톨로지 관리 클래스
- `data/ontology/base_ontology.json`: 기본 온톨로지 정의

### 4.2 OneDrive 통합 시스템

#### 핵심 기능
- OneDrive 문서 감지 및 메타데이터 추출
- Excel, PowerPoint, Word 문서 내용 처리
- 변경 사항 모니터링 및 알림
- 문서 카탈로그화 및 인덱싱

#### AI 개발 접근 방식
```
// Codex 프롬프트 예시
"O365 API를 사용하여 OneDrive에서 Excel 문서를 읽고, 
지정된 시트의 데이터를 Python 데이터프레임으로 변환한 후
데이터 유효성을 검사하는 함수를 작성해줘.
누락된 필수 열이 있으면 오류를 발생시켜야 함."
```

#### 주요 파일
- `analysis/atoms/onedrive_connector.py`: OneDrive 연결 모듈
- `analysis/molecules/document_processor.py`: 문서 처리 로직
- `config/onedrive.yaml`: OneDrive 연결 설정

### 4.3 데이터 파이프라인 시스템 (Apache Airflow)

#### 핵심 기능
- 데이터 처리 워크플로우 자동화
- 스케줄링 및 모니터링
- 파이프라인 버전 관리
- 오류 처리 및 재시도 메커니즘

#### 주요 파이프라인
- 문서 처리 파이프라인: `analysis/airflow/dags/document_processing_pipeline.py`
- 데이터 품질 모니터링 파이프라인: `analysis/airflow/dags/data_quality_pipeline.py`
- 온톨로지 관리 파이프라인: `analysis/airflow/dags/ontology_management_pipeline.py`

### 4.4 웹 대시보드 인터페이스

#### 핵심 기능
- 홈 대시보드: 시스템 개요 및 주요 지표
- 문서 관리: 문서 브라우징 및 검색
- 온톨로지 뷰: 온톨로지 시각화 및 관리
- 성능 분석: 성능 테스트 결과 시각화

#### AI 개발 접근 방식
```
// Cursor AI 및 Codex CLI 프롬프트 예시
"Dash와 Plotly를 사용하여 온톨로지 네트워크를 시각화하는 대시보드 컴포넌트를 
설계해줘. 노드는 객체 타입별로 색상이 다르게 표시되어야 하고, 관계는 선으로 
표시되어야 함. 사용자가 노드를 클릭하면 세부 정보가 사이드바에 표시되어야 함."
```

#### 주요 파일
- `analysis/tools/organisms/dashboard.py`: 대시보드 메인 모듈
- `analysis/tools/organisms/dashboard/app.py`: Dash 애플리케이션
- `analysis/tools/organisms/templates/`: HTML 템플릿

### 4.5 데이터 품질 시스템 (Great Expectations)

#### 핵심 기능
- 데이터 품질 규칙 정의
- 데이터 검증 자동화
- 품질 보고서 생성
- 품질 이슈 알림

#### 주요 파일
- `analysis/molecules/quality_validator.py`: 품질 검증 클래스
- `data/quality/expectations`: 데이터 품질 기대치 정의
- `output/reports/validation_results`: 검증 결과 보고서

### 4.6 RAG 시스템

#### 핵심 기능
- 문서 임베딩 생성
- 벡터 저장소 관리
- 의미 기반 검색
- 컨텍스트 강화 생성

#### AI 개발 접근 방식
```
// Cursor AI와 Codex 혼합 접근법
"문서에서 텍스트를 추출하고 ChromaDB에 저장하는 RAG 파이프라인을 구현해줘.
1. 문서 청크 분할
2. Sentence-Transformers로 임베딩 생성
3. ChromaDB에 저장
4. 유사도 검색 함수 구현"
```

#### 주요 파일
- `analysis/molecules/embedding_generator.py`: 임베딩 생성 모듈
- `analysis/molecules/vector_store.py`: 벡터 저장소 관리
- `analysis/tools/organisms/rag_system.py`: RAG 시스템 통합 모듈

## 5. 개발 우선순위 및 일정

### 5.1 1주차: 기반 구축 및 OneDrive 통합
- Day 1-2: 개발 환경 설정 및 AI 도구 통합
- Day 3-6: OneDrive 통합 시스템 개발
- Day 7: 첫 주 검토 및 조정

### 5.2 2-3주차: 웹 대시보드 및 RAG 시스템
- Day 8-12: 웹 대시보드 기본 프레임워크 개발
- Day 13-18: RAG 시스템 구현
- Day 19: 중간 검토 및 조정

### 5.3 4주차: LLM 통합 및 시스템 통합
- Day 20-24: LLM 통합 및 자가 개선 시스템 구현
- Day 25-27: 전체 시스템 통합 및 테스트
- Day 28: 최종 검토 및 문서화 완료

### 5.4 작업 체크리스트
1. **OneDrive 통합**
   - [ ] Excel 문서 처리기 구현
   - [ ] PowerPoint 문서 처리기 구현
   - [ ] SharePoint 연결 구현
   - [ ] 통합 테스트 및 디버깅

2. **웹 대시보드**
   - [ ] 기본 레이아웃 구현
   - [ ] 온톨로지 시각화 컴포넌트 구현
   - [ ] 문서 관리 페이지 구현
   - [ ] 품질 모니터링 페이지 구현

3. **RAG 시스템**
   - [ ] 임베딩 생성 모듈 구현
   - [ ] 벡터 저장소 설정
   - [ ] 검색 알고리즘 구현
   - [ ] 컨텍스트 통합 메커니즘 구현

4. **LLM 통합**
   - [ ] Cursor AI 및 Codex CLI API 연동 클래스 구현
   - [ ] 프롬프트 템플릿 시스템 구현
   - [ ] RAG 시스템 기본 구조 구현
   - [ ] 코드 생성 워크플로 구현

## 6. 디버깅 및 문제 해결

### 6.1 로그 확인
로그 파일은 `logs` 디렉토리에서 모듈별로 확인할 수 있습니다:
- Neo4j 로그: `logs/neo4j.log`
- Airflow 로그: `logs/airflow/*.log`
- 문서 처리 로그: `logs/document_processing.log`
- 품질 검증 로그: `logs/quality_validator.log`

### 6.2 AI 도구를 활용한 디버깅
- Cursor AI의 코드 설명 기능으로 복잡한 로직 이해
- 오류 메시지를 Cursor AI에 입력하여 해결책 제안 받기
- Cursor AI 및 Codex CLI에게 디버깅 전략과 잠재적 문제 영역 식별 요청

### 6.3 Neo4j 디버깅
```cypher
// 온톨로지 구조 확인
MATCH (n) RETURN n LIMIT 100

// 관계 확인
MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100

// 특정 타입의 객체 확인
MATCH (d:Document) RETURN d LIMIT 20
```

### 6.4 일반 문제 해결
- **연결 오류**: 서비스 실행 상태 확인
- **구성 오류**: 구성 파일의 설정 값 확인
- **권한 오류**: 필요한 API 키 및 인증 정보 확인
- **성능 문제**: 로그 파일에서 병목 현상 식별

## 7. 코드 품질 및 표준

### 7.1 코드 품질 관리
- **일관된 코딩 스타일**: PEP 8 및 프로젝트 특화 스타일 가이드 준수
- **자동화된 테스트**: 단위 테스트 및 통합 테스트 자동 생성
- **코드 리뷰**: Cursor AI 및 Codex CLI를 통한 정기적 코드 리뷰
- **문서화**: 모든 모듈 및 함수에 명확한 문서 문자열 포함

### 7.2 AI 생성 코드 품질 체크리스트
- 코드가 요구사항을 충족하는가?
- 오류 처리 및 예외 상황이 적절히 고려되었는가?
- 성능이 최적화되었는가?
- 확장 가능하고 유지보수가 용이한가?
- 보안 취약점이 있는가?
- 코드 스타일이 일관되게 유지되는가?

### 7.3 코드 작성 패턴
- **원자-분자-유기체 패턴**: 단일 책임, 합성 및 시스템 수준 모듈화
- **설정 주입**: 모든 설정은 외부에서 주입, 하드코딩 금지
- **명시적 오류 처리**: 모든 예외 상황에 대한 명시적 처리
- **풍부한 로깅**: 디버깅 및 모니터링을 위한 자세한 로그

### 7.4 문서화 표준
- 모든 모듈 및 클래스에 문서 문자열 작성
- 복잡한 함수에는 세부 설명과 예제 포함
- 프로젝트 문서는 Markdown 형식으로 작성

## 8. 1인 개발자를 위한 작업 관리 전략

### 8.1 집중적 개발 세션
- 2-3시간 집중 개발 세션 후 짧은 휴식
- 단일 기능에 집중하여 컨텍스트 전환 최소화
- 하루 작업 시작 전 목표 설정 및 작업 종료 후 회고

### 8.2 자동화 최대화
- 반복 작업의 완전 자동화
- CI/CD 파이프라인 구축으로 배포 자동화
- 테스트 및 품질 확인 자동화

### 8.3 부담 관리
- 복잡한 작업은 작은 단위로 분할
- 우선순위 명확화 및 불필요한 작업 제거
- 정기적인 진행 상황 평가 및 일정 조정

## 9. 지속적 개선 및 학습

### 9.1 AI 프롬프트 최적화
- 효과적인 프롬프트 패턴 식별 및 문서화
- 프롬프트 템플릿 지속적 개선
- 프롬프트-응답 DB 구축

### 9.2 코드 패턴 라이브러리
- 재사용 가능한 코드 패턴 식별
- AI 도구가 이해하기 쉬운 형태로 문서화
- 프로젝트 전반에 일관된 패턴 적용

## 부록: 효과적인 AI 프롬프트 예시

### Cursor AI 프롬프트
```
# 기능 구현
"다음 요구사항에 맞는 함수를 구현해줘:
- 함수명: process_excel_file
- 입력: 파일 경로 (문자열), 시트명 (문자열, 기본값='Sheet1')
- 기능: Excel 파일을 읽고 지정된 시트의 데이터를 Pandas DataFrame으로 변환
- 예외 처리: 파일이 없거나 시트가 없는 경우 적절한 예외 발생
- 로깅: 주요 단계에서 로그 남기기"

# 코드 개선
"/edit 이 코드를 더 효율적으로 만들어줘. 특히 대용량 파일 처리를 고려해서 
메모리 사용량을 줄이고, 처리 속도를 높이는 방향으로 개선해줘."

# 버그 수정
"/fix 이 코드의 버그를 찾아서 수정해줘. 특히 데이터 입력이 null일 때 
처리 방식에 문제가 있는 것 같아."
```

### Codex 프롬프트
```
# 알고리즘 구현
"Neo4j 그래프 데이터베이스에서 두 노드 간의 모든 경로를 찾고, 
경로의 가중치를 계산하는 알고리즘을 Python으로 구현해줘. 
가중치는 경로 상의 각 관계에 있는 'weight' 속성의 합으로 계산됨."

# 복잡한 기능 개발
"RAG 시스템에서 문서 청크를 최적으로 분할하는 함수를 구현해줘.
- 의미적 일관성을 유지하면서 분할해야 함
- 청크의 길이는 약 1000자로 유지
- 문단이나 제목 경계를 존중해야 함
- 필요하면 겹치는 부분을 포함할 수 있음"
```

### Cursor AI 및 Codex CLI 프롬프트
```
# 시스템 설계
"OneDrive 통합, Neo4j 온톨로지 관리, RAG 시스템을 연결하는 아키텍처를 설계해줘.
각 시스템 간의 데이터 흐름, 주요 인터페이스, 그리고 구현해야 할 핵심 모듈을
상세히 설명해줘."

# 코드 리뷰
"다음 코드를 리뷰하고 개선점을 제안해줘:
[코드 붙여넣기]
특히 다음 측면을 고려해줘:
1. 코드 품질 및 가독성
2. 성능 최적화
3. 오류 처리
4. 확장성 및 유지보수성"
```

---

문서 버전: 4.0  
최종 업데이트: 2025-05-17

# 시스템 아키텍처

[Architecture Quick Start](quick_start_architecture.md)
> ⚠️ 본 문서에서 언급된 Cursor AI 및 Codex CLI 관련 내용은 Cursor AI 및 Codex CLI가 담당하는 것으로 변경되었습니다.

## 1. 아키텍처 개요

![Architecture Diagram](diagram.png)

팔란티어 파운드리 프로젝트는 모듈화된 구성요소로 이루어진 확장 가능한 시스템입니다. 시스템은 '원자-분자-유기체' 패턴을 따라 구성되어 있으며, 각 구성요소는 명확한 경계와 책임을 가지고 독립적으로 운영되면서도 필요에 따라 통합될 수 있습니다.

### 1.1 핵심 원칙
- **모듈성**: 각 구성요소는 독립적으로 개발, 테스트, 배포 가능
- **확장성**: 시스템은 새로운 데이터 소스, 분석 기능, 시각화 방법을 쉽게 통합할 수 있도록 설계
- **데이터 중심**: 모든 구성요소는 공통된 데이터 모델을 중심으로 상호작용
- **자동화**: 반복 작업을 최대한 자동화하여 효율성 극대화
- **AI 주도 개발**: AI 도구를 개발 프로세스에 깊이 통합

## 2. 시스템 구성요소

### 2.1 데이터 레이어
- **온톨로지 관리 시스템** (Neo4j)
  - 데이터 객체 및 관계 정의
  - 메타데이터 관리
  - 온톨로지 질의 및 탐색
  
- **문서 관리 시스템** (OneDrive 통합)
  - 문서 수집 및 처리
  - 메타데이터 추출
  - 콘텐츠 색인화
  
- **벡터 저장소** (ChromaDB)
  - 문서 임베딩 저장
  - 시맨틱 검색 지원
  - 벡터 인덱싱 및 관리

### 2.2 처리 레이어
- **데이터 파이프라인 시스템** (Apache Airflow)
  - 워크플로우 자동화
  - 데이터 처리 및 변환
  - 스케줄링 및 모니터링
  
- **데이터 품질 시스템** (Great Expectations)
  - 데이터 검증 및 품질 관리
  - 품질 보고서 생성
  - 이상 감지 및 알림
  
- **RAG 시스템**
  - 문서 청크 분할
  - 임베딩 생성
  - 컨텍스트 증강 검색

### 2.3 표현 레이어
- **웹 대시보드 인터페이스** (Dash)
  - 데이터 시각화
  - 사용자 인터페이스
  - 대화형 분석
  
- **API 시스템** (FastAPI)
  - 데이터 액세스 인터페이스
  - 시스템 간 통합
  - 외부 시스템 연결

### 2.4 지원 레이어
- **LLM 통합 및 자가 개선 시스템** (Cursor AI 및 Codex CLI)
  - AI 페어 프로그래밍
  - 코드 생성 및 최적화
  - 설계 지원 및 문서화
  
- **MCP(Model Context Protocol) 시스템**
  - AI 도구 통합 프로토콜
  - 도구 등록 및 관리
  - 워크플로우 조정
  
- **로깅 및 모니터링 시스템**
  - 오류 추적
  - 성능 모니터링
  - 사용 패턴 분석

## 3. 구성요소 간 상호작용

### 3.1 데이터 흐름
```
[OneDrive] → [문서 처리 파이프라인] → [온톨로지 관리 시스템]
                    ↓
[문서 처리 파이프라인] → [RAG 시스템] → [벡터 저장소]
                    ↓
[데이터 품질 시스템] ← [데이터 파이프라인] → [웹 대시보드]
                                      ↑
                                [API 시스템]
```

### 3.2 주요 인터페이스
- **온톨로지 관리 API**: Neo4j 데이터베이스와의 상호작용
- **문서 처리 API**: OneDrive 문서 처리 및 메타데이터 추출
- **파이프라인 API**: Airflow 워크플로우 관리 및 실행
- **벡터 검색 API**: ChromaDB를 통한 시맨틱 검색
- **대시보드 API**: Dash 애플리케이션과의 데이터 교환
- **LLM API**: Cursor AI 및 Codex CLI와의 통신
- **MCP API**: AI 도구와의 통합 인터페이스

## 4. 모듈 구조

### 4.1 원자 모듈 (기본 기능 단위)
- **neo4j_connector.py**: Neo4j 데이터베이스 연결 및 기본 작업
- **onedrive_connector.py**: OneDrive 연결 및 기본 작업
- **airflow_connector.py**: Airflow 연결 및 DAG 관리
- **embeddings_generator.py**: 문서 임베딩 생성
- **quality_checker.py**: 기본 데이터 품질 검사

### 4.2 분자 모듈 (복합 기능 단위)
- **ontology_manager.py**: 온톨로지 관리 및 조작
- **document_processor.py**: 문서 처리 및 메타데이터 추출
- **pipeline_manager.py**: 데이터 파이프라인 관리
- **quality_validator.py**: 고급 데이터 검증 및 보고
- **vector_store.py**: 벡터 저장소 관리

### 4.3 유기체 모듈 (통합 시스템)
- **dashboard.py**: 웹 대시보드 애플리케이션
- **api_server.py**: REST API 서버
- **rag_system.py**: RAG 시스템 통합 모듈
- **ai_pair_system.py**: AI 페어 프로그래밍 시스템
- **mcp_init.py**: MCP 시스템 초기화 및 관리

## 5. 기술 스택

### 5.1 핵심 기술
- **데이터베이스**: Neo4j (그래프 데이터베이스)
- **워크플로우 엔진**: Apache Airflow
- **데이터 품질**: Great Expectations
- **벡터 저장소**: ChromaDB
- **웹 프레임워크**: Dash, FastAPI
- **AI 모델**: Cursor AI 및 Codex CLI, OpenAI Codex
- **개발 도구**: Cursor AI, OpenAI Codex CLI
- **통합 프로토콜**: MCP(Model Context Protocol)

### 5.2 언어 및 라이브러리
- **주 개발 언어**: Python 3.10
- **데이터 처리**: Pandas, NumPy
- **시각화**: Plotly, Dash
- **AI 통합**: Anthropic API, OpenAI API
- **임베딩**: Sentence-Transformers
- **RPC 통신**: JSON-RPC 2.0

## 6. 배포 아키텍처

### 6.1 개발 환경
- 로컬 Windows 개발 환경
- 로컬 Neo4j 인스턴스
- 로컬 Airflow 서버
- OneDrive API 통합
- 로컬 MCP 서버

### 6.2 프로덕션 환경 계획
- 클라우드 기반 Neo4j 인스턴스 (필요시)
- 확장 가능한 문서 처리 파이프라인
- 자동화된 배포 및 확장
- 분산 MCP 서버 구성

## 7. 보안 아키텍처

### 7.1 인증 및 권한
- API 키 기반 인증
- 세분화된 접근 제어
- 로깅 및 감사

### 7.2 데이터 보안
- 민감 데이터 암호화
- 안전한 API 통신
- 데이터 권한 관리

## 8. 확장성 및 성능 고려사항

### 8.1 확장 계획
- 다양한 데이터 소스 통합 (SharePoint, Google Drive 등)
- 고급 분석 및 머신 러닝 모듈 추가
- 실시간 처리 및 알림 기능
- 추가 MCP 도구 통합

### 8.2 성능 최적화
- 온톨로지 쿼리 최적화
- 대용량 문서 처리 효율화
- 벡터 검색 성능 향상
- MCP 통신 최적화

## 9. AI 개발 아키텍처

### 9.1 Cursor AI 통합
- 코드 편집 및 AI 코드 생성
- 실시간 코드 제안 및 리팩토링
- 프로젝트 컨텍스트 인식

### 9.2 Codex 통합
- 복잡한 알고리즘 생성
- 특수 패턴 구현
- 코드 품질 검증

### 9.3 Cursor AI 및 Codex CLI 통합
- 시스템 설계 지원
- 코드 리뷰 및 최적화
- 문서화 및 지식 관리

### 9.4 MCP 시스템
- AI 도구 등록 및 관리
- 통합된 인터페이스 제공
- 도구 간 상호작용 조정
- 워크플로우 자동화

## 10. 개발 워크플로우

### 10.1 계획 및 설계
- Cursor AI 및 Codex CLI를 통한 모듈 설계
- 아키텍처 정의 및 문서화
- 인터페이스 명세

### 10.2 구현
- Cursor AI로 기본 코드 구조 생성
- Codex로 복잡한 로직 구현
- 단위별 개발 및 테스트
- MCP 도구를 통한 코드 생성 및 리팩토링

### 10.3 검토 및 최적화
- Cursor AI 및 Codex CLI를 통한 코드 리뷰
- 성능 및 품질 평가
- 리팩토링 및 최적화

### 10.4 통합 및 배포
- 모듈 통합 및 테스트
- 문서화 및 지식 공유
- 배포 및 모니터링

---

문서 버전: 2.1  
최종 업데이트: 2025-05-18

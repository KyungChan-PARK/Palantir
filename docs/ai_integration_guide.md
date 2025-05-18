# AI 도구 통합 가이드
> ⚠️ 본 문서에서 언급된 Cursor AI 및 Codex CLI 관련 내용은 Cursor AI 및 Codex CLI가 담당하는 것으로 변경되었습니다.

## 1. 개요

팔란티어 파운드리 프로젝트는 AI 도구를 통합적으로 활용하여 개발 효율성을 극대화하고 코드 품질을 높은 수준으로 유지합니다. 이 가이드는 프로젝트에서 활용하는 AI 도구(Cursor AI, OpenAI Codex, Cursor AI 및 Codex CLI)의 설정, 통합, 최적화 방법을 상세히 설명합니다.

## 2. AI 개발 환경 아키텍처

### 2.1 통합 아키텍처
```
[개발자] ←→ [Cursor AI] ←→ [코드베이스]
    ↑           ↑           ↑
    ↓           ↓           ↓
[Cursor AI 및 Codex CLI] ←→ [Codex CLI] ←→ [MCP 시스템]
    ↑           ↑           ↑
    ↓           ↓           ↓
[프롬프트 라이브러리] ←→ [코드 스니펫 라이브러리] ←→ [AI 도구 서비스]
```

### 2.2 도구별 역할

- **Cursor AI**: 주 코드 편집기, 실시간 코드 완성, 리팩토링
- **OpenAI Codex**: 복잡한 알고리즘 생성, 특수 패턴 구현
- **Cursor AI 및 Codex CLI**: 시스템 설계, 코드 리뷰, 문서화 지원
- **MCP 시스템**: AI 도구 통합 프로토콜, 도구 등록 및 관리, 워크플로우 조정

### 2.3 데이터 흐름
1. 개발자는 Cursor AI로 코드 작성 및 편집
2. 복잡한 알고리즘이나 패턴은 Codex로 생성
3. 시스템 설계 및 아키텍처 결정은 Cursor AI 및 Codex CLI와 상의
4. 생성된 코드는 코드베이스에 통합
5. 코드 리뷰 및 최적화는 Cursor AI 및 Codex CLI로 수행
6. MCP 시스템을 통해 다양한 AI 도구 활용
7. 자동화된 워크플로우로 AI 도구 간 상호작용 조정

## 3. Cursor AI 설정 및 활용

### 3.1 Cursor AI 설치 및 설정

1. **설치 방법**
   - [Cursor AI 공식 웹사이트](https://cursor.sh/)에서 다운로드
   - 설치 마법사 실행 및 기본 설정 완료
   - 초기 실행 시 GitHub 또는 이메일로 로그인

2. **프로젝트 설정**
   ```
   1. Cursor AI 실행
   2. File > Open Folder 선택
   3. C:\Users\packr\OneDrive\palantir 디렉토리 선택
   4. 첫 실행 시 프로젝트 인덱싱 진행
   ```

3. **AI 모델 설정**
   - Settings > AI > Model: `Cursor AI 및 Codex CLI-3-7-Sonnet` 선택
   - Context Length: `Maximum` 설정
   - Auto Complete: 활성화
   - Chat & Inline Chat: 활성화

### 3.2 Cursor AI 주요 기능

#### 코드 생성 및 편집
- `cmd/ctrl + k`: AI와 대화창 열기
- `/generate`: 코드 생성 요청
- `/edit`: 코드 수정 요청
- `/explain`: 코드 설명 요청
- `/fix`: 오류 수정 요청
- `/test`: 테스트 코드 생성 요청
- `tab`: 제안된 코드 수락

#### 프로젝트 탐색
- `cmd/ctrl + p`: 파일 탐색
- `cmd/ctrl + shift + f`: 프로젝트 전체 검색
- `cmd/ctrl + b`: 정의로 이동
- `alt + click`: 여러 커서 생성

#### AI 프롬프트 지침
- 명확하고 구체적인 지시어 사용
- 필요한 입력과 출력 명시
- 코드 스타일 및 패턴 지정
- 오류 처리 및 로깅 요구사항 포함

### 3.3 효과적인 활용 전략

1. **컨텍스트 인식 향상**
   - 작업 시작 전 프로젝트 구조에 대해 AI에게 설명
   - 중요한 디자인 패턴 및 코딩 스타일 알려주기
   - 관련 파일 참조 및 링크 제공

2. **단계적 개발**
   - 큰 기능을 작은 단계로 분할
   - 각 단계별로 명확한 지시 제공
   - 중간 결과물 검토 및 피드백

3. **팁과 트릭**
   - 코드블록 선택 후 `/edit` 명령으로 특정 부분만 수정
   - 복잡한 기능은 테스트 케이스부터 작성
   - 유사한 기존 코드 참조하여 일관성 유지

## 4. OpenAI Codex 설정 및 활용

### 4.1 Codex CLI 설치 및 설정

1. **설치 방법**
   ```bash
   npm install -g @openai/codex
   ```

2. **기본 설정**
   ```bash
   # API 키 설정
   codex config set api-key your_api_key

   # 기본 모델 설정
   codex config set model o4-mini

   # 승인 모드 설정
   codex config set approval-mode auto-edit

   # 기본 디렉토리 설정
   codex config set default-directory "C:\Users\packr\OneDrive\palantir"
   ```

3. **환경 변수 설정**
   ```
   CODEX_API_KEY=your_api_key
   CODEX_DEFAULT_MODEL=o4-mini
   CODEX_APPROVAL_MODE=auto-edit
   ```

### 4.2 Codex CLI 주요 명령어

#### 기본 명령어
- `codex "명령어"`: 자연어 명령 실행
- `codex --file file.py "명령어"`: 특정 파일에 대한 명령 실행
- `codex --dir directory "명령어"`: 특정 디렉토리에 대한 명령 실행
- `codex --help`: 도움말 표시

#### 승인 모드
- `codex --approval-mode suggest "명령어"`: 제안만 표시(기본값)
- `codex --approval-mode auto-edit "명령어"`: 자동 편집(사용자 확인 필요)
- `codex --approval-mode full-auto "명령어"`: 완전 자동(사용자 확인 없음)

#### 고급 명령어
- `codex --context "컨텍스트" "명령어"`: 추가 컨텍스트 제공
- `codex --execute "명령어"`: 코드 생성 후 자동 실행
- `codex --no-cache "명령어"`: 캐시 사용하지 않음

### 4.3 효과적인 활용 전략

1. **복잡한 알고리즘 생성**
   ```bash
   codex "Neo4j 그래프에서 두 노드 간의 모든 경로를 찾고, 경로의 가중치를 계산하는 알고리즘을 Python으로 구현해줘. 가중치는 경로 상의 각 관계에 있는 'weight' 속성의 합으로 계산됨."
   ```

2. **파일 탐색 및 수정**
   ```bash
   codex --dir analysis/molecules "온톨로지 관리 클래스에 두 노드 간의 관계를 생성하는 메서드를 추가해줘."
   ```

3. **프로젝트 이해**
   ```bash
   codex --dir analysis "이 디렉토리의 구조와 주요 모듈의 기능을 설명해줘."
   ```

4. **대량 코드 생성**
   ```bash
   codex --approval-mode auto-edit "다양한 문서 형식(Word, Excel, PowerPoint, PDF, 텍스트)을 처리할 수 있는 팩토리 패턴 기반의 문서 처리 시스템을 구현해줘."
   ```

## 5. Cursor AI 및 Codex CLI 통합 및 활용

### 5.1 Cursor AI 및 Codex CLI API 설정

1. **API 키 설정**
   ```python
   # config/llm.yaml
   claude:
     api_key: "your_claude_api_key"
     model: "claude-3-7-sonnet-20250219"
     max_tokens: 4000
   ```

2. **클라이언트 구현**
   ```python
   # analysis/tools/organisms/ai_pair_system/claude_client.py
   import anthropic
   import yaml
   import logging
   from pathlib import Path

   class Cursor AI 및 Codex CLIClient:
       def __init__(self, config_path):
           """Cursor AI 및 Codex CLI API 클라이언트 초기화"""
           self.config = self._load_config(config_path)
           self.client = anthropic.Anthropic(api_key=self.config['claude']['api_key'])
           self.model = self.config['claude']['model']
           self.max_tokens = self.config['claude']['max_tokens']
           self.logger = logging.getLogger(__name__)
       
       def _load_config(self, config_path):
           """구성 파일 로드"""
           with open(config_path, 'r') as f:
               return yaml.safe_load(f)
       
       async def ask(self, prompt, system_prompt=None):
           """Cursor AI 및 Codex CLI에게 질문하기"""
           try:
               message = self.client.messages.create(
                   model=self.model,
                   max_tokens=self.max_tokens,
                   messages=[{"role": "user", "content": prompt}],
                   system=system_prompt
               )
               return message.content
           except Exception as e:
               self.logger.error(f"Cursor AI 및 Codex CLI API 오류: {str(e)}")
               return None
   ```

### 5.2 주요 사용 사례

#### 시스템 설계 및 아키텍처
```python
system_prompt = """
당신은 소프트웨어 설계 전문가입니다. 사용자의 요구사항에 맞는 아키텍처와 설계를 제안해주세요.
설계는 확장성, 유지보수성, 테스트 가능성을 고려해야 합니다.
원자-분자-유기체 패턴을 따르는 모듈식 설계를 선호합니다.
"""

design_prompt = """
OneDrive 통합, Neo4j 온톨로지 관리, RAG 시스템을 연결하는 아키텍처를 설계해줘.
각 시스템 간의 데이터 흐름, 주요 인터페이스, 그리고 구현해야 할 핵심 모듈을 상세히 설명해줘.
"""

design = await claude_client.ask(design_prompt, system_prompt)
```

#### 코드 리뷰
```python
system_prompt = """
당신은 코드 리뷰 전문가입니다. 제공된 코드를 검토하고 개선 사항을 제안해주세요.
다음 측면을 고려해주세요:
1. 코드 품질 및 가독성
2. 성능 최적화
3. 오류 처리
4. 확장성 및 유지보수성
5. 보안 취약점
"""

code_review_prompt = f"""
다음 코드를 리뷰하고 개선점을 제안해줘:

```python
{code_to_review}
```
"""

review = await claude_client.ask(code_review_prompt, system_prompt)
```

#### 문서화
```python
system_prompt = """
당신은 기술 문서 작성 전문가입니다. 제공된 코드나 설계를 바탕으로 명확하고 이해하기 쉬운 문서를 작성해주세요.
문서는 Markdown 형식으로 작성하고, 다음 요소를 포함해야 합니다:
1. 개요 및 목적
2. 아키텍처 및 구성요소
3. 사용 방법 및 예제
4. 주요 API 설명
5. 주의사항 및 제한사항
"""

documentation_prompt = f"""
다음 코드에 대한 문서를 작성해줘:

```python
{code_to_document}
```
"""

documentation = await claude_client.ask(documentation_prompt, system_prompt)
```

### 5.3 효과적인 활용 전략

1. **시스템 프롬프트 최적화**
   - 역할과 전문성 명확히 정의
   - 기대하는 출력 형식 지정
   - 프로젝트 컨텍스트 및 패턴 포함

2. **컨텍스트 제공**
   - 관련 코드 및 설계 문서 포함
   - 프로젝트 특화 용어 및 패턴 설명
   - 이전 작업 및 결정 사항 요약

3. **대화형 개발**
   - 초기 설계 후 단계적 세부사항 구체화
   - 생성된 코드에 대한 피드백 및 개선 요청
   - 문제점 해결을 위한 대화식 접근

## 6. AI 자원 관리

### 6.1 프롬프트 라이브러리

프롬프트 라이브러리는 효과적인 프롬프트 패턴을 수집하고 재사용하기 위한 리포지토리입니다. `ai_resources/prompts` 디렉토리에 저장됩니다.

#### 디렉토리 구조
```
ai_resources/prompts/
│
├── cursor/                          # Cursor AI 프롬프트
│   ├── neo4j_prompts.md             # Neo4j 관련 프롬프트
│   ├── onedrive_prompts.md          # OneDrive 관련 프롬프트
│   └── rag_prompts.md               # RAG 시스템 관련 프롬프트
│
├── codex/                           # Codex 프롬프트
│   ├── algorithm_prompts.md         # 알고리즘 생성 프롬프트
│   ├── pattern_prompts.md           # 디자인 패턴 프롬프트
│   └── refactoring_prompts.md       # 리팩토링 프롬프트
│
└── claude/                          # Cursor AI 및 Codex CLI 프롬프트
    ├── design_prompts.md            # 설계 관련 프롬프트
    ├── review_prompts.md            # 코드 리뷰 프롬프트
    └── documentation_prompts.md     # 문서화 프롬프트
```

#### 프롬프트 형식
```markdown
# 프롬프트 제목

## 목적
이 프롬프트의 목적 및 용도에 대한 설명

## 포함 요소
- 역할 및 전문성
- 작업 설명
- 제약 조건
- 출력 형식

## 템플릿
```
[프롬프트 템플릿]
```

## 예시
```
[예시 프롬프트]
```

## 결과
[성공적인 결과 예시]

## 팁
- 사용 시 주의사항
- 최적화 방법
```

### 6.2 코드 스니펫 라이브러리

코드 스니펫 라이브러리는 재사용 가능한 코드 조각을 수집하고 관리하기 위한 리포지토리입니다. `ai_resources/snippets` 디렉토리에 저장됩니다.

#### 디렉토리 구조
```
ai_resources/snippets/
│
├── neo4j/                           # Neo4j 관련 스니펫
│   ├── connection.py                # 연결 관련 스니펫
│   ├── queries.py                   # 쿼리 관련 스니펫
│   └── transactions.py              # 트랜잭션 관련 스니펫
│
├── onedrive/                        # OneDrive 관련 스니펫
│   ├── authentication.py            # 인증 관련 스니펫
│   ├── file_operations.py           # 파일 작업 관련 스니펫
│   └── change_tracking.py           # 변경 추적 관련 스니펫
│
└── patterns/                        # 디자인 패턴 스니펫
    ├── factory.py                   # 팩토리 패턴 스니펫
    ├── observer.py                  # 옵저버 패턴 스니펫
    └── strategy.py                  # 전략 패턴 스니펫
```

#### 스니펫 형식
```python
"""
제목: [스니펫 제목]
설명: [스니펫 설명]
사용법: [사용 방법]
의존성: [필요한 패키지 및 모듈]
작성자: [작성자]
버전: [버전]
업데이트: [최종 업데이트 날짜]
"""

# 코드 스니펫
def example_function():
    """함수 문서화"""
    pass

class ExampleClass:
    """클래스 문서화"""
    pass

# 사용 예시
if __name__ == "__main__":
    # 사용 예시 코드
    pass
```

### 6.3 패턴 라이브러리

패턴 라이브러리는 프로젝트에서 사용하는 설계 패턴과 코딩 패턴을 문서화한 리포지토리입니다. `ai_resources/patterns` 디렉토리에 저장됩니다.

#### 디렉토리 구조
```
ai_resources/patterns/
│
├── atomic_patterns.md               # 원자 수준 패턴
├── molecular_patterns.md            # 분자 수준 패턴
└── organism_patterns.md             # 유기체 수준 패턴
```

#### 패턴 형식
```markdown
# 패턴 제목

## 분류
[원자/분자/유기체] 패턴

## 목적
이 패턴의 목적 및 해결하는 문제

## 구조
패턴의 구조 및 구성요소에 대한 설명

## 적용 상황
이 패턴이 적합한 상황

## 구현 방법
패턴 구현 방법 및 예시

## 장단점
- 장점 1
- 장점 2
- 단점 1
- 단점 2

## 관련 패턴
- 관련 패턴 1
- 관련 패턴 2

## 예시 코드
```python
# 예시 코드
```
```

## 7. AI 통합 워크플로우

### 7.1 계획 및 설계 단계

1. **요구사항 분석**
   - Cursor AI 및 Codex CLI를 사용하여 요구사항 명확화
   - 기능 및 비기능 요구사항 정의
   - 우선순위 및 제약 조건 식별

2. **아키텍처 설계**
   ```
   Cursor AI 및 Codex CLI에 프롬프트 전송:
   "OneDrive 통합, Neo4j 온톨로지 관리, RAG 시스템을 연결하는 아키텍처를 설계해줘.
   각 시스템 간의 데이터 흐름, 주요 인터페이스, 그리고 구현해야 할 핵심 모듈을
   상세히 설명해줘."
   ```

3. **모듈 설계**
   ```
   Cursor AI 및 Codex CLI에 프롬프트 전송:
   "OneDrive에서 문서를 가져와 처리하는 모듈의 클래스 다이어그램을 설계해줘.
   다양한 문서 형식(Word, Excel, PowerPoint, PDF)을 처리할 수 있어야 하고,
   전략 패턴을 사용하여 각 문서 유형에 맞는 처리기를 선택할 수 있어야 해."
   ```

### 7.2 구현 단계

1. **기본 구조 생성**
   ```
   Cursor AI에 명령:
   "/generate 아래 클래스 다이어그램에 따라 문서 처리 시스템의 기본 구조를 구현해줘.
   인터페이스, 추상 클래스, 구체 클래스를 포함해야 하고,
   각 클래스의 주요 메서드는 구현 없이 문서화만 포함하면 됨."
   
   [클래스 다이어그램 붙여넣기]
   ```

2. **복잡한 알고리즘 구현**
   ```
   Codex CLI 명령:
   codex "Neo4j 그래프에서 두 노드 간의 모든 경로를 찾고,
   경로의 가중치를 계산하는 알고리즘을 Python으로 구현해줘.
   가중치는 경로 상의 각 관계에 있는 'weight' 속성의 합으로 계산됨."
   ```

3. **특정 기능 구현**
   ```
   Cursor AI에 명령:
   "/generate Excel 파일을 처리하는 ExcelHandler 클래스를 구현해줘.
   pandas와 openpyxl을 사용하여 다음 기능을 구현해야 함:
   1. 시트 목록 조회
   2. 시트 데이터를 DataFrame으로 변환
   3. 헤더 및 데이터 타입 자동 감지
   4. 차트 및 그래프 정보 추출
   5. 오류 처리 (손상된 파일, 비밀번호 보호 등)"
   ```

### 7.3 검토 및 최적화 단계

1. **코드 리뷰**
   ```
   Cursor AI 및 Codex CLI에 프롬프트 전송:
   "다음 코드를 리뷰하고 개선점을 제안해줘.
   특히 다음 측면을 고려해줘:
   1. 코드 품질 및 가독성
   2. 성능 최적화
   3. 오류 처리
   4. 확장성 및 유지보수성"
   
   [코드 붙여넣기]
   ```

2. **성능 최적화**
   ```
   Cursor AI에 명령:
   "/edit 이 코드를 더 효율적으로 만들어줘.
   특히 대용량 파일 처리를 고려해서 메모리 사용량을 줄이고,
   처리 속도를 높이는 방향으로 개선해줘."
   
   [코드 블록 선택]
   ```

3. **테스트 코드 생성**
   ```
   Cursor AI에 명령:
   "/test 이 클래스에 대한 단위 테스트를 작성해줘.
   주요 메서드의 정상 동작, 경계 조건, 오류 케이스를 포함해야 함.
   unittest 프레임워크를 사용하고, 필요한 경우 mock 객체를 활용해줘."
   
   [클래스 코드 블록 선택]
   ```

### 7.4 통합 및 배포 단계

1. **시스템 통합**
   ```
   Cursor AI에 명령:
   "/generate 다음 모듈들을 통합하는 시스템 수준 클래스를 구현해줘.
   OneDriveConnector, DocumentProcessor, OntologyManager를 사용하여
   OneDrive에서 문서를 가져와 처리하고 온톨로지에 저장하는 통합 시스템이어야 함."
   ```

2. **문서화**
   ```
   Cursor AI 및 Codex CLI에 프롬프트 전송:
   "다음 코드에 대한 사용자 가이드를 작성해줘.
   일반 사용자와 개발자 두 가지 대상으로 작성하고,
   주요 기능, 사용 방법, API 문서, 설정 방법을 포함해줘."
   
   [코드 붙여넣기]
   ```

3. **배포 스크립트 생성**
   ```
   Codex CLI 명령:
   codex "이 Python 프로젝트를 패키징하고 배포하는 setup.py 파일을 작성해줘.
   필요한 의존성은 requirements.txt에서 가져오고,
   콘솔 스크립트로 실행할 수 있도록 entry_points를 설정해줘."
   ```

## 8. 문제 해결 및 최적화

### 8.1 일반적인 AI 도구 문제 해결

#### Cursor AI 문제 해결
- **문제**: AI가 프로젝트 컨텍스트를 이해하지 못하는 경우
  - **해결책**: 작업 전 프로젝트 구조 및 목적 설명, 관련 파일 참조 제공
  
- **문제**: 코드 생성이 불완전하거나 오류가 있는 경우
  - **해결책**: 요구사항을 더 명확히 제공, 단계별 접근, 기존 코드 참조

- **문제**: 응답이 지나치게 길거나 복잡한 경우
  - **해결책**: 요청을 더 작은 단위로 분할, 출력 형식 명시

#### Codex 문제 해결
- **문제**: API 제한 또는 오류 발생 시
  - **해결책**: 지수 백오프 재시도, 캐싱 메커니즘, 오프라인 모드 활용

- **문제**: 생성된 코드가 프로젝트 스타일과 맞지 않는 경우
  - **해결책**: 프로젝트 스타일 가이드 참조 제공, 예시 코드 포함

- **문제**: 복잡한 요청에 대한 응답이 부족한 경우
  - **해결책**: 요청을 더 작은 단위로 분할, 컨텍스트 제공

#### Cursor AI 및 Codex CLI 문제 해결
- **문제**: 토큰 제한으로 인한 불완전한 응답
  - **해결책**: 요청 분할, 필수 정보만 포함, 후속 요청 활용

- **문제**: 프로젝트 특화 지식 부족
  - **해결책**: 시스템 프롬프트에 프로젝트 컨텍스트 포함, 필요한 배경 지식 제공

- **문제**: API 연결 문제
  - **해결책**: 오류 처리 및 재시도 메커니즘, 로컬 캐싱, 대체 API 엔드포인트

### 8.2 AI 통합 최적화 전략

1. **프롬프트 최적화**
   - 명확하고 구체적인 지시어 사용
   - 예시 및 반례 포함
   - 출력 형식 명시
   - 프로젝트 컨텍스트 제공

2. **워크플로우 최적화**
   - 각 AI 도구의 강점에 맞는 작업 할당
   - 반복적인 작업 템플릿화
   - 효과적인 프롬프트 재사용
   - 생성된 코드의 품질 검증 자동화

3. **리소스 최적화**
   - API 호출 최소화를 위한 효율적인 요청
   - 결과 캐싱으로 중복 작업 방지
   - 병렬 처리를 통한 성능 향상
   - 비용 효율적인 모델 선택

## 9. MCP 시스템 및 OpenAI Codex MCP 통합

### 9.1 MCP 시스템 개요
MCP(Model Context Protocol) 시스템은 다양한 AI 도구를 통합하고 관리하기 위한 프로토콜입니다. 이 시스템은 일관된 인터페이스를 통해 여러 AI 도구의 기능에 접근할 수 있게 해줍니다.

#### MCP 시스템 아키텍처
```
[MCP 초기화 모듈] ←→ [도구 레지스트리]
          ↑
          ↓
[MCP 서비스 1] ←→ [MCP 서비스 2] ←→ [MCP 서비스 3]
      ↑                 ↑                 ↑
      ↓                 ↓                 ↓
[AI 도구 1]        [AI 도구 2]        [AI 도구 3]
```

#### MCP 시스템 구성요소
- **MCP 초기화 모듈**: MCP 시스템 초기화 및 구성 관리
- **도구 레지스트리**: AI 도구 등록 및 관리
- **MCP 서비스**: 개별 AI 도구에 대한 인터페이스 제공
- **AI 도구**: 실제 AI 기능을 수행하는 외부 도구

### 9.2 OpenAI Codex MCP 통합
OpenAI Codex MCP 도구는 JSON-RPC 2.0 프로토콜을 통해 OpenAI Codex CLI와 통합됩니다. 이 통합을 통해 코드 생성, 코드 설명, 디버깅 등의 기능을 MCP 시스템을 통해 활용할 수 있습니다.

#### 주요 기능
- **코드 생성**: 프롬프트로부터 코드 생성
- **코드 설명**: 복잡한 코드 설명
- **코드 디버깅**: 버그 식별 및 수정
- **코드 리팩토링**: 코드 개선 및 최적화
- **테스트 생성**: 테스트 코드 생성

#### 사용 예시
```python
from analysis.mcp_init import mcp

async def generate_code():
    result = await mcp.tools["openai_codex_write_code"]["function"](
        prompt="Neo4j 연결 클래스 구현",
        model="o4-mini"  # o4-mini, o4-preview, o4-pro 선택 가능
    )
    return result["code"]
```

자세한 사용 방법은 `docs/openai_codex_mcp_guide.md` 문서를 참조하세요.

## 10. AI 개발 지원 시스템 확장 계획

### 10.1 추가 AI 통합 계획
- GPT-4.5 및 Llama 3 기반 모델 통합
- 코드 생성 및 품질 평가 벤치마크
- 멀티모달 AI 지원 (코드-이미지 변환, 다이어그램 생성)
- 코드 자동화 테스트 및 버그 예측

### 10.2 자가 개선 메커니즘
- 생성된 코드 품질 평가 및 피드백 루프
- 프로젝트 특화 언어 모델 파인튜닝
- 효과적인 프롬프트 패턴 자동 학습
- 코드 생성 히스토리 분석 및 최적화

### 10.3 개발자 경험 개선
- AI 도구 통합 대시보드
- 프롬프트 및 결과 공유 메커니즘
- 실시간 협업 및 코드 리뷰
- 자동화된 문서화 및 지식 관리

## 부록: 시스템 프롬프트 템플릿

### Cursor AI 시스템 프롬프트
```
당신은 숙련된 Python 개발자로서 팔란티어 파운드리 프로젝트를 지원합니다.
이 프로젝트는 다음 구성요소를 포함합니다:
- Neo4j 기반 온톨로지 관리 시스템
- OneDrive 통합 문서 관리 시스템
- Apache Airflow 기반 데이터 파이프라인
- Dash 기반 웹 대시보드
- RAG(Retrieval Augmented Generation) 시스템

프로젝트는 '원자-분자-유기체' 패턴을 따르며, 코드는 PEP 8 스타일 가이드를 준수합니다.
모든 클래스와 함수는 명확한 문서 문자열을 포함해야 하며, 오류 처리 및 로깅이 필수적입니다.
설정 값은 외부 구성 파일에서 로드하고, 하드코딩을 피합니다.

코드 생성 시, 다음 사항을 고려하세요:
1. 모듈성 및 재사용성
2. 테스트 가능성
3. 성능 및 확장성
4. 견고한 오류 처리
5. 명확한 문서화
```

### Codex 시스템 프롬프트
```
이 프로젝트는 Python 기반의 '팔란티어 파운드리' 플랫폼으로, Neo4j 온톨로지 관리, Airflow 데이터 파이프라인, OneDrive 문서 통합, Dash 웹 대시보드, RAG 시스템을 포함합니다.

코드 생성 시 다음 패턴 및 가이드라인을 따르세요:
- '원자-분자-유기체' 계층 구조
- PEP 8 스타일 가이드
- 명확한 문서화 및 타입 힌트
- 설정 주입 및 구성 외부화
- 적절한 오류 처리 및 로깅
- 단위 테스트 가능한 설계

모든 구성요소는 독립적으로 운영되면서도 통합될 수 있어야 합니다.
```

### Cursor AI 및 Codex CLI 시스템 프롬프트
```
당신은 소프트웨어 아키텍처 및 개발 전문가로서 팔란티어 파운드리 프로젝트를 지원합니다.
이 프로젝트는 데이터 온톨로지 관리, 문서 처리, 데이터 파이프라인, 대시보드 시각화 등의
기능을 제공하는 종합적인 플랫폼입니다.

당신의 역할은 다음과 같습니다:
1. 시스템 설계 및 아키텍처 제안
2. 코드 리뷰 및 개선 제안
3. 문서화 및 지식 관리 지원
4. 구현 전략 및 최적화 제안
5. 문제 해결 및 디버깅 지원

모든 설계와 코드는 다음 원칙을 따라야 합니다:
- '원자-분자-유기체' 패턴 기반 모듈화
- 확장성 및 유지보수성 최우선
- 철저한 오류 처리 및 로깅
- 명확한 문서화 및 설명
- 효율적인 리소스 사용

응답은 명확하고 구조화된 형식으로 제공하며, 필요에 따라 코드, 다이어그램, 예시를 포함합니다.
```

---

문서 버전: 2.1  
최종 업데이트: 2025-05-18

# 통합 확장판 Palantir‑Style Local Dev 환경 구축 & AI 자율 개선 시스템 (Windows + Cursor AI + Codex CLI)

> **한 눈에 보기**  | 목표 | 주요 도구 | AI 에이전트 | 주기 |
> ---------------- | ------------------------------------------- | ------------------------ | ------------------------------- | ----- |
> 환경 세팅 자동화 | **Conda/venv + Foundry‑Dev‑Tools** 설치·설정 (순수 Windows) | **Codex CLI** | "Bootstrap Agent" | 수동 1회 |
> 코드/데이터 개발 | Transforms 작성·테스트·온톨로지 시뮬 | **Cursor AI** (IDE) | "Dev Copilot" | 상시 대화 |
> 일일 지식 업데이트 | 외부 오픈소스·논문 검색 → 개선 커밋 | Cursor + Codex 합동 | **"Daily Update Agent"** | **매일 01:00 (Task Scheduler)** |
> 경영진 보고 생성 | 교육/기술 콘텐츠 요약·시각화 | Cursor AI | **"Summary Agent"** | 요구 또는 주간 |

---
## 1 프로젝트 비전
- **Foundry 클라우드 미사용** 환경에서 최대한 유사한 데이터 관리·변환 경험 제공.
- **AI 주도** 개발: IDE 코딩은 Cursor AI, 터미널/Git 자동화는 Codex CLI.
- **자율 개선 루프**: 로컬 코드를 분석→외부 최신 자료 검색→코드·문서 개선→PR/테스트.
- **비기술 의사결정권자**에게 핵심 가치 전달을 위한 요약/시각화 자동화.
- **WSL2 완전 배제**: 모든 툴은 Windows 네이티브(PowerShell, cmd)에서 실행.

---
## 2 시스템 아키텍처(상세)
```mermaid
graph LR
  subgraph Windows Local
    P[PowerShell / CMD] -->|pip/conda| B[foundry-dev-tools]
    B --> C[Transforms 로컬 실행]
    D[Cursor IDE] -->|AI 코딩| C
    E[Codex CLI] -->|shell/git| C
    E --> D
    C --> F[data/ CSV·Parquet]
  end

  subgraph Agents
    U[Daily Update Agent\n(Task Scheduler)] -->|웹 검색| WEB[(Open Web)]
    U -->|코드 개선| C
    S[Summary Agent] -->|콘텐츠 수집| EDU[(교육 자료)]
    S -->|Markdown/PDF 보고| REP[CEO Report]
  end
```

---
## 3 환경 구축 단계 (Windows PowerShell 기준)
### 3.1 부트스트랩 스크립트
```powershell
# 1. 필수 도구
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e   # 3.x 64‑bit
winget install --id EclipseAdoptium.Temurin.17.JDK -e

# 2. 가상환경 및 패키지
python -m venv C:\dev\foundry_env
C:\dev\foundry_env\Scripts\activate.ps1

pip install --upgrade pip
pip install "foundry-dev-tools[full]" pytest rich pyyaml chromadb duckduckgo-search

# 3. Codex CLI
pip install openai-codex-cli
setx OPENAI_API_KEY "sk-..."

# 4. Cursor 설치 (수동 .exe 다운로드 후 설치)
```
> **TIP**: Conda를 선호할 경우 `winget install Anaconda.Miniconda3` 후 `conda create -n foundry_env python=3.10` 로 대체.

### 3.2 JAVA_HOME 설정
```powershell
setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17\" /m
```

---
## 4 프로젝트 구조
```
foundry_local_project/
├─ transforms/
│  └─ flight_delays/flight_delay_summary.py
├─ data/flight_delays_raw.csv
├─ ontology/FlightDelay.yaml
├─ agents/
│  ├─ daily_update.py
│  └─ summary_agent.py
├─ tests/test_flight_delay_summary.py
├─ datasets.yaml
└─ README.md
```

---
## 5 AI 에이전트 설계
### 5.1 Codex CLI Bootstrap Agent
| 단계 | 명령 예시 | 자동 행동 |
|---|---|---|
| 1 | `codex "새 Python 프로젝트 생성"` | 폴더/README/Licence → `git init` |
| 2 | `codex "foundry-dev-tools 설치"` | pip 실행, req.txt 업데이트 |
| 3 | `codex "transform 스캐폴딩"` | transforms 구조 및 샘플 코드 작성 |
| 4 | `codex "pytest 실행"` | 실패 시 Cursor AI 패치 지시 |

### 5.2 Cursor IDE Dev Copilot
- `Ctrl+K`: 코드 생성·리팩터링
- `Ctrl+L`: 대화/Agent 모드 (YOLO Off)

### 5.3 Daily Update Agent (Windows Task Scheduler)
1. 코드베이스 분석 → 의존성/TODO 추출
2. `duckduckgo-search`·GitHub API로 최신 릴리스·논문·블로그 수집
3. `chromadb` 로컬 임베딩 DB 저장 → 의미 매칭
4. Cursor 프롬프트: "코드/문서 업데이트"
5. Codex CLI → 테스트 → 통과 시 `git commit -m "daily update"` & 푸시

`agents/daily_update.py` 샘플:
```python
from duckduckgo_search import ddg
from chromadb import Client
...
```

### 5.4 Summary Agent (교육/경영진 보고)
- 트리거: `python agents/summary_agent.py` 혹은 주간 스케줄
- 외부 교육 자료 → Cursor: `tl;dr for executives`
- Markdown↔PDF 변환 후 이메일 공유

---
## 6 테스트·CI 파이프라인
GitHub Actions 예시 동일 (Ubuntu runner에서 테스트) — 코드 생략.

---
## 7 안전 & ガバナンス
1. Cursor YOLO Off, Codex 위험 명령 승인 필요.
2. Daily Update Agent는 `update/*` 브랜치로 PR.
3. `.env` 비밀 관리 + API 키 마스킹.
4. 태그·reflog·venv 스냅샷으로 롤백.

---
## 8 확장 & 향후 로드맵
| 단계 | 내용 | 효과 |
|---|---|---|
| Docker‑Spark (Windows Docker Desktop) | 로컬 Spark 클러스터→대용량 PySpark 테스트 | 확장성 ↑ |
| MinIO 연동 | FDT S3 API→로컬 MinIO | 데이터 버전 관리 |
| LLM 교체 전략 | Claude → GPT‑4o, Gemini Pro 등 | 비용/품질 최적화 |

---
## 9 결론
**Cursor AI + Codex CLI**만으로 순수 Windows 환경에서 Palantir Foundry와 유사한 데이터 파이프라인 개발·운영·자율 개선 체계를 구축할 수 있습니다. WSL2 없이도 설치 복잡성을 줄이고, 매일 최신 지식을 코드에 반영하며, 경영진용 요약까지 자동 생성해 1인 개발자의 생산성을 극대화합니다.

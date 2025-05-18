"""
LLM 통합 기본 도구 모듈

Claude 3.7 Sonnet API와의 통신 및 LLM 모델을 활용한 코드 생성, 개선 등의 
기본 도구 함수들을 제공합니다.
"""

import json
import logging
import os
try:
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore
from typing import Any, Dict, List, Optional, Union

try:
    import anthropic
except Exception:  # pragma: no cover - optional
    anthropic = None  # type: ignore

# 로깅 설정
logger = logging.getLogger("llm_tools")

class ClaudeClient:
    """Claude API 클라이언트 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: LLM 구성 파일 경로
        """
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """LLM 구성 파일 로드
        
        Args:
            config_path: 구성 파일 경로
            
        Returns:
            구성 정보가 담긴 딕셔너리
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"구성 파일 로드 오류: {e}")
            raise
    
    def _initialize_client(self) -> Any:
        """Anthropic Claude 클라이언트 초기화
        
        Returns:
            Anthropic 클라이언트 인스턴스
        """
        if anthropic is None:
            raise ImportError("anthropic package is required for ClaudeClient")
        try:
            client = anthropic.Anthropic(api_key=self.config["claude"]["api_key"])
            logger.info("Claude API 클라이언트가 초기화되었습니다.")
            return client
        except Exception as e:
            logger.error(f"Claude API 클라이언트 초기화 오류: {e}")
            raise
    
    def get_model(self) -> str:
        """구성된 모델 이름 반환
        
        Returns:
            모델 이름
        """
        return self.config["claude"]["model"]

async def create_completion(client: ClaudeClient, prompt: str, max_tokens: int = None, 
                     temperature: float = 0.7, system_prompt: str = None) -> str:
    """Claude 모델을 사용하여 텍스트 생성
    
    Args:
        client: Claude 클라이언트 인스턴스
        prompt: 사용자 프롬프트
        max_tokens: 최대 토큰 수 (None이면 구성 파일의 값 사용)
        temperature: 온도 값 (0~1)
        system_prompt: 시스템 프롬프트
        
    Returns:
        생성된 텍스트
    """
    max_tokens = max_tokens or client.config["claude"]["max_tokens"]
    
    try:
        response = client.client.messages.create(
            model=client.get_model(),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        logger.info(f"Claude API 응답 받음: {len(response.content)} 문자")
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude API 호출 오류: {e}")
        raise

async def generate_code(client: ClaudeClient, prompt: str, language: str, 
                 context: str = None, max_tokens: int = None, 
                 temperature: float = 0.2) -> str:
    """코드 생성
    
    Args:
        client: Claude 클라이언트 인스턴스
        prompt: 코드 생성 지시사항
        language: 프로그래밍 언어
        context: 추가 컨텍스트
        max_tokens: 최대 토큰 수
        temperature: 온도 값 (0~1)
        
    Returns:
        생성된 코드
    """
    system_prompt = f"""당신은 경험이 풍부한 {language} 프로그래머로, 명확하고 효율적인 코드를 작성하는 전문가입니다.
요청에 따라 주어진 컨텍스트와 모범 사례를 고려하여 상세한 {language} 코드를 제공하세요.
코드는 모듈식, 재사용 가능하고, 신중한 에러 처리를 포함해야 합니다.
코드를 마크다운 포맷 안에 제공하세요 (예: ```{language} ... ```)."""

    context_text = f"\n\n컨텍스트:\n{context}" if context else ""
    prompt_text = f"{prompt}{context_text}\n\n{language} 코드로 작성해주세요."
    
    response = await create_completion(
        client=client,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt
    )
    
    return response

async def review_code(client: ClaudeClient, code: str, language: str, 
               max_tokens: int = None, temperature: float = 0.3) -> str:
    """코드 검토
    
    Args:
        client: Claude 클라이언트 인스턴스
        code: 검토할 코드
        language: 프로그래밍 언어
        max_tokens: 최대 토큰 수
        temperature: 온도 값 (0~1)
        
    Returns:
        코드 검토 결과
    """
    system_prompt = f"""당신은 숙련된 {language} 코드 리뷰어입니다. 
제시된 코드를 철저하게 분석하여 다음 항목에 대한 구체적인 피드백을 제공하세요:
1. 코드 품질 및 가독성
2. 잠재적인 버그 및 오류
3. 성능 및 효율성 문제
4. 보안 취약점
5. 모범 사례 준수
6. 특정 개선 제안

피드백은 명확하고 구체적이며 건설적이어야 합니다."""

    prompt_text = f"""다음 {language} 코드를 검토해주세요:

```{language}
{code}
```

코드의 장점과 개선이 필요한 부분에 대해 자세히 알려주세요."""
    
    response = await create_completion(
        client=client,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt
    )
    
    return response

async def refine_code(client: ClaudeClient, code: str, feedback: str, language: str,
               max_tokens: int = None, temperature: float = 0.3) -> str:
    """코드 개선
    
    Args:
        client: Claude 클라이언트 인스턴스
        code: 원본 코드
        feedback: 개선 피드백
        language: 프로그래밍 언어
        max_tokens: 최대 토큰 수
        temperature: 온도 값 (0~1)
        
    Returns:
        개선된 코드
    """
    system_prompt = f"""당신은 뛰어난 {language} 개발자로, 코드 품질과 모범 사례에 관한 전문가입니다.
주어진 코드와 피드백을 바탕으로 개선된 버전의 코드를 작성해야 합니다.
원본 코드와 피드백을 철저히 검토하고, 문제를 해결하면서 코드를 최적화하세요.
개선된 코드를 마크다운 포맷으로 제공하고, 변경 사항과 그 이유를 간략히 설명해주세요."""

    prompt_text = f"""다음은 원본 {language} 코드입니다:

```{language}
{code}
```

이에 대한 피드백은 다음과 같습니다:

{feedback}

이 피드백을 바탕으로 코드를 개선해주세요. 개선된 코드와 함께 주요 변경 사항을 설명해주세요."""
    
    response = await create_completion(
        client=client,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt
    )
    
    return response

async def explain_code(client: ClaudeClient, code: str, language: str,
                max_tokens: int = None, temperature: float = 0.5) -> str:
    """코드 설명
    
    Args:
        client: Claude 클라이언트 인스턴스
        code: 설명할 코드
        language: 프로그래밍 언어
        max_tokens: 최대 토큰 수
        temperature: 온도 값 (0~1)
        
    Returns:
        코드 설명
    """
    system_prompt = """당신은 프로그래밍 교육 전문가로, 복잡한 코드를 명확하고 이해하기 쉽게 설명하는 능력이 있습니다.
주어진 코드를 라인별로 분석하며, 코드가 어떻게 작동하는지 상세히 설명하세요.
기술적인 정확성을 유지하면서도 초보자가 이해할 수 있는 언어로 설명해야 합니다.
코드의 중요한 부분, 프로그래밍 패턴, 그리고 설계 결정에 대해 설명하세요."""

    prompt_text = f"""다음 {language} 코드를 상세히 설명해주세요:

```{language}
{code}
```

코드의 목적, 작동 방식, 그리고 주요 구성 요소에 대해 명확하게 설명해주세요."""
    
    response = await create_completion(
        client=client,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt
    )
    
    return response

async def save_generated_code(code: str, filename: str, directory: str) -> str:
    """생성된 코드를 파일로 저장
    
    Args:
        code: 저장할 코드
        filename: 파일 이름
        directory: 디렉토리 경로
        
    Returns:
        저장된 파일 경로
    """
    try:
        # 코드 텍스트 추출 (마크다운 코드 블록에서)
        code_text = code
        
        # 마크다운 코드 블록 제거
        code_start = code.find("```")
        if code_start != -1:
            code_end = code.find("```", code_start + 3)
            if code_end != -1:
                first_newline = code.find("\n", code_start)
                if first_newline != -1 and first_newline < code_end:
                    code_text = code[first_newline + 1:code_end].strip()
        
        # 디렉토리 생성
        os.makedirs(directory, exist_ok=True)
        
        # 파일 저장
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(code_text)
        
        logger.info(f"생성된 코드가 저장되었습니다: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"코드 저장 오류: {e}")
        raise

async def load_prompt_template(client: ClaudeClient, template_name: str) -> str:
    """프롬프트 템플릿 로드
    
    Args:
        client: Claude 클라이언트 인스턴스
        template_name: 템플릿 이름
        
    Returns:
        템플릿 내용
    """
    try:
        template_dir = client.config["prompts"]["template_directory"]
        template_path = os.path.join(template_dir, f"{template_name}.txt")
        
        with open(template_path, 'r', encoding='utf-8') as file:
            template = file.read()
        
        logger.info(f"프롬프트 템플릿을 로드했습니다: {template_name}")
        return template
    except Exception as e:
        logger.error(f"프롬프트 템플릿 로드 오류: {e}")
        raise

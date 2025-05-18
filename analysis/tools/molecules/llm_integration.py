"""
LLM 통합 및 자가 개선 시스템 모듈

Claude 3.7 Sonnet 기반 LLM 통합 및 자가 개선 시스템을 제공합니다. 이 시스템은 코드 생성 및 
최적화를 지원하며 RAG를 통한 컨텍스트 강화 기능을 제공합니다.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional
    SentenceTransformer = None  # type: ignore
try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional
    chromadb = None  # type: ignore

from analysis.tools.atoms.llm_tools import (
    ClaudeClient, create_completion, explain_code, generate_code, 
    load_prompt_template, refine_code, review_code, save_generated_code
)
from analysis.mcp_init import mcp
from common.helpers import load_config

# 로깅 설정
logger = logging.getLogger("llm_integration")

class ClaudeAIPair:
    """Claude AI 페어 프로그래밍 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: LLM 구성 파일 경로
        """
        self.client = ClaudeClient(config_path)
        self.config_path = config_path
        logger.info("Claude AI 페어 프로그래밍 시스템이 초기화되었습니다.")
    
    async def ask_for_code(self, prompt: str, context: str = None, 
                    language: str = "python") -> str:
        """코드 생성 요청
        
        Args:
            prompt: 코드 생성 지시사항
            context: 추가 컨텍스트
            language: 프로그래밍 언어
            
        Returns:
            생성된 코드
        """
        try:
            response = await generate_code(
                client=self.client,
                prompt=prompt,
                language=language,
                context=context
            )
            
            logger.info(f"{language} 코드 생성 완료: {len(response)} 문자")
            return response
        except Exception as e:
            logger.error(f"코드 생성 오류: {e}")
            raise
    
    async def review_generated_code(self, code: str, language: str = "python") -> str:
        """생성된 코드 검토
        
        Args:
            code: 검토할 코드
            language: 프로그래밍 언어
            
        Returns:
            코드 검토 결과
        """
        try:
            review = await review_code(
                client=self.client,
                code=code,
                language=language
            )
            
            logger.info(f"코드 검토 완료: {len(review)} 문자")
            return review
        except Exception as e:
            logger.error(f"코드 검토 오류: {e}")
            raise
    
    async def self_refine_code(self, code: str, iterations: int = 1, 
                       language: str = "python") -> Dict[str, Any]:
        """자가 개선 피드백 루프
        
        Args:
            code: 개선할 코드
            iterations: 반복 횟수
            language: 프로그래밍 언어
            
        Returns:
            개선 과정 정보
        """
        try:
            current_code = code
            refinement_history = []
            
            for i in range(iterations):
                logger.info(f"코드 자가 개선 반복 {i+1}/{iterations} 시작")
                
                # 코드 검토
                review = await self.review_generated_code(current_code, language)
                
                # 코드 개선
                improved_code = await refine_code(
                    client=self.client,
                    code=current_code,
                    feedback=review,
                    language=language
                )
                
                # 기록 저장
                refinement_history.append({
                    "iteration": i + 1,
                    "review": review,
                    "refined_code": improved_code
                })
                
                # 현재 코드 업데이트
                current_code = improved_code
                
                logger.info(f"코드 자가 개선 반복 {i+1}/{iterations} 완료")
            
            result = {
                "original_code": code,
                "refined_code": current_code,
                "refinement_history": refinement_history
            }
            
            logger.info(f"코드 자가 개선 완료: {iterations} 반복")
            return result
        except Exception as e:
            logger.error(f"코드 자가 개선 오류: {e}")
            raise
    
    async def save_code_generation(self, code: str, filename: str, 
                           is_improved: bool = False) -> str:
        """생성된 코드 저장
        
        Args:
            code: 저장할 코드
            filename: 파일 이름
            is_improved: 개선된 코드인지 여부
            
        Returns:
            저장된 파일 경로
        """
        try:
            directory = os.path.join(
                self.client.config["output"]["generated_code_dir"],
                "improved" if is_improved else "original"
            )
            
            file_path = await save_generated_code(
                code=code,
                filename=filename,
                directory=directory
            )
            
            logger.info(f"생성된 코드가 저장되었습니다: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"코드 저장 오류: {e}")
            raise
    
    async def load_prompt_from_template(self, template_name: str, 
                                 replacements: Dict[str, str] = None) -> str:
        """템플릿에서 프롬프트 로드
        
        Args:
            template_name: 템플릿 이름
            replacements: 대체할 변수 딕셔너리
            
        Returns:
            완성된 프롬프트
        """
        try:
            template = await load_prompt_template(
                client=self.client,
                template_name=template_name
            )
            
            if replacements:
                for key, value in replacements.items():
                    template = template.replace(f"{{{key}}}", value)
            
            logger.info(f"프롬프트 템플릿 '{template_name}'에서 프롬프트 로드 완료")
            return template
        except Exception as e:
            logger.error(f"프롬프트 로드 오류: {e}")
            raise
    
    async def explain_generated_code(self, code: str, language: str = "python") -> str:
        """생성된 코드 설명
        
        Args:
            code: 설명할 코드
            language: 프로그래밍 언어
            
        Returns:
            코드 설명
        """
        try:
            explanation = await explain_code(
                client=self.client,
                code=code,
                language=language
            )
            
            logger.info(f"코드 설명 완료: {len(explanation)} 문자")
            return explanation
        except Exception as e:
            logger.error(f"코드 설명 오류: {e}")
            raise

class LocalKnowledgeRAG:
    """로컬 지식 베이스 RAG 시스템 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: RAG 시스템 구성 파일 경로
        """
        self.config = load_config(config_path)
        self.embeddings_model = None
        self.vector_db = None
        self.collection = None
        
        self._initialize_vector_db()
        logger.info("로컬 지식 베이스 RAG 시스템이 초기화되었습니다.")
    
    
    def _initialize_vector_db(self) -> None:
        """벡터 데이터베이스 초기화"""
        try:
            # 임베딩 모델 로드
            self.embeddings_model = SentenceTransformer(self.config["embeddings"]["model"])
            
            # ChromaDB 클라이언트 초기화
            self.vector_db = chromadb.Client()
            
            # 컬렉션 생성 또는 조회
            self.collection = self.vector_db.get_or_create_collection(
                name=self.config["vector_db"]["collection_name"]
            )
            
            logger.info("벡터 데이터베이스가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"벡터 데이터베이스 초기화 오류: {e}")
            raise
    
    async def index_documents(self) -> int:
        """프로젝트 문서 인덱싱
        
        Returns:
            인덱싱된 문서 수
        """
        try:
            # 지식 베이스 디렉토리 및 파일 패턴
            kb_dir = self.config["knowledge_base"]["directory"]
            file_patterns = self.config["knowledge_base"]["file_patterns"]
            
            # 지식 베이스 디렉토리 생성 확인
            os.makedirs(kb_dir, exist_ok=True)
            
            # 파일 검색 및 처리
            documents = []
            ids = []
            texts = []
            metadatas = []
            
            doc_count = 0
            
            for root, _, files in os.walk(kb_dir):
                for file in files:
                    # 파일 패턴 확인
                    if not any(file.endswith(pattern.replace("*", "")) for pattern in file_patterns):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        # 파일 내용 읽기
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 텍스트 청크 분할 (예: 4000자 단위)
                        chunk_size = 4000
                        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                        
                        # 각 청크 처리
                        for i, chunk in enumerate(chunks):
                            doc_id = f"{os.path.relpath(file_path, kb_dir)}_{i}"
                            documents.append(chunk)
                            ids.append(doc_id)
                            # 임베딩 모델을 사용한 임베딩 생성은 벡터 DB 내부에서 처리됨
                            texts.append(chunk)
                            metadatas.append({
                                "file_path": file_path,
                                "chunk_index": i,
                                "file_type": os.path.splitext(file)[1],
                                "file_name": file
                            })
                            doc_count += 1
                    except Exception as e:
                        logger.error(f"파일 '{file_path}' 처리 오류: {e}")
            
            # 벡터 데이터베이스에 문서 추가
            if documents:
                # 임베딩 생성
                embeddings = [self.embeddings_model.encode(text).tolist() for text in texts]
                
                # 청크 단위로 추가 (최대 100개씩)
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch_end = min(i + batch_size, len(documents))
                    
                    self.collection.add(
                        ids=ids[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                        documents=documents[i:batch_end]
                    )
            
            logger.info(f"{doc_count}개 문서 청크가 인덱싱되었습니다.")
            return doc_count
        except Exception as e:
            logger.error(f"문서 인덱싱 오류: {e}")
            raise
    
    async def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """쿼리 실행
        
        Args:
            query_text: 검색 쿼리
            n_results: 반환할 결과 수
            
        Returns:
            유사 문서 목록
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings_model.encode(query_text).tolist()
            
            # 벡터 검색 실행
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 결과 포맷팅
            formatted_results = []
            
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "document": results["documents"][0][i],
                        "score": results.get("distances", [[0] * len(results["ids"][0])])[0][i]
                    })
            
            logger.info(f"쿼리 '{query_text}'에 대해 {len(formatted_results)}개 결과 반환")
            return formatted_results
        except Exception as e:
            logger.error(f"쿼리 실행 오류: {e}")
            raise
    
    async def enhance_prompt_with_context(self, query: str, claude_pair: ClaudeAIPair, 
                                   task_description: str, n_results: int = 3) -> str:
        """컨텍스트로 프롬프트 강화
        
        Args:
            query: 검색 쿼리
            claude_pair: Claude AI 페어 인스턴스
            task_description: 태스크 설명
            n_results: 사용할 결과 수
            
        Returns:
            강화된 응답
        """
        try:
            # 관련 컨텍스트 검색
            context_results = await self.query(query, n_results)
            
            # 컨텍스트 추출
            contexts = []
            for result in context_results:
                document = result["document"]
                metadata = result["metadata"]
                context_str = f"파일: {metadata['file_name']}\n내용:\n{document}\n"
                contexts.append(context_str)
            
            context_text = "\n---\n".join(contexts)
            
            # 프롬프트 구성
            prompt = f"""다음은 프로젝트 관련 컨텍스트입니다:

------- 컨텍스트 시작 -------
{context_text}
------- 컨텍스트 끝 -------

위 컨텍스트를 활용하여 다음 작업을 수행해주세요:

{task_description}

컨텍스트에서 발견한 관련 정보를 최대한 활용하세요. 컨텍스트에 없는 내용은 일반적인 모범 사례를 따르세요.
"""
            
            # LLM에 컨텍스트 강화 쿼리 전송
            response = await create_completion(
                client=claude_pair.client,
                prompt=prompt
            )
            
            logger.info(f"컨텍스트로 강화된 응답 생성 완료: {len(response)} 문자")
            return response
        except Exception as e:
            logger.error(f"컨텍스트 강화 오류: {e}")
            raise

@mcp.llm_agent(
    name="claude_code_generator",
    model="claude-3-7-sonnet-20250219",
    description="Claude 기반 코드 생성 에이전트",
    context_tags=["code", "python", "foundry"]
)
async def generate_code_with_claude(prompt: str, context: str = None, 
                             language: str = "python") -> str:
    """Claude를 사용한 코드 생성
    
    Args:
        prompt: 코드 생성 지시사항
        context: 추가 컨텍스트
        language: 프로그래밍 언어
        
    Returns:
        생성된 코드
    """
    client = ClaudeAIPair("C:\\Users\\packr\\OneDrive\\palantir\\config\\llm.yaml")
    generated_code = await client.ask_for_code(prompt, context, language)
    return generated_code

@mcp.rag_source(
    name="project_knowledge_base",
    source_type="file_system",
    description="프로젝트 문서 및 코드 소스",
    patterns=["*.md", "*.py", "*.yaml", "*.toml"]
)
async def query_project_knowledge(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """프로젝트 지식 베이스 쿼리
    
    Args:
        query: 검색 쿼리
        limit: 반환할 결과 수
        
    Returns:
        검색 결과 목록
    """
    rag_system = LocalKnowledgeRAG("C:\\Users\\packr\\OneDrive\\palantir\\config\\rag.yaml")
    results = await rag_system.query(query, limit)
    return results

@mcp.workflow(
    name="self_improving_code_generation",
    description="자가 개선 코드 생성 워크플로우"
)
async def generate_self_improving_code(prompt: str, filename: str,
                               language: str = "python",
                               iterations: int = 2) -> Dict[str, Any]:
    """자가 개선 코드 생성 워크플로우
    
    Args:
        prompt: 코드 생성 지시사항
        filename: 저장할 파일 이름
        language: 프로그래밍 언어
        iterations: 자가 개선 반복 횟수
        
    Returns:
        코드 생성 및 개선 결과
    """
    # Claude AI 페어 초기화
    claude_pair = ClaudeAIPair("C:\\Users\\packr\\OneDrive\\palantir\\config\\llm.yaml")
    
    # 초기 코드 생성
    initial_code = await claude_pair.ask_for_code(prompt, language=language)
    
    # 초기 코드 저장
    original_path = await claude_pair.save_code_generation(
        code=initial_code,
        filename=filename,
        is_improved=False
    )
    
    # 자가 개선 실행
    improved = await claude_pair.self_refine_code(
        code=initial_code,
        iterations=iterations,
        language=language
    )
    
    # 개선된 코드 저장
    improved_path = await claude_pair.save_code_generation(
        code=improved["refined_code"],
        filename=filename,
        is_improved=True
    )
    
    # 결과 반환
    result = {
        "original_code": initial_code,
        "original_path": original_path,
        "improved_code": improved["refined_code"],
        "improved_path": improved_path,
        "iterations": iterations,
        "language": language
    }
    
    return result

@mcp.workflow(
    name="context_enhanced_code_generation",
    description="컨텍스트 강화 코드 생성 워크플로우"
)
async def generate_context_enhanced_code(query: str, task: str, filename: str,
                                 language: str = "python") -> Dict[str, Any]:
    """컨텍스트 강화 코드 생성 워크플로우
    
    Args:
        query: 지식 베이스 검색 쿼리
        task: 코드 생성 태스크 설명
        filename: 저장할 파일 이름
        language: 프로그래밍 언어
        
    Returns:
        컨텍스트 강화 코드 생성 결과
    """
    # RAG 시스템 및 Claude AI 페어 초기화
    rag_system = LocalKnowledgeRAG("C:\\Users\\packr\\OneDrive\\palantir\\config\\rag.yaml")
    claude_pair = ClaudeAIPair("C:\\Users\\packr\\OneDrive\\palantir\\config\\llm.yaml")
    
    # 문서 인덱싱
    await rag_system.index_documents()
    
    # 컨텍스트 강화 응답 생성
    enhanced_response = await rag_system.enhance_prompt_with_context(
        query=query,
        claude_pair=claude_pair,
        task_description=task
    )
    
    # 응답에서 코드 추출
    # (코드는 마크다운 형식으로 반환되므로 ```로 된 코드 블록을 찾기)
    code_start = enhanced_response.find("```")
    generated_code = enhanced_response
    
    if code_start != -1:
        code_end = enhanced_response.find("```", code_start + 3)
        if code_end != -1:
            first_newline = enhanced_response.find("\n", code_start)
            if first_newline != -1 and first_newline < code_end:
                extracted_code = enhanced_response[first_newline + 1:code_end].strip()
                generated_code = f"```{language}\n{extracted_code}\n```"
    
    # 코드 저장
    file_path = await claude_pair.save_code_generation(
        code=generated_code,
        filename=filename,
        is_improved=False
    )
    
    # 결과 반환
    result = {
        "enhanced_response": enhanced_response,
        "generated_code": generated_code,
        "file_path": file_path,
        "language": language
    }
    
    return result

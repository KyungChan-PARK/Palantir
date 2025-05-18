"""
Model Context Protocol (MCP) 초기화 모듈

팔란티어 파운드리 시스템에서 사용되는 MCP(Model Context Protocol) 시스템을 
초기화하고 관리하는 모듈입니다.
"""

import asyncio
import functools
import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'mcp.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("mcp")

class MCP:
    """MCP(Model Context Protocol) 시스템 클래스"""
    
    def __init__(self):
        self.tools = {}
        self.workflows = {}
        self.systems = {}
        self.llm_agents = {}
        self.rag_sources = {}
        self.logger = logger
        
        self.logger.info("MCP 시스템 초기화 완료")
    
    def tool(self, name: str = None, description: str = None, tags: List[str] = None):
        """도구 수준 데코레이터
        
        Args:
            name: 도구 이름
            description: 도구 설명
            tags: 관련 태그 목록
        """
        def decorator(func):
            func_name = name or func.__name__
            func_tags = tags or []
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                self.logger.info(f"도구 실행: {func_name}")
                result = await func(*args, **kwargs)
                return result
            
            self.tools[func_name] = {
                "function": wrapper,
                "description": description or func.__doc__,
                "tags": func_tags,
                "signature": inspect.signature(func)
            }
            
            self.logger.info(f"도구 등록: {func_name}")
            return wrapper
        
        return decorator
    
    def workflow(self, name: str = None, description: str = None):
        """워크플로우 수준 데코레이터
        
        Args:
            name: 워크플로우 이름
            description: 워크플로우 설명
        """
        def decorator(func):
            func_name = name or func.__name__
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                self.logger.info(f"워크플로우 실행: {func_name}")
                result = await func(*args, **kwargs)
                return result
            
            self.workflows[func_name] = {
                "function": wrapper,
                "description": description or func.__doc__,
                "signature": inspect.signature(func)
            }
            
            self.logger.info(f"워크플로우 등록: {func_name}")
            return wrapper
        
        return decorator
    
    def system(self, name: str = None, description: str = None):
        """시스템 수준 데코레이터
        
        Args:
            name: 시스템 이름
            description: 시스템 설명
        """
        def decorator(func):
            func_name = name or func.__name__
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                self.logger.info(f"시스템 실행: {func_name}")
                result = await func(*args, **kwargs)
                return result
            
            self.systems[func_name] = {
                "function": wrapper,
                "description": description or func.__doc__,
                "signature": inspect.signature(func)
            }
            
            self.logger.info(f"시스템 등록: {func_name}")
            return wrapper
        
        return decorator
    
    def llm_agent(self, name: str = None, model: str = None, 
                 description: str = None, context_tags: List[str] = None):
        """LLM 에이전트 데코레이터
        
        Args:
            name: 에이전트 이름
            model: 사용할 LLM 모델
            description: 에이전트 설명
            context_tags: 관련 컨텍스트 태그 목록
        """
        def decorator(func):
            func_name = name or func.__name__
            func_context_tags = context_tags or []
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                self.logger.info(f"LLM 에이전트 실행: {func_name}")
                result = await func(*args, **kwargs)
                return result
            
            self.llm_agents[func_name] = {
                "function": wrapper,
                "description": description or func.__doc__,
                "model": model,
                "context_tags": func_context_tags,
                "signature": inspect.signature(func)
            }
            
            self.logger.info(f"LLM 에이전트 등록: {func_name}")
            return wrapper
        
        return decorator
    
    def rag_source(self, name: str = None, source_type: str = None, 
                  description: str = None, patterns: List[str] = None):
        """RAG 소스 데코레이터
        
        Args:
            name: 소스 이름
            source_type: 소스 유형 (예: file_system, database)
            description: 소스 설명
            patterns: 파일 패턴 목록
        """
        def decorator(func):
            func_name = name or func.__name__
            func_patterns = patterns or []
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                self.logger.info(f"RAG 소스 실행: {func_name}")
                result = await func(*args, **kwargs)
                return result
            
            self.rag_sources[func_name] = {
                "function": wrapper,
                "description": description or func.__doc__,
                "source_type": source_type,
                "patterns": func_patterns,
                "signature": inspect.signature(func)
            }
            
            self.logger.info(f"RAG 소스 등록: {func_name}")
            return wrapper
        
        return decorator
    
    def list_tools(self):
        """등록된 모든 도구 목록 반환"""
        return {name: {k: v for k, v in details.items() if k != "function"} 
                for name, details in self.tools.items()}
    
    def list_workflows(self):
        """등록된 모든 워크플로우 목록 반환"""
        return {name: {k: v for k, v in details.items() if k != "function"} 
                for name, details in self.workflows.items()}
    
    def list_systems(self):
        """등록된 모든 시스템 목록 반환"""
        return {name: {k: v for k, v in details.items() if k != "function"} 
                for name, details in self.systems.items()}
    
    def list_llm_agents(self):
        """등록된 모든 LLM 에이전트 목록 반환"""
        return {name: {k: v for k, v in details.items() if k != "function"} 
                for name, details in self.llm_agents.items()}
    
    def list_rag_sources(self):
        """등록된 모든 RAG 소스 목록 반환"""
        return {name: {k: v for k, v in details.items() if k != "function"} 
                for name, details in self.rag_sources.items()}

# MCP 인스턴스 생성
mcp = MCP()

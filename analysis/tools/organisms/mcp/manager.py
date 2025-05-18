"""
MCP(Model Context Protocol) 도구 관리자

모든 MCP 도구를 중앙에서 관리하고 구성하는 모듈입니다.
"""

import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# 프로젝트 루트 디렉토리 설정
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# MCP 모듈 임포트
from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("mcp.manager")

class MCPManager:
    """MCP 도구 관리자 클래스"""
    
    def __init__(self):
        """MCP 관리자 초기화"""
        self.logger = logger
        self.tools_config = self._load_tools_config()
        self.initialized_tools = set()
        
        self.logger.info("MCP 관리자 초기화 완료")
    
    def _load_tools_config(self) -> Dict[str, Any]:
        """MCP 도구 구성 파일 로드"""
        config_path = project_root / "config" / "ai_tools.yaml"
        
        if not config_path.exists():
            self.logger.warning(f"MCP 도구 구성 파일을 찾을 수 없음: {config_path}")
            return {"mcp": {"services": []}}
        
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            return config
        except Exception as e:
            self.logger.error(f"MCP 도구 구성 파일 로드 오류: {str(e)}")
            return {"mcp": {"services": []}}
    
    def available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 MCP 도구 목록 반환"""
        services = self.tools_config.get("mcp", {}).get("services", [])
        return services
    
    def is_tool_available(self, tool_name: str) -> bool:
        """특정 MCP 도구가 사용 가능한지 확인"""
        services = self.available_tools()
        return any(service["name"] == tool_name for service in services)
    
    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """특정 MCP 도구의 구성 정보 반환"""
        services = self.available_tools()
        for service in services:
            if service["name"] == tool_name:
                return service
        return None
    
    def initialize_tool(self, tool_name: str) -> bool:
        """특정 MCP 도구 초기화"""
        if tool_name in self.initialized_tools:
            self.logger.info(f"MCP 도구가 이미 초기화됨: {tool_name}")
            return True
        
        if not self.is_tool_available(tool_name):
            self.logger.error(f"MCP 도구를 찾을 수 없음: {tool_name}")
            return False
        
        try:
            # 도구 모듈 동적 임포트
            import_path = f"analysis.tools.organisms.mcp.{tool_name}"
            importlib.import_module(import_path)
            
            self.initialized_tools.add(tool_name)
            self.logger.info(f"MCP 도구 초기화 성공: {tool_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"MCP 도구 초기화 오류 ({tool_name}): {str(e)}")
            return False
    
    def initialize_all_tools(self) -> Dict[str, bool]:
        """모든 MCP 도구 초기화"""
        results = {}
        for service in self.available_tools():
            tool_name = service["name"]
            results[tool_name] = self.initialize_tool(tool_name)
        
        return results
    
    def list_initialized_tools(self) -> List[str]:
        """초기화된 MCP 도구 목록 반환"""
        return list(self.initialized_tools)
    
    def list_registered_methods(self, tool_name: Optional[str] = None) -> Dict[str, List[str]]:
        """등록된 MCP 메서드 목록 반환"""
        result = {}
        
        # MCP에 등록된 모든 도구 조회
        all_tools = mcp.list_tools()
        
        if tool_name:
            # 특정 도구의 메서드만 필터링
            prefix = f"{tool_name}_"
            filtered_tools = {name: details for name, details in all_tools.items() 
                             if name.startswith(prefix)}
            result[tool_name] = list(filtered_tools.keys())
        else:
            # 도구별로 그룹화
            for name in all_tools:
                if "_" in name:
                    t_name, method = name.split("_", 1)
                    if t_name not in result:
                        result[t_name] = []
                    result[t_name].append(name)
                else:
                    if "general" not in result:
                        result["general"] = []
                    result["general"].append(name)
        
        return result
    
    def cleanup_unused_tools(self) -> List[str]:
        """사용하지 않는 MCP 도구 정리"""
        # 구성 파일에 정의된 도구 이름 목록
        configured_tools = [service["name"] for service in self.available_tools()]
        
        # 등록된 메서드에서 도구 이름 추출
        registered_methods = self.list_registered_methods()
        registered_tools = list(registered_methods.keys())
        
        # 구성에 없지만 등록된 도구 식별 (general 카테고리 제외)
        unused_tools = [tool for tool in registered_tools 
                       if tool not in configured_tools and tool != "general"]
        
        if unused_tools:
            self.logger.info(f"사용하지 않는 MCP 도구 식별됨: {unused_tools}")
        
        return unused_tools
    
    def export_tools_status(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """MCP 도구 상태 내보내기"""
        status = {
            "available_tools": self.available_tools(),
            "initialized_tools": self.list_initialized_tools(),
            "registered_methods": self.list_registered_methods(),
            "unused_tools": self.cleanup_unused_tools()
        }
        
        if output_path:
            try:
                with open(output_path, "w") as f:
                    json.dump(status, f, indent=2)
                self.logger.info(f"MCP 도구 상태를 파일에 저장함: {output_path}")
            except Exception as e:
                self.logger.error(f"MCP 도구 상태 저장 오류: {str(e)}")
        
        return status

# MCP 관리자 인스턴스
mcp_manager = MCPManager()

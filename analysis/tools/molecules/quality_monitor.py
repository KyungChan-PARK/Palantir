"""
데이터 품질 모니터링 시스템 모듈

Great Expectations 기반 데이터 품질 모니터링 시스템을 제공합니다.
이 시스템은 데이터의 무결성을 검증하고 모니터링합니다.
"""

import json
import logging
import os
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("quality_monitor")

class QualityMonitoringSystem:
    """데이터 품질 모니터링 시스템 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 품질 모니터링 구성 파일 경로
        """
        self.config = self._load_config(config_path)
        self.expectations_dir = self.config["expectations_dir"]
        self.validation_dir = self.config["validation_dir"]
        
        # 필요한 디렉토리 생성
        os.makedirs(self.expectations_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        
        logger.info(f"데이터 품질 모니터링 시스템 초기화: config_path={config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """구성 파일 로드
        
        Args:
            config_path: 구성 파일 경로
            
        Returns:
            구성 정보 딕셔너리
        """
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"구성 파일 로드 오류: {e}")
            raise
    
    async def create_expectation_suite(self, suite_name: str, expectations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """기대치 스위트 생성
        
        Args:
            suite_name: 스위트 이름
            expectations: 기대치 목록
            
        Returns:
            생성된 기대치 스위트 정보
        """
        try:
            # 기대치 스위트 구성
            suite = {
                "expectation_suite_name": suite_name,
                "expectations": expectations,
                "meta": {
                    "created_at": datetime.now().isoformat(),
                    "created_by": "foundry"
                }
            }
            
            # 기대치 스위트 저장
            suite_path = os.path.join(self.expectations_dir, f"{suite_name}.json")
            with open(suite_path, "w", encoding="utf-8") as file:
                json.dump(suite, file, indent=2, ensure_ascii=False)
            
            logger.info(f"기대치 스위트 생성 완료: {suite_name}")
            
            return {
                "name": suite_name,
                "path": suite_path,
                "expectations_count": len(expectations),
                "created_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"기대치 스위트 생성 오류: {e}")
            raise
    
    async def get_expectation_suite(self, suite_name: str) -> Dict[str, Any]:
        """기대치 스위트 조회
        
        Args:
            suite_name: 스위트 이름
            
        Returns:
            기대치 스위트 정보
        """
        try:
            # 기대치 스위트 파일 경로
            suite_path = os.path.join(self.expectations_dir, f"{suite_name}.json")
            
            # 파일 존재 여부 확인
            if not os.path.exists(suite_path):
                raise FileNotFoundError(f"기대치 스위트를 찾을 수 없습니다: {suite_name}")
            
            # 기대치 스위트 읽기
            with open(suite_path, "r", encoding="utf-8") as file:
                suite = json.load(file)
            
            return suite
        except Exception as e:
            logger.error(f"기대치 스위트 조회 오류: {e}")
            raise
    
    async def list_expectation_suites(self) -> List[Dict[str, Any]]:
        """기대치 스위트 목록 조회
        
        Returns:
            기대치 스위트 목록
        """
        try:
            suites = []
            
            # 디렉토리 내 모든 JSON 파일 탐색
            for file_name in os.listdir(self.expectations_dir):
                if file_name.endswith(".json"):
                    file_path = os.path.join(self.expectations_dir, file_name)
                    
                    # 파일 생성 및 수정 시간
                    stat = os.stat(file_path)
                    created_time = datetime.fromtimestamp(stat.st_ctime)
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    # 기대치 스위트 정보 추출
                    with open(file_path, "r", encoding="utf-8") as file:
                        suite = json.load(file)
                    
                    suite_name = suite.get("expectation_suite_name", os.path.splitext(file_name)[0])
                    expectations_count = len(suite.get("expectations", []))
                    
                    suites.append({
                        "name": suite_name,
                        "file_name": file_name,
                        "file_path": file_path,
                        "expectations_count": expectations_count,
                        "created_at": created_time.isoformat(),
                        "modified_at": modified_time.isoformat()
                    })
            
            return suites
        except Exception as e:
            logger.error(f"기대치 스위트 목록 조회 오류: {e}")
            raise
    
    async def validate_data(self, data: Dict[str, Any], suite_name: str) -> Dict[str, Any]:
        """데이터 검증
        
        Args:
            data: 검증할 데이터
            suite_name: 기대치 스위트 이름
            
        Returns:
            검증 결과
        """
        try:
            # 데이터 변환
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            else:
                raise ValueError("지원되지 않는 데이터 형식입니다.")
            
            # 기대치 스위트 조회
            suite = await self.get_expectation_suite(suite_name)
            expectations = suite.get("expectations", [])
            
            # 검증 결과
            validation_results = []
            
            # 각 기대치 검증
            for expectation in expectations:
                expectation_type = expectation.get("expectation_type")
                kwargs = expectation.get("kwargs", {})
                
                result = await self._validate_expectation(df, expectation_type, kwargs)
                validation_results.append(result)
            
            # 전체 검증 결과
            success = all(result.get("success") for result in validation_results)
            
            # 검증 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(
                self.validation_dir, 
                f"{suite_name}_validation_{timestamp}.json"
            )
            
            validation_result = {
                "suite_name": suite_name,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "results": validation_results,
                "meta": {
                    "data_shape": df.shape,
                    "columns": df.columns.tolist()
                }
            }
            
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(validation_result, file, indent=2, ensure_ascii=False)
            
            logger.info(f"데이터 검증 완료: {suite_name}, 성공: {success}")
            
            return {
                "suite_name": suite_name,
                "success": success,
                "result_path": result_path,
                "results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"데이터 검증 오류: {e}")
            raise
    
    async def validate_dataframe(self, df: pd.DataFrame, suite_name: str) -> Dict[str, Any]:
        """데이터프레임 검증
        
        Args:
            df: 검증할 데이터프레임
            suite_name: 기대치 스위트 이름
            
        Returns:
            검증 결과
        """
        try:
            # 기대치 스위트 조회
            suite = await self.get_expectation_suite(suite_name)
            expectations = suite.get("expectations", [])
            
            # 검증 결과
            validation_results = []
            
            # 각 기대치 검증
            for expectation in expectations:
                expectation_type = expectation.get("expectation_type")
                kwargs = expectation.get("kwargs", {})
                
                result = await self._validate_expectation(df, expectation_type, kwargs)
                validation_results.append(result)
            
            # 전체 검증 결과
            success = all(result.get("success") for result in validation_results)
            
            # 검증 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(
                self.validation_dir, 
                f"{suite_name}_validation_{timestamp}.json"
            )
            
            validation_result = {
                "suite_name": suite_name,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "results": validation_results,
                "meta": {
                    "data_shape": df.shape,
                    "columns": df.columns.tolist()
                }
            }
            
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(validation_result, file, indent=2, ensure_ascii=False)
            
            logger.info(f"데이터프레임 검증 완료: {suite_name}, 성공: {success}")
            
            return {
                "suite_name": suite_name,
                "success": success,
                "result_path": result_path,
                "results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"데이터프레임 검증 오류: {e}")
            raise
    
    async def validate_csv_file(self, file_path: str, suite_name: str) -> Dict[str, Any]:
        """CSV 파일 검증
        
        Args:
            file_path: 검증할 CSV 파일 경로
            suite_name: 기대치 스위트 이름
            
        Returns:
            검증 결과
        """
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            
            # 데이터프레임 검증
            result = await self.validate_dataframe(df, suite_name)
            
            # 파일 정보 추가
            result["file_path"] = file_path
            result["file_name"] = os.path.basename(file_path)
            
            logger.info(f"CSV 파일 검증 완료: {file_path}, 성공: {result['success']}")
            
            return result
        except Exception as e:
            logger.error(f"CSV 파일 검증 오류: {e}")
            raise
    
    async def create_data_quality_report(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """데이터 품질 보고서 생성
        
        Args:
            validation_results: 검증 결과 목록
            
        Returns:
            데이터 품질 보고서
        """
        try:
            # 보고서 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.config["reports_dir"], 
                f"quality_report_{timestamp}.json"
            )
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # 보고서 내용
            report = {
                "timestamp": datetime.now().isoformat(),
                "validation_count": len(validation_results),
                "success_count": sum(1 for result in validation_results if result.get("success")),
                "failure_count": sum(1 for result in validation_results if not result.get("success")),
                "validation_results": validation_results
            }
            
            # 보고서 저장
            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2, ensure_ascii=False)
            
            logger.info(f"데이터 품질 보고서 생성 완료: {report_path}")
            
            return {
                "report_path": report_path,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "validation_count": len(validation_results),
                    "success_count": report["success_count"],
                    "failure_count": report["failure_count"],
                    "success_rate": report["success_count"] / len(validation_results) if validation_results else 0
                }
            }
        except Exception as e:
            logger.error(f"데이터 품질 보고서 생성 오류: {e}")
            raise
    
    async def create_document_expectation_suite(self) -> Dict[str, Any]:
        """문서 기대치 스위트 생성
        
        Returns:
            생성된 기대치 스위트 정보
        """
        try:
            suite_name = "document_expectations"
            
            # 기대치 목록
            expectations = [
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {
                        "column": "title"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": "title"
                    }
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {
                        "column": "content"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": "content"
                    }
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {
                        "column": "doc_type"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "doc_type",
                        "value_set": ["report", "analysis", "memo"]
                    }
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {
                        "column": "status"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": "status",
                        "value_set": ["draft", "review", "approved", "published", "archived"]
                    }
                }
            ]
            
            # 기대치 스위트 생성
            result = await self.create_expectation_suite(suite_name, expectations)
            
            logger.info(f"문서 기대치 스위트 생성 완료: {suite_name}")
            
            return result
        except Exception as e:
            logger.error(f"문서 기대치 스위트 생성 오류: {e}")
            raise
    
    async def create_ontology_expectation_suite(self) -> Dict[str, Any]:
        """온톨로지 기대치 스위트 생성
        
        Returns:
            생성된 기대치 스위트 정보
        """
        try:
            suite_name = "ontology_expectations"
            
            # 기대치 목록
            expectations = [
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {
                        "column": "nodes"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": "nodes"
                    }
                },
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {
                        "column": "relationships"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": "relationships"
                    }
                },
                {
                    "expectation_type": "expect_table_row_count_to_be_greater_than",
                    "kwargs": {
                        "min_value": 1
                    }
                }
            ]
            
            # 기대치 스위트 생성
            result = await self.create_expectation_suite(suite_name, expectations)
            
            logger.info(f"온톨로지 기대치 스위트 생성 완료: {suite_name}")
            
            return result
        except Exception as e:
            logger.error(f"온톨로지 기대치 스위트 생성 오류: {e}")
            raise
    
    async def _validate_expectation(self, df: pd.DataFrame, expectation_type: str, 
                           kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """기대치 검증
        
        Args:
            df: 검증할 데이터프레임
            expectation_type: 기대치 유형
            kwargs: 기대치 매개변수
            
        Returns:
            검증 결과
        """
        try:
            result = {
                "expectation_type": expectation_type,
                "kwargs": kwargs,
                "success": False,
                "exception_info": None
            }
            
            # 기대치 유형별 검증
            if expectation_type == "expect_column_to_exist":
                column = kwargs.get("column")
                result["success"] = column in df.columns
            elif expectation_type == "expect_column_values_to_not_be_null":
                column = kwargs.get("column")
                if column in df.columns:
                    result["success"] = not df[column].isnull().any()
                    result["result"] = {
                        "null_count": df[column].isnull().sum(),
                        "total_count": len(df)
                    }
            elif expectation_type == "expect_column_values_to_be_in_set":
                column = kwargs.get("column")
                value_set = kwargs.get("value_set", [])
                if column in df.columns:
                    result["success"] = df[column].isin(value_set).all()
                    result["result"] = {
                        "unexpected_values": df[~df[column].isin(value_set)][column].unique().tolist(),
                        "unexpected_count": (~df[column].isin(value_set)).sum(),
                        "total_count": len(df)
                    }
            elif expectation_type == "expect_table_row_count_to_be_greater_than":
                min_value = kwargs.get("min_value", 0)
                result["success"] = len(df) > min_value
                result["result"] = {
                    "row_count": len(df),
                    "min_value": min_value
                }
            elif expectation_type == "expect_column_values_to_match_regex":
                column = kwargs.get("column")
                regex = kwargs.get("regex")
                if column in df.columns:
                    result["success"] = df[column].str.match(regex).all()
                    result["result"] = {
                        "unexpected_count": (~df[column].str.match(regex)).sum(),
                        "total_count": len(df)
                    }
            elif expectation_type == "expect_column_values_to_be_unique":
                column = kwargs.get("column")
                if column in df.columns:
                    result["success"] = df[column].is_unique
                    result["result"] = {
                        "duplicate_count": len(df) - df[column].nunique(),
                        "total_count": len(df)
                    }
            else:
                result["exception_info"] = f"지원되지 않는 기대치 유형: {expectation_type}"
            
            return result
        except Exception as e:
            logger.error(f"기대치 검증 오류: {e}")
            return {
                "expectation_type": expectation_type,
                "kwargs": kwargs,
                "success": False,
                "exception_info": str(e)
            }

@mcp.workflow(
    name="create_document_expectations",
    description="문서 기대치 스위트 생성 워크플로우"
)
async def create_document_expectations(config_path: str = "C:\\Users\\packr\\OneDrive\\palantir\\config\\quality.yaml") -> Dict[str, Any]:
    """문서 기대치 스위트 생성 워크플로우
    
    Args:
        config_path: 품질 모니터링 구성 파일 경로
        
    Returns:
        생성된 기대치 스위트 정보
    """
    system = QualityMonitoringSystem(config_path)
    return await system.create_document_expectation_suite()

@mcp.workflow(
    name="create_ontology_expectations",
    description="온톨로지 기대치 스위트 생성 워크플로우"
)
async def create_ontology_expectations(config_path: str = "C:\\Users\\packr\\OneDrive\\palantir\\config\\quality.yaml") -> Dict[str, Any]:
    """온톨로지 기대치 스위트 생성 워크플로우
    
    Args:
        config_path: 품질 모니터링 구성 파일 경로
        
    Returns:
        생성된 기대치 스위트 정보
    """
    system = QualityMonitoringSystem(config_path)
    return await system.create_ontology_expectation_suite()

@mcp.workflow(
    name="validate_document_data",
    description="문서 데이터 검증 워크플로우"
)
async def validate_document_data(data: Dict[str, Any], 
                        config_path: str = "C:\\Users\\packr\\OneDrive\\palantir\\config\\quality.yaml") -> Dict[str, Any]:
    """문서 데이터 검증 워크플로우
    
    Args:
        data: 검증할 문서 데이터
        config_path: 품질 모니터링 구성 파일 경로
        
    Returns:
        검증 결과
    """
    system = QualityMonitoringSystem(config_path)
    return await system.validate_data(data, "document_expectations")

@mcp.workflow(
    name="validate_ontology_data",
    description="온톨로지 데이터 검증 워크플로우"
)
async def validate_ontology_data(data: Dict[str, Any], 
                        config_path: str = "C:\\Users\\packr\\OneDrive\\palantir\\config\\quality.yaml") -> Dict[str, Any]:
    """온톨로지 데이터 검증 워크플로우
    
    Args:
        data: 검증할 온톨로지 데이터
        config_path: 품질 모니터링 구성 파일 경로
        
    Returns:
        검증 결과
    """
    system = QualityMonitoringSystem(config_path)
    return await system.validate_data(data, "ontology_expectations")

# 직접 실행 시
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    import asyncio
    
    async def main():
        # 구성 파일 경로
        config_path = "C:\\Users\\packr\\OneDrive\\palantir\\config\\quality.yaml"
        
        # 품질 모니터링 시스템 초기화
        system = QualityMonitoringSystem(config_path)
        
        # 기대치 스위트 생성
        doc_suite = await system.create_document_expectation_suite()
        print(f"문서 기대치 스위트 생성 결과: {doc_suite}")
        
        onto_suite = await system.create_ontology_expectation_suite()
        print(f"온톨로지 기대치 스위트 생성 결과: {onto_suite}")
        
        # 예시 문서 데이터 검증
        doc_data = {
            "title": "테스트 문서",
            "content": "테스트 내용입니다.",
            "doc_type": "report",
            "status": "draft",
            "created_at": datetime.now().isoformat()
        }
        
        doc_result = await system.validate_data(doc_data, "document_expectations")
        print(f"문서 데이터 검증 결과: {doc_result}")
    
    asyncio.run(main())

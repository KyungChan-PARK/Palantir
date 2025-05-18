"""
문서 성능 테스트 도구

팔란티어 파운드리 시스템의 문서 처리 성능을 테스트합니다.
"""

import json
import logging
import os
import time
from datetime import datetime
from common.path_utils import get_palantir_root
from typing import Dict, List, Optional, Union
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("test_document_performance")

class DocumentPerformanceTest:
    """문서 성능 테스트 클래스"""
    
    def __init__(self, test_dir: str, results_dir: str):
        """
        Args:
            test_dir: 테스트 문서 디렉토리
            results_dir: 결과 저장 디렉토리
        """
        self.test_dir = test_dir
        self.results_dir = results_dir
        
        # 결과 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"문서 성능 테스트 초기화: test_dir={test_dir}, results_dir={results_dir}")

    async def run_test(self, doc_dir: str, test_name: str) -> Dict[str, Union[float, int, str]]:
        """단일 테스트 실행
        
        Args:
            doc_dir: 테스트할 문서 디렉토리
            test_name: 테스트 이름
            
        Returns:
            테스트 결과
        """
        try:
            logger.info(f"테스트 실행: {test_name}, 문서 디렉토리: {doc_dir}")
            
            # 문서 수 확인
            doc_files = glob.glob(os.path.join(doc_dir, "*.txt"))
            doc_count = len(doc_files)
            
            if doc_count == 0:
                logger.warning(f"문서가 없습니다: {doc_dir}")
                return {
                    "test_name": test_name,
                    "doc_count": 0,
                    "doc_dir": doc_dir,
                    "status": "error",
                    "error": "문서가 없습니다."
                }
            
            # 성능 측정 시작
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # 문서 처리 (예시: 모든 문서 읽기 및 간단한 분석)
            total_chars = 0
            total_words = 0
            total_lines = 0
            doc_size_distribution = []
            doc_types = {"report": 0, "analysis": 0, "memo": 0, "unknown": 0}
            
            for doc_file in doc_files:
                # 문서 유형 확인
                file_name = os.path.basename(doc_file)
                doc_type = "unknown"
                for known_type in ["report", "analysis", "memo"]:
                    if file_name.startswith(known_type):
                        doc_type = known_type
                        break
                
                doc_types[doc_type] += 1
                
                # 문서 내용 읽기
                with open(doc_file, "r", encoding="utf-8") as file:
                    content = file.read()
                    
                    # 문서 통계
                    chars = len(content)
                    words = len(content.split())
                    lines = content.count("\n") + 1
                    
                    total_chars += chars
                    total_words += words
                    total_lines += lines
                    
                    doc_size_distribution.append(chars)
            
            # 성능 측정 종료
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # 결과 계산
            processing_time = (end_time - start_time) * 1000  # 밀리초
            memory_usage = end_memory - start_memory  # 메가바이트
            
            # 결과 저장
            result = {
                "test_name": test_name,
                "timestamp": datetime.now().isoformat(),
                "doc_count": doc_count,
                "doc_dir": doc_dir,
                "processing_time_ms": processing_time,
                "memory_usage_mb": memory_usage,
                "total_chars": total_chars,
                "total_words": total_words,
                "total_lines": total_lines,
                "avg_chars_per_doc": total_chars / doc_count,
                "avg_words_per_doc": total_words / doc_count,
                "avg_lines_per_doc": total_lines / doc_count,
                "doc_types": doc_types,
                "doc_size_distribution": {
                    "min": min(doc_size_distribution),
                    "max": max(doc_size_distribution),
                    "mean": sum(doc_size_distribution) / len(doc_size_distribution),
                    "median": sorted(doc_size_distribution)[len(doc_size_distribution) // 2],
                    "std": self._std(doc_size_distribution)
                },
                "status": "success"
            }
            
            # 결과 파일 저장
            result_file = os.path.join(self.results_dir, f"{test_name}_result.json")
            with open(result_file, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2, ensure_ascii=False)
            
            logger.info(f"테스트 완료: {test_name}, 처리 시간: {processing_time:.2f}ms, "
                      f"메모리 사용: {memory_usage:.2f}MB")
            
            return result
        except Exception as e:
            logger.error(f"테스트 실행 오류: {e}")
            return {
                "test_name": test_name,
                "doc_dir": doc_dir,
                "status": "error",
                "error": str(e)
            }
    
    async def run_all_tests(self, document_counts: List[int] = None) -> List[Dict[str, Union[float, int, str]]]:
        """여러 문서 수에 대한 테스트 실행
        
        Args:
            document_counts: 테스트할 문서 수 목록 (None이면 기본값 [10, 50, 100] 사용)
            
        Returns:
            테스트 결과 목록
        """
        if document_counts is None:
            document_counts = [10, 50, 100]
        
        logger.info(f"모든 테스트 실행: 문서 수 = {document_counts}")
        
        results = []
        
        for count in document_counts:
            doc_dir = os.path.join(self.test_dir, f"docs_{count}")
            
            # 해당 문서 수에 대한 디렉토리가 없으면 건너뜀
            if not os.path.exists(doc_dir):
                logger.warning(f"디렉토리가 없습니다: {doc_dir}")
                continue
            
            # 테스트 실행
            test_name = f"performance_test_{count}_docs"
            result = await self.run_test(doc_dir, test_name)
            results.append(result)
        
        # 종합 결과 저장
        summary_file = os.path.join(self.results_dir, "performance_summary.json")
        with open(summary_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        
        # 성능 그래프 생성
        await self.generate_performance_graphs(results)
        
        logger.info(f"모든 테스트 완료: {len(results)}개 테스트 실행")
        
        return results
    
    async def generate_performance_graphs(self, results: List[Dict[str, Union[float, int, str]]]) -> str:
        """성능 테스트 결과를 그래프로 시각화
        
        Args:
            results: 테스트 결과 목록
            
        Returns:
            그래프 파일 저장 경로
        """
        try:
            # 성공한 결과만 필터링
            success_results = [r for r in results if r["status"] == "success"]
            
            if not success_results:
                logger.warning("성공한 테스트 결과가 없어 그래프를 생성할 수 없습니다.")
                return ""
            
            # 데이터 프레임 생성
            df = pd.DataFrame([
                {
                    "문서 수": r["doc_count"],
                    "처리 시간(ms)": r["processing_time_ms"],
                    "메모리 사용(MB)": r["memory_usage_mb"],
                    "평균 문서 크기(char)": r["avg_chars_per_doc"]
                }
                for r in success_results
            ])
            
            # 정렬
            df = df.sort_values("문서 수")
            
            # 그래프 생성
            plt.figure(figsize=(15, 10))
            
            # 처리 시간 그래프
            plt.subplot(2, 2, 1)
            plt.plot(df["문서 수"], df["처리 시간(ms)"], marker="o", linewidth=2, markersize=8)
            plt.title("문서 수에 따른 처리 시간")
            plt.xlabel("문서 수")
            plt.ylabel("처리 시간 (ms)")
            plt.grid(True)
            
            # 메모리 사용 그래프
            plt.subplot(2, 2, 2)
            plt.plot(df["문서 수"], df["메모리 사용(MB)"], marker="o", linewidth=2, markersize=8, color="green")
            plt.title("문서 수에 따른 메모리 사용량")
            plt.xlabel("문서 수")
            plt.ylabel("메모리 사용 (MB)")
            plt.grid(True)
            
            # 문서 당 처리 시간 그래프
            plt.subplot(2, 2, 3)
            df["문서 당 처리 시간(ms)"] = df["처리 시간(ms)"] / df["문서 수"]
            plt.plot(df["문서 수"], df["문서 당 처리 시간(ms)"], marker="o", linewidth=2, markersize=8, color="red")
            plt.title("문서 수에 따른 문서 당 처리 시간")
            plt.xlabel("문서 수")
            plt.ylabel("문서 당 처리 시간 (ms)")
            plt.grid(True)
            
            # 문서 당 메모리 사용 그래프
            plt.subplot(2, 2, 4)
            df["문서 당 메모리 사용(MB)"] = df["메모리 사용(MB)"] / df["문서 수"]
            plt.plot(df["문서 수"], df["문서 당 메모리 사용(MB)"], marker="o", linewidth=2, markersize=8, color="purple")
            plt.title("문서 수에 따른 문서 당 메모리 사용량")
            plt.xlabel("문서 수")
            plt.ylabel("문서 당 메모리 사용 (MB)")
            plt.grid(True)
            
            plt.tight_layout()
            
            # 그래프 저장
            graph_file = os.path.join(self.results_dir, "performance_graphs.png")
            plt.savefig(graph_file)
            plt.close()
            
            logger.info(f"성능 그래프 생성 완료: {graph_file}")
            
            # 결과 표 저장
            table_file = os.path.join(self.results_dir, "performance_table.csv")
            df.to_csv(table_file, index=False)
            
            return graph_file
        except Exception as e:
            logger.error(f"그래프 생성 오류: {e}")
            return ""
    
    def _get_memory_usage(self) -> float:
        """현재 프로세스의 메모리 사용량 반환 (MB)"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # 바이트를 MB로 변환
        except ImportError:
            logger.warning("psutil 패키지가 설치되지 않아 메모리 사용량을 정확히 측정할 수 없습니다.")
            return 0.0
    
    def _std(self, data: List[float]) -> float:
        """표준 편차 계산"""
        if not data:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance ** 0.5

@mcp.tool(
    name="test_document_performance",
    description="문서 처리 성능 테스트 도구",
    tags=["test", "performance", "document"]
)
async def test_document_performance(test_dir: str, results_dir: str,
                             document_counts: List[int] = None) -> Dict[str, Union[List, str]]:
    """문서 처리 성능 테스트
    
    Args:
        test_dir: 테스트 문서 디렉토리
        results_dir: 결과 저장 디렉토리
        document_counts: 테스트할 문서 수 목록 (None이면 기본값 [10, 50, 100] 사용)
        
    Returns:
        테스트 결과
    """
    try:
        # 테스트 실행
        tester = DocumentPerformanceTest(test_dir, results_dir)
        results = await tester.run_all_tests(document_counts)
        
        return {
            "status": "success",
            "results": results,
            "summary_file": os.path.join(results_dir, "performance_summary.json"),
            "graph_file": os.path.join(results_dir, "performance_graphs.png")
        }
    except Exception as e:
        logger.error(f"성능 테스트 오류: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

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
        result = await test_document_performance(
            test_dir=os.path.join(get_palantir_root().as_posix(), "temp", "test_documents"),
            results_dir=os.path.join(get_palantir_root().as_posix(), "output", "reports", "performance"),
            document_counts=[5]  # 테스트 목적으로 적은 수만 사용
        )
        print(f"결과: {result}")
    
    asyncio.run(main())

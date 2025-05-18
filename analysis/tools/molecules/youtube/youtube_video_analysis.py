"""
YouTube 동영상 분석 워크플로우 모듈
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from analysis.mcp_init import mcp
# config 패키지에서 직접 임포트
from config.youtube_config import TEMP_DOWNLOAD_DIR, ANALYSIS_OUTPUT_DIR

# 원자 도구 가져오기
from analysis.tools.atoms.youtube.youtube_data_api import get_video_metadata, search_youtube_videos
from analysis.tools.atoms.youtube.youtube_downloader import download_youtube_video, get_video_download_info
from analysis.tools.atoms.youtube.video_intelligence_api import upload_video_to_gcs, analyze_video_content, get_video_highlights

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube_video_analysis")

@mcp.workflow(
    name="analyze_youtube_video",
    description="YouTube 동영상을 다운로드하고 Video Intelligence API로 분석하는 전체 워크플로우"
)
async def analyze_youtube_video(
    video_id: str,
    analysis_features: Optional[List[str]] = None,
    download_resolution: str = "720p",
    extract_highlights: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    YouTube 동영상을 다운로드하고 Video Intelligence API로 분석하는 전체 워크플로우
    
    Parameters:
    -----------
    video_id : str
        분석할 YouTube 비디오 ID
    analysis_features : List[str], optional
        분석할 기능 목록 (기본값: ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "OBJECT_TRACKING"])
    download_resolution : str, optional
        다운로드할 해상도 (기본값: "720p")
    extract_highlights : bool, optional
        하이라이트 추출 여부 (기본값: True)
    output_dir : str, optional
        결과를 저장할 디렉토리 (기본값: ANALYSIS_OUTPUT_DIR)
        
    Returns:
    --------
    Dict[str, Any]
        분석 결과를 포함하는 딕셔너리
    """
    start_time = time.time()
    
    # 기본값 설정
    if analysis_features is None:
        analysis_features = ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "OBJECT_TRACKING"]
    
    if output_dir is None:
        output_dir = ANALYSIS_OUTPUT_DIR
    
    # 저장 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프 생성 (중복 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. 비디오 메타데이터 가져오기
        logger.info(f"비디오 '{video_id}' 메타데이터 가져오기")
        metadata_result = await get_video_metadata(video_id)
        
        if not metadata_result.get("success", False):
            return {
                "success": False,
                "error": f"비디오 메타데이터 가져오기 실패: {metadata_result.get('error')}"
            }
        
        metadata = metadata_result
        video_title = metadata.get("title", "Unknown Title").replace('/', '_').replace('\\', '_')
        
        # 2. 비디오 다운로드
        logger.info(f"비디오 '{video_id}' 다운로드 시작")
        download_result = await download_youtube_video(
            video_id=video_id,
            output_dir=TEMP_DOWNLOAD_DIR,
            resolution=download_resolution
        )
        
        if not download_result.get("success", False):
            return {
                "success": False,
                "error": f"비디오 다운로드 실패: {download_result.get('error')}"
            }
        
        video_file_path = download_result.get("file_path")
        
        # 3. 비디오를 GCS에 업로드
        logger.info(f"비디오를 GCS에 업로드")
        blob_name = f"youtube_videos/{video_id}_{timestamp}.mp4"
        upload_result = await upload_video_to_gcs(
            file_path=video_file_path,
            destination_blob_name=blob_name
        )
        
        if not upload_result.get("success", False):
            return {
                "success": False,
                "error": f"GCS 업로드 실패: {upload_result.get('error')}"
            }
        
        gcs_uri = upload_result.get("gcs_uri")
        
        # 4. Video Intelligence API를 사용하여 비디오 분석
        logger.info(f"Video Intelligence API로 비디오 분석 시작")
        
        # 분석 결과 저장 경로
        analysis_output_file = os.path.join(output_dir, f"analysis_{video_id}_{timestamp}.json")
        
        analysis_result = await analyze_video_content(
            gcs_uri=gcs_uri,
            features=analysis_features,
            output_file=analysis_output_file
        )
        
        if not analysis_result.get("success", False):
            return {
                "success": False,
                "error": f"비디오 분석 실패: {analysis_result.get('error')}"
            }
        
        # 5. 하이라이트 추출 (선택 사항)
        highlight_result = None
        
        if extract_highlights:
            logger.info("비디오 하이라이트 추출")
            highlight_result = await get_video_highlights(analysis_result)
            
            # 하이라이트 결과 저장
            if highlight_result.get("success", False):
                highlight_output_file = os.path.join(output_dir, f"highlights_{video_id}_{timestamp}.json")
                
                with open(highlight_output_file, 'w', encoding='utf-8') as f:
                    json.dump(highlight_result, f, ensure_ascii=False, indent=2)
                
                highlight_result["output_file"] = highlight_output_file
        
        # 6. 종합 보고서 생성
        report = {
            "video_id": video_id,
            "video_title": video_title,
            "timestamp": timestamp,
            "metadata": metadata,
            "analysis_features": analysis_features,
            "analysis_summary": analysis_result.get("summary"),
            "highlights": highlight_result.get("highlights") if highlight_result else None,
            "files": {
                "video": video_file_path,
                "gcs_uri": gcs_uri,
                "analysis": analysis_output_file,
                "highlights": highlight_result.get("output_file") if highlight_result else None
            }
        }
        
        # 보고서 저장
        report_file = os.path.join(output_dir, f"report_{video_id}_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"비디오 '{video_id}' 분석 완료 (실행 시간: {execution_time:.2f}초)")
        
        return {
            "success": True,
            "video_id": video_id,
            "video_title": video_title,
            "timestamp": timestamp,
            "execution_time": execution_time,
            "report_file": report_file,
            "metadata": metadata,
            "analysis_summary": analysis_result.get("summary"),
            "highlights": highlight_result.get("highlights") if highlight_result else None,
            "files": {
                "video": video_file_path,
                "gcs_uri": gcs_uri,
                "analysis": analysis_output_file,
                "highlights": highlight_result.get("output_file") if highlight_result else None
            }
        }
    
    except Exception as e:
        import traceback
        error_message = f"분석 워크플로우 중 오류 발생: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "video_id": video_id,
            "error": error_message,
            "traceback": traceback.format_exc()
        }

@mcp.workflow(
    name="batch_analyze_youtube_videos",
    description="여러 YouTube 동영상을 일괄 분석하는 워크플로우"
)
async def batch_analyze_youtube_videos(
    video_ids: List[str],
    analysis_features: Optional[List[str]] = None,
    download_resolution: str = "720p",
    extract_highlights: bool = True,
    output_dir: Optional[str] = None,
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """
    여러 YouTube 동영상을 일괄 분석하는 워크플로우
    
    Parameters:
    -----------
    video_ids : List[str]
        분석할 YouTube 비디오 ID 목록
    analysis_features : List[str], optional
        분석할 기능 목록 (기본값: ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "OBJECT_TRACKING"])
    download_resolution : str, optional
        다운로드할 해상도 (기본값: "720p")
    extract_highlights : bool, optional
        하이라이트 추출 여부 (기본값: True)
    output_dir : str, optional
        결과를 저장할 디렉토리 (기본값: ANALYSIS_OUTPUT_DIR)
    max_concurrent : int, optional
        동시 실행 최대 작업 수 (기본값: 3)
        
    Returns:
    --------
    Dict[str, Any]
        일괄 분석 결과를 포함하는 딕셔너리
    """
    start_time = time.time()
    
    # 저장 디렉토리가 없으면 생성
    if output_dir is None:
        output_dir = ANALYSIS_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프 생성 (중복 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # 세마포어를 사용하여 동시 실행 제한
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def analyze_with_semaphore(video_id):
        async with semaphore:
            return await analyze_youtube_video(
                video_id=video_id,
                analysis_features=analysis_features,
                download_resolution=download_resolution,
                extract_highlights=extract_highlights,
                output_dir=batch_dir
            )
    
    try:
        logger.info(f"{len(video_ids)}개 비디오 일괄 분석 시작 (최대 동시 실행: {max_concurrent})")
        
        # 비동기로 모든 비디오 분석 실행
        tasks = [analyze_with_semaphore(video_id) for video_id in video_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            video_id = video_ids[i]
            
            # 예외 처리
            if isinstance(result, Exception):
                failed.append({
                    "video_id": video_id,
                    "error": str(result)
                })
                continue
            
            if not result.get("success", False):
                failed.append({
                    "video_id": video_id,
                    "error": result.get("error", "Unknown error")
                })
                continue
            
            # 성공 결과 추가
            successful.append({
                "video_id": video_id,
                "video_title": result.get("video_title"),
                "execution_time": result.get("execution_time"),
                "report_file": result.get("report_file")
            })
        
        # 최종 보고서 생성
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        batch_report = {
            "timestamp": timestamp,
            "total_videos": len(video_ids),
            "successful": len(successful),
            "failed": len(failed),
            "total_execution_time": total_execution_time,
            "successful_videos": successful,
            "failed_videos": failed
        }
        
        # 보고서 저장
        batch_report_file = os.path.join(output_dir, f"batch_report_{timestamp}.json")
        with open(batch_report_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"일괄 분석 완료: {len(successful)}개 성공, {len(failed)}개 실패 (총 실행 시간: {total_execution_time:.2f}초)")
        
        return {
            "success": True,
            "timestamp": timestamp,
            "total_videos": len(video_ids),
            "successful": len(successful),
            "failed": len(failed),
            "total_execution_time": total_execution_time,
            "batch_report_file": batch_report_file,
            "batch_dir": batch_dir,
            "successful_videos": successful,
            "failed_videos": failed
        }
    
    except Exception as e:
        import traceback
        error_message = f"일괄 분석 워크플로우 중 오류 발생: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": error_message,
            "traceback": traceback.format_exc()
        }

@mcp.workflow(
    name="search_and_analyze_youtube_videos",
    description="키워드로 YouTube 동영상을 검색하고 분석하는 워크플로우"
)
async def search_and_analyze_youtube_videos(
    query: str,
    max_videos: int = 5,
    analysis_features: Optional[List[str]] = None,
    download_resolution: str = "720p",
    extract_highlights: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    키워드로 YouTube 동영상을 검색하고 분석하는 워크플로우
    
    Parameters:
    -----------
    query : str
        검색할 키워드
    max_videos : int, optional
        분석할 최대 비디오 수 (기본값: 5)
    analysis_features : List[str], optional
        분석할 기능 목록 (기본값: ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "OBJECT_TRACKING"])
    download_resolution : str, optional
        다운로드할 해상도 (기본값: "720p")
    extract_highlights : bool, optional
        하이라이트 추출 여부 (기본값: True)
    output_dir : str, optional
        결과를 저장할 디렉토리 (기본값: ANALYSIS_OUTPUT_DIR)
        
    Returns:
    --------
    Dict[str, Any]
        검색 및 분석 결과를 포함하는 딕셔너리
    """
    try:
        # 1. 키워드로 YouTube 비디오 검색
        logger.info(f"키워드 '{query}'로 YouTube 비디오 검색")
        search_result = await search_youtube_videos(query=query, max_results=max_videos)
        
        if not search_result.get("success", False) or not search_result.get("videos"):
            return {
                "success": False,
                "error": f"비디오 검색 실패 또는 결과 없음: {search_result.get('error', '검색 결과가 없습니다.')}"
            }
        
        # 2. 검색된 비디오 ID 목록 추출
        videos = search_result.get("videos", [])
        video_ids = [video["video_id"] for video in videos]
        
        # 3. 검색 결과에 해당하는 비디오 일괄 분석
        logger.info(f"검색된 {len(video_ids)}개 비디오 분석 시작")
        
        # 검색 키워드를 포함하는 저장 디렉토리 생성
        if output_dir is None:
            output_dir = ANALYSIS_OUTPUT_DIR
        
        search_query_safe = query.replace('/', '_').replace('\\', '_').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_output_dir = os.path.join(output_dir, f"search_{search_query_safe}_{timestamp}")
        
        # 일괄 분석 실행
        batch_result = await batch_analyze_youtube_videos(
            video_ids=video_ids,
            analysis_features=analysis_features,
            download_resolution=download_resolution,
            extract_highlights=extract_highlights,
            output_dir=search_output_dir
        )
        
        # 4. 결과 구성
        if not batch_result.get("success", False):
            return {
                "success": False,
                "error": f"비디오 분석 실패: {batch_result.get('error')}",
                "search_results": search_result
            }
        
        # 검색 결과와 분석 결과 결합
        combined_result = {
            "success": True,
            "query": query,
            "timestamp": timestamp,
            "search_results": search_result,
            "analysis_results": batch_result,
            "output_dir": search_output_dir
        }
        
        # 보고서 저장
        report_file = os.path.join(output_dir, f"search_analysis_{search_query_safe}_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)
        
        combined_result["report_file"] = report_file
        
        logger.info(f"키워드 '{query}'에 대한 검색 및 분석 완료")
        
        return combined_result
    
    except Exception as e:
        import traceback
        error_message = f"검색 및 분석 워크플로우 중 오류 발생: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "query": query,
            "error": error_message,
            "traceback": traceback.format_exc()
        }

# 도구 테스트 함수 (직접 실행 시 사용)
async def test_youtube_analysis_workflow():
    """YouTube 분석 워크플로우를 테스트합니다."""
    # 1. 단일 비디오 분석 테스트
    video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    print(f"비디오 '{video_id}' 분석 테스트:")
    result = await analyze_youtube_video(
        video_id=video_id,
        analysis_features=["LABEL_DETECTION", "SHOT_CHANGE_DETECTION"]
    )
    
    if result.get("success", False):
        print(f"분석 성공: {result.get('report_file')}")
        print(f"분석 시간: {result.get('execution_time'):.2f}초")
    else:
        print(f"분석 실패: {result.get('error')}")
    
    print("-" * 50)
    
    # 2. 키워드 검색 및 분석 테스트
    query = "machine learning tutorial short"
    max_videos = 2
    
    print(f"'{query}' 검색 및 분석 테스트 (최대 {max_videos}개 비디오):")
    result = await search_and_analyze_youtube_videos(
        query=query,
        max_videos=max_videos,
        analysis_features=["LABEL_DETECTION"]
    )
    
    if result.get("success", False):
        print(f"검색 및 분석 성공!")
        search_results = result.get("search_results", {})
        analysis_results = result.get("analysis_results", {})
        
        print(f"검색된 비디오: {search_results.get('video_count', 0)}개")
        print(f"분석 성공: {analysis_results.get('successful', 0)}개")
        print(f"분석 실패: {analysis_results.get('failed', 0)}개")
        print(f"총 실행 시간: {analysis_results.get('total_execution_time', 0):.2f}초")
    else:
        print(f"검색 및 분석 실패: {result.get('error')}")

# 모듈 직접 실행 시 테스트 수행
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_youtube_analysis_workflow())

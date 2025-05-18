"""
YouTube 동영상 다운로드 모듈
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from pytube import YouTube
from pytube.exceptions import PytubeError

from analysis.mcp_init import mcp
# config 패키지에서 직접 임포트
from config.youtube_config import TEMP_DOWNLOAD_DIR

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube_downloader")

@mcp.tool(
    name="download_youtube_video",
    description="YouTube 동영상을 다운로드합니다 (YouTube ToS 준수 필수)",
    tags=["youtube", "download", "video"]
)
async def download_youtube_video(
    video_id: str,
    output_dir: Optional[str] = None,
    resolution: str = "720p",
    file_extension: str = "mp4"
) -> Dict[str, Any]:
    """
    YouTube 동영상을 다운로드합니다.
    
    주의: YouTube 서비스 약관(ToS)을 준수해야 합니다.
    상업적 용도나 저작권 침해 목적의 다운로드는 금지됩니다.
    
    Parameters:
    -----------
    video_id : str
        다운로드할 YouTube 비디오 ID
    output_dir : str, optional
        동영상을 저장할 디렉토리 (기본값: TEMP_DOWNLOAD_DIR)
    resolution : str, optional
        다운로드할 해상도 (기본값: "720p")
    file_extension : str, optional
        파일 확장자 (기본값: "mp4")
        
    Returns:
    --------
    Dict[str, Any]
        다운로드 결과 정보를 포함하는 딕셔너리
    """
    if output_dir is None:
        output_dir = TEMP_DOWNLOAD_DIR
    
    # 저장 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # YouTube 동영상 URL 생성
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        start_time = time.time()
        
        # YouTube 객체 생성
        yt = YouTube(video_url)
        
        # 현재 시간으로 파일명 생성 (중복 방지)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{video_id}_{timestamp}.{file_extension}"
        file_path = os.path.join(output_dir, filename)
        
        # 동영상 스트림 선택 및 다운로드
        if resolution:
            # 특정 해상도로 다운로드
            video_stream = yt.streams.filter(
                progressive=True,
                file_extension=file_extension,
                resolution=resolution
            ).first()
            
            # 해당 해상도가 없으면 가장 가까운 해상도 선택
            if not video_stream:
                logger.warning(f"해상도 {resolution}을(를) 찾을 수 없습니다. 가장 높은 해상도로 다운로드합니다.")
                video_stream = yt.streams.filter(
                    progressive=True,
                    file_extension=file_extension
                ).order_by('resolution').desc().first()
        else:
            # 기본: 가장 높은 해상도로 다운로드
            video_stream = yt.streams.filter(
                progressive=True,
                file_extension=file_extension
            ).order_by('resolution').desc().first()
        
        # 다운로드
        if video_stream:
            video_stream.download(output_path=output_dir, filename=filename)
            
            # 다운로드 완료 시간 계산
            end_time = time.time()
            download_time = end_time - start_time
            
            # 파일 크기 확인
            file_size = os.path.getsize(file_path)
            
            logger.info(f"비디오 '{video_id}' 다운로드 성공: {file_path} ({file_size/1024/1024:.2f} MB)")
            
            return {
                "success": True,
                "video_id": video_id,
                "title": yt.title,
                "file_path": file_path,
                "file_size": file_size,
                "resolution": video_stream.resolution,
                "download_time": download_time,
                "author": yt.author
            }
        else:
            error_message = f"비디오 '{video_id}'에 적합한 다운로드 스트림을 찾을 수 없습니다."
            logger.error(error_message)
            return {
                "success": False,
                "video_id": video_id,
                "error": error_message
            }
    
    except PytubeError as e:
        error_message = f"PyTube 오류: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "video_id": video_id,
            "error": error_message
        }
    
    except Exception as e:
        error_message = f"동영상 다운로드 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "video_id": video_id,
            "error": error_message
        }

@mcp.tool(
    name="get_video_download_info",
    description="다운로드 전에 YouTube 동영상의 사용 가능한 형식 정보를 가져옵니다",
    tags=["youtube", "info", "video"]
)
async def get_video_download_info(video_id: str) -> Dict[str, Any]:
    """
    다운로드 전에 YouTube 동영상의 사용 가능한 형식 정보를 가져옵니다.
    
    Parameters:
    -----------
    video_id : str
        정보를 확인할 YouTube 비디오 ID
        
    Returns:
    --------
    Dict[str, Any]
        사용 가능한 비디오 형식 정보를 포함하는 딕셔너리
    """
    # YouTube 동영상 URL 생성
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        # YouTube 객체 생성
        yt = YouTube(video_url)
        
        # 동영상 기본 정보
        basic_info = {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,  # 초 단위
            "publish_date": yt.publish_date.isoformat() if yt.publish_date else None,
            "views": yt.views,
            "rating": yt.rating,
            "description": yt.description
        }
        
        # 사용 가능한 스트림 정보 수집
        streams_info = []
        
        # 프로그레시브(오디오+비디오) 스트림
        progressive_streams = yt.streams.filter(progressive=True)
        for stream in progressive_streams:
            stream_info = {
                "itag": stream.itag,
                "mime_type": stream.mime_type,
                "resolution": stream.resolution,
                "fps": stream.fps,
                "abr": getattr(stream, 'abr', None),  # 오디오 비트레이트
                "file_size": stream.filesize,
                "file_size_mb": stream.filesize / (1024 * 1024),
                "type": "progressive",
                "codecs": stream.codecs
            }
            streams_info.append(stream_info)
        
        # 비디오 전용 스트림(DASH)
        video_streams = yt.streams.filter(adaptive=True, only_video=True)
        for stream in video_streams:
            stream_info = {
                "itag": stream.itag,
                "mime_type": stream.mime_type,
                "resolution": stream.resolution,
                "fps": stream.fps,
                "file_size": stream.filesize,
                "file_size_mb": stream.filesize / (1024 * 1024),
                "type": "video_only",
                "codecs": stream.codecs
            }
            streams_info.append(stream_info)
        
        # 오디오 전용 스트림
        audio_streams = yt.streams.filter(adaptive=True, only_audio=True)
        for stream in audio_streams:
            stream_info = {
                "itag": stream.itag,
                "mime_type": stream.mime_type,
                "abr": getattr(stream, 'abr', None),
                "file_size": stream.filesize,
                "file_size_mb": stream.filesize / (1024 * 1024),
                "type": "audio_only",
                "codecs": stream.codecs
            }
            streams_info.append(stream_info)
        
        logger.info(f"비디오 '{video_id}' 다운로드 정보 가져오기 성공")
        
        return {
            "success": True,
            "video_id": video_id,
            "basic_info": basic_info,
            "streams": streams_info,
            "available_resolutions": sorted(list(set([s.get("resolution") for s in streams_info if s.get("resolution")])), key=lambda x: int(x[:-1]) if x and x[:-1].isdigit() else 0)
        }
    
    except PytubeError as e:
        error_message = f"PyTube 오류: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "video_id": video_id,
            "error": error_message
        }
    
    except Exception as e:
        error_message = f"동영상 정보 가져오기 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "video_id": video_id,
            "error": error_message
        }

# 도구 테스트 함수 (직접 실행 시 사용)
async def test_youtube_downloader():
    """YouTube 다운로더 도구를 테스트합니다."""
    # 테스트용 비디오 ID (Big Buck Bunny - 오픈 소스 비디오)
    video_id = "aqz-KE-bpKQ"
    
    # 다운로드 정보 가져오기 테스트
    print(f"비디오 '{video_id}' 다운로드 정보 가져오기 테스트:")
    result = await get_video_download_info(video_id)
    
    # 결과 요약 출력
    if result.get("success", False):
        basic_info = result.get("basic_info", {})
        print(f"제목: {basic_info.get('title')}")
        print(f"작성자: {basic_info.get('author')}")
        print(f"길이: {basic_info.get('length')}초")
        print(f"조회수: {basic_info.get('views')}")
        
        print("\n사용 가능한 해상도:")
        for resolution in result.get("available_resolutions", []):
            print(f"- {resolution}")
    else:
        print(f"오류: {result.get('error')}")
    
    print("-" * 50)
    
    # 동영상 다운로드 테스트
    print(f"비디오 '{video_id}' 다운로드 테스트:")
    download_result = await download_youtube_video(video_id)
    
    if download_result.get("success", False):
        print(f"다운로드 성공: {download_result.get('file_path')}")
        print(f"파일 크기: {download_result.get('file_size')/1024/1024:.2f} MB")
        print(f"해상도: {download_result.get('resolution')}")
        print(f"다운로드 시간: {download_result.get('download_time'):.2f}초")
    else:
        print(f"다운로드 실패: {download_result.get('error')}")

# 모듈 직접 실행 시 테스트 수행
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_youtube_downloader())

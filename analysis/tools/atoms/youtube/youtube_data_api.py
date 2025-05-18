"""
YouTube Data API v3 활용 모듈
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from analysis.mcp_init import mcp
# config 패키지에서 직접 임포트
from config.youtube_config import YOUTUBE_API_KEY

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube_data_api")

@mcp.tool(
    name="get_video_metadata",
    description="YouTube Data API를 통해 특정 비디오의 메타데이터를 가져옵니다",
    tags=["youtube", "data", "metadata"]
)
async def get_video_metadata(video_id: str) -> Dict[str, Any]:
    """
    YouTube Data API를 통해 특정 비디오의 메타데이터를 가져옵니다.
    
    Parameters:
    -----------
    video_id : str
        분석할 YouTube 비디오 ID
        
    Returns:
    --------
    Dict[str, Any]
        비디오 메타데이터를 포함하는 딕셔너리
    """
    try:
        # API 서비스 객체 생성
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # 비디오 정보 요청
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        
        # 유효한 비디오 ID인지 확인
        if not response.get('items'):
            return {
                "success": False,
                "error": f"비디오 ID '{video_id}'에 해당하는 비디오를 찾을 수 없습니다."
            }
        
        # 필요한 메타데이터 추출
        video_data = response['items'][0]
        
        metadata = {
            "success": True,
            "video_id": video_id,
            "title": video_data['snippet']['title'],
            "description": video_data['snippet']['description'],
            "publish_date": video_data['snippet']['publishedAt'],
            "channel_id": video_data['snippet']['channelId'],
            "channel_title": video_data['snippet']['channelTitle'],
            "tags": video_data['snippet'].get('tags', []),
            "category_id": video_data['snippet'].get('categoryId', ''),
            "duration": video_data['contentDetails']['duration'],
            "view_count": int(video_data['statistics'].get('viewCount', 0)),
            "like_count": int(video_data['statistics'].get('likeCount', 0)),
            "comment_count": int(video_data['statistics'].get('commentCount', 0)),
            "thumbnail_url": video_data['snippet']['thumbnails']['high']['url']
        }
        
        # 결과 로깅
        logger.info(f"비디오 '{video_id}' 메타데이터 수집 성공")
        
        return metadata
    
    except HttpError as e:
        error_message = f"YouTube API 오류: {e.reason}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }
    
    except Exception as e:
        error_message = f"비디오 메타데이터 수집 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }

@mcp.tool(
    name="search_youtube_videos",
    description="YouTube Data API를 통해 검색어에 해당하는 비디오 목록을 가져옵니다",
    tags=["youtube", "data", "search"]
)
async def search_youtube_videos(
    query: str,
    max_results: int = 10,
    order: str = "relevance",
    published_after: Optional[str] = None
) -> Dict[str, Any]:
    """
    YouTube Data API를 통해 검색어에 해당하는 비디오 목록을 가져옵니다.
    
    Parameters:
    -----------
    query : str
        검색할 키워드
    max_results : int, optional
        반환할 최대 결과 수 (기본값: 10)
    order : str, optional
        결과 정렬 방식 (기본값: "relevance")
        옵션: "date", "rating", "relevance", "title", "viewCount"
    published_after : str, optional
        특정 날짜 이후 발행된 비디오만 검색 (ISO 8601 형식: YYYY-MM-DDThh:mm:ssZ)
        
    Returns:
    --------
    Dict[str, Any]
        검색 결과 비디오 목록을 포함하는 딕셔너리
    """
    try:
        # API 서비스 객체 생성
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # 검색 요청 매개변수 구성
        request_params = {
            "q": query,
            "part": "snippet",
            "maxResults": max_results,
            "order": order,
            "type": "video"
        }
        
        # 발행 날짜 필터 추가 (제공된 경우)
        if published_after:
            request_params["publishedAfter"] = published_after
        
        # 검색 요청 실행
        request = youtube.search().list(**request_params)
        response = request.execute()
        
        # 검색 결과 처리
        video_list = []
        
        for item in response.get('items', []):
            video_info = {
                "video_id": item['id']['videoId'],
                "title": item['snippet']['title'],
                "description": item['snippet']['description'],
                "publish_date": item['snippet']['publishedAt'],
                "channel_id": item['snippet']['channelId'],
                "channel_title": item['snippet']['channelTitle'],
                "thumbnail_url": item['snippet']['thumbnails']['high']['url']
            }
            video_list.append(video_info)
        
        # 결과 로깅
        logger.info(f"'{query}' 검색어로 {len(video_list)}개 비디오 검색 성공")
        
        return {
            "success": True,
            "query": query,
            "video_count": len(video_list),
            "videos": video_list
        }
    
    except HttpError as e:
        error_message = f"YouTube API 오류: {e.reason}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }
    
    except Exception as e:
        error_message = f"YouTube 비디오 검색 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }

@mcp.tool(
    name="get_channel_videos",
    description="YouTube Data API를 통해 특정 채널의 비디오 목록을 가져옵니다",
    tags=["youtube", "data", "channel"]
)
async def get_channel_videos(
    channel_id: str,
    max_results: int = 20
) -> Dict[str, Any]:
    """
    YouTube Data API를 통해 특정 채널의 비디오 목록을 가져옵니다.
    
    Parameters:
    -----------
    channel_id : str
        비디오를 가져올 YouTube 채널 ID
    max_results : int, optional
        반환할 최대 결과 수 (기본값: 20)
        
    Returns:
    --------
    Dict[str, Any]
        채널 비디오 목록을 포함하는 딕셔너리
    """
    try:
        # API 서비스 객체 생성
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # 1단계: 채널의 업로드 재생목록 ID 가져오기
        channel_request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        channel_response = channel_request.execute()
        
        # 유효한 채널 ID인지 확인
        if not channel_response.get('items'):
            return {
                "success": False,
                "error": f"채널 ID '{channel_id}'에 해당하는 채널을 찾을 수 없습니다."
            }
        
        # 업로드 재생목록 ID 추출
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # 2단계: 업로드 재생목록에서 비디오 목록 가져오기
        playlist_request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results
        )
        playlist_response = playlist_request.execute()
        
        # 비디오 정보 추출
        video_list = []
        
        for item in playlist_response.get('items', []):
            video_info = {
                "video_id": item['contentDetails']['videoId'],
                "title": item['snippet']['title'],
                "description": item['snippet']['description'],
                "publish_date": item['snippet']['publishedAt'],
                "thumbnail_url": item['snippet']['thumbnails']['high']['url']
            }
            video_list.append(video_info)
        
        # 채널 이름 가져오기
        channel_title = ""
        if playlist_response.get('items'):
            channel_title = playlist_response['items'][0]['snippet']['channelTitle']
        
        # 결과 로깅
        logger.info(f"채널 '{channel_id}' ({channel_title})에서 {len(video_list)}개 비디오 가져오기 성공")
        
        return {
            "success": True,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "uploads_playlist_id": uploads_playlist_id,
            "video_count": len(video_list),
            "videos": video_list
        }
    
    except HttpError as e:
        error_message = f"YouTube API 오류: {e.reason}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }
    
    except Exception as e:
        error_message = f"채널 비디오 가져오기 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }

# 도구 테스트 함수 (직접 실행 시 사용)
async def test_youtube_api():
    """YouTube Data API 도구를 테스트합니다."""
    # 비디오 메타데이터 가져오기 테스트
    video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    print(f"비디오 '{video_id}' 메타데이터 가져오기 테스트:")
    result = await get_video_metadata(video_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 50)
    
    # 비디오 검색 테스트
    query = "machine learning tutorial"
    print(f"'{query}' 검색어로 비디오 검색 테스트:")
    result = await search_youtube_videos(query, max_results=3)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-" * 50)
    
    # 채널 비디오 가져오기 테스트
    channel_id = "UC-lHJZR3Gqxm24_Vd_AJ5Yw"  # PewDiePie
    print(f"채널 '{channel_id}' 비디오 가져오기 테스트:")
    result = await get_channel_videos(channel_id, max_results=3)
    print(json.dumps(result, indent=2, ensure_ascii=False))

# 모듈 직접 실행 시 테스트 수행
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_youtube_api())

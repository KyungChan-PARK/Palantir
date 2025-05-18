"""
YouTube 인사이트 시스템 - 유기체 레벨 모듈
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from analysis.mcp_init import mcp
# config 패키지에서 직접 임포트
from config.youtube_config import ANALYSIS_OUTPUT_DIR

# 분자 레벨 워크플로우 가져오기
from analysis.tools.molecules.youtube.youtube_video_analysis import (
    analyze_youtube_video,
    batch_analyze_youtube_videos,
    search_and_analyze_youtube_videos
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube_insights_system")

@mcp.system(
    name="youtube_insights_system",
    description="YouTube 동영상 분석 결과를 기반으로 종합적인 인사이트를 생성하는 시스템"
)
async def generate_youtube_insights(
    video_ids: Optional[List[str]] = None,
    search_query: Optional[str] = None,
    max_videos: int = 5,
    analysis_type: str = "comprehensive",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    YouTube 동영상 분석 결과를 기반으로 종합적인 인사이트를 생성합니다.
    
    Parameters:
    -----------
    video_ids : List[str], optional
        분석할 특정 YouTube 비디오 ID 목록
    search_query : str, optional
        비디오를 검색할 키워드 (video_ids가 제공되지 않은 경우 사용)
    max_videos : int, optional
        검색할 최대 비디오 수 (기본값: 5)
    analysis_type : str, optional
        분석 유형 (기본값: "comprehensive")
        옵션: "basic", "comprehensive", "content_moderation", "audience_insights"
    output_dir : str, optional
        결과를 저장할 디렉토리 (기본값: ANALYSIS_OUTPUT_DIR)
        
    Returns:
    --------
    Dict[str, Any]
        YouTube 인사이트를 포함하는 딕셔너리
    """
    if output_dir is None:
        output_dir = ANALYSIS_OUTPUT_DIR
    
    # 타임스탬프 생성 (중복 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 분석 유형에 따른 기능 선택
    if analysis_type == "basic":
        analysis_features = ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION"]
        extract_highlights = False
    elif analysis_type == "content_moderation":
        analysis_features = ["EXPLICIT_CONTENT_DETECTION", "OBJECT_TRACKING", "TEXT_DETECTION"]
        extract_highlights = False
    elif analysis_type == "audience_insights":
        analysis_features = ["LABEL_DETECTION", "OBJECT_TRACKING", "FACE_DETECTION", "PERSON_DETECTION"]
        extract_highlights = True
    else:  # comprehensive (기본값)
        analysis_features = ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "OBJECT_TRACKING", "TEXT_DETECTION"]
        extract_highlights = True
    
    try:
        # 1. 비디오 분석 수행
        analysis_results = None
        
        if video_ids:
            # 특정 비디오 ID들을 분석
            logger.info(f"{len(video_ids)}개 비디오 분석 시작")
            
            # 분석을 위한 디렉토리 준비
            insights_dir = os.path.join(output_dir, f"insights_{timestamp}")
            os.makedirs(insights_dir, exist_ok=True)
            
            analysis_results = await batch_analyze_youtube_videos(
                video_ids=video_ids,
                analysis_features=analysis_features,
                extract_highlights=extract_highlights,
                output_dir=insights_dir
            )
            
            analysis_mode = "specific_videos"
        
        elif search_query:
            # 검색 쿼리로 비디오를 찾아 분석
            logger.info(f"'{search_query}' 검색 및 분석 시작")
            
            # 검색 결과 저장을 위한 디렉토리 준비
            search_query_safe = search_query.replace('/', '_').replace('\\', '_').replace(' ', '_')
            insights_dir = os.path.join(output_dir, f"insights_search_{search_query_safe}_{timestamp}")
            
            analysis_results = await search_and_analyze_youtube_videos(
                query=search_query,
                max_videos=max_videos,
                analysis_features=analysis_features,
                extract_highlights=extract_highlights,
                output_dir=insights_dir
            )
            
            analysis_mode = "search_query"
        
        else:
            return {
                "success": False,
                "error": "video_ids 또는 search_query 중 하나를 제공해야 합니다."
            }
        
        if not analysis_results.get("success", False):
            return {
                "success": False,
                "error": f"비디오 분석 실패: {analysis_results.get('error')}"
            }
        
        # 2. 분석 결과 처리 및 인사이트 도출
        insights = {}
        
        if analysis_mode == "specific_videos":
            # 개별 비디오 보고서 파일 읽기
            report_files = []
            
            for video_result in analysis_results.get("successful_videos", []):
                report_file = video_result.get("report_file")
                if report_file and os.path.exists(report_file):
                    report_files.append(report_file)
            
            # 보고서 파일들로부터 데이터 추출
            if report_files:
                insights = await extract_insights_from_reports(
                    report_files=report_files,
                    analysis_type=analysis_type,
                    output_dir=insights_dir
                )
            
        elif analysis_mode == "search_query":
            # 검색 결과 및 분석 결과 처리
            search_results = analysis_results.get("search_results", {})
            batch_results = analysis_results.get("analysis_results", {})
            
            # 검색 인사이트 추가
            insights["search"] = {
                "query": search_query,
                "video_count": search_results.get("video_count", 0),
                "videos": search_results.get("videos", [])
            }
            
            # 개별 비디오 보고서 파일 읽기
            report_files = []
            
            for video_result in batch_results.get("successful_videos", []):
                report_file = video_result.get("report_file")
                if report_file and os.path.exists(report_file):
                    report_files.append(report_file)
            
            # 보고서 파일들로부터 데이터 추출
            if report_files:
                analysis_insights = await extract_insights_from_reports(
                    report_files=report_files,
                    analysis_type=analysis_type,
                    output_dir=insights_dir
                )
                
                # 분석 인사이트 병합
                insights.update(analysis_insights)
        
        # 3. 인사이트 시각화 및 보고서 생성 
        visualizations = await create_insight_visualizations(
            insights=insights,
            analysis_type=analysis_type,
            output_dir=insights_dir
        )
        
        # 4. 최종 보고서 작성
        report = {
            "timestamp": timestamp,
            "analysis_type": analysis_type,
            "mode": analysis_mode,
            "analysis_results": {
                "success": analysis_results.get("success", False),
                "total_videos": analysis_results.get("total_videos", 0) if analysis_mode == "specific_videos" else analysis_results.get("analysis_results", {}).get("total_videos", 0),
                "successful": analysis_results.get("successful", 0) if analysis_mode == "specific_videos" else analysis_results.get("analysis_results", {}).get("successful", 0),
                "failed": analysis_results.get("failed", 0) if analysis_mode == "specific_videos" else analysis_results.get("analysis_results", {}).get("failed", 0)
            },
            "insights": insights,
            "visualizations": visualizations
        }
        
        # 검색 쿼리 또는 비디오 ID 정보 추가
        if search_query:
            report["search_query"] = search_query
        if video_ids:
            report["video_ids"] = video_ids
        
        # 보고서 저장
        if analysis_mode == "specific_videos":
            report_file = os.path.join(output_dir, f"insights_report_{timestamp}.json")
        else:
            search_query_safe = search_query.replace('/', '_').replace('\\', '_').replace(' ', '_')
            report_file = os.path.join(output_dir, f"insights_report_{search_query_safe}_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"인사이트 생성 완료: {report_file}")
        
        # HTML 보고서 생성 (옵션)
        html_report = await generate_html_report(
            report=report,
            output_dir=insights_dir
        )
        
        return {
            "success": True,
            "timestamp": timestamp,
            "analysis_type": analysis_type,
            "mode": analysis_mode,
            "insights_dir": insights_dir,
            "report_file": report_file,
            "html_report": html_report,
            "insights": insights,
            "visualizations": visualizations
        }
    
    except Exception as e:
        import traceback
        error_message = f"인사이트 생성 중 오류 발생: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": error_message,
            "traceback": traceback.format_exc()
        }

async def extract_insights_from_reports(
    report_files: List[str],
    analysis_type: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    비디오 분석 보고서에서 통합 인사이트를 추출합니다.
    
    Parameters:
    -----------
    report_files : List[str]
        분석 보고서 파일 경로 목록
    analysis_type : str
        분석 유형
    output_dir : str
        출력 파일을 저장할 디렉토리
        
    Returns:
    --------
    Dict[str, Any]
        추출된 인사이트를 포함하는 딕셔너리
    """
    logger.info(f"{len(report_files)}개 보고서에서 인사이트 추출 시작")
    
    # 보고서 데이터 로드
    reports_data = []
    
    for file_path in report_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                reports_data.append(report_data)
        except Exception as e:
            logger.error(f"보고서 파일 '{file_path}' 로드 중 오류: {str(e)}")
    
    if not reports_data:
        return {"error": "유효한 보고서를 찾을 수 없습니다."}
    
    # 인사이트 초기화
    insights = {
        "video_count": len(reports_data),
        "videos": [],
        "common_labels": {},
        "common_objects": {},
        "common_text_themes": {},
        "explicit_content": {},
        "content_diversity": {},
        "engagement_factors": {}
    }
    
    # 집계 데이터
    all_labels = {}
    all_objects = {}
    all_texts = {}
    explicit_content_scores = []
    shot_counts = []
    
    # 비디오별 데이터 처리
    for report in reports_data:
        video_id = report.get("video_id", "unknown")
        video_title = report.get("video_title", "Unknown Title")
        metadata = report.get("metadata", {})
        analysis_summary = report.get("analysis_summary", {})
        highlights = report.get("highlights", [])
        
        # 비디오 기본 정보 추가
        video_info = {
            "video_id": video_id,
            "title": video_title,
            "duration": metadata.get("duration", ""),
            "view_count": metadata.get("view_count", 0),
            "like_count": metadata.get("like_count", 0),
            "publish_date": metadata.get("publish_date", "")
        }
        
        # 분석 유형에 따른 추가 정보
        if analysis_type == "comprehensive" or analysis_type == "basic":
            # 레이블 추가
            labels = analysis_summary.get("labels", [])
            video_info["top_labels"] = labels[:5] if labels else []
            
            # 레이블 집계
            for label in labels:
                description = label.get("description", "")
                confidence = label.get("confidence", 0)
                
                if description:
                    if description not in all_labels:
                        all_labels[description] = {"count": 0, "confidence_sum": 0}
                    
                    all_labels[description]["count"] += 1
                    all_labels[description]["confidence_sum"] += confidence
            
            # 샷 정보 추가
            shots = analysis_summary.get("shots", {})
            shot_count = shots.get("count", 0)
            video_info["shot_count"] = shot_count
            
            if shot_count > 0:
                shot_counts.append(shot_count)
        
        if analysis_type == "comprehensive" or analysis_type == "audience_insights":
            # 객체 정보 추가
            objects = analysis_summary.get("objects", [])
            video_info["top_objects"] = objects[:5] if objects else []
            
            # 객체 집계
            for obj in objects:
                description = obj.get("description", "")
                confidence = obj.get("confidence", 0)
                
                if description:
                    if description not in all_objects:
                        all_objects[description] = {"count": 0, "confidence_sum": 0}
                    
                    all_objects[description]["count"] += 1
                    all_objects[description]["confidence_sum"] += confidence
        
        if analysis_type == "comprehensive" or analysis_type == "content_moderation":
            # 텍스트 정보 추가
            texts = analysis_summary.get("texts", [])
            video_info["detected_texts"] = texts[:5] if texts else []
            
            # 텍스트 집계
            for text_item in texts:
                text = text_item.get("text", "")
                confidence = text_item.get("confidence", 0)
                
                if text:
                    if text not in all_texts:
                        all_texts[text] = {"count": 0, "confidence_sum": 0}
                    
                    all_texts[text]["count"] += 1
                    all_texts[text]["confidence_sum"] += confidence
            
            # 명시적 콘텐츠 정보 추가
            explicit_content = analysis_summary.get("explicit_content", {})
            
            if explicit_content:
                adult_detected = explicit_content.get("adult_content_detected", False)
                video_info["adult_content_detected"] = adult_detected
                
                if adult_detected:
                    explicit_content_scores.append(1)
                else:
                    explicit_content_scores.append(0)
        
        # 하이라이트 정보 추가
        if highlights:
            video_info["highlights_count"] = len(highlights)
            
            # 상위 3개 하이라이트만 포함
            top_highlights = sorted(highlights, key=lambda x: x.get("score", 0), reverse=True)[:3]
            
            video_info["top_highlights"] = [
                {
                    "start": highlight.get("start", 0),
                    "end": highlight.get("end", 0),
                    "duration": highlight.get("end", 0) - highlight.get("start", 0),
                    "objects": [obj.get("description", "") for obj in highlight.get("objects", [])[:3]],
                    "labels": [label.get("description", "") for label in highlight.get("labels", [])[:3]]
                }
                for highlight in top_highlights
            ]
        
        # 비디오 정보 추가
        insights["videos"].append(video_info)
    
    # 공통 레이블 분석
    if all_labels:
        # 빈도와 신뢰도를 기준으로 레이블 정렬
        sorted_labels = [
            {
                "description": label,
                "count": data["count"],
                "frequency": data["count"] / len(reports_data),
                "avg_confidence": data["confidence_sum"] / data["count"]
            }
            for label, data in all_labels.items()
        ]
        sorted_labels.sort(key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)
        
        insights["common_labels"] = {
            "top_labels": sorted_labels[:10],
            "unique_labels_count": len(all_labels),
            "label_diversity": len(all_labels) / max(1, sum(data["count"] for data in all_labels.values()))
        }
    
    # 공통 객체 분석
    if all_objects:
        # 빈도와 신뢰도를 기준으로 객체 정렬
        sorted_objects = [
            {
                "description": obj,
                "count": data["count"],
                "frequency": data["count"] / len(reports_data),
                "avg_confidence": data["confidence_sum"] / data["count"]
            }
            for obj, data in all_objects.items()
        ]
        sorted_objects.sort(key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)
        
        insights["common_objects"] = {
            "top_objects": sorted_objects[:10],
            "unique_objects_count": len(all_objects),
            "object_diversity": len(all_objects) / max(1, sum(data["count"] for data in all_objects.values()))
        }
    
    # 공통 텍스트 테마 분석
    if all_texts:
        # 빈도와 신뢰도를 기준으로 텍스트 정렬
        sorted_texts = [
            {
                "text": text,
                "count": data["count"],
                "frequency": data["count"] / len(reports_data),
                "avg_confidence": data["confidence_sum"] / data["count"]
            }
            for text, data in all_texts.items()
        ]
        sorted_texts.sort(key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)
        
        insights["common_text_themes"] = {
            "top_texts": sorted_texts[:10],
            "unique_texts_count": len(all_texts)
        }
    
    # 명시적 콘텐츠 분석
    if explicit_content_scores:
        adult_content_ratio = sum(explicit_content_scores) / len(explicit_content_scores)
        
        insights["explicit_content"] = {
            "adult_content_ratio": adult_content_ratio,
            "adult_content_videos": sum(explicit_content_scores),
            "safe_content_videos": len(explicit_content_scores) - sum(explicit_content_scores)
        }
    
    # 콘텐츠 다양성 분석
    if shot_counts:
        avg_shots = sum(shot_counts) / len(shot_counts)
        max_shots = max(shot_counts)
        min_shots = min(shot_counts)
        
        insights["content_diversity"] = {
            "avg_shot_count": avg_shots,
            "max_shot_count": max_shots,
            "min_shot_count": min_shots,
            "shot_count_variance": np.var(shot_counts) if len(shot_counts) > 1 else 0
        }
    
    # 참여도 요인 분석 (메타데이터 기반)
    view_counts = [video.get("view_count", 0) for video in insights["videos"]]
    like_counts = [video.get("like_count", 0) for video in insights["videos"]]
    
    if view_counts and like_counts:
        # 좋아요 대 조회수 비율 계산
        like_view_ratios = []
        
        for i, view_count in enumerate(view_counts):
            if view_count > 0:
                like_view_ratios.append(like_counts[i] / view_count)
        
        if like_view_ratios:
            avg_like_view_ratio = sum(like_view_ratios) / len(like_view_ratios)
            
            insights["engagement_factors"] = {
                "avg_view_count": sum(view_counts) / len(view_counts),
                "avg_like_count": sum(like_counts) / len(like_counts),
                "avg_like_view_ratio": avg_like_view_ratio,
                "max_views_video": insights["videos"][view_counts.index(max(view_counts))]["title"] if max(view_counts) > 0 else None,
                "max_likes_video": insights["videos"][like_counts.index(max(like_counts))]["title"] if max(like_counts) > 0 else None,
                "max_engagement_video": insights["videos"][like_view_ratios.index(max(like_view_ratios))]["title"] if like_view_ratios and max(like_view_ratios) > 0 else None
            }
    
    logger.info("인사이트 추출 완료")
    
    return insights

async def create_insight_visualizations(
    insights: Dict[str, Any],
    analysis_type: str,
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    인사이트에 기반한 시각화를 생성합니다.
    
    Parameters:
    -----------
    insights : Dict[str, Any]
        분석 인사이트 데이터
    analysis_type : str
        분석 유형
    output_dir : str
        시각화 파일을 저장할 디렉토리
        
    Returns:
    --------
    List[Dict[str, Any]]
        생성된 시각화 정보 목록
    """
    logger.info("인사이트 시각화 생성 시작")
    
    # 시각화 정보 저장 목록
    visualizations = []
    
    # 시각화 저장 디렉토리 확인
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # 1. 상위 레이블 막대 차트
        if "common_labels" in insights and insights["common_labels"].get("top_labels"):
            top_labels = insights["common_labels"]["top_labels"][:10]
            
            plt.figure(figsize=(10, 6))
            labels = [item["description"] for item in top_labels]
            frequencies = [item["frequency"] * 100 for item in top_labels]  # 백분율로 변환
            
            plt.barh(labels, frequencies, color='skyblue')
            plt.xlabel('출현 빈도 (%)')
            plt.ylabel('레이블')
            plt.title('상위 10개 콘텐츠 레이블')
            plt.tight_layout()
            
            # 파일 저장
            viz_file = os.path.join(viz_dir, "top_labels.png")
            plt.savefig(viz_file)
            plt.close()
            
            visualizations.append({
                "title": "상위 콘텐츠 레이블",
                "description": "분석된 비디오에서 가장 자주 나타나는 콘텐츠 레이블",
                "file_path": viz_file,
                "type": "bar_chart"
            })
        
        # 2. 상위 객체 막대 차트
        if "common_objects" in insights and insights["common_objects"].get("top_objects"):
            top_objects = insights["common_objects"]["top_objects"][:10]
            
            plt.figure(figsize=(10, 6))
            objects = [item["description"] for item in top_objects]
            frequencies = [item["frequency"] * 100 for item in top_objects]  # 백분율로 변환
            
            plt.barh(objects, frequencies, color='lightgreen')
            plt.xlabel('출현 빈도 (%)')
            plt.ylabel('객체')
            plt.title('상위 10개 감지된 객체')
            plt.tight_layout()
            
            # 파일 저장
            viz_file = os.path.join(viz_dir, "top_objects.png")
            plt.savefig(viz_file)
            plt.close()
            
            visualizations.append({
                "title": "상위 감지된 객체",
                "description": "분석된 비디오에서 가장 자주 나타나는 객체",
                "file_path": viz_file,
                "type": "bar_chart"
            })
        
        # 3. 안전 vs 성인 콘텐츠 원형 차트
        if "explicit_content" in insights and "adult_content_ratio" in insights["explicit_content"]:
            adult_ratio = insights["explicit_content"]["adult_content_ratio"]
            
            plt.figure(figsize=(8, 8))
            plt.pie(
                [1 - adult_ratio, adult_ratio],
                labels=['안전 콘텐츠', '성인 콘텐츠'],
                colors=['lightgreen', 'salmon'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0, 0.1)
            )
            plt.title('콘텐츠 안전성 분포')
            plt.axis('equal')  # 원형 차트를 위한 비율 조정
            
            # 파일 저장
            viz_file = os.path.join(viz_dir, "content_safety.png")
            plt.savefig(viz_file)
            plt.close()
            
            visualizations.append({
                "title": "콘텐츠 안전성 분포",
                "description": "안전한 콘텐츠와 성인 콘텐츠의 비율",
                "file_path": viz_file,
                "type": "pie_chart"
            })
        
        # 4. 샷 카운트 분포 히스토그램
        if "content_diversity" in insights and "avg_shot_count" in insights["content_diversity"]:
            shot_counts = [video.get("shot_count", 0) for video in insights["videos"] if "shot_count" in video]
            
            if shot_counts:
                plt.figure(figsize=(10, 6))
                plt.hist(shot_counts, bins=10, alpha=0.7, color='purple')
                plt.axvline(insights["content_diversity"]["avg_shot_count"], color='red', linestyle='dashed', linewidth=1)
                plt.text(insights["content_diversity"]["avg_shot_count"] * 1.1, plt.ylim()[1] * 0.9, f'평균: {insights["content_diversity"]["avg_shot_count"]:.1f}', color='red')
                plt.xlabel('샷 수')
                plt.ylabel('비디오 수')
                plt.title('비디오별 샷 수 분포')
                plt.grid(axis='y', alpha=0.75)
                
                # 파일 저장
                viz_file = os.path.join(viz_dir, "shot_count_distribution.png")
                plt.savefig(viz_file)
                plt.close()
                
                visualizations.append({
                    "title": "비디오별 샷 수 분포",
                    "description": "각 비디오에서 감지된 샷의 수 분포",
                    "file_path": viz_file,
                    "type": "histogram"
                })
        
        # 5. 참여도 요인 막대 차트
        if "videos" in insights and len(insights["videos"]) > 0:
            # 조회수와 좋아요 수가 있는 비디오만 선택
            filtered_videos = [v for v in insights["videos"] if v.get("view_count", 0) > 0 and v.get("like_count", 0) > 0]
            
            if filtered_videos:
                # 상위 5개 비디오 선택 (조회수 기준)
                top_videos = sorted(filtered_videos, key=lambda x: x.get("view_count", 0), reverse=True)[:5]
                
                # 비디오 제목 및 데이터 준비
                titles = [v.get("title", "")[:20] + "..." if len(v.get("title", "")) > 20 else v.get("title", "") for v in top_videos]
                view_counts = [v.get("view_count", 0) for v in top_videos]
                like_counts = [v.get("like_count", 0) for v in top_videos]
                
                # 참여율 계산 (좋아요/조회수)
                engagement_rates = [(like / view) * 100 for like, view in zip(like_counts, view_counts)]
                
                plt.figure(figsize=(12, 6))
                
                x = np.arange(len(titles))
                width = 0.35
                
                # 첫 번째 축: 조회수 (로그 스케일)
                ax1 = plt.subplot(111)
                bars1 = ax1.bar(x - width/2, view_counts, width, color='skyblue', label='조회수')
                ax1.set_yscale('log')
                ax1.set_ylabel('조회수 (로그 스케일)')
                
                # 두 번째 축: 참여율
                ax2 = ax1.twinx()
                bars2 = ax2.bar(x + width/2, engagement_rates, width, color='salmon', label='참여율 (%)')
                ax2.set_ylabel('참여율 (%)')
                
                # 레이블 및 제목
                ax1.set_xticks(x)
                ax1.set_xticklabels(titles, rotation=45, ha='right')
                plt.title('상위 5개 비디오의 조회수 및 참여율')
                plt.tight_layout()
                
                # 범례
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                # 파일 저장
                viz_file = os.path.join(viz_dir, "engagement_factors.png")
                plt.savefig(viz_file)
                plt.close()
                
                visualizations.append({
                    "title": "상위 비디오 참여도 분석",
                    "description": "가장 많은 조회수를 가진 비디오의 참여율 비교",
                    "file_path": viz_file,
                    "type": "bar_chart"
                })
        
        logger.info(f"총 {len(visualizations)}개 시각화 생성 완료")
        return visualizations
    
    except Exception as e:
        logger.error(f"시각화 생성 중 오류 발생: {str(e)}")
        return visualizations

async def generate_html_report(
    report: Dict[str, Any],
    output_dir: str
) -> str:
    """
    인사이트 보고서의 HTML 버전을 생성합니다.
    
    Parameters:
    -----------
    report : Dict[str, Any]
        인사이트 보고서 데이터
    output_dir : str
        HTML 파일을 저장할 디렉토리
        
    Returns:
    --------
    str
        생성된 HTML 파일 경로
    """
    logger.info("HTML 보고서 생성 시작")
    
    try:
        # 타임스탬프
        timestamp = report.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # 분석 유형
        analysis_type = report.get("analysis_type", "comprehensive")
        
        # 모드 (특정 비디오 또는 검색 쿼리)
        mode = report.get("mode", "specific_videos")
        
        # HTML 템플릿 생성
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YouTube 인사이트 보고서 - {timestamp}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                header {{
                    background-color: #ff0000;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h1, h2, h3 {{
                    margin-top: 30px;
                }}
                .card {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .flex-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .flex-item {{
                    flex: 1;
                    min-width: 300px;
                }}
                .stats {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    gap: 10px;
                }}
                .stat-item {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    text-align: center;
                    flex: 1;
                    min-width: 150px;
                }}
                .stat-number {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #ff0000;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                footer {{
                    background-color: #f8f9fa;
                    color: #555;
                    text-align: center;
                    padding: 10px;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>YouTube 인사이트 보고서</h1>
                <p>생성 시간: {timestamp}</p>
            </header>
            
            <div class="container">
                <div class="card">
                    <h2>보고서 개요</h2>
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{report.get('analysis_results', {}).get('total_videos', 0)}</div>
                            <div>분석된 비디오</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{report.get('analysis_results', {}).get('successful', 0)}</div>
                            <div>성공한 분석</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{report.get('analysis_results', {}).get('failed', 0)}</div>
                            <div>실패한 분석</div>
                        </div>
                    </div>
                    
                    <h3>분석 정보</h3>
                    <p><strong>분석 유형:</strong> {analysis_type}</p>
                    <p><strong>분석 모드:</strong> {mode}</p>
        """
        
        # 검색 쿼리 또는 비디오 ID 정보 추가
        if mode == "search_query" and "search_query" in report:
            html_content += f'<p><strong>검색 쿼리:</strong> {report["search_query"]}</p>'
        elif mode == "specific_videos" and "video_ids" in report:
            video_ids = ', '.join(report["video_ids"])
            html_content += f'<p><strong>비디오 ID:</strong> {video_ids}</p>'
        
        html_content += """
                </div>
        """
        
        # 시각화 섹션
        if "visualizations" in report and report["visualizations"]:
            html_content += """
                <div class="card">
                    <h2>인사이트 시각화</h2>
                    <div class="flex-container">
            """
            
            for viz in report["visualizations"]:
                title = viz.get("title", "시각화")
                description = viz.get("description", "")
                file_path = viz.get("file_path", "")
                
                # 상대 경로로 변환
                rel_path = os.path.relpath(file_path, output_dir)
                
                html_content += f"""
                        <div class="flex-item">
                            <div class="visualization">
                                <h3>{title}</h3>
                                <p>{description}</p>
                                <img src="{rel_path}" alt="{title}">
                            </div>
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
            """
        
        # 인사이트 섹션
        insights = report.get("insights", {})
        
        # 공통 레이블 인사이트
        if "common_labels" in insights and insights["common_labels"].get("top_labels"):
            html_content += """
                <div class="card">
                    <h2>콘텐츠 레이블 인사이트</h2>
            """
            
            # 레이블 다양성 통계
            label_diversity = insights["common_labels"].get("label_diversity", 0)
            unique_labels = insights["common_labels"].get("unique_labels_count", 0)
            
            html_content += f"""
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{unique_labels}</div>
                            <div>고유 레이블 수</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{label_diversity:.2f}</div>
                            <div>레이블 다양성 지수</div>
                        </div>
                    </div>
                    
                    <h3>상위 레이블</h3>
                    <table>
                        <tr>
                            <th>레이블</th>
                            <th>출현 빈도</th>
                            <th>평균 신뢰도</th>
                        </tr>
            """
            
            for label in insights["common_labels"]["top_labels"][:10]:
                html_content += f"""
                        <tr>
                            <td>{label.get("description", "")}</td>
                            <td>{label.get("frequency", 0) * 100:.1f}%</td>
                            <td>{label.get("avg_confidence", 0) * 100:.1f}%</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # 공통 객체 인사이트
        if "common_objects" in insights and insights["common_objects"].get("top_objects"):
            html_content += """
                <div class="card">
                    <h2>객체 감지 인사이트</h2>
            """
            
            # 객체 다양성 통계
            object_diversity = insights["common_objects"].get("object_diversity", 0)
            unique_objects = insights["common_objects"].get("unique_objects_count", 0)
            
            html_content += f"""
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{unique_objects}</div>
                            <div>고유 객체 수</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{object_diversity:.2f}</div>
                            <div>객체 다양성 지수</div>
                        </div>
                    </div>
                    
                    <h3>상위 객체</h3>
                    <table>
                        <tr>
                            <th>객체</th>
                            <th>출현 빈도</th>
                            <th>평균 신뢰도</th>
                        </tr>
            """
            
            for obj in insights["common_objects"]["top_objects"][:10]:
                html_content += f"""
                        <tr>
                            <td>{obj.get("description", "")}</td>
                            <td>{obj.get("frequency", 0) * 100:.1f}%</td>
                            <td>{obj.get("avg_confidence", 0) * 100:.1f}%</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # 콘텐츠 안전성 인사이트
        if "explicit_content" in insights:
            adult_ratio = insights["explicit_content"].get("adult_content_ratio", 0)
            adult_videos = insights["explicit_content"].get("adult_content_videos", 0)
            safe_videos = insights["explicit_content"].get("safe_content_videos", 0)
            
            html_content += f"""
                <div class="card">
                    <h2>콘텐츠 안전성 인사이트</h2>
                    
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{adult_ratio * 100:.1f}%</div>
                            <div>성인 콘텐츠 비율</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{adult_videos}</div>
                            <div>성인 콘텐츠 비디오</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{safe_videos}</div>
                            <div>안전한 콘텐츠 비디오</div>
                        </div>
                    </div>
                </div>
            """
        
        # 콘텐츠 다양성 인사이트
        if "content_diversity" in insights:
            avg_shots = insights["content_diversity"].get("avg_shot_count", 0)
            max_shots = insights["content_diversity"].get("max_shot_count", 0)
            min_shots = insights["content_diversity"].get("min_shot_count", 0)
            shot_variance = insights["content_diversity"].get("shot_count_variance", 0)
            
            html_content += f"""
                <div class="card">
                    <h2>콘텐츠 다양성 인사이트</h2>
                    
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{avg_shots:.1f}</div>
                            <div>평균 샷 수</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{max_shots}</div>
                            <div>최대 샷 수</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{min_shots}</div>
                            <div>최소 샷 수</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{shot_variance:.1f}</div>
                            <div>샷 수 분산</div>
                        </div>
                    </div>
                </div>
            """
        
        # 참여도 요인 인사이트
        if "engagement_factors" in insights:
            avg_views = insights["engagement_factors"].get("avg_view_count", 0)
            avg_likes = insights["engagement_factors"].get("avg_like_count", 0)
            avg_ratio = insights["engagement_factors"].get("avg_like_view_ratio", 0)
            max_views_video = insights["engagement_factors"].get("max_views_video", "")
            max_likes_video = insights["engagement_factors"].get("max_likes_video", "")
            
            html_content += f"""
                <div class="card">
                    <h2>참여도 인사이트</h2>
                    
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{avg_views:,.0f}</div>
                            <div>평균 조회수</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{avg_likes:,.0f}</div>
                            <div>평균 좋아요</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{avg_ratio * 100:.2f}%</div>
                            <div>평균 참여율</div>
                        </div>
                    </div>
                    
                    <h3>주목할 비디오</h3>
                    <ul>
                        <li><strong>최다 조회수:</strong> {max_views_video}</li>
                        <li><strong>최다 좋아요:</strong> {max_likes_video}</li>
                    </ul>
                </div>
            """
        
        # 비디오 목록
        if "videos" in insights and insights["videos"]:
            html_content += """
                <div class="card">
                    <h2>분석된 비디오</h2>
                    <table>
                        <tr>
                            <th>제목</th>
                            <th>비디오 ID</th>
                            <th>조회수</th>
                            <th>좋아요</th>
                        </tr>
            """
            
            for video in insights["videos"]:
                video_id = video.get("video_id", "")
                title = video.get("title", "")
                views = video.get("view_count", 0)
                likes = video.get("like_count", 0)
                
                html_content += f"""
                        <tr>
                            <td>{title}</td>
                            <td><a href="https://youtube.com/watch?v={video_id}" target="_blank">{video_id}</a></td>
                            <td>{views:,}</td>
                            <td>{likes:,}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # HTML 마무리
        html_content += """
            </div>
            
            <footer>
                <p>YouTube Data API v3 및 Cloud Video Intelligence API를 사용하여 생성된 보고서</p>
                <p>© 2025 YouTube 인사이트 시스템</p>
            </footer>
        </body>
        </html>
        """
        
        # HTML 파일 저장
        if mode == "search_query" and "search_query" in report:
            search_query_safe = report["search_query"].replace('/', '_').replace('\\', '_').replace(' ', '_')
            html_file = os.path.join(output_dir, f"report_{search_query_safe}_{timestamp}.html")
        else:
            html_file = os.path.join(output_dir, f"report_{timestamp}.html")
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML 보고서 생성 완료: {html_file}")
        
        return html_file
    
    except Exception as e:
        logger.error(f"HTML 보고서 생성 중 오류 발생: {str(e)}")
        return ""

# 도구 테스트 함수 (직접 실행 시 사용)
async def test_youtube_insights_system():
    """YouTube 인사이트 시스템을 테스트합니다."""
    # 방법 1: 특정 비디오 ID 목록으로 테스트
    video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0"]  # 테스트용 비디오 ID
    
    print(f"{len(video_ids)}개 비디오에 대한 인사이트 생성 테스트:")
    result = await generate_youtube_insights(
        video_ids=video_ids,
        analysis_type="basic"  # 빠른 테스트를 위해 기본 분석만 수행
    )
    
    if result.get("success", False):
        print(f"인사이트 생성 성공!")
        print(f"보고서 파일: {result.get('report_file')}")
        print(f"HTML 보고서: {result.get('html_report')}")
    else:
        print(f"인사이트 생성 실패: {result.get('error')}")
    
    print("-" * 50)
    
    # 방법 2: 검색 쿼리를 통한 테스트
    search_query = "machine learning short tutorial"
    
    print(f"'{search_query}' 검색 기반 인사이트 생성 테스트:")
    result = await generate_youtube_insights(
        search_query=search_query,
        max_videos=2,  # 테스트를 위해 적은 수의 비디오만 사용
        analysis_type="basic"  # 빠른 테스트를 위해 기본 분석만 수행
    )
    
    if result.get("success", False):
        print(f"인사이트 생성 성공!")
        print(f"보고서 파일: {result.get('report_file')}")
        print(f"HTML 보고서: {result.get('html_report')}")
    else:
        print(f"인사이트 생성 실패: {result.get('error')}")

# 모듈 직접 실행 시 테스트 수행
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_youtube_insights_system())

"""
Google Cloud Video Intelligence API 연동 모듈
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from google.cloud import videointelligence
from google.cloud import storage
from google.protobuf.json_format import MessageToDict

from analysis.mcp_init import mcp
# config 패키지에서 직접 임포트
from config.youtube_config import PROJECT_ID, GCS_BUCKET, SERVICE_ACCOUNT_FILE

# 서비스 계정 키 파일 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("video_intelligence_api")

@mcp.tool(
    name="upload_video_to_gcs",
    description="로컬 비디오 파일을 Google Cloud Storage에 업로드합니다",
    tags=["gcs", "upload", "video"]
)
async def upload_video_to_gcs(
    file_path: str,
    bucket_name: Optional[str] = None,
    destination_blob_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    로컬 비디오 파일을 Google Cloud Storage에 업로드합니다.
    
    Parameters:
    -----------
    file_path : str
        업로드할 로컬 비디오 파일 경로
    bucket_name : str, optional
        대상 GCS 버킷 이름 (기본값: GCS_BUCKET)
    destination_blob_name : str, optional
        대상 GCS 경로 (기본값: 파일 이름)
        
    Returns:
    --------
    Dict[str, Any]
        업로드 결과 정보를 포함하는 딕셔너리
    """
    if bucket_name is None:
        bucket_name = GCS_BUCKET
    
    if destination_blob_name is None:
        destination_blob_name = os.path.basename(file_path)
    
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"파일이 존재하지 않습니다: {file_path}"
            }
        
        # Storage 클라이언트 초기화
        storage_client = storage.Client()
        
        # 버킷이 존재하는지 확인, 없으면 생성
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except Exception:
            bucket = storage_client.create_bucket(bucket_name)
            logger.info(f"버킷 생성: {bucket_name}")
        
        # Blob 객체 생성 및 파일 업로드
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        
        # GCS URI 생성
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        
        logger.info(f"'{file_path}'를 '{gcs_uri}'로 업로드 성공")
        
        return {
            "success": True,
            "file_path": file_path,
            "bucket_name": bucket_name,
            "blob_name": destination_blob_name,
            "gcs_uri": gcs_uri
        }
    
    except Exception as e:
        error_message = f"GCS 업로드 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }

@mcp.tool(
    name="analyze_video_content",
    description="Google Cloud Video Intelligence API를 사용하여 비디오 콘텐츠를 분석합니다",
    tags=["video", "analysis", "ai"]
)
async def analyze_video_content(
    gcs_uri: str,
    features: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Google Cloud Video Intelligence API를 사용하여 비디오 콘텐츠를 분석합니다.
    
    Parameters:
    -----------
    gcs_uri : str
        분석할 비디오의 Google Cloud Storage URI (gs://bucket_name/file_name)
    features : List[str], optional
        분석할 기능 목록 (기본값: ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION"])
        옵션: "LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "EXPLICIT_CONTENT_DETECTION",
              "OBJECT_TRACKING", "TEXT_DETECTION", "LOGO_RECOGNITION", "FACE_DETECTION",
              "PERSON_DETECTION", "SPEECH_TRANSCRIPTION"
    output_file : str, optional
        분석 결과를 저장할 JSON 파일 경로
        
    Returns:
    --------
    Dict[str, Any]
        비디오 분석 결과를 포함하는 딕셔너리
    """
    # 기본 분석 기능 설정
    if features is None:
        features = ["LABEL_DETECTION", "SHOT_CHANGE_DETECTION"]
    
    try:
        # Video Intelligence 클라이언트 초기화
        client = videointelligence.VideoIntelligenceServiceClient()
        
        # 기능 열거형 변환
        feature_enum_map = {
            "LABEL_DETECTION": videointelligence.Feature.LABEL_DETECTION,
            "SHOT_CHANGE_DETECTION": videointelligence.Feature.SHOT_CHANGE_DETECTION,
            "EXPLICIT_CONTENT_DETECTION": videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
            "OBJECT_TRACKING": videointelligence.Feature.OBJECT_TRACKING,
            "TEXT_DETECTION": videointelligence.Feature.TEXT_DETECTION,
            "LOGO_RECOGNITION": videointelligence.Feature.LOGO_RECOGNITION,
            "FACE_DETECTION": videointelligence.Feature.FACE_DETECTION,
            "PERSON_DETECTION": videointelligence.Feature.PERSON_DETECTION,
            "SPEECH_TRANSCRIPTION": videointelligence.Feature.SPEECH_TRANSCRIPTION
        }
        
        video_features = [feature_enum_map[f] for f in features if f in feature_enum_map]
        
        if not video_features:
            return {
                "success": False,
                "error": "유효한 분석 기능이 지정되지 않았습니다."
            }
        
        # 음성 인식이 포함된 경우 추가 설정
        if videointelligence.Feature.SPEECH_TRANSCRIPTION in video_features:
            config = videointelligence.SpeechTranscriptionConfig(
                language_code="ko-KR",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=True,
                diarization_speaker_count=2
            )
            context = videointelligence.VideoContext(speech_transcription_config=config)
        else:
            context = None
        
        # 분석 요청
        logger.info(f"'{gcs_uri}' 분석 시작 (기능: {', '.join(features)})")
        
        # 분석 요청 생성
        request = {
            "input_uri": gcs_uri,
            "features": video_features
        }
        
        if context:
            request["video_context"] = context
        
        # 분석 실행 (비동기)
        operation = client.annotate_video(request=request)
        
        logger.info("비디오 분석 중... (완료까지 몇 분 소요될 수 있습니다)")
        result = operation.result(timeout=1800)  # 최대 30분 대기
        
        # 프로토버프 메시지를 딕셔너리로 변환
        result_dict = MessageToDict(result._pb)
        
        # 분석 결과 요약 생성
        summary = create_analysis_summary(result_dict)
        
        # 결과를 JSON 파일로 저장 (지정된 경우)
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"분석 결과 저장 완료: {output_file}")
        
        logger.info(f"'{gcs_uri}' 분석 완료")
        
        return {
            "success": True,
            "gcs_uri": gcs_uri,
            "features": features,
            "summary": summary,
            "raw_result": result_dict,
            "output_file": output_file
        }
    
    except Exception as e:
        error_message = f"비디오 분석 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }

def create_analysis_summary(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Video Intelligence API 분석 결과에서 주요 정보를 추출하여 요약합니다.
    
    Parameters:
    -----------
    result_dict : Dict[str, Any]
        분석 결과 딕셔너리
        
    Returns:
    --------
    Dict[str, Any]
        요약된 분석 결과
    """
    summary = {}
    
    # 주석 결과가 있는지 확인
    annotation_results = result_dict.get("annotationResults", [])
    if not annotation_results or len(annotation_results) == 0:
        return {"message": "분석 결과가 없습니다."}
    
    # 첫 번째 주석 결과 사용
    annotation = annotation_results[0]
    
    # 레이블 분석 요약
    if "segmentLabelAnnotations" in annotation:
        labels = []
        for label in annotation["segmentLabelAnnotations"]:
            if "entity" in label:
                labels.append({
                    "description": label["entity"].get("description", ""),
                    "confidence": label.get("confidence", 0),
                    "segments": len(label.get("segments", []))
                })
        
        # 신뢰도 기준 내림차순 정렬
        labels.sort(key=lambda x: x["confidence"], reverse=True)
        summary["labels"] = labels[:10]  # 상위 10개만 포함
    
    # 샷 변경 감지 요약
    if "shotAnnotations" in annotation:
        shots = len(annotation["shotAnnotations"])
        summary["shots"] = {
            "count": shots,
            "details": [
                {
                    "start": shot.get("startTimeOffset", {}).get("seconds", 0),
                    "end": shot.get("endTimeOffset", {}).get("seconds", 0)
                }
                for shot in annotation["shotAnnotations"][:5]  # 처음 5개만 포함
            ]
        }
    
    # 객체 추적 요약
    if "objectAnnotations" in annotation:
        objects = {}
        for obj in annotation["objectAnnotations"]:
            if "entity" in obj:
                description = obj["entity"].get("description", "unknown")
                confidence = obj.get("confidence", 0)
                
                if description not in objects:
                    objects[description] = {
                        "count": 0,
                        "avg_confidence": 0,
                        "appearances": []
                    }
                
                objects[description]["count"] += 1
                objects[description]["avg_confidence"] = (
                    (objects[description]["avg_confidence"] * (objects[description]["count"] - 1) + confidence) / 
                    objects[description]["count"]
                )
                
                # 프레임 정보 추가 (최대 3개)
                if len(objects[description]["appearances"]) < 3:
                    if "frames" in obj and len(obj["frames"]) > 0:
                        frame = obj["frames"][0]
                        if "timeOffset" in frame:
                            objects[description]["appearances"].append(
                                frame["timeOffset"].get("seconds", 0)
                            )
        
        # 객체 목록 생성 (신뢰도 기준 정렬)
        object_list = [
            {
                "description": desc,
                "count": info["count"],
                "confidence": info["avg_confidence"],
                "appearances": info["appearances"]
            }
            for desc, info in objects.items()
        ]
        object_list.sort(key=lambda x: x["confidence"], reverse=True)
        summary["objects"] = object_list[:10]  # 상위 10개만 포함
    
    # 텍스트 감지 요약
    if "textAnnotations" in annotation:
        texts = []
        for text in annotation["textAnnotations"]:
            if "text" in text:
                texts.append({
                    "text": text["text"],
                    "confidence": text.get("confidence", 0),
                    "segments": len(text.get("segments", []))
                })
        summary["texts"] = texts[:10]  # 상위 10개만 포함
    
    # 음성 인식 요약
    if "speechTranscriptions" in annotation:
        transcriptions = []
        for trans in annotation["speechTranscriptions"]:
            if "alternatives" in trans and len(trans["alternatives"]) > 0:
                alt = trans["alternatives"][0]
                if "transcript" in alt:
                    transcriptions.append({
                        "transcript": alt["transcript"],
                        "confidence": alt.get("confidence", 0)
                    })
        summary["speech"] = transcriptions
    
    # 명시적 콘텐츠 감지 요약
    if "explicitAnnotation" in annotation:
        explicit_frames = annotation["explicitAnnotation"].get("frames", [])
        if explicit_frames:
            summary["explicit_content"] = {
                "frame_count": len(explicit_frames),
                "adult_content_detected": any(
                    frame.get("pornographyLikelihood", "") in ["LIKELY", "VERY_LIKELY"] 
                    for frame in explicit_frames
                )
            }
    
    return summary

@mcp.tool(
    name="get_video_highlights",
    description="비디오 분석 결과를 기반으로 하이라이트 구간을 추출합니다",
    tags=["video", "highlights", "analysis"]
)
async def get_video_highlights(
    analysis_result: Dict[str, Any],
    max_highlights: int = 5,
    min_duration: int = 5,
    max_duration: int = 30
) -> Dict[str, Any]:
    """
    비디오 분석 결과를 기반으로 하이라이트 구간을 추출합니다.
    
    Parameters:
    -----------
    analysis_result : Dict[str, Any]
        analyze_video_content 함수의 결과
    max_highlights : int, optional
        추출할 최대 하이라이트 수 (기본값: 5)
    min_duration : int, optional
        하이라이트 구간의 최소 길이(초) (기본값: 5)
    max_duration : int, optional
        하이라이트 구간의 최대 길이(초) (기본값: 30)
        
    Returns:
    --------
    Dict[str, Any]
        추출된 하이라이트 정보를 포함하는 딕셔너리
    """
    try:
        if not analysis_result.get("success", False):
            return {
                "success": False,
                "error": "유효한 분석 결과가 아닙니다."
            }
        
        raw_result = analysis_result.get("raw_result", {})
        annotation_results = raw_result.get("annotationResults", [])
        
        if not annotation_results:
            return {
                "success": False,
                "error": "분석 결과에 주석이 없습니다."
            }
        
        # 첫 번째 주석 결과 사용
        annotation = annotation_results[0]
        
        # 샷 정보 추출
        shots = []
        if "shotAnnotations" in annotation:
            for shot in annotation["shotAnnotations"]:
                start_time = int(shot.get("startTimeOffset", {}).get("seconds", 0))
                end_time = int(shot.get("endTimeOffset", {}).get("seconds", 0))
                duration = end_time - start_time
                
                # 유효한 길이의 샷만 포함
                if min_duration <= duration <= max_duration:
                    shots.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                        "score": 0  # 초기 점수
                    })
        
        # 객체 정보로 샷 점수 계산
        if "objectAnnotations" in annotation:
            for obj in annotation["objectAnnotations"]:
                confidence = obj.get("confidence", 0)
                if "frames" in obj:
                    for frame in obj["frames"]:
                        if "timeOffset" in frame:
                            frame_time = int(frame["timeOffset"].get("seconds", 0))
                            
                            # 해당 시간을 포함하는 샷 찾기
                            for shot in shots:
                                if shot["start"] <= frame_time <= shot["end"]:
                                    # 객체 신뢰도에 따라 점수 증가
                                    shot["score"] += confidence
        
        # 레이블 정보로 샷 점수 보완
        if "segmentLabelAnnotations" in annotation:
            for label in annotation["segmentLabelAnnotations"]:
                label_confidence = label.get("confidence", 0)
                if "segments" in label:
                    for segment in label["segments"]:
                        if "segment" in segment:
                            seg_start = int(segment["segment"].get("startTimeOffset", {}).get("seconds", 0))
                            seg_end = int(segment["segment"].get("endTimeOffset", {}).get("seconds", 0))
                            
                            # 해당 세그먼트와 겹치는 샷 찾기
                            for shot in shots:
                                # 세그먼트와 샷이 겹치는 경우
                                if not (shot["end"] < seg_start or shot["start"] > seg_end):
                                    # 레이블 신뢰도에 따라 점수 증가
                                    shot["score"] += label_confidence * 0.5
        
        # 점수 기준 내림차순 정렬
        shots.sort(key=lambda x: x["score"], reverse=True)
        
        # 상위 하이라이트 선택
        highlights = shots[:max_highlights]
        
        # 하이라이트에 연관 객체 및 레이블 추가
        for highlight in highlights:
            highlight["objects"] = []
            highlight["labels"] = []
            
            # 객체 정보 추가
            if "objectAnnotations" in annotation:
                for obj in annotation["objectAnnotations"]:
                    if "entity" in obj and "frames" in obj:
                        obj_description = obj["entity"].get("description", "")
                        obj_confidence = obj.get("confidence", 0)
                        
                        # 이 하이라이트 구간에 나타나는 객체인지 확인
                        obj_in_highlight = False
                        for frame in obj["frames"]:
                            if "timeOffset" in frame:
                                frame_time = int(frame["timeOffset"].get("seconds", 0))
                                if highlight["start"] <= frame_time <= highlight["end"]:
                                    obj_in_highlight = True
                                    break
                        
                        if obj_in_highlight and obj_description:
                            # 이미 추가된 객체인지 확인
                            existing_obj = next((o for o in highlight["objects"] if o["description"] == obj_description), None)
                            if existing_obj:
                                existing_obj["count"] += 1
                            else:
                                highlight["objects"].append({
                                    "description": obj_description,
                                    "confidence": obj_confidence,
                                    "count": 1
                                })
            
            # 레이블 정보 추가
            if "segmentLabelAnnotations" in annotation:
                for label in annotation["segmentLabelAnnotations"]:
                    if "entity" in label and "segments" in label:
                        label_description = label["entity"].get("description", "")
                        label_confidence = label.get("confidence", 0)
                        
                        # 이 하이라이트 구간과 겹치는 레이블인지 확인
                        label_in_highlight = False
                        for segment in label["segments"]:
                            if "segment" in segment:
                                seg_start = int(segment["segment"].get("startTimeOffset", {}).get("seconds", 0))
                                seg_end = int(segment["segment"].get("endTimeOffset", {}).get("seconds", 0))
                                
                                # 세그먼트와 하이라이트 구간이 겹치는 경우
                                if not (highlight["end"] < seg_start or highlight["start"] > seg_end):
                                    label_in_highlight = True
                                    break
                        
                        if label_in_highlight and label_description:
                            # 이미 추가된 레이블인지 확인
                            existing_label = next((l for l in highlight["labels"] if l["description"] == label_description), None)
                            if not existing_label:
                                highlight["labels"].append({
                                    "description": label_description,
                                    "confidence": label_confidence
                                })
            
            # 객체와 레이블 정렬
            highlight["objects"].sort(key=lambda x: x["confidence"], reverse=True)
            highlight["labels"].sort(key=lambda x: x["confidence"], reverse=True)
        
        logger.info(f"하이라이트 추출 완료: {len(highlights)}개 구간")
        
        return {
            "success": True,
            "highlights_count": len(highlights),
            "highlights": highlights
        }
    
    except Exception as e:
        error_message = f"하이라이트 추출 중 오류 발생: {str(e)}"
        logger.error(error_message)
        return {
            "success": False,
            "error": error_message
        }

# 도구 테스트 함수 (직접 실행 시 사용)
async def test_video_intelligence():
    """Video Intelligence API 도구를 테스트합니다."""
    # 경고: 실제 서비스 계정 키가 필요합니다
    
    # 업로드 테스트 (로컬 파일이 있다고 가정)
    test_video_path = "test_video.mp4"
    
    if os.path.exists(test_video_path):
        print(f"'{test_video_path}' GCS 업로드 테스트:")
        upload_result = await upload_video_to_gcs(test_video_path)
        
        if upload_result.get("success", False):
            print(f"업로드 성공: {upload_result.get('gcs_uri')}")
            
            # 분석 테스트
            print("\n비디오 분석 테스트:")
            analysis_result = await analyze_video_content(
                upload_result.get("gcs_uri"),
                features=["LABEL_DETECTION", "SHOT_CHANGE_DETECTION", "OBJECT_TRACKING"]
            )
            
            if analysis_result.get("success", False):
                print("분석 성공!")
                
                # 요약 출력
                summary = analysis_result.get("summary", {})
                
                if "labels" in summary:
                    print("\n상위 레이블:")
                    for label in summary["labels"][:5]:
                        print(f"- {label['description']} ({label['confidence']:.2f})")
                
                if "shots" in summary:
                    print(f"\n감지된 샷 수: {summary['shots']['count']}")
                
                if "objects" in summary:
                    print("\n감지된 객체:")
                    for obj in summary["objects"][:5]:
                        print(f"- {obj['description']} ({obj['confidence']:.2f})")
                
                # 하이라이트 추출 테스트
                print("\n하이라이트 추출 테스트:")
                highlights_result = await get_video_highlights(analysis_result)
                
                if highlights_result.get("success", False):
                    print(f"하이라이트 추출 성공: {highlights_result.get('highlights_count')}개 구간")
                    
                    for i, highlight in enumerate(highlights_result.get("highlights", [])):
                        print(f"\n하이라이트 #{i+1} ({highlight['start']}s - {highlight['end']}s):")
                        print(f"- 점수: {highlight['score']:.2f}")
                        print(f"- 객체: {', '.join([obj['description'] for obj in highlight['objects'][:3]])}")
                        print(f"- 레이블: {', '.join([label['description'] for label in highlight['labels'][:3]])}")
                else:
                    print(f"하이라이트 추출 실패: {highlights_result.get('error')}")
            else:
                print(f"분석 실패: {analysis_result.get('error')}")
        else:
            print(f"업로드 실패: {upload_result.get('error')}")
    else:
        print(f"테스트 비디오 파일이 없습니다: {test_video_path}")
        print("테스트를 건너뜁니다.")

# 모듈 직접 실행 시 테스트 수행
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_video_intelligence())

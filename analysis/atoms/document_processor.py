"""
문서 처리 기본 모듈
"""

import os
import sys
import logging
import json
import re
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

logger = logging.getLogger(__name__)

def scan_documents(input_path: str, file_patterns: List[str] = None) -> Dict[str, Any]:
    """
    지정된 경로에서 문서 스캔
    
    Args:
        input_path (str): 입력 문서 경로 (파일 또는 디렉토리)
        file_patterns (List[str], optional): 포함할 파일 패턴 목록 (예: ["*.pdf", "*.docx"])
    
    Returns:
        Dict[str, Any]: 스캔 결과
    """
    if file_patterns is None:
        file_patterns = ["*.txt", "*.pdf", "*.docx", "*.md", "*.json", "*.html"]
    
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_path": input_path,
        "file_patterns": file_patterns,
        "documents": [],
        "stats": {
            "total_files": 0,
            "by_extension": {},
            "by_size_range": {
                "small (<100KB)": 0,
                "medium (100KB-1MB)": 0,
                "large (1MB-10MB)": 0,
                "very_large (>10MB)": 0
            }
        }
    }
    
    try:
        # 입력 경로가 파일인 경우
        if os.path.isfile(input_path):
            file_info = _get_file_info(input_path)
            result["documents"].append(file_info)
            
            # 통계 업데이트
            result["stats"]["total_files"] = 1
            
            ext = file_info["extension"].lower()
            result["stats"]["by_extension"][ext] = 1
            
            size_range = _get_size_range(file_info["size_bytes"])
            result["stats"]["by_size_range"][size_range] = 1
        
        # 입력 경로가 디렉토리인 경우
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    # 패턴 매칭 확인
                    if not any(fnmatch(file, pattern) for pattern in file_patterns):
                        continue
                    
                    file_path = os.path.join(root, file)
                    file_info = _get_file_info(file_path)
                    result["documents"].append(file_info)
                    
                    # 통계 업데이트
                    result["stats"]["total_files"] += 1
                    
                    ext = file_info["extension"].lower()
                    result["stats"]["by_extension"][ext] = result["stats"]["by_extension"].get(ext, 0) + 1
                    
                    size_range = _get_size_range(file_info["size_bytes"])
                    result["stats"]["by_size_range"][size_range] += 1
        else:
            logger.warning(f"입력 경로가 존재하지 않음: {input_path}")
            result["error"] = f"입력 경로가 존재하지 않음: {input_path}"
        
        logger.info(f"문서 스캔 완료: {result['stats']['total_files']}개 파일 발견")
        return result
    except Exception as e:
        logger.error(f"문서 스캔 중 오류: {str(e)}")
        result["error"] = str(e)
        return result

def process_document_text(scan_result: Dict[str, Any] = None, output_path: str = None, **kwargs) -> Dict[str, Any]:
    """
    문서 텍스트 처리
    
    Args:
        scan_result (Dict[str, Any], optional): 스캔 결과 (None이면 XCom에서 가져옴)
        output_path (str, optional): 출력 디렉토리 경로
        **kwargs: 추가 인수 (Airflow 컨텍스트)
    
    Returns:
        Dict[str, Any]: 처리 결과
    """
    # XCom에서 스캔 결과 가져오기
    if scan_result is None and 'ti' in kwargs:
        scan_result = kwargs['ti'].xcom_pull(task_ids='scan_documents')
    
    if not scan_result or not isinstance(scan_result, dict) or "documents" not in scan_result:
        error_msg = "유효한 스캔 결과를 찾을 수 없음"
        logger.error(error_msg)
        return {"error": error_msg}
    
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "processed_documents": [],
        "stats": {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_extension": {}
        }
    }
    
    try:
        for doc in scan_result["documents"]:
            doc_result = {
                "path": doc["path"],
                "processed": False,
                "extracted_text": None,
                "error": None
            }
            
            try:
                # 파일 확장자에 따라 다른 처리 방법 적용
                extension = doc["extension"].lower()
                
                if extension == ".txt":
                    with open(doc["path"], 'r', encoding='utf-8', errors='ignore') as f:
                        doc_result["extracted_text"] = f.read()
                    doc_result["processed"] = True
                
                elif extension == ".md":
                    with open(doc["path"], 'r', encoding='utf-8', errors='ignore') as f:
                        doc_result["extracted_text"] = f.read()
                    doc_result["processed"] = True
                
                elif extension == ".json":
                    with open(doc["path"], 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                    
                    # JSON 데이터를 문자열로 변환
                    if isinstance(data, (dict, list)):
                        doc_result["extracted_text"] = json.dumps(data, indent=2, ensure_ascii=False)
                    else:
                        doc_result["extracted_text"] = str(data)
                    
                    doc_result["processed"] = True
                
                elif extension == ".html":
                    # HTML 파일에서 텍스트 추출 (기본 구현)
                    with open(doc["path"], 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    
                    # 간단한 HTML 태그 제거
                    text = re.sub(r'<[^>]+>', ' ', html_content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    doc_result["extracted_text"] = text
                    doc_result["processed"] = True
                
                elif extension in [".pdf", ".docx"]:
                    # PDF 및 DOCX 처리는 추가 라이브러리 필요
                    doc_result["error"] = f"{extension} 파일 처리를 위한 라이브러리가 구현되지 않음"
                
                else:
                    doc_result["error"] = f"지원되지 않는 파일 형식: {extension}"
                
                # 통계 업데이트
                result["stats"]["total_processed"] += 1
                
                if doc_result["processed"]:
                    result["stats"]["successful"] += 1
                    # 텍스트 길이 및 단어 수 추가
                    if doc_result["extracted_text"]:
                        doc_result["text_length"] = len(doc_result["extracted_text"])
                        doc_result["word_count"] = len(doc_result["extracted_text"].split())
                else:
                    result["stats"]["failed"] += 1
                
                # 확장자별 통계
                ext_key = extension or "unknown"
                if ext_key not in result["stats"]["by_extension"]:
                    result["stats"]["by_extension"][ext_key] = {
                        "total": 0,
                        "successful": 0,
                        "failed": 0
                    }
                
                result["stats"]["by_extension"][ext_key]["total"] += 1
                if doc_result["processed"]:
                    result["stats"]["by_extension"][ext_key]["successful"] += 1
                else:
                    result["stats"]["by_extension"][ext_key]["failed"] += 1
            
            except Exception as e:
                doc_result["error"] = str(e)
                doc_result["processed"] = False
                result["stats"]["failed"] += 1
                logger.error(f"문서 '{doc['path']}' 처리 중 오류: {str(e)}")
            
            result["processed_documents"].append(doc_result)
        
        # 결과 XCom에 저장
        if 'ti' in kwargs:
            kwargs['ti'].xcom_push(key='processed_documents', value=result)
        
        # 출력 경로가 제공된 경우 결과 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_path = os.path.join(output_path, "text_processing_result.json")
            
            with open(result_path, 'w', encoding='utf-8') as f:
                # 큰 텍스트는 제외하고 저장
                compact_result = result.copy()
                compact_result["processed_documents"] = [
                    {k: v for k, v in doc.items() if k != "extracted_text"} 
                    for doc in result["processed_documents"]
                ]
                
                json.dump(compact_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"텍스트 처리 결과 저장됨: {result_path}")
            result["result_path"] = result_path
        
        logger.info(f"문서 텍스트 처리 완료: {result['stats']['successful']}/{result['stats']['total_processed']} 성공")
        return result
    except Exception as e:
        logger.error(f"문서 텍스트 처리 중 오류: {str(e)}")
        result["error"] = str(e)
        return result

def extract_document_metadata(scan_result: Dict[str, Any] = None, output_path: str = None, **kwargs) -> Dict[str, Any]:
    """
    문서 메타데이터 추출
    
    Args:
        scan_result (Dict[str, Any], optional): 스캔 결과 (None이면 XCom에서 가져옴)
        output_path (str, optional): 출력 디렉토리 경로
        **kwargs: 추가 인수 (Airflow 컨텍스트)
    
    Returns:
        Dict[str, Any]: 메타데이터 추출 결과
    """
    # XCom에서 스캔 결과 가져오기
    if scan_result is None and 'ti' in kwargs:
        scan_result = kwargs['ti'].xcom_pull(task_ids='scan_documents')
    
    if not scan_result or not isinstance(scan_result, dict) or "documents" not in scan_result:
        error_msg = "유효한 스캔 결과를 찾을 수 없음"
        logger.error(error_msg)
        return {"error": error_msg}
    
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "document_metadata": [],
        "stats": {
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
    }
    
    try:
        for doc in scan_result["documents"]:
            metadata = {
                "path": doc["path"],
                "filename": doc["filename"],
                "extension": doc["extension"],
                "size_bytes": doc["size_bytes"],
                "created_at": doc["created_at"],
                "modified_at": doc["modified_at"],
                "extracted_metadata": {},
                "processed": False,
                "error": None
            }
            
            try:
                # 파일 확장자에 따라 다른 메타데이터 추출 방법 적용
                extension = doc["extension"].lower()
                
                if extension in [".txt", ".md", ".json", ".html"]:
                    # 기본 메타데이터만 포함
                    metadata["processed"] = True
                
                elif extension == ".pdf":
                    # PDF 메타데이터 추출 (추가 라이브러리 필요)
                    metadata["error"] = "PDF 메타데이터 추출을 위한 라이브러리가 구현되지 않음"
                
                elif extension == ".docx":
                    # DOCX 메타데이터 추출 (추가 라이브러리 필요)
                    metadata["error"] = "DOCX 메타데이터 추출을 위한 라이브러리가 구현되지 않음"
                
                else:
                    metadata["error"] = f"지원되지 않는 파일 형식: {extension}"
                
                # 통계 업데이트
                result["stats"]["total_processed"] += 1
                
                if metadata["processed"]:
                    result["stats"]["successful"] += 1
                else:
                    result["stats"]["failed"] += 1
            
            except Exception as e:
                metadata["error"] = str(e)
                metadata["processed"] = False
                result["stats"]["failed"] += 1
                logger.error(f"문서 '{doc['path']}' 메타데이터 추출 중 오류: {str(e)}")
            
            result["document_metadata"].append(metadata)
        
        # 결과 XCom에 저장
        if 'ti' in kwargs:
            kwargs['ti'].xcom_push(key='document_metadata', value=result)
        
        # 출력 경로가 제공된 경우 결과 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_path = os.path.join(output_path, "metadata_extraction_result.json")
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"메타데이터 추출 결과 저장됨: {result_path}")
            result["result_path"] = result_path
        
        logger.info(f"문서 메타데이터 추출 완료: {result['stats']['successful']}/{result['stats']['total_processed']} 성공")
        return result
    except Exception as e:
        logger.error(f"문서 메타데이터 추출 중 오류: {str(e)}")
        result["error"] = str(e)
        return result

def save_processed_documents(processed_text: Dict[str, Any] = None, document_metadata: Dict[str, Any] = None,
                           output_path: str = None, **kwargs) -> Dict[str, Any]:
    """
    처리된 문서 저장
    
    Args:
        processed_text (Dict[str, Any], optional): 처리된 텍스트 결과 (None이면 XCom에서 가져옴)
        document_metadata (Dict[str, Any], optional): 메타데이터 추출 결과 (None이면 XCom에서 가져옴)
        output_path (str, optional): 출력 디렉토리 경로
        **kwargs: 추가 인수 (Airflow 컨텍스트)
    
    Returns:
        Dict[str, Any]: 저장 결과
    """
    # XCom에서 처리 결과 가져오기
    if 'ti' in kwargs:
        if processed_text is None:
            try:
                processed_text = kwargs['ti'].xcom_pull(task_ids='process_document_text', key='processed_documents')
            except:
                processed_text = None
        
        if document_metadata is None:
            try:
                document_metadata = kwargs['ti'].xcom_pull(task_ids='extract_document_metadata', key='document_metadata')
            except:
                document_metadata = None
    
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "saved_documents": [],
        "stats": {
            "total_saved": 0,
            "with_text": 0,
            "with_metadata": 0,
            "with_both": 0,
            "failed": 0
        }
    }
    
    try:
        # 출력 경로 생성
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = os.path.join(os.getcwd(), "processed_documents")
            os.makedirs(output_path, exist_ok=True)
        
        # 처리된 텍스트 문서 정보 수집
        text_docs = {}
        if processed_text and "processed_documents" in processed_text:
            for doc in processed_text["processed_documents"]:
                if doc["processed"] and "path" in doc:
                    text_docs[doc["path"]] = doc
        
        # 메타데이터 문서 정보 수집
        metadata_docs = {}
        if document_metadata and "document_metadata" in document_metadata:
            for doc in document_metadata["document_metadata"]:
                if doc["processed"] and "path" in doc:
                    metadata_docs[doc["path"]] = doc
        
        # 모든 문서 경로 수집
        all_paths = set(list(text_docs.keys()) + list(metadata_docs.keys()))
        
        for path in all_paths:
            save_result = {
                "original_path": path,
                "saved": False,
                "saved_path": None,
                "has_text": path in text_docs,
                "has_metadata": path in metadata_docs,
                "error": None
            }
            
            try:
                # 파일 이름 및 확장자 추출
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                
                # 저장 파일 경로 생성
                save_dir = os.path.join(output_path, name)
                os.makedirs(save_dir, exist_ok=True)
                
                # 처리된 텍스트 저장
                if path in text_docs and "extracted_text" in text_docs[path] and text_docs[path]["extracted_text"]:
                    text_path = os.path.join(save_dir, f"{name}_text.txt")
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text_docs[path]["extracted_text"])
                    save_result["text_path"] = text_path
                
                # 메타데이터 저장
                if path in metadata_docs:
                    metadata_path = os.path.join(save_dir, f"{name}_metadata.json")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata_docs[path], f, indent=2, ensure_ascii=False)
                    save_result["metadata_path"] = metadata_path
                
                # 원본 파일 복사
                if os.path.exists(path):
                    original_copy_path = os.path.join(save_dir, filename)
                    shutil.copy2(path, original_copy_path)
                    save_result["original_copy_path"] = original_copy_path
                
                # 통계 업데이트
                save_result["saved"] = True
                result["stats"]["total_saved"] += 1
                
                if path in text_docs:
                    result["stats"]["with_text"] += 1
                
                if path in metadata_docs:
                    result["stats"]["with_metadata"] += 1
                
                if path in text_docs and path in metadata_docs:
                    result["stats"]["with_both"] += 1
            
            except Exception as e:
                save_result["error"] = str(e)
                save_result["saved"] = False
                result["stats"]["failed"] += 1
                logger.error(f"문서 '{path}' 저장 중 오류: {str(e)}")
            
            result["saved_documents"].append(save_result)
        
        # 결과 요약 저장
        summary_path = os.path.join(output_path, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        result["summary_path"] = summary_path
        
        logger.info(f"처리된 문서 저장 완료: {result['stats']['total_saved']} 문서 저장됨")
        return result
    except Exception as e:
        logger.error(f"처리된 문서 저장 중 오류: {str(e)}")
        result["error"] = str(e)
        return result

def _get_file_info(file_path: str) -> Dict[str, Any]:
    """
    파일 정보 가져오기
    
    Args:
        file_path (str): 파일 경로
    
    Returns:
        Dict[str, Any]: 파일 정보
    """
    stat = os.stat(file_path)
    filename = os.path.basename(file_path)
    _, extension = os.path.splitext(filename)
    
    return {
        "path": file_path,
        "filename": filename,
        "extension": extension,
        "size_bytes": stat.st_size,
        "created_at": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    }

def _get_size_range(size_bytes: int) -> str:
    """
    파일 크기 범위 결정
    
    Args:
        size_bytes (int): 파일 크기 (바이트)
    
    Returns:
        str: 크기 범위 문자열
    """
    kb = size_bytes / 1024
    mb = kb / 1024
    
    if kb < 100:
        return "small (<100KB)"
    elif kb < 1024:
        return "medium (100KB-1MB)"
    elif mb < 10:
        return "large (1MB-10MB)"
    else:
        return "very_large (>10MB)"

def fnmatch(filename: str, pattern: str) -> bool:
    """
    파일 이름이 패턴과 일치하는지 확인
    
    Args:
        filename (str): 파일 이름
        pattern (str): 패턴 (*, ? 와일드카드 지원)
    
    Returns:
        bool: 일치 여부
    """
    import re
    
    # 와일드카드 패턴을 정규식으로 변환
    pattern = pattern.replace(".", "\\.")
    pattern = pattern.replace("*", ".*")
    pattern = pattern.replace("?", ".")
    pattern = f"^{pattern}$"
    
    return re.match(pattern, filename, re.IGNORECASE) is not None

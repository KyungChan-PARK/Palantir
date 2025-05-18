"""
대규모 문서 세트 처리를 위한 최적화 모듈
메모리 효율적인 문서 처리 및 성능 최적화 기능 제공
"""

import os
import sys
import json
import logging
import time
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Generator, Callable
from datetime import datetime
import asyncio
import multiprocessing
from functools import partial

# 모듈 로드 경로에 MCP 초기화 모듈 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from analysis.mcp_init import mcp

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('document_optimization')

# ===== 문서 청킹 및 처리 최적화 도구 =====

@mcp.tool(
    name="chunk_document_iterator",
    description="Memory-efficient document chunk iterator",
    tags=["document", "optimization", "memory"]
)
async def chunk_document_iterator(file_path: str, chunk_size: int = 1000, encoding: str = 'utf-8') -> Generator[str, None, None]:
    """
    대용량 문서를 청크 단위로 처리하는 메모리 효율적인 이터레이터를 제공합니다.
    
    Parameters:
        file_path (str): 문서 파일 경로
        chunk_size (int): 한 번에 처리할 문자 수 (바이트 단위가 아님)
        encoding (str): 파일 인코딩
        
    Yields:
        str: 문서 청크
    """
    logger.info(f"Creating chunk iterator for {file_path} with chunk size {chunk_size}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            chunk = ""
            for line in f:
                chunk += line
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = ""
            
            # 마지막 청크가 남아 있으면 반환
            if chunk:
                yield chunk
                
        logger.info(f"Finished iterating through {file_path}")
    except Exception as e:
        logger.error(f"Error in chunk_document_iterator for {file_path}: {e}")
        raise

@mcp.tool(
    name="get_memory_usage",
    description="Get current memory usage",
    tags=["optimization", "memory"]
)
async def get_memory_usage() -> Dict[str, float]:
    """
    현재 프로세스의 메모리 사용량 정보를 반환합니다.
    
    Returns:
        dict: 메모리 사용량 정보 (MB 단위)
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / (1024 * 1024),  # Resident Set Size (MB)
        "vms": memory_info.vms / (1024 * 1024),  # Virtual Memory Size (MB)
        "percent": process.memory_percent(),     # 시스템 메모리 사용 비율 (%)
        "available": psutil.virtual_memory().available / (1024 * 1024),  # 사용 가능한 메모리 (MB)
        "total": psutil.virtual_memory().total / (1024 * 1024)           # 총 메모리 (MB)
    }

@mcp.tool(
    name="optimize_document_loading",
    description="Load multiple documents with optimized memory usage",
    tags=["document", "optimization", "memory"]
)
async def optimize_document_loading(file_paths: List[str], encoding: str = 'utf-8', max_workers: int = None, batch_size: int = 10, max_memory_pct: float = 80.0) -> Dict[str, Any]:
    """
    여러 문서를 메모리 사용량을 최적화하여 로드합니다.
    
    Parameters:
        file_paths (list): 문서 파일 경로 목록
        encoding (str): 파일 인코딩
        max_workers (int, optional): 최대 동시 작업자 수 (None이면 CPU 코어 수의 3/4)
        batch_size (int): 한 번에 처리할 문서 수
        max_memory_pct (float): 최대 메모리 사용 비율 (백분율)
        
    Returns:
        dict: 로드된 문서 및 성능 정보
    """
    logger.info(f"Optimized loading for {len(file_paths)} documents")
    
    start_time = time.time()
    total_size = 0
    loaded_count = 0
    docs_content = {}
    errors = []
    
    # 최대 작업자 수 설정
    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    logger.info(f"Using {max_workers} workers for document loading")
    
    # 병렬 로드 함수
    async def load_document(path):
        try:
            # 메모리 사용량 체크
            memory_info = await get_memory_usage()
            if memory_info["percent"] > max_memory_pct:
                logger.warning(f"Memory usage ({memory_info['percent']:.1f}%) exceeds threshold ({max_memory_pct}%). Pausing loading.")
                await asyncio.sleep(1)  # 메모리 사용량이 많으면 잠시 대기
            
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return path, content, len(content)
        except Exception as e:
            logger.error(f"Error loading document {path}: {e}")
            return path, None, 0
    
    # 배치 단위로 처리
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        tasks = [load_document(path) for path in batch]
        
        # 병렬 실행
        results = await asyncio.gather(*tasks)
        
        # 결과 처리
        for path, content, size in results:
            if content is not None:
                docs_content[path] = content
                total_size += size
                loaded_count += 1
            else:
                errors.append(path)
        
        # 메모리 사용량 로깅
        if (i + batch_size) % 50 == 0 or i + batch_size >= len(file_paths):
            memory_info = await get_memory_usage()
            logger.info(f"Loaded {loaded_count}/{len(file_paths)} documents, Memory: {memory_info['rss']:.1f} MB ({memory_info['percent']:.1f}%)")
            
            # 메모리 임계값 초과 시 가비지 컬렉션 촉진
            if memory_info["percent"] > max_memory_pct * 0.9:
                import gc
                gc.collect()
                logger.info("Forced garbage collection")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 결과 정보
    result = {
        "loaded_count": loaded_count,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "elapsed_seconds": elapsed,
        "docs_per_second": loaded_count / elapsed if elapsed > 0 else 0,
        "mb_per_second": (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0,
        "errors": errors
    }
    
    logger.info(f"Loaded {loaded_count} documents ({result['total_size_mb']:.1f} MB) in {elapsed:.2f} seconds")
    return result

@mcp.tool(
    name="process_documents_in_batches",
    description="Process multiple documents in batches with memory optimization",
    tags=["document", "optimization", "memory", "processing"]
)
async def process_documents_in_batches(file_paths: List[str], processor_func: Callable, batch_size: int = 10, max_workers: int = None, max_memory_pct: float = 80.0, **kwargs) -> Dict[str, Any]:
    """
    다수의 문서를 메모리 최적화된 방식으로 배치 처리합니다.
    
    Parameters:
        file_paths (list): 문서 파일 경로 목록
        processor_func (callable): 각 문서를 처리할 함수
        batch_size (int): 한 번에 처리할 문서 수
        max_workers (int, optional): 최대 동시 작업자 수 (None이면 CPU 코어 수의 3/4)
        max_memory_pct (float): 최대 메모리 사용 비율 (백분율)
        **kwargs: processor_func에 전달할 추가 인자
        
    Returns:
        dict: 처리 결과 및 성능 정보
    """
    logger.info(f"Batch processing {len(file_paths)} documents with batch size {batch_size}")
    
    start_time = time.time()
    processed_count = 0
    results = {}
    errors = []
    
    # 최대 작업자 수 설정
    if max_workers is None:
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    logger.info(f"Using {max_workers} workers for document processing")
    
    # 병렬 처리 함수
    async def process_document(path):
        try:
            # 메모리 사용량 체크
            memory_info = await get_memory_usage()
            if memory_info["percent"] > max_memory_pct:
                logger.warning(f"Memory usage ({memory_info['percent']:.1f}%) exceeds threshold ({max_memory_pct}%). Pausing processing.")
                await asyncio.sleep(1)  # 메모리 사용량이 많으면 잠시 대기
            
            result = await processor_func(path, **kwargs)
            return path, result, True
        except Exception as e:
            logger.error(f"Error processing document {path}: {e}")
            return path, str(e), False
    
    # 세마포어로 동시 작업 수 제한
    semaphore = asyncio.Semaphore(max_workers)
    
    async def limited_process(path):
        async with semaphore:
            return await process_document(path)
    
    # 배치 단위로 처리
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        tasks = [limited_process(path) for path in batch]
        
        # 병렬 실행
        batch_results = await asyncio.gather(*tasks)
        
        # 결과 처리
        for path, result, success in batch_results:
            if success:
                results[path] = result
                processed_count += 1
            else:
                errors.append((path, result))
        
        # 메모리 사용량 및 진행 상황 로깅
        if (i + batch_size) % 50 == 0 or i + batch_size >= len(file_paths):
            memory_info = await get_memory_usage()
            progress = (i + len(batch)) / len(file_paths) * 100
            logger.info(f"Progress: {progress:.1f}%, Processed {processed_count}/{len(file_paths)} documents, Memory: {memory_info['rss']:.1f} MB ({memory_info['percent']:.1f}%)")
            
            # 메모리 임계값 초과 시 가비지 컬렉션 촉진
            if memory_info["percent"] > max_memory_pct * 0.9:
                import gc
                gc.collect()
                logger.info("Forced garbage collection")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 결과 정보
    result = {
        "processed_count": processed_count,
        "elapsed_seconds": elapsed,
        "docs_per_second": processed_count / elapsed if elapsed > 0 else 0,
        "results": results,
        "errors": errors
    }
    
    logger.info(f"Processed {processed_count} documents in {elapsed:.2f} seconds")
    return result

@mcp.tool(
    name="semantic_chunking",
    description="Perform semantic chunking of documents",
    tags=["document", "optimization", "chunking"]
)
async def semantic_chunking(text: str, min_chunk_size: int = 100, max_chunk_size: int = 2000, overlap: int = 20) -> List[Dict[str, Any]]:
    """
    문서를 의미론적으로 적절한 청크로 분할합니다.
    
    Parameters:
        text (str): 청킹할 문서 텍스트
        min_chunk_size (int): 최소 청크 크기 (단어 수)
        max_chunk_size (int): 최대 청크 크기 (단어 수)
        overlap (int): 청크 간 겹치는 단어 수
        
    Returns:
        list: 청크 목록 (텍스트, 시작 위치, 종료 위치, 메타데이터)
    """
    logger.info(f"Performing semantic chunking with size range {min_chunk_size}-{max_chunk_size} words")
    
    # NLTK 사용 시 필요한 패키지
    try:
        import nltk
        nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    except ImportError:
        logger.warning("NLTK not available. Falling back to basic chunking.")
        nltk = None
    
    # 청크 결과 저장 목록
    chunks = []
    
    try:
        # NLTK 이용한 문장 분할
        if nltk:
            sentences = nltk.sent_tokenize(text)
            
            # 문장 길이 계산 (단어 수)
            sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
            sentence_positions = []
            
            # 각 문장의 시작 위치 계산
            pos = 0
            for i, sentence in enumerate(sentences):
                sentence_positions.append(pos)
                pos += len(sentence)
            
            # 의미 있는 경계를 찾기 위한 청킹
            current_chunk = []
            current_chunk_word_count = 0
            current_chunk_start = 0
            
            for i, sentence in enumerate(sentences):
                words = sentence_lengths[i]
                
                # 첫 문장이거나 청크 크기가 최대 크기를 초과하지 않는 경우
                if not current_chunk or current_chunk_word_count + words <= max_chunk_size:
                    current_chunk.append(sentence)
                    if not current_chunk_word_count:
                        current_chunk_start = sentence_positions[i]
                    current_chunk_word_count += words
                else:
                    # 청크가 최소 크기를 넘으면 저장
                    if current_chunk_word_count >= min_chunk_size:
                        chunk_text = " ".join(current_chunk)
                        chunk_end = sentence_positions[i] - 1
                        
                        chunks.append({
                            "text": chunk_text,
                            "start": current_chunk_start,
                            "end": chunk_end,
                            "word_count": current_chunk_word_count
                        })
                        
                        # 겹침을 위해 청크의 마지막 몇 문장을 유지
                        overlap_word_count = 0
                        overlap_sentences = []
                        
                        for j in range(len(current_chunk) - 1, -1, -1):
                            overlap_word_count += len(nltk.word_tokenize(current_chunk[j]))
                            overlap_sentences.insert(0, current_chunk[j])
                            
                            if overlap_word_count >= overlap:
                                break
                        
                        # 새 청크 시작
                        current_chunk = overlap_sentences + [sentence]
                        current_chunk_start = sentence_positions[i - len(overlap_sentences)]
                        current_chunk_word_count = overlap_word_count + words
                    else:
                        # 청크가 너무 작으면 현재 문장을 추가
                        current_chunk.append(sentence)
                        current_chunk_word_count += words
            
            # 마지막 청크 처리
            if current_chunk and current_chunk_word_count >= min_chunk_size:
                chunk_text = " ".join(current_chunk)
                
                chunks.append({
                    "text": chunk_text,
                    "start": current_chunk_start,
                    "end": len(text),
                    "word_count": current_chunk_word_count
                })
        else:
            # NLTK가 없는 경우 간단한 청킹 (단락 기준)
            paragraphs = text.split("\n\n")
            pos = 0
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    pos += len(paragraph) + 2  # 단락 + "\n\n"
                    continue
                
                word_count = len(paragraph.split())
                
                # 단락이 최대 크기를 초과하는 경우 분할
                if word_count > max_chunk_size:
                    sentences = paragraph.split(". ")
                    temp_chunk = []
                    temp_word_count = 0
                    temp_start = pos
                    
                    for sentence in sentences:
                        sentence_words = len(sentence.split())
                        
                        if temp_word_count + sentence_words <= max_chunk_size:
                            temp_chunk.append(sentence)
                            temp_word_count += sentence_words
                        else:
                            if temp_word_count >= min_chunk_size:
                                temp_text = ". ".join(temp_chunk) + ("." if temp_chunk[-1][-1] != "." else "")
                                temp_end = temp_start + len(temp_text)
                                
                                chunks.append({
                                    "text": temp_text,
                                    "start": temp_start,
                                    "end": temp_end,
                                    "word_count": temp_word_count
                                })
                                
                                temp_start = temp_end
                            
                            temp_chunk = [sentence]
                            temp_word_count = sentence_words
                    
                    # 마지막 청크 처리
                    if temp_chunk and temp_word_count >= min_chunk_size:
                        temp_text = ". ".join(temp_chunk) + ("." if temp_chunk[-1][-1] != "." else "")
                        temp_end = temp_start + len(temp_text)
                        
                        chunks.append({
                            "text": temp_text,
                            "start": temp_start,
                            "end": temp_end,
                            "word_count": temp_word_count
                        })
                        
                        pos = temp_end
                else:
                    # 단락이 적절한 크기인 경우
                    chunks.append({
                        "text": paragraph,
                        "start": pos,
                        "end": pos + len(paragraph),
                        "word_count": word_count
                    })
                    
                    pos += len(paragraph) + 2  # 단락 + "\n\n"
        
        # 청크 ID 할당 및 메타데이터 추가
        for i, chunk in enumerate(chunks):
            chunk["id"] = f"chunk_{i+1}"
            
            # 청크 유형 분류
            word_count = chunk["word_count"]
            if word_count < min_chunk_size * 1.2:
                chunk["size_category"] = "small"
            elif word_count > max_chunk_size * 0.8:
                chunk["size_category"] = "large"
            else:
                chunk["size_category"] = "medium"
        
        logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
        return chunks
    except Exception as e:
        logger.error(f"Error in semantic chunking: {e}")
        
        # 오류 발생 시 기본 청크 하나 반환
        return [{
            "id": "chunk_1",
            "text": text,
            "start": 0,
            "end": len(text),
            "word_count": len(text.split()),
            "size_category": "large",
            "error": str(e)
        }]

@mcp.tool(
    name="cache_document_processing",
    description="Cache document processing results",
    tags=["document", "optimization", "cache"]
)
async def cache_document_processing(cache_key: str, processor_func: Callable, force_refresh: bool = False, cache_dir: str = "temp/doc_cache", **kwargs) -> Dict[str, Any]:
    """
    문서 처리 결과를 캐싱하여 재사용합니다.
    
    Parameters:
        cache_key (str): 캐시 키
        processor_func (callable): 문서 처리 함수
        force_refresh (bool): 캐시가 있어도 강제로 재처리할지 여부
        cache_dir (str): 캐시 디렉토리
        **kwargs: processor_func에 전달할 추가 인자
        
    Returns:
        dict: 처리 결과
    """
    import hashlib
    
    # 캐시 디렉토리 생성
    os.makedirs(cache_dir, exist_ok=True)
    
    # 캐시 키 해싱
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_hash}.json")
    
    # 캐시 확인
    if os.path.exists(cache_file) and not force_refresh:
        try:
            logger.info(f"Loading cached result for '{cache_key}'")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # 캐시 메타데이터 추가
            cached_data["_cache_info"] = {
                "cache_hit": True,
                "cache_file": cache_file,
                "cached_at": cached_data.get("_cache_info", {}).get("processed_at", "unknown")
            }
            
            return cached_data
        except Exception as e:
            logger.warning(f"Error loading cache, will reprocess: {e}")
    
    # 캐시 없거나 강제 갱신인 경우 처리
    logger.info(f"Processing data for '{cache_key}'")
    result = await processor_func(**kwargs)
    
    # 처리 결과 캐싱
    try:
        # NumPy 배열 및 특수 객체 변환
        def json_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, set):
                return list(obj)
            else:
                return str(obj)
        
        # 캐시 메타데이터 추가
        result["_cache_info"] = {
            "cache_hit": False,
            "cache_file": cache_file,
            "cache_key": cache_key,
            "processed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, default=json_serializable, indent=2)
            
        logger.info(f"Cached result saved to {cache_file}")
    except Exception as e:
        logger.warning(f"Error saving cache: {e}")
    
    return result

# ===== 워크플로우 수준 도구 (Molecules) =====

@mcp.workflow(
    name="optimize_document_set_processing",
    description="Optimize processing of large document sets",
    
)
async def optimize_document_set_processing(doc_directory: str, processor_func: Callable, cache_enabled: bool = True, batch_size: int = 10, max_workers: int = None, use_semantic_chunking: bool = True, **kwargs) -> Dict[str, Any]:
    """
    대규모 문서 세트 처리를 최적화합니다.
    
    Parameters:
        doc_directory (str): 문서 디렉토리 경로
        processor_func (callable): 각 문서를 처리할 함수
        cache_enabled (bool): 캐싱 사용 여부
        batch_size (int): 배치 크기
        max_workers (int, optional): 최대 작업자 수
        use_semantic_chunking (bool): 의미론적 청킹 사용 여부
        **kwargs: processor_func에 전달할 추가 인자
        
    Returns:
        dict: 처리 결과 및 성능 정보
    """
    logger.info(f"Starting optimized document set processing for {doc_directory}")
    start_time = time.time()
    
    # 문서 파일 목록 가져오기
    def get_file_list(directory, extensions=['.txt', '.md', '.html', '.json', '.csv']):
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_list.append(os.path.join(root, file))
        return file_list
    
    file_paths = get_file_list(doc_directory)
    logger.info(f"Found {len(file_paths)} documents to process")
    
    if not file_paths:
        return {
            "error": f"No documents found in {doc_directory}",
            "elapsed_seconds": 0,
            "processed_count": 0
        }
    
    # 메모리 정보 로깅
    memory_info = await get_memory_usage()
    logger.info(f"Initial memory usage: {memory_info['rss']:.1f} MB ({memory_info['percent']:.1f}%)")
    
    # 처리 함수 래핑
    async def process_document_wrapper(file_path, **proc_kwargs):
        # 캐싱 사용 시
        if cache_enabled:
            cache_key = f"{os.path.basename(file_path)}_{hash(str(proc_kwargs))}"
            return await cache_document_processing(
                cache_key=cache_key,
                processor_func=processor_func,
                file_path=file_path,
                **proc_kwargs
            )
        else:
            # 의미론적 청킹 사용 시
            if use_semantic_chunking:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    chunks = await semantic_chunking(text)
                    chunk_results = []
                    
                    for chunk in chunks:
                        chunk_result = await processor_func(
                            text=chunk["text"],
                            file_path=file_path,
                            chunk_info=chunk,
                            **proc_kwargs
                        )
                        chunk_results.append(chunk_result)
                    
                    return {
                        "file_path": file_path,
                        "chunks": chunks,
                        "results": chunk_results
                    }
                except Exception as e:
                    logger.error(f"Error processing document {file_path} with chunking: {e}")
                    raise
            else:
                # 일반 처리
                return await processor_func(file_path=file_path, **proc_kwargs)
    
    # 배치 처리 실행
    results = await process_documents_in_batches(
        file_paths=file_paths,
        processor_func=process_document_wrapper,
        batch_size=batch_size,
        max_workers=max_workers,
        **kwargs
    )
    
    # 종료 시간 및 메모리 사용량
    end_time = time.time()
    elapsed = end_time - start_time
    memory_info = await get_memory_usage()
    
    # 결과 정보 확장
    results["total_elapsed_seconds"] = elapsed
    results["memory_usage"] = {
        "rss_mb": memory_info["rss"],
        "percent": memory_info["percent"]
    }
    results["use_semantic_chunking"] = use_semantic_chunking
    results["cache_enabled"] = cache_enabled
    
    logger.info(f"Completed document set processing in {elapsed:.2f} seconds")
    logger.info(f"Final memory usage: {memory_info['rss']:.1f} MB ({memory_info['percent']:.1f}%)")
    
    return results

# 테스트 용 함수
async def test_optimization():
    """
    문서 최적화 도구 테스트
    """
    logger.info("Testing document optimization module")
    
    try:
        # 테스트 디렉토리 생성
        test_dir = "temp/optimization_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # 테스트 문서 생성
        for i in range(5):
            file_path = os.path.join(test_dir, f"test_doc_{i+1}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Test document {i+1}\n\n")
                f.write("This is a test document for optimization testing.\n\n")
                for j in range(10):
                    f.write(f"Paragraph {j+1}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. ")
                    f.write("Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n\n")
        
        logger.info("Created test documents")
        
        # 테스트 처리 함수
        async def test_processor(file_path=None, text=None, **kwargs):
            if file_path and not text:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            word_count = len(text.split())
            char_count = len(text)
            
            return {
                "file_path": file_path,
                "word_count": word_count,
                "char_count": char_count,
                "processed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # 1. 청크 이터레이터 테스트
        file_path = os.path.join(test_dir, "test_doc_1.txt")
        chunk_gen = chunk_document_iterator(file_path, chunk_size=100)
        
        async for chunk in chunk_gen:
            logger.info(f"Chunk size: {len(chunk)} characters")
            break  # 첫 번째 청크만 테스트
        
        logger.info("SUCCESS: chunk_document_iterator test passed")
        
        # 2. 메모리 사용량 테스트
        memory_info = await get_memory_usage()
        logger.info(f"Memory usage: {memory_info['rss']:.1f} MB ({memory_info['percent']:.1f}%)")
        logger.info("SUCCESS: get_memory_usage test passed")
        
        # 3. 문서 로딩 최적화 테스트
        file_paths = [os.path.join(test_dir, f"test_doc_{i+1}.txt") for i in range(5)]
        load_results = await optimize_document_loading(file_paths)
        logger.info(f"Loaded {load_results['loaded_count']} documents in {load_results['elapsed_seconds']:.2f} seconds")
        logger.info("SUCCESS: optimize_document_loading test passed")
        
        # 4. 배치 처리 테스트
        batch_results = await process_documents_in_batches(file_paths, test_processor, batch_size=2)
        logger.info(f"Processed {batch_results['processed_count']} documents in {batch_results['elapsed_seconds']:.2f} seconds")
        logger.info("SUCCESS: process_documents_in_batches test passed")
        
        # 5. 의미론적 청킹 테스트
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = await semantic_chunking(text, min_chunk_size=50, max_chunk_size=200)
        logger.info(f"Created {len(chunks)} semantic chunks")
        logger.info("SUCCESS: semantic_chunking test passed")
        
        # 6. 캐싱 테스트
        cache_result = await cache_document_processing("test_key", test_processor, file_path=file_path)
        logger.info(f"Cache result: {cache_result['word_count']} words, {cache_result['char_count']} characters")
        
        # 캐시 적중 테스트
        cache_hit = await cache_document_processing("test_key", test_processor, file_path=file_path)
        logger.info(f"Cache hit: {cache_hit.get('_cache_info', {}).get('cache_hit', False)}")
        logger.info("SUCCESS: cache_document_processing test passed")
        
        # 7. 워크플로우 테스트
        workflow_result = await optimize_document_set_processing(
            doc_directory=test_dir,
            processor_func=test_processor,
            batch_size=2,
            use_semantic_chunking=True
        )
        
        logger.info(f"Workflow processed {workflow_result['processed_count']} documents")
        logger.info("SUCCESS: optimize_document_set_processing workflow test passed")
        
        logger.info("All document optimization tests passed successfully")
        return True
    except Exception as e:
        logger.error(f"Document optimization test failed: {e}")
        raise

# 테스트 실행
if __name__ == "__main__":
    asyncio.run(test_optimization())

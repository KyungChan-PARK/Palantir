"""
웹 대시보드 애플리케이션
시스템 관리 및 모니터링을 위한 웹 인터페이스
"""

import os
import sys
import json
import time
import logging
import platform
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 모듈 로드 경로에 MCP 초기화 모듈 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from analysis.mcp_init import mcp

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/dashboard.log',
    filemode='a'
)
logger = logging.getLogger('dashboard')

# Flask 애플리케이션 초기화
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.urandom(24).hex()

# 시스템 상태 및 활동 데이터
system_status = {
    'mcp_server': 'active',
    'data_analysis': 'active',
    'youtube_api': 'inactive',
    'document_test': 'active'
}

# 활동 데이터 저장
activity_data = {
    'timestamps': [],
    'data_analysis': [],
    'document_test': [],
    'youtube_api': []
}

# 최근 작업 목록
recent_tasks = []

# 시스템 설정
CONFIG = {
    'output_dir': 'output',
    'data_dir': 'data',
    'temp_dir': 'temp',
    'logs_dir': 'logs',
    'document_types': ['report', 'analysis', 'memo', 'data', 'code', 'other'],
    'default_page_size': 9,
    'max_page_size': 50,
    'default_sort': 'date_desc',
    'cache_enabled': True,
    'cache_dir': 'temp/dashboard_cache',
    'default_folders': [
        {'name': 'All Documents', 'path': ''},
        {'name': 'Reports', 'path': 'reports'},
        {'name': 'Analysis', 'path': 'analysis'},
        {'name': 'Memos', 'path': 'memos'},
        {'name': 'Data', 'path': 'data'}
    ],
    'default_tags': [
        {'name': 'important', 'count': 5},
        {'name': 'todo', 'count': 3},
        {'name': 'reviewed', 'count': 8},
        {'name': 'archived', 'count': 12}
    ]
}

# MCP 시스템 가져오기
async def import_mcp_systems():
    """MCP 시스템 모듈 가져오기"""
    try:
        from analysis.tools.organisms import data_analysis_system
        logger.info("Imported data_analysis_system")
    except ImportError:
        logger.warning("Could not import data_analysis_system")
    
    try:
        from analysis.tools.organisms import youtube_api_system
        logger.info("Imported youtube_api_system")
    except ImportError:
        logger.warning("Could not import youtube_api_system")
    
    try:
        from analysis.tools.organisms import document_test_system
        logger.info("Imported document_test_system")
    except ImportError:
        logger.warning("Could not import document_test_system")
    
    try:
        from analysis.tools.molecules import document_optimization
        logger.info("Imported document_optimization")
    except ImportError:
        logger.warning("Could not import document_optimization")

# 대시보드 초기화
def initialize_dashboard():
    """대시보드 초기화 및 설정"""
    # 필요한 디렉토리 생성
    for dir_path in [
        CONFIG['output_dir'],
        CONFIG['data_dir'],
        CONFIG['temp_dir'],
        CONFIG['logs_dir'],
        CONFIG['cache_dir'],
        os.path.join(CONFIG['output_dir'], 'reports'),
        os.path.join(CONFIG['output_dir'], 'viz'),
        os.path.join(CONFIG['output_dir'], 'reports', 'performance'),
        os.path.join(CONFIG['output_dir'], 'reports', 'optimization'),
        os.path.join(CONFIG['output_dir'], 'decisions', 'context_optimization')
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    # MCP 시스템 비동기 가져오기
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(import_mcp_systems())
    
    # 시스템 상태 초기화
    update_system_status()
    
    # 초기 시스템 활동 데이터 생성
    generate_sample_activity_data()
    
    # 초기 최근 작업 목록 생성
    generate_sample_tasks()
    
    logger.info("Dashboard initialized successfully")

# 시스템 상태 업데이트
def update_system_status():
    """시스템 상태 정보 업데이트"""
    global system_status
    
    # MCP 서버 상태 확인
    try:
        if hasattr(mcp, 'tools') and hasattr(mcp, 'workflows') and hasattr(mcp, 'systems'):
            system_status['mcp_server'] = 'active'
        else:
            system_status['mcp_server'] = 'warning'
    except Exception:
        system_status['mcp_server'] = 'error'
    
    # 데이터 분석 시스템 상태 확인
    try:
        from analysis.tools.organisms import data_analysis_system
        if hasattr(data_analysis_system, 'decision_support_system'):
            system_status['data_analysis'] = 'active'
        else:
            system_status['data_analysis'] = 'warning'
    except ImportError:
        system_status['data_analysis'] = 'inactive'
    except Exception:
        system_status['data_analysis'] = 'error'
    
    # YouTube API 시스템 상태 확인
    try:
        from analysis.tools.organisms import youtube_api_system
        if hasattr(youtube_api_system, 'generate_youtube_insights'):
            system_status['youtube_api'] = 'active'
        else:
            system_status['youtube_api'] = 'warning'
    except ImportError:
        system_status['youtube_api'] = 'inactive'
    except Exception:
        system_status['youtube_api'] = 'error'
    
    # 문서 테스트 시스템 상태 확인
    try:
        from analysis.tools.organisms import document_test_system
        if hasattr(document_test_system, 'test_system'):
            system_status['document_test'] = 'active'
        else:
            system_status['document_test'] = 'warning'
    except ImportError:
        system_status['document_test'] = 'inactive'
    except Exception:
        system_status['document_test'] = 'error'
    
    logger.info(f"System status updated: {system_status}")

# 리소스 사용량 정보 가져오기
def get_resource_usage():
    """시스템 리소스 사용량 정보 반환"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # 디스크 사용량
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'disk_percent': disk_percent,
        'memory_used': memory.used / (1024 * 1024),  # MB
        'memory_total': memory.total / (1024 * 1024),  # MB
        'disk_used': disk.used / (1024 * 1024 * 1024),  # GB
        'disk_total': disk.total / (1024 * 1024 * 1024)  # GB
    }

# 시스템 정보 가져오기
def get_system_info():
    """시스템 정보 반환"""
    python_version = platform.python_version()
    os_info = f"{platform.system()} {platform.release()}"
    
    # CPU 정보
    cpu_count = psutil.cpu_count(logical=True)
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 * 1024)  # MB
    
    # MCP 정보
    mcp_info = {
        'mcp_mode': getattr(mcp, 'mode', 'standalone'),
        'tools_count': len(getattr(mcp, 'tools', {})),
        'workflows_count': len(getattr(mcp, 'workflows', {})),
        'systems_count': len(getattr(mcp, 'systems', {}))
    }
    
    return {
        'python_version': python_version,
        'os': os_info,
        'cpu_count': cpu_count,
        'total_memory': round(total_memory),
        **mcp_info
    }

# 사용 가능한 시스템 가져오기
def get_available_systems():
    """사용 가능한 MCP 시스템 정보 반환"""
    available_systems = {}
    
    # 데이터 분석 시스템
    if system_status['data_analysis'] in ['active', 'warning']:
        available_systems['data_analysis'] = {
            'name': 'Data Analysis System',
            'status': system_status['data_analysis'],
            'description': 'Structured data analysis and insights generation',
            'module': 'analysis.tools.organisms.data_analysis_system'
        }
    
    # YouTube API 시스템
    if system_status['youtube_api'] in ['active', 'warning']:
        available_systems['youtube_api'] = {
            'name': 'YouTube API System',
            'status': system_status['youtube_api'],
            'description': 'YouTube video content analysis',
            'module': 'analysis.tools.organisms.youtube_api_system'
        }
    
    # 문서 테스트 시스템
    if system_status['document_test'] in ['active', 'warning']:
        available_systems['document_test'] = {
            'name': 'Document Test System',
            'status': system_status['document_test'],
            'description': 'Large document set testing and performance evaluation',
            'module': 'analysis.tools.organisms.document_test_system'
        }
    
    return available_systems

# 샘플 활동 데이터 생성
def generate_sample_activity_data():
    """샘플 활동 데이터 생성 (실제 구현 시 실제 데이터로 대체)"""
    global activity_data
    
    # 최근 24시간에 대한 시간별 레이블 생성
    now = datetime.now()
    timestamps = [(now - timedelta(hours=i)).strftime('%H:00') for i in range(24, 0, -1)]
    
    # 시스템별 활동 데이터 생성
    np.random.seed(42)  # 재현성을 위한 시드 고정
    data_analysis = np.random.randint(0, 10, 24).tolist()
    document_test = np.random.randint(0, 5, 24).tolist()
    youtube_api = np.random.randint(0, 3, 24).tolist()
    
    activity_data = {
        'timestamps': timestamps,
        'data_analysis': data_analysis,
        'document_test': document_test,
        'youtube_api': youtube_api
    }

# 샘플 작업 목록 생성
def generate_sample_tasks():
    """샘플 최근 작업 목록 생성 (실제 구현 시 실제 데이터로 대체)"""
    global recent_tasks
    
    tasks = [
        {
            'id': 'task-001',
            'name': 'Exploratory Data Analysis',
            'system': 'Data Analysis',
            'status': 'completed',
            'timestamp': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'task-002',
            'name': 'Document Performance Test',
            'system': 'Document Test',
            'status': 'completed',
            'timestamp': (datetime.now() - timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'task-003',
            'name': 'Context Optimization',
            'system': 'Document Test',
            'status': 'running',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'task-004',
            'name': 'YouTube Channel Analysis',
            'system': 'YouTube API',
            'status': 'failed',
            'timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
        },
        {
            'id': 'task-005',
            'name': 'Predictive Modeling',
            'system': 'Data Analysis',
            'status': 'completed',
            'timestamp': (datetime.now() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M')
        }
    ]
    
    recent_tasks = tasks

# 샘플 문서 목록 생성
def get_sample_documents(folder=None, tag=None, doc_type=None, query=None, sort_order='date_desc', page=1, page_size=9):
    """샘플 문서 목록 생성 (실제 구현 시 실제 데이터로 대체)"""
    # 샘플 문서 데이터
    documents = [
        {
            'id': f'doc-{i:03d}',
            'name': f'Sample Document {i}',
            'type': CONFIG['document_types'][i % len(CONFIG['document_types'])],
            'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
            'description': f'This is a sample document for testing purposes. Document {i}.',
            'tags': ['sample', CONFIG['default_tags'][i % len(CONFIG['default_tags'])]['name']],
            'folder': CONFIG['default_folders'][i % len(CONFIG['default_folders'])]['path']
        }
        for i in range(1, 51)
    ]
    
    # 필터링
    if folder:
        documents = [doc for doc in documents if doc['folder'] == folder]
    
    if tag:
        documents = [doc for doc in documents if tag in doc['tags']]
    
    if doc_type:
        documents = [doc for doc in documents if doc['type'] == doc_type]
    
    if query:
        query = query.lower()
        documents = [doc for doc in documents if query in doc['name'].lower() or 
                    query in doc['description'].lower() or 
                    any(query in tag.lower() for tag in doc['tags'])]
    
    # 정렬
    if sort_order == 'date_desc':
        documents.sort(key=lambda x: x['date'], reverse=True)
    elif sort_order == 'date_asc':
        documents.sort(key=lambda x: x['date'])
    elif sort_order == 'name_asc':
        documents.sort(key=lambda x: x['name'])
    elif sort_order == 'name_desc':
        documents.sort(key=lambda x: x['name'], reverse=True)
    
    # 페이지네이션
    total_count = len(documents)
    total_pages = (total_count + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    return {
        'documents': documents[start_idx:end_idx],
        'total_count': total_count,
        'total_pages': total_pages,
        'current_page': page
    }

# ==== 라우트 설정 ====

@app.route('/')
def index():
    """대시보드 메인 페이지"""
    update_system_status()
    resource_usage = get_resource_usage()
    system_info = get_system_info()
    available_systems = get_available_systems()
    
    # JSON 직렬화를 위한 변환
    activity_data_json = {
        'timestamps': json.dumps(activity_data['timestamps']),
        'data_analysis': json.dumps(activity_data['data_analysis']),
        'document_test': json.dumps(activity_data['document_test']),
        'youtube_api': json.dumps(activity_data['youtube_api'])
    }
    
    return render_template('index.html',
                          active_page='dashboard',
                          mcp_status=system_status['mcp_server'],
                          system_status=system_status,
                          resource_usage=resource_usage,
                          system_info=system_info,
                          available_systems=available_systems,
                          activity_data=activity_data_json,
                          recent_tasks=recent_tasks)

@app.route('/documents')
def documents():
    """문서 관리 페이지"""
    folder = request.args.get('folder', '')
    tag = request.args.get('tag', '')
    doc_type = request.args.get('type', '')
    query = request.args.get('query', '')
    sort_order = request.args.get('sort', CONFIG['default_sort'])
    page = int(request.args.get('page', 1))
    
    # 문서 데이터 가져오기
    docs_data = get_sample_documents(
        folder=folder,
        tag=tag,
        doc_type=doc_type,
        query=query,
        sort_order=sort_order,
        page=page,
        page_size=CONFIG['default_page_size']
    )
    
    # 모든 폴더 정보 가져오기
    folders = [
        {'name': folder['name'], 'path': folder['path'], 'count': sum(1 for doc in get_sample_documents()['documents'] if doc['folder'] == folder['path'])}
        for folder in CONFIG['default_folders']
    ]
    
    # 모든 태그 정보 가져오기
    tags = CONFIG['default_tags']
    
    return render_template('documents.html',
                          active_page='documents',
                          mcp_status=system_status['mcp_server'],
                          documents=docs_data['documents'],
                          total_pages=docs_data['total_pages'],
                          current_page=docs_data['current_page'],
                          folders=folders,
                          tags=tags,
                          document_types=CONFIG['document_types'],
                          current_folder=folder,
                          current_tag=tag,
                          current_type=doc_type,
                          search_query=query,
                          sort_order=sort_order)

@app.route('/analytics')
def analytics():
    """분석 페이지"""
    return render_template('analytics.html',
                          active_page='analytics',
                          mcp_status=system_status['mcp_server'])

@app.route('/context')
def context():
    """컨텍스트 생성 페이지"""
    return render_template('context.html',
                          active_page='context',
                          mcp_status=system_status['mcp_server'])

@app.route('/chat')
def chat():
    """채팅 인터페이스 페이지"""
    return render_template('chat.html',
                          active_page='chat',
                          mcp_status=system_status['mcp_server'])

@app.route('/settings')
def settings():
    """설정 페이지"""
    return render_template('settings.html',
                          active_page='settings',
                          mcp_status=system_status['mcp_server'])

@app.route('/logs')
def logs():
    """로그 페이지"""
    return render_template('logs.html',
                          active_page='logs',
                          mcp_status=system_status['mcp_server'])

@app.route('/system/data_analysis')
def system_data_analysis():
    """데이터 분석 시스템 페이지"""
    return render_template('system_data_analysis.html',
                          active_page='data_analysis',
                          mcp_status=system_status['mcp_server'])

@app.route('/system/youtube_api')
def system_youtube_api():
    """YouTube API 시스템 페이지"""
    return render_template('system_youtube_api.html',
                          active_page='youtube_api',
                          mcp_status=system_status['mcp_server'])

@app.route('/system/document_test')
def system_document_test():
    """문서 테스트 시스템 페이지"""
    return render_template('system_document_test.html',
                          active_page='document_test',
                          mcp_status=system_status['mcp_server'])

# ==== API 엔드포인트 ====

@app.route('/api/status')
def api_status():
    """시스템 상태 정보 API"""
    update_system_status()
    return jsonify({
        'system_status': system_status,
        'resource_usage': get_resource_usage(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/documents')
def api_documents():
    """문서 목록 API"""
    folder = request.args.get('folder', '')
    tag = request.args.get('tag', '')
    doc_type = request.args.get('type', '')
    query = request.args.get('query', '')
    sort_order = request.args.get('sort', CONFIG['default_sort'])
    page = int(request.args.get('page', 1))
    page_size = min(int(request.args.get('page_size', CONFIG['default_page_size'])), CONFIG['max_page_size'])
    
    docs_data = get_sample_documents(
        folder=folder,
        tag=tag,
        doc_type=doc_type,
        query=query,
        sort_order=sort_order,
        page=page,
        page_size=page_size
    )
    
    return jsonify(docs_data)

@app.route('/api/context/generate', methods=['POST'])
def api_generate_context():
    """컨텍스트 생성 API"""
    data = request.json
    documents = data.get('documents', [])
    query = data.get('query', '')
    persona = data.get('persona', 'technical')
    max_tokens = int(data.get('max_tokens', 4000))
    
    # 실제 구현에서는 컨텍스트 최적화 모듈 호출
    # 여기서는 예제 응답 반환
    context = {
        'optimized_context': f"This is an optimized context for query: {query} with persona: {persona}",
        'source_documents': documents,
        'token_count': max_tokens // 2,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify({
        'status': 'success',
        'context': context
    })

@app.route('/api/chat/message', methods=['POST'])
def api_chat_message():
    """채팅 메시지 API"""
    data = request.json
    message = data.get('message', '')
    context_id = data.get('context_id', '')
    
    # 실제 구현에서는 채팅 모듈 호출
    # 여기서는 예제 응답 반환
    response = {
        'text': f"This is a response to: {message}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify({
        'status': 'success',
        'response': response
    })

@app.route('/api/generate-documents')
def api_generate_documents():
    """테스트 문서 생성 API"""
    count = int(request.args.get('count', 10))
    output_dir = request.args.get('output_dir', 'temp/test_documents')
    
    # 실제 구현에서는 문서 생성 모듈 호출
    # 여기서는 예제 응답 반환
    return jsonify({
        'status': 'success',
        'message': f"Generated {count} test documents in {output_dir}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/run-tests')
def api_run_tests():
    """성능 테스트 실행 API"""
    test_type = request.args.get('test_type', 'document_performance')
    
    # 실제 구현에서는 테스트 모듈 호출
    # 여기서는 예제 응답 반환
    return jsonify({
        'status': 'success',
        'message': f"Started {test_type} test",
        'task_id': f"task-{int(time.time())}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# ==== 주요 기능 라우트 ====

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """문서 업로드 처리"""
    # 실제 구현에서는 문서 업로드 처리
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/create_document', methods=['POST'])
def create_document():
    """문서 생성 처리"""
    # 실제 구현에서는 문서 생성 처리
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/delete_document', methods=['POST'])
def delete_document():
    """문서 삭제 처리"""
    # 실제 구현에서는 문서 삭제 처리
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/view_document')
def view_document():
    """문서 조회 페이지"""
    doc_id = request.args.get('doc_id', '')
    
    # 실제 구현에서는 문서 조회 처리
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/edit_document')
def edit_document():
    """문서 편집 페이지"""
    doc_id = request.args.get('doc_id', '')
    
    # 실제 구현에서는 문서 편집 페이지로 이동
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/download_document')
def download_document():
    """문서 다운로드 처리"""
    doc_id = request.args.get('doc_id', '')
    
    # 실제 구현에서는 문서 다운로드 처리
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/analyze_document')
def analyze_document():
    """문서 분석 페이지"""
    doc_id = request.args.get('doc_id', '')
    
    # 실제 구현에서는 문서 분석 페이지로 이동
    # 여기서는 문서 목록 페이지로 리디렉션
    return redirect(url_for('documents'))

@app.route('/add_to_context')
def add_to_context():
    """컨텍스트에 문서 추가 처리"""
    doc_id = request.args.get('doc_id', '')
    
    # 실제 구현에서는 컨텍스트에 문서 추가 처리
    # 여기서는 컨텍스트 페이지로 리디렉션
    return redirect(url_for('context'))

@app.route('/get_document_content')
def get_document_content():
    """문서 내용 조회 API"""
    doc_id = request.args.get('doc_id', '')
    
    # 실제 구현에서는 문서 내용 조회 처리
    # 여기서는 예제 내용 반환
    return f"Sample content for document {doc_id}\n\nThis is a sample document content for testing purposes."

@app.route('/data_analysis_run')
def data_analysis_run():
    """데이터 분석 실행 페이지"""
    # 실제 구현에서는 데이터 분석 실행 페이지로 이동
    # 여기서는 데이터 분석 시스템 페이지로 리디렉션
    return redirect(url_for('system_data_analysis'))

@app.route('/generate_context')
def generate_context():
    """컨텍스트 생성 페이지"""
    # 실제 구현에서는 컨텍스트 생성 페이지로 이동
    # 여기서는 컨텍스트 페이지로 리디렉션
    return redirect(url_for('context'))

@app.route('/run_document_test')
def run_document_test():
    """문서 테스트 실행 페이지"""
    # 실제 구현에서는 문서 테스트 실행 페이지로 이동
    # 여기서는 문서 테스트 시스템 페이지로 리디렉션
    return redirect(url_for('system_document_test'))

@app.route('/youtube_search')
def youtube_search():
    """YouTube 검색 페이지"""
    # 실제 구현에서는 YouTube 검색 페이지로 이동
    # 여기서는 YouTube API 시스템 페이지로 리디렉션
    return redirect(url_for('system_youtube_api'))

# ==== 애플리케이션 실행 ====

def run_dashboard(host='127.0.0.1', port=5000, debug=False):
    """웹 대시보드 실행"""
    # 대시보드 초기화
    initialize_dashboard()
    
    # Flask 앱 실행
    app.run(host=host, port=port, debug=debug)

# 직접 실행 시 대시보드 시작
if __name__ == '__main__':
    run_dashboard(debug=True)

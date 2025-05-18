"""
OneDrive 커넥터 클래스 - OneDrive와의 통합을 위한 모듈

이 모듈은 OneDrive(Microsoft 365)와의 통합을 위한 기능을 제공합니다.
파일 목록 조회, 파일 다운로드/업로드, 변경 감지 등의 기능을 구현합니다.
"""

import os
import json
import logging
import datetime
import hashlib
import time
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

class OneDriveConnector:
    """
    OneDrive와의 통합을 위한 클래스
    간소화된 버전으로, 실제 API 대신 로컬 파일 시스템을 통한 접근 제공
    """
    
    def __init__(self, config=None):
        """
        OneDriveConnector 초기화
        
        Args:
            config (dict, optional): OneDrive 연결 설정
        """
        self.logger = logger
        self.config = config or {}
        
        # OneDrive 루트 경로 (로컬 동기화 폴더)
        self.root_path = self.config.get('root_path', os.path.join(os.path.expanduser('~'), 'OneDrive'))
        
        # 캐시 저장 디렉토리
        self.cache_dir = self.config.get('cache_dir', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache', 'onedrive'))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 파일 상태 캐시 파일
        self.file_status_cache = os.path.join(self.cache_dir, 'file_status.json')
        
        # 로드된 파일 상태 캐시
        self.file_status = self._load_file_status_cache()
    
    def _load_file_status_cache(self):
        """
        파일 상태 캐시 로드
        
        Returns:
            dict: 파일 상태 캐시
        """
        if os.path.exists(self.file_status_cache):
            try:
                with open(self.file_status_cache, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                self.logger.warning("파일 상태 캐시 로드 실패, 새로운 캐시 생성")
        
        return {}
    
    def _save_file_status_cache(self):
        """
        파일 상태 캐시 저장
        """
        try:
            with open(self.file_status_cache, 'w', encoding='utf-8') as f:
                json.dump(self.file_status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"파일 상태 캐시 저장 실패: {str(e)}")
    
    def _get_onedrive_path(self, relative_path):
        """
        OneDrive 상대 경로를 절대 경로로 변환
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            
        Returns:
            str: 절대 경로
        """
        # 경로 정규화
        relative_path = relative_path.replace('\\', '/').strip('/')
        
        # 절대 경로 반환
        return os.path.join(self.root_path, relative_path)
    
    def _calculate_file_hash(self, file_path):
        """
        파일의 MD5 해시 계산
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            str: MD5 해시 값
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"파일 해시 계산 중 오류 발생: {str(e)}")
            return None
    
    def _get_file_info(self, file_path):
        """
        파일 정보 조회
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            dict: 파일 정보
        """
        try:
            file_stat = os.stat(file_path)
            file_info = {
                'path': file_path,
                'size': file_stat.st_size,
                'created_time': datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified_time': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'hash': self._calculate_file_hash(file_path),
                'last_checked': datetime.datetime.now().isoformat()
            }
            return file_info
        except Exception as e:
            self.logger.error(f"파일 정보 조회 중 오류 발생: {str(e)}")
            return None
    
    def list_files(self, relative_path, file_pattern=None, recursive=False):
        """
        OneDrive 디렉토리 내 파일 목록 조회
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            file_pattern (str, optional): 파일 패턴 (예: '*.docx')
            recursive (bool): 하위 디렉토리까지 검색 여부
            
        Returns:
            list: 파일 경로 목록
        """
        try:
            onedrive_path = self._get_onedrive_path(relative_path)
            
            # 검색 시작 디렉토리가 존재하는지 확인
            if not os.path.exists(onedrive_path):
                self.logger.error(f"디렉토리가 존재하지 않음: {onedrive_path}")
                return []
            
            # 모든 파일 경로 수집
            files = []
            
            if recursive:
                # 재귀적 검색
                for root, _, filenames in os.walk(onedrive_path):
                    for filename in filenames:
                        # 패턴 일치 여부 확인
                        if file_pattern is None or Path(filename).match(file_pattern):
                            files.append(os.path.join(root, filename))
            else:
                # 비재귀적 검색 (현재 디렉토리만)
                for filename in os.listdir(onedrive_path):
                    file_path = os.path.join(onedrive_path, filename)
                    if os.path.isfile(file_path):
                        # 패턴 일치 여부 확인
                        if file_pattern is None or Path(filename).match(file_pattern):
                            files.append(file_path)
            
            return files
        except Exception as e:
            self.logger.error(f"파일 목록 조회 중 오류 발생: {str(e)}")
            return []
    
    def list_new_files(self, relative_path, file_pattern=None, recursive=False):
        """
        OneDrive 디렉토리 내 신규/변경된 파일 목록 조회
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            file_pattern (str, optional): 파일 패턴 (예: '*.docx')
            recursive (bool): 하위 디렉토리까지 검색 여부
            
        Returns:
            list: 신규/변경된 파일 경로 목록
        """
        # 모든 파일 목록 조회
        all_files = self.list_files(relative_path, file_pattern, recursive)
        
        # 신규/변경된 파일 필터링
        new_files = []
        
        for file_path in all_files:
            # 파일 정보 조회
            file_info = self._get_file_info(file_path)
            
            if file_info is None:
                continue
            
            # 캐시에 없거나 변경된 경우 신규 파일로 간주
            if file_path not in self.file_status or \
               self.file_status[file_path]['hash'] != file_info['hash'] or \
               self.file_status[file_path]['modified_time'] != file_info['modified_time']:
                
                new_files.append(file_path)
                
                # 파일 상태 캐시 업데이트
                self.file_status[file_path] = file_info
        
        # 캐시 저장
        self._save_file_status_cache()
        
        return new_files
    
    def get_file_content(self, relative_path):
        """
        OneDrive 파일 내용 조회
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            
        Returns:
            bytes: 파일 내용
        """
        try:
            file_path = self._get_onedrive_path(relative_path)
            
            # 파일 존재 여부 확인
            if not os.path.isfile(file_path):
                self.logger.error(f"파일이 존재하지 않음: {file_path}")
                return None
            
            # 파일 내용 읽기
            with open(file_path, 'rb') as f:
                content = f.read()
            
            return content
        except Exception as e:
            self.logger.error(f"파일 내용 조회 중 오류 발생: {str(e)}")
            return None
    
    def get_file_text(self, relative_path, encoding='utf-8'):
        """
        OneDrive 텍스트 파일 내용 조회
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            encoding (str): 파일 인코딩
            
        Returns:
            str: 파일 텍스트 내용
        """
        try:
            file_path = self._get_onedrive_path(relative_path)
            
            # 파일 존재 여부 확인
            if not os.path.isfile(file_path):
                self.logger.error(f"파일이 존재하지 않음: {file_path}")
                return None
            
            # 파일 내용 읽기
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return content
        except Exception as e:
            self.logger.error(f"파일 텍스트 내용 조회 중 오류 발생: {str(e)}")
            return None
    
    def create_directory(self, relative_path):
        """
        OneDrive 디렉토리 생성
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            dir_path = self._get_onedrive_path(relative_path)
            
            # 디렉토리가 이미 존재하는 경우
            if os.path.exists(dir_path):
                return True
            
            # 디렉토리 생성
            os.makedirs(dir_path, exist_ok=True)
            
            return True
        except Exception as e:
            self.logger.error(f"디렉토리 생성 중 오류 발생: {str(e)}")
            return False
    
    def write_file(self, relative_path, content, overwrite=True):
        """
        OneDrive 파일 쓰기
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            content (bytes or str): 파일 내용
            overwrite (bool): 기존 파일 덮어쓰기 여부
            
        Returns:
            bool: 성공 여부
        """
        try:
            file_path = self._get_onedrive_path(relative_path)
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 파일이 이미 존재하고 덮어쓰기가 비활성화된 경우
            if os.path.exists(file_path) and not overwrite:
                self.logger.error(f"파일이 이미 존재함 (덮어쓰기 비활성화됨): {file_path}")
                return False
            
            # 파일 쓰기 모드 결정
            mode = 'wb' if isinstance(content, bytes) else 'w'
            encoding = None if isinstance(content, bytes) else 'utf-8'
            
            # 파일 쓰기
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            # 파일 상태 캐시 업데이트
            file_info = self._get_file_info(file_path)
            if file_info:
                self.file_status[file_path] = file_info
                self._save_file_status_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"파일 쓰기 중 오류 발생: {str(e)}")
            return False
    
    def delete_file(self, relative_path):
        """
        OneDrive 파일 삭제
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            file_path = self._get_onedrive_path(relative_path)
            
            # 파일 존재 여부 확인
            if not os.path.isfile(file_path):
                self.logger.error(f"파일이 존재하지 않음: {file_path}")
                return False
            
            # 파일 삭제
            os.remove(file_path)
            
            # 파일 상태 캐시에서 제거
            if file_path in self.file_status:
                del self.file_status[file_path]
                self._save_file_status_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"파일 삭제 중 오류 발생: {str(e)}")
            return False
    
    def copy_file(self, source_path, target_path, overwrite=True):
        """
        OneDrive 파일 복사
        
        Args:
            source_path (str): 소스 파일 상대 경로
            target_path (str): 대상 파일 상대 경로
            overwrite (bool): 기존 파일 덮어쓰기 여부
            
        Returns:
            bool: 성공 여부
        """
        try:
            source_full_path = self._get_onedrive_path(source_path)
            target_full_path = self._get_onedrive_path(target_path)
            
            # 소스 파일 존재 여부 확인
            if not os.path.isfile(source_full_path):
                self.logger.error(f"소스 파일이 존재하지 않음: {source_full_path}")
                return False
            
            # 대상 파일이 이미 존재하고 덮어쓰기가 비활성화된 경우
            if os.path.exists(target_full_path) and not overwrite:
                self.logger.error(f"대상 파일이 이미 존재함 (덮어쓰기 비활성화됨): {target_full_path}")
                return False
            
            # 대상 디렉토리 생성
            os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
            
            # 파일 복사
            import shutil
            shutil.copy2(source_full_path, target_full_path)
            
            # 파일 상태 캐시 업데이트
            file_info = self._get_file_info(target_full_path)
            if file_info:
                self.file_status[target_full_path] = file_info
                self._save_file_status_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"파일 복사 중 오류 발생: {str(e)}")
            return False
    
    def move_file(self, source_path, target_path, overwrite=True):
        """
        OneDrive 파일 이동
        
        Args:
            source_path (str): 소스 파일 상대 경로
            target_path (str): 대상 파일 상대 경로
            overwrite (bool): 기존 파일 덮어쓰기 여부
            
        Returns:
            bool: 성공 여부
        """
        try:
            source_full_path = self._get_onedrive_path(source_path)
            target_full_path = self._get_onedrive_path(target_path)
            
            # 소스 파일 존재 여부 확인
            if not os.path.isfile(source_full_path):
                self.logger.error(f"소스 파일이 존재하지 않음: {source_full_path}")
                return False
            
            # 대상 파일이 이미 존재하고 덮어쓰기가 비활성화된 경우
            if os.path.exists(target_full_path) and not overwrite:
                self.logger.error(f"대상 파일이 이미 존재함 (덮어쓰기 비활성화됨): {target_full_path}")
                return False
            
            # 대상 디렉토리 생성
            os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
            
            # 파일 이동
            import shutil
            shutil.move(source_full_path, target_full_path)
            
            # 파일 상태 캐시 업데이트
            if source_full_path in self.file_status:
                del self.file_status[source_full_path]
            
            file_info = self._get_file_info(target_full_path)
            if file_info:
                self.file_status[target_full_path] = file_info
                self._save_file_status_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"파일 이동 중 오류 발생: {str(e)}")
            return False
    
    def wait_for_sync(self, timeout=60):
        """
        OneDrive 동기화 대기
        
        Args:
            timeout (int): 최대 대기 시간 (초)
            
        Returns:
            bool: 성공 여부
        """
        # 참고: 이 메서드는 실제 OneDrive 동기화 상태를 확인할 수 없으므로 의미 있는 구현 제공 불가
        # 실제 구현에서는 OneDrive API 또는 이벤트 모니터링을 통해 동기화 상태를 확인해야 함
        self.logger.warning("OneDrive 동기화 상태 확인 기능은 이 구현에서 지원되지 않음, 대신 지정된 시간만큼 대기함")
        
        # 지정된 시간만큼 대기
        time.sleep(timeout)
        
        return True
    
    def get_file_metadata(self, relative_path):
        """
        OneDrive 파일 메타데이터 조회
        
        Args:
            relative_path (str): OneDrive 내 상대 경로
            
        Returns:
            dict: 파일 메타데이터
        """
        try:
            file_path = self._get_onedrive_path(relative_path)
            
            # 파일 존재 여부 확인
            if not os.path.isfile(file_path):
                self.logger.error(f"파일이 존재하지 않음: {file_path}")
                return None
            
            # 파일 메타데이터 조회
            file_stat = os.stat(file_path)
            
            metadata = {
                'name': os.path.basename(file_path),
                'path': file_path,
                'size': file_stat.st_size,
                'created_time': datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified_time': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'file_extension': os.path.splitext(file_path)[1][1:].lower()
            }
            
            return metadata
        except Exception as e:
            self.logger.error(f"파일 메타데이터 조회 중 오류 발생: {str(e)}")
            return None

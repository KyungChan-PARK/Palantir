"""
알림 관리자 클래스 - 시스템 알림 관리를 위한 모듈

이 모듈은 다양한 알림 매체(이메일, 로그 파일, 콘솔)를 통해 알림을 전송합니다.
시스템 이벤트, 데이터 품질 이슈, 온톨로지 문제 등에 대한 알림 생성 및 관리를 담당합니다.
"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

class NotificationManager:
    """
    시스템 알림 관리 클래스
    여러 채널을 통한 알림 전송을 지원
    """
    
    def __init__(self, config=None):
        """
        NotificationManager 초기화
        
        Args:
            config (dict, optional): 알림 설정
        """
        self.logger = logger
        self.config = config or {}
        
        # 알림 이력 저장 디렉토리
        self.history_dir = self.config.get('history_dir', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'notifications'))
        os.makedirs(self.history_dir, exist_ok=True)
    
    def send_notification(self, notification):
        """
        알림 전송
        
        Args:
            notification (dict): 알림 내용
                필수 키:
                - level: 알림 레벨 (info, warning, error)
                - title: 알림 제목
                - message: 알림 메시지
                선택적 키:
                - timestamp: 알림 발생 시간 (ISO 형식)
                - details: 추가 세부 정보
                
        Returns:
            bool: 성공 여부
        """
        if not self._validate_notification(notification):
            return False
        
        # 타임스탬프 추가 (없는 경우)
        if 'timestamp' not in notification:
            notification['timestamp'] = datetime.now().isoformat()
        
        # 알림 전송 (구성된 모든 채널로)
        success = True
        
        # 콘솔 로깅
        if self.config.get('log_to_console', True):
            success = success and self._log_to_console(notification)
        
        # 파일 로깅
        if self.config.get('log_to_file', True):
            success = success and self._log_to_file(notification)
        
        # 이메일 전송
        if self.config.get('send_email', False) and notification['level'] in self.config.get('email_levels', ['error']):
            success = success and self._send_email(notification)
        
        # 알림 이력 저장
        self._save_notification_history(notification)
        
        return success
    
    def _validate_notification(self, notification):
        """
        알림 형식 유효성 검사
        
        Args:
            notification (dict): 검증할 알림
            
        Returns:
            bool: 유효성 여부
        """
        required_fields = ['level', 'title', 'message']
        for field in required_fields:
            if field not in notification:
                self.logger.error(f"알림에 필수 필드가 누락됨: {field}")
                return False
        
        valid_levels = ['info', 'warning', 'error']
        if notification['level'] not in valid_levels:
            self.logger.error(f"유효하지 않은 알림 레벨: {notification['level']}")
            return False
        
        return True
    
    def _log_to_console(self, notification):
        """
        콘솔에 알림 로깅
        
        Args:
            notification (dict): 알림 내용
            
        Returns:
            bool: 성공 여부
        """
        try:
            level = notification['level']
            message = f"[{notification['timestamp']}] [{level.upper()}] {notification['title']}: {notification['message']}"
            
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            
            return True
        except Exception as e:
            self.logger.error(f"콘솔 로깅 중 오류 발생: {str(e)}")
            return False
    
    def _log_to_file(self, notification):
        """
        파일에 알림 로깅
        
        Args:
            notification (dict): 알림 내용
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 로그 디렉토리
            log_dir = self.config.get('log_dir', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs'))
            os.makedirs(log_dir, exist_ok=True)
            
            # 날짜별 로그 파일
            log_date = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(log_dir, f"notifications_{log_date}.log")
            
            # 로그 메시지 형식
            timestamp = notification['timestamp']
            level = notification['level'].upper()
            title = notification['title']
            message = notification['message']
            log_message = f"[{timestamp}] [{level}] {title}: {message}\n"
            
            # 파일에 추가
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
            
            return True
        except Exception as e:
            self.logger.error(f"파일 로깅 중 오류 발생: {str(e)}")
            return False
    
    def _send_email(self, notification):
        """
        이메일로 알림 전송
        
        Args:
            notification (dict): 알림 내용
            
        Returns:
            bool: 성공 여부
        """
        if 'email' not in self.config:
            self.logger.warning("이메일 설정이 구성되지 않음")
            return False
        
        try:
            email_config = self.config['email']
            
            # 이메일 설정 확인
            required_fields = ['smtp_server', 'smtp_port', 'username', 'password', 'from_addr', 'to_addrs']
            for field in required_fields:
                if field not in email_config:
                    self.logger.error(f"이메일 설정에 필수 필드가 누락됨: {field}")
                    return False
            
            # 이메일 내용 생성
            msg = MIMEMultipart()
            msg['From'] = email_config['from_addr']
            msg['To'] = ", ".join(email_config['to_addrs'])
            msg['Subject'] = f"[{notification['level'].upper()}] {notification['title']}"
            
            # 이메일 본문
            body = f"""
            <html>
            <body>
                <h2>{notification['title']}</h2>
                <p><strong>레벨:</strong> {notification['level'].upper()}</p>
                <p><strong>시간:</strong> {notification['timestamp']}</p>
                <p><strong>메시지:</strong> {notification['message']}</p>
                
                {self._format_details_html(notification) if 'details' in notification else ''}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # 이메일 전송
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            self.logger.error(f"이메일 전송 중 오류 발생: {str(e)}")
            return False
    
    def _format_details_html(self, notification):
        """
        알림 세부 정보의 HTML 포맷팅
        
        Args:
            notification (dict): 알림 내용
            
        Returns:
            str: HTML 형식의 세부 정보
        """
        details = notification.get('details', {})
        if not details:
            return ""
        
        # 세부 정보가 딕셔너리인 경우
        if isinstance(details, dict):
            html = "<h3>세부 정보:</h3><table border='1'><tr><th>속성</th><th>값</th></tr>"
            for key, value in details.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            html += "</table>"
            return html
        
        # 세부 정보가 리스트인 경우
        elif isinstance(details, list):
            html = "<h3>세부 정보:</h3><ul>"
            for item in details:
                if isinstance(item, dict):
                    html += "<li><table border='1'><tr><th>속성</th><th>값</th></tr>"
                    for key, value in item.items():
                        html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                    html += "</table></li>"
                else:
                    html += f"<li>{item}</li>"
            html += "</ul>"
            return html
        
        # 그 외의 경우
        else:
            return f"<h3>세부 정보:</h3><p>{details}</p>"
    
    def _save_notification_history(self, notification):
        """
        알림 이력 저장
        
        Args:
            notification (dict): 알림 내용
        """
        try:
            # 이력 파일명 (날짜별)
            history_date = datetime.now().strftime("%Y%m%d")
            history_file = os.path.join(self.history_dir, f"notification_history_{history_date}.jsonl")
            
            # 파일에 추가
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(notification, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"알림 이력 저장 중 오류 발생: {str(e)}")
    
    def get_recent_notifications(self, count=10, level=None):
        """
        최근 알림 이력 조회
        
        Args:
            count (int): 조회할 알림 수
            level (str, optional): 특정 레벨만 조회
            
        Returns:
            list: 알림 목록
        """
        notifications = []
        
        try:
            # 이력 파일 목록 (최신순)
            history_files = sorted([f for f in os.listdir(self.history_dir) if f.startswith('notification_history_')], reverse=True)
            
            # 파일별로 알림 로드
            for file in history_files:
                file_path = os.path.join(self.history_dir, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            notification = json.loads(line.strip())
                            
                            # 레벨 필터링
                            if level is None or notification['level'] == level:
                                notifications.append(notification)
                                
                                # 충분한 알림을 로드한 경우 종료
                                if len(notifications) >= count:
                                    return notifications
                        except:
                            continue
        except Exception as e:
            self.logger.error(f"알림 이력 조회 중 오류 발생: {str(e)}")
        
        return notifications

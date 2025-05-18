import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_PATH = "."  # 감시할 경로

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # .git, .venv, __pycache__ 등은 무시
        if any(x in event.src_path for x in ['.git', '.venv', '__pycache__']):
            return
        try:
            subprocess.run(["git", "add", "-A"])
            subprocess.run(["git", "commit", "-m", "auto: Codex 실시간 동기화"], check=False)
            # push는 post-commit hook에서 처리됨
        except Exception as e:
            print(f"[오류] {e}")

if __name__ == "__main__":
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_PATH, recursive=True)
    observer.start()
    print("[실행중] 파일 변경 감지 및 자동 git 커밋...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 
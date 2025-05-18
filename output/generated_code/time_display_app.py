import tkinter as tk
from datetime import datetime
import threading
import time

class TimeDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("시간 표시 앱")
        self.root.geometry("400x200")
        self.root.resizable(False, False)
        
        # 시간 표시 레이블
        self.time_label = tk.Label(
            root, 
            text="", 
            font=("Helvetica", 48),
            fg="#333333"
        )
        self.time_label.pack(expand=True)
        
        # 날짜 표시 레이블
        self.date_label = tk.Label(
            root, 
            text="", 
            font=("Helvetica", 16),
            fg="#666666"
        )
        self.date_label.pack(expand=True)
        
        # 시간 업데이트 시작
        self.update_time()
        
        # 배경색 변경 버튼
        self.color_button = tk.Button(
            root,
            text="배경색 변경",
            command=self.change_background,
            font=("Helvetica", 12),
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.color_button.pack(pady=20)
        
        # 색상 목록
        self.colors = ["#f0f0f0", "#e6f7ff", "#fff0f0", "#f0fff0", "#fff0f5"]
        self.current_color = 0
        
        # 초기 배경색 설정
        self.root.configure(bg=self.colors[self.current_color])
        self.time_label.configure(bg=self.colors[self.current_color])
        self.date_label.configure(bg=self.colors[self.current_color])
    
    def update_time(self):
        """시간과 날짜 업데이트"""
        now = datetime.now()
        
        # 시간 형식 (24시간)
        time_string = now.strftime("%H:%M:%S")
        self.time_label.config(text=time_string)
        
        # 날짜 형식 (년-월-일 요일)
        weekday_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        weekday = weekday_names[now.weekday()]
        date_string = now.strftime("%Y년 %m월 %d일") + f" ({weekday})"
        self.date_label.config(text=date_string)
        
        # 1초마다 업데이트
        self.root.after(1000, self.update_time)
    
    def change_background(self):
        """배경색 변경"""
        self.current_color = (self.current_color + 1) % len(self.colors)
        new_color = self.colors[self.current_color]
        
        # 배경색 적용
        self.root.configure(bg=new_color)
        self.time_label.configure(bg=new_color)
        self.date_label.configure(bg=new_color)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeDisplayApp(root)
    root.mainloop()

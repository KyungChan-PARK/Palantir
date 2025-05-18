import os
import subprocess
import sys
import tempfile
import json

def get_api_key():
    """환경 변수에서 API 키를 가져오거나 사용자에게 입력받음"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    api_key = input("OpenAI API 키를 입력하세요: ")
    os.environ['OPENAI_API_KEY'] = api_key
    return api_key

def check_codex_installation():
    """Codex CLI가 설치되어 있는지 확인하고 없으면 설치"""
    try:
        subprocess.run(['codex', '--version'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
        print("Codex CLI가 이미 설치되어 있습니다.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Codex CLI가 설치되어 있지 않습니다. 설치를 시작합니다...")
        try:
            subprocess.run(['npm', 'install', '-g', '@openai/codex'], check=True)
            print("Codex CLI가 성공적으로 설치되었습니다.")
            return True
        except subprocess.CalledProcessError:
            print("Codex CLI 설치에 실패했습니다.")
            return False

def run_codex_command(prompt, model='o4-mini', approval_mode='auto-edit'):
    """Codex 명령어 실행"""
    api_key = get_api_key()
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = api_key
    
    # 임시 스크립트 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        script_path = f.name
    
    # Codex 명령어 구성
    command = [
        'codex',
        f'--model={model}',
        f'--approval-mode={approval_mode}',
        '--quiet',
        f'Python으로 다음을 구현하세요: {prompt}. 결과 파일을 {script_path}에 저장하세요.'
    ]
    
    print(f"\n실행 중인 명령어: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("명령어 실행 결과:")
        print(f"반환 코드: {result.returncode}")
        
        if result.stdout:
            print(f"표준 출력:\n{result.stdout}")
        
        if result.stderr:
            print(f"표준 오류:\n{result.stderr}")
        
        # 생성된 스크립트 파일 확인
        if os.path.exists(script_path) and os.path.getsize(script_path) > 0:
            print(f"\n생성된 스크립트 ({script_path}):")
            with open(script_path, 'r') as f:
                script_content = f.read()
                print(script_content)
            
            # 스크립트를 프로젝트 디렉토리에 복사
            output_path = os.path.join(os.getcwd(), 'output', 'generated_code', os.path.basename(script_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(script_content)
            
            print(f"\n스크립트가 다음 위치에 저장되었습니다: {output_path}")
            
            # 생성된 스크립트 실행 여부 확인
            run_script = input("생성된 스크립트를 실행하시겠습니까? (y/n): ")
            if run_script.lower() == 'y':
                print("\n스크립트 실행 결과:")
                subprocess.run([sys.executable, script_path])
        else:
            print(f"생성된 스크립트가 없거나 비어 있습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
    
    # 임시 파일 삭제
    try:
        os.unlink(script_path)
    except:
        pass

def main():
    """메인 함수"""
    print("OpenAI Codex CLI 통합 도구")
    print("=" * 30)
    
    if not check_codex_installation():
        print("Codex CLI 설치 실패로 프로그램을 종료합니다.")
        return
    
    while True:
        print("\n옵션:")
        print("1. 대화형 모드 실행")
        print("2. 명령어 모드 실행")
        print("3. 프로그램 종료")
        
        choice = input("\n선택: ")
        
        if choice == '1':
            model = input("모델 선택 (기본값: o4-mini): ") or 'o4-mini'
            approval_mode = input("승인 모드 선택 (suggest/auto-edit/full-auto, 기본값: auto-edit): ") or 'auto-edit'
            
            # 대화형 모드 실행
            env = os.environ.copy()
            env['OPENAI_API_KEY'] = get_api_key()
            
            subprocess.run(['codex', f'--model={model}', f'--approval-mode={approval_mode}'], env=env)
            
        elif choice == '2':
            model = input("모델 선택 (기본값: o4-mini): ") or 'o4-mini'
            approval_mode = input("승인 모드 선택 (suggest/auto-edit/full-auto, 기본값: auto-edit): ") or 'auto-edit'
            prompt = input("프롬프트 입력: ")
            
            if prompt:
                run_codex_command(prompt, model, approval_mode)
            else:
                print("프롬프트가 비어 있습니다.")
                
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
            
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()

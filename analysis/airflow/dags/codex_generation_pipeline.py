from datetime import datetime
import os
import subprocess
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# 기본 설정
DAG_ID = 'codex_code_generation_pipeline'
DEFAULT_ARGS = {
    'owner': 'palantir',
    'depends_on_past': False,
    'retries': 0,
}

# Codex CLI 실행 함수
def run_codex_cli(**context):
    """Airflow 태스크에서 Codex CLI를 호출해 스캐폴딩 코드 생성"""
    # 1) 프롬프트 결정: Airflow Variable 에 저장하거나 XCom 으로 전달
    prompt_text = Variable.get('codex_prompt_text', default_var=None)
    prompt_file = Variable.get('codex_prompt_file', default_var=None)

    if not prompt_text and not prompt_file:
        raise ValueError('codex_prompt_text 또는 codex_prompt_file 변수를 Airflow에 설정해야 합니다.')

    if prompt_file:
        # Codex CLI 에 --prompt <file> 방식 사용
        command = [
            'codex',
            '--model=o4-mini',
            '--approval-mode=suggest',
            f'--file={prompt_file}'
        ]
    else:
        # 직접 프롬프트 텍스트 전달
        command = [
            'codex',
            '--model=o4-mini',
            '--approval-mode=suggest',
            prompt_text
        ]

    # Codex 작업 결과물을 저장할 경로
    output_dir = os.path.join(os.getcwd(), 'output', 'generated_code', datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()

    # 실행
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # 결과 로그 저장
    with open(os.path.join(output_dir, 'codex_stdout.log'), 'w', encoding='utf-8') as f_out:
        f_out.write(result.stdout)
    with open(os.path.join(output_dir, 'codex_stderr.log'), 'w', encoding='utf-8') as f_err:
        f_err.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f'Codex CLI 실패: {result.stderr}')

    context['ti'].xcom_push(key='codex_output_dir', value=output_dir)


def notify_completion(**context):
    """Codex 생성 완료 알림(추후 Slack/Webhook 등으로 확장 가능)"""
    output_dir = context['ti'].xcom_pull(key='codex_output_dir')
    print(f'Codex 코드 생성 완료. 결과 경로: {output_dir}')


with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2025, 5, 18),
    catchup=False,
    tags=['codex', 'automation'],
) as dag:

    generate_code = PythonOperator(
        task_id='generate_code_with_codex',
        python_callable=run_codex_cli,
        provide_context=True,
    )

    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
    )

    generate_code >> notify 
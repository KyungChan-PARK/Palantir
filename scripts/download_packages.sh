#!/usr/bin/env bash
# 목적: requirements.txt + extras(airflow[docker], dash, fastapi-offline 등)를 wheel 형태로 vendor/에 미리 내려받음
set -e
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

# 1) poetry.lock가 있으면 export, 없으면 requirements.txt 사용
if [ -f pyproject.toml ]; then
  poetry export --without-hashes -f requirements.txt > /tmp/req.txt
else
  cp requirements.txt /tmp/req.txt
fi

# 2) 필수/추가 패키지 병합
echo "apache-airflow[celery,postgres]==3.0.*"       >> /tmp/req.txt
echo "dash>=2.17,<3"                                >> /tmp/req.txt
echo "fastapi-offline"                              >> /tmp/req.txt

# 3) 모든 dependency 를 wheel 형태로 vendor/에 저장
mkdir -p vendor
python -m pip download -r /tmp/req.txt --dest vendor

if [ $? -eq 0 ]; then
  echo "✅  vendor/ 에 $(ls vendor | wc -l) 개의 wheel 다운로드 완료"
else
  echo "❌  wheel 다운로드 실패"
  exit 1
fi 
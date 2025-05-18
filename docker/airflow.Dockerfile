# Custom Airflow image with project deps
FROM apache/airflow:2.8.1-python3.11
USER root
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt
USER airflow 
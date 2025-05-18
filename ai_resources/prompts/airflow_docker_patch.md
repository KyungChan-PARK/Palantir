# Prompt — Build Custom Airflow Image & Patch Compose

**Goal**
1) Generate `docker/airflow.Dockerfile` that installs all Python deps listed in `requirements.txt`.
2) Produce a YAML patch snippet to replace the Airflow service in `docker-compose.yml` so it uses the built image (`airflow-custom:latest`) and removes on-startup `pip install`.

**Context**
Repo root contains `requirements.txt` including pandas, pyspark, boto3, neo4j, great_expectations, psycopg2-binary, apache-airflow==2.8.1 등.

**Output Format**
```dockerfile
# docker/airflow.Dockerfile
(complete file here)
```
```yaml
# docker-compose-patch.yaml (Airflow service only)
(airflow: ...)
```

Use best practices: switch to root, pip install, clean cache, switch back to airflow user. 
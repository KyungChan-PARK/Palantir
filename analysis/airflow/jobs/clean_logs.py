"""clean_logs.py
Spark Job: 원시 LMS 로그(csv) → 정제(parquet)

Usage (Airflow BashOperator):
    spark-submit analysis/airflow/jobs/clean_logs.py

Input : data/source/raw_logs.csv
Output: data/interim/clean.parquet + data/interim/clean.csv (Neo4j용)
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pathlib import Path

RAW_PATH = "data/source/raw_logs.csv"
PARQUET_PATH = "data/interim/clean.parquet"
CSV_PATH = "data/interim/clean.csv"

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("clean_logs")
        .getOrCreate()
    )

    df = (
        spark.read.option("header", True).csv(RAW_PATH)
    )

    # 간단 정제 예시: null 제거, 불필요 컬럼 삭제
    clean_df = (
        df.dropna(subset=["user_id", "course", "timestamp"])
          .select(
              col("user_id").alias("student_id"),
              col("course"),
              col("event"),
              col("timestamp")
          )
    )

    # 저장
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    clean_df.write.mode("overwrite").parquet(PARQUET_PATH)
    clean_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        CSV_PATH,
        sep=",",
    )

    spark.stop() 
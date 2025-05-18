"""Postgres 접속 및 간단 CRUD 테스트"""
import uuid
import psycopg2
import os

PG_CONN = {
    "dbname": os.getenv("POSTGRES_DB", "ontology"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "pass"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
}


def test_pg_connection():
    table = f"tmp_{uuid.uuid4().hex[:8]}"
    conn = psycopg2.connect(**PG_CONN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table}(id INT PRIMARY KEY);")
    cur.execute(f"INSERT INTO {table}(id) VALUES(1);")
    cur.execute(f"SELECT id FROM {table};")
    row = cur.fetchone()
    assert row[0] == 1
    cur.execute(f"DROP TABLE {table};")
    cur.close()
    conn.close() 
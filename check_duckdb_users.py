import duckdb

DB = r"C:/Users/packr/OneDrive/palantir/data/duckdb/palantir.db"
con = duckdb.connect(DB)

# 1. 테이블 목록
print(con.sql("SHOW TABLES").fetchall())

# 2. users 테이블 스키마
print(con.sql("DESCRIBE users").fetchdf())

# 3. users row 수
rows = con.sql("SELECT COUNT(*) AS n FROM users").fetchone()[0]
print(f"users rows: {rows}") 
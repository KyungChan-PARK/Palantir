import duckdb

DB = r"C:/Users/packr/OneDrive/palantir/data/duckdb/palantir.db"
con = duckdb.connect(DB)

con.execute("""
ALTER TABLE users
  ALTER COLUMN address SET DATA TYPE VARCHAR USING CAST(address AS VARCHAR);
""")

con.execute("""
ALTER TABLE users
  ALTER COLUMN company SET DATA TYPE VARCHAR USING CAST(company AS VARCHAR);
""")

print("✅ address, company 컬럼을 VARCHAR로 변환 완료") 
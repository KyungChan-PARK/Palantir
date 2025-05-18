from great_expectations.data_context import get_context

# loads great_expectations/ or creates one on first run
context = get_context()

# register or update a DuckDB datasource named "duckdb_conn"
context.sources.add_or_update_sql(
    name="duckdb_conn",
    connection_string="duckdb:///C:/Users/packr/OneDrive/palantir/data/duckdb/palantir.db",
)
print("✅ DuckDB datasource registered!") 
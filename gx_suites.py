from great_expectations.data_context import get_context

def build_users_suite():
    ctx = get_context()
    # DuckDB datasource 객체 가져오기 (Fluent API)
    ds = ctx.get_datasource("duckdb_conn")
    # users 테이블을 asset으로 등록 (이미 등록되어 있으면 무시)
    if "users" not in [a.name for a in ds.assets]:
        ds.add_table_asset(name="users", table_name="users")
    suite = ctx.add_or_update_expectation_suite("users.basic")
    v = ctx.get_validator(
        datasource_name="duckdb_conn",
        data_asset_name="users",
        expectation_suite_name=suite.expectation_suite_name
    )
    v.expect_column_values_to_not_be_null("id")
    v.expect_column_values_to_match_regex("website", r"^[a-z0-9.-]+\\.[a-z]{2,}$")
    v.save_expectation_suite()
    print("✅ users.basic suite built and saved.")

if __name__ == "__main__":
    build_users_suite() 
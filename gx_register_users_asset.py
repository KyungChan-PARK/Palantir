from great_expectations.data_context import get_context

ctx = get_context()
ds  = ctx.get_datasource("duckdb_conn")

# add the asset once; skip if already present
if "users" not in [a.name for a in ds.assets]:
    ds.add_table_asset(name="users", table_name="users")
    print("✅  users asset registered")
else:
    print("users asset already exists") 
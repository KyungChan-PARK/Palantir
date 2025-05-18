from great_expectations.data_context import get_context

context = get_context()

suite = context.add_or_update_expectation_suite(
    expectation_suite_name="users.basic"
)
print("✅ Suite created:", suite.expectation_suite_name) 
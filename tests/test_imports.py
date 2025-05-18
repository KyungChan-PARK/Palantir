import importlib
import pytest

MODULES = [
    "analysis.atoms.data_connector",
    "analysis.atoms.neo4j_connector",
    "analysis.atoms.airflow_connector",
    "analysis.atoms.onedrive_connector",
    "analysis.atoms.document_processor",
]

@pytest.mark.parametrize("module_name", MODULES)
def test_import_module(module_name):
    """Ensure key modules can be imported without errors."""
    importlib.import_module(module_name)

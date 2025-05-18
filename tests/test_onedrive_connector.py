import os
import hashlib
from analysis.atoms.onedrive_connector import OneDriveConnector


def _setup_connector(tmp_path):
    root = tmp_path / "onedrive"
    root.mkdir()
    cache = tmp_path / "cache"
    return OneDriveConnector({"root_path": str(root), "cache_dir": str(cache)})


def test_calculate_file_hash(tmp_path):
    connector = _setup_connector(tmp_path)
    f = tmp_path / "onedrive" / "file.txt"
    content = b"hello"
    f.write_bytes(content)
    calculated = connector._calculate_file_hash(str(f))
    assert calculated == hashlib.md5(content).hexdigest()


def test_list_new_files(tmp_path, monkeypatch):
    connector = _setup_connector(tmp_path)
    data_dir = tmp_path / "onedrive" / "data"
    data_dir.mkdir()
    f = data_dir / "a.txt"
    f.write_text("one")

    # first call should report the new file
    new_files = connector.list_new_files("data", file_pattern="*.txt", recursive=False)
    assert [str(f)] == new_files

    # subsequent call without changes should give empty list
    new_files = connector.list_new_files("data", file_pattern="*.txt", recursive=False)
    assert new_files == []

    # modify file contents to trigger change detection
    f.write_text("two")
    new_files = connector.list_new_files("data", file_pattern="*.txt", recursive=False)
    assert [str(f)] == new_files

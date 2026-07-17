"""Tests for Autonomous Lab retrieve (undo agent file edits)."""

from pathlib import Path

from pages.AutonomousLab.retrieve import (
    apply_retrieve_batch,
    capture_retrieve_batch,
    capture_retrieve_entry,
    push_retrieve_batch,
)


def test_capture_and_retrieve_modify(tmp_path: Path):
    path = tmp_path / "demo.py"
    path.write_text("alpha\n", encoding="utf-8")
    entry = capture_retrieve_entry(
        "modify_file",
        {"filepath": "demo.py", "new_content": "beta\n"},
        tmp_path,
    )
    assert entry is not None
    assert entry["kind"] == "modify"
    assert entry["original_content"] == "alpha\n"

    path.write_text("beta\n", encoding="utf-8")
    result = apply_retrieve_batch({"entries": [entry]})
    assert path.read_text(encoding="utf-8") == "alpha\n"
    assert "restored" in result[0]


def test_capture_and_retrieve_create_via_write(tmp_path: Path):
    entry = capture_retrieve_entry(
        "write_file",
        {"filepath": "new.txt", "content": "hello"},
        tmp_path,
    )
    assert entry is not None
    assert entry["kind"] == "create"
    assert entry["file_existed"] is False

    created = tmp_path / "new.txt"
    created.write_text("hello", encoding="utf-8")
    apply_retrieve_batch({"entries": [entry]})
    assert not created.exists()


def test_capture_and_retrieve_delete(tmp_path: Path):
    path = tmp_path / "keep_me.md"
    path.write_text("# keep\n", encoding="utf-8")
    entry = capture_retrieve_entry("delete_file", {"filepath": "keep_me.md"}, tmp_path)
    assert entry is not None
    assert entry["kind"] == "delete"

    path.unlink()
    apply_retrieve_batch({"entries": [entry]})
    assert path.read_text(encoding="utf-8") == "# keep\n"


def test_capture_and_retrieve_rename(tmp_path: Path):
    src = tmp_path / "old.py"
    dst = tmp_path / "new.py"
    src.write_text("x = 1\n", encoding="utf-8")
    entry = capture_retrieve_entry(
        "rename_file",
        {"filepath": "old.py", "new_filepath": "new.py"},
        tmp_path,
    )
    assert entry is not None
    assert entry["kind"] == "rename"

    src.rename(dst)
    apply_retrieve_batch({"entries": [entry]})
    assert src.exists()
    assert not dst.exists()
    assert src.read_text(encoding="utf-8") == "x = 1\n"


def test_capture_batch_from_action_requests(tmp_path: Path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    batch = capture_retrieve_batch(
        [
            {"name": "modify_file", "args": {"filepath": "a.py", "new_content": "b\n"}},
            {"name": "write_file", "args": {"filepath": "c.py", "content": "c\n"}},
            {"name": "compile_simulation", "args": {"case": "x"}},
        ],
        tmp_path,
    )
    assert batch is not None
    assert batch["count"] == 2
    assert len(batch["entries"]) == 2


def test_directory_delete_not_snapshotted(tmp_path: Path):
    d = tmp_path / "folder"
    d.mkdir()
    (d / "f.txt").write_text("x", encoding="utf-8")
    assert capture_retrieve_entry(
        "delete_file",
        {"filepath": "folder", "recursive": True},
        tmp_path,
    ) is None


def test_push_retrieve_batch_caps_stack():
    stack = []
    for i in range(25):
        stack = push_retrieve_batch(stack, {
            "id": f"b{i}",
            "entries": [],
            "label": str(i),
            "count": 0,
        })
    assert len(stack) == 20
    assert stack[0]["id"] == "b5"
    assert stack[-1]["id"] == "b24"

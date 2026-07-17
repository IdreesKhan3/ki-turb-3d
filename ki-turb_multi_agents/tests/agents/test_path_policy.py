"""
Path policy tests — writes must stay inside approved roots, and traversal /
symlink escapes must be denied.
"""

import os

import pytest

from agents.security.path_policy import PathPolicy, PathPolicyError, default_policy


def test_allows_write_inside_root(tmp_path):
    policy = PathPolicy([tmp_path])
    target = tmp_path / "exports" / "figure.png"
    assert policy.is_write_allowed(target) is True
    resolved = policy.resolve_write(target)
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_allows_nonexistent_file_inside_root(tmp_path):
    # Writes create new files; the target need not exist yet.
    policy = PathPolicy([tmp_path])
    target = tmp_path / "new_dir" / "new_file.dat"
    assert policy.is_write_allowed(target) is True


def test_denies_absolute_path_outside_root(tmp_path):
    policy = PathPolicy([tmp_path])
    assert policy.is_write_allowed("/etc/passwd") is False
    with pytest.raises(PathPolicyError):
        policy.resolve_write("/etc/passwd")


def test_denies_directory_traversal(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    policy = PathPolicy([root])
    escape = root / ".." / "secret.txt"
    assert policy.is_write_allowed(escape) is False
    with pytest.raises(PathPolicyError):
        policy.resolve_write(escape)


def test_denies_symlink_escape(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link = root / "link_to_outside"
    try:
        os.symlink(outside_dir, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    # A path that goes through the symlink resolves outside the root -> denied.
    target = link / "evil.txt"
    assert policy_write_allowed(root, target) is False


def policy_write_allowed(root, target):
    return PathPolicy([root]).is_write_allowed(target)


def test_multiple_roots(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    policy = PathPolicy([root_a, root_b])
    assert policy.is_write_allowed(root_a / "x.txt") is True
    assert policy.is_write_allowed(root_b / "y.txt") is True
    assert policy.is_write_allowed(tmp_path / "c" / "z.txt") is False


def test_default_policy_includes_project_and_export_case(tmp_path):
    policy = default_policy(tmp_path)
    assert policy.is_write_allowed(tmp_path / "app.py") is True
    assert policy.is_write_allowed(tmp_path / "exports" / "report.pdf") is True
    assert policy.is_write_allowed(tmp_path / "cases" / "case1" / "input.xml") is True
    assert policy.is_write_allowed("/tmp/evil.sh") is False


def test_empty_roots_rejected():
    with pytest.raises(ValueError):
        PathPolicy([])

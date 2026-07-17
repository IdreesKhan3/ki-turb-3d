"""
Shell policy tests — allowlist behavior and rejection of shell escapes.
"""

import pytest

from agents.security.shell_policy import (
    ShellPolicyError,
    check_command,
    is_command_allowed,
    to_argv,
)


@pytest.mark.parametrize("cmd", [
    "git status",
    "git log --oneline -n 5",
    "ls -la",
    "cat README.md",
    "grep -r pattern .",
    "find . -name '*.py'",
    "pwd",
])
def test_allows_common_inspection_commands(cmd):
    ok, reason = is_command_allowed(cmd)
    assert ok is True, reason
    assert to_argv(cmd)[0] == cmd.split()[0]


@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "sudo rm -rf /",
    "mkfs.ext4 /dev/sda",
    "shutdown now",
    "python -c 'import os; os.system(\"rm -rf /\")'",
    "python3 evil.py",
    "bash -c 'echo hi'",
    "pip install evil",
])
def test_denies_non_allowlisted_programs(cmd):
    ok, _ = is_command_allowed(cmd)
    assert ok is False
    with pytest.raises(ShellPolicyError):
        to_argv(cmd)


@pytest.mark.parametrize("cmd", [
    "ls | sh",
    "cat file; rm -rf /",
    "echo $(rm -rf /)",
    "cat a && rm b",
    "ls > /etc/passwd",
    "cat < /etc/shadow",
    "echo `whoami`",
    "ls & disown",
])
def test_denies_shell_metacharacters(cmd):
    ok, reason = is_command_allowed(cmd)
    assert ok is False, f"should reject: {cmd}"
    assert "operator" in reason or "allowlist" in reason or "parse" in reason


def test_denies_path_qualified_program():
    ok, reason = is_command_allowed("/usr/bin/python3 evil.py")
    assert ok is False
    assert "path-qualified" in reason or "operator" in reason


def test_denies_relative_path_program():
    ok, reason = is_command_allowed("./malware.sh")
    assert ok is False


def test_empty_command_denied():
    ok, _ = is_command_allowed("")
    assert ok is False
    assert is_command_allowed("   ")[0] is False


def test_check_command_raises_on_bad():
    with pytest.raises(ShellPolicyError):
        check_command("rm -rf /")
    # Allowed command returns None (no raise).
    assert check_command("git status") is None


def test_custom_allowlist():
    ok, _ = is_command_allowed("mytool --run", allowed=frozenset({"mytool"}))
    assert ok is True
    ok, _ = is_command_allowed("git status", allowed=frozenset({"mytool"}))
    assert ok is False

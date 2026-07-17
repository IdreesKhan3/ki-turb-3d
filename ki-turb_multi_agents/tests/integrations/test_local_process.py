"""Unit tests for LocalProcessRunner polling edge cases."""

from integrations.local_process import LocalProcessRunner


def test_poll_treats_zombie_as_exited(monkeypatch):
    runner = LocalProcessRunner()
    monkeypatch.setattr(runner, "_proc_state", lambda pid: "Z")
    monkeypatch.setattr(
        LocalProcessRunner,
        "_pid_alive",
        staticmethod(lambda pid: True),
    )
    assert runner.poll(424242) == 0

"""OpenLB compile lock must not block on dead holder processes."""
from __future__ import annotations

import os
from pathlib import Path

from agents.tools.simulation.openlb_compile_agent import OpenLBCompileAgent


def test_stale_build_lock_is_removed(tmp_path: Path):
    lock = tmp_path / ".kiturb-build.lock"
    lock.write_text("999999999", encoding="utf-8")
    agent = OpenLBCompileAgent()

    with agent._lock(lock, timeout=2):
        assert lock.is_file()
        assert lock.read_text().strip() == str(os.getpid())

    assert not lock.exists()

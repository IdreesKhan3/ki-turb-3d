"""Checkpoint/restart regression test.

Set both environment variables to NPZ files containing identically named arrays
from an uninterrupted reference run and a checkpoint/restarted run.
"""

import os
from pathlib import Path

import numpy as np
import pytest


def test_checkpoint_restart_matches_uninterrupted_run():
    reference = os.environ.get("KITURB_RESTART_REFERENCE")
    resumed = os.environ.get("KITURB_RESTART_RESUMED")
    if not reference or not resumed:
        pytest.skip("set KITURB_RESTART_REFERENCE and KITURB_RESTART_RESUMED to run")
    reference_path = Path(reference)
    resumed_path = Path(resumed)
    if not reference_path.is_file() or not resumed_path.is_file():
        pytest.skip("restart comparison files are not available")

    with np.load(reference_path) as expected, np.load(resumed_path) as actual:
        assert set(expected.files) == set(actual.files)
        for name in expected.files:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1.0e-9, atol=1.0e-11)

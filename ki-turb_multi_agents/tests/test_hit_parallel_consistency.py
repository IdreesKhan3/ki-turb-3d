"""Serial/OpenMP/MPI regression comparison for deterministic mini-cases."""

import os
from pathlib import Path

import numpy as np
import pytest


def test_serial_openmp_mpi_consistency():
    paths = {
        "serial": os.environ.get("KITURB_HIT_SERIAL_RESULT"),
        "openmp": os.environ.get("KITURB_HIT_OPENMP_RESULT"),
        "mpi": os.environ.get("KITURB_HIT_MPI_RESULT"),
    }
    if not all(paths.values()):
        pytest.skip("set serial, OpenMP and MPI result environment variables to run")
    for value in paths.values():
        if not Path(value).is_file():
            pytest.skip(f"parallel comparison file is missing: {value}")

    loaded = {name: np.load(path) for name, path in paths.items()}
    try:
        keys = set(loaded["serial"].files)
        assert set(loaded["openmp"].files) == keys
        assert set(loaded["mpi"].files) == keys
        for key in keys:
            np.testing.assert_allclose(
                loaded["openmp"][key], loaded["serial"][key], rtol=1.0e-8, atol=1.0e-10
            )
            np.testing.assert_allclose(
                loaded["mpi"][key], loaded["serial"][key], rtol=1.0e-8, atol=1.0e-10
            )
    finally:
        for item in loaded.values():
            item.close()

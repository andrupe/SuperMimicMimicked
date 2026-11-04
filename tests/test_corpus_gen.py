"""Unit test for generate_corpus and dataset storage."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tempfile
import numpy as np
import pytest
from generate_corpus import generate_corpus, HAS_H5PY
if HAS_H5PY:
    import h5py


def test_dry_run_generation():
    """Verify dry-run solves tasks and achieves high accept rate."""
    summary = generate_corpus(
        num_trajectories=3,
        tasks=["point_to_point", "hover"],
        workers=1,
        dt=0.04,
        dry_run=True,
        seed=123
    )

    assert summary["total"] == 3
    assert summary["accept_rate"] >= 0.66
    assert summary["successful"] >= 2


def test_hdf5_storage():
    """Verify saving to HDF5 and reloading preserves state and control shapes."""
    if not HAS_H5PY:
        pytest.skip("h5py not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = generate_corpus(
            num_trajectories=2,
            tasks=["hover", "point_to_point"],
            workers=1,
            dt=0.04,
            out_dir=tmpdir,
            output_format="h5",
            seed=42,
            dry_run=False
        )

        h5_files = [f for f in os.listdir(tmpdir) if f.endswith(".h5")]
        assert len(h5_files) == 1
        h5_path = os.path.join(tmpdir, h5_files[0])

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["num_trajectories"] == summary["successful"]
            assert "states" in f or "trajectories" in f
            assert "task_types" in f
            assert "max_residuals" in f
            residuals = f["max_residuals"][:]
            assert np.all(residuals < 1e-5)

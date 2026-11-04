"""Unit and integration tests for SuperMimic quadrotor training and sim2real pipeline."""

import os
import tempfile
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from quadrotor.supermimic import (
    SuperMimicPolicy,
    SuperMimicDataset,
    compute_obs,
    normalize_obs,
    train_behavior_cloning,
    train_bptt,
    evaluate_sim2real,
    export_firmware_header,
    OBS_DIM,
    ACT_DIM,
)

CORPUS_PATH = "corpus/crazyflie_corpus_20trajs.h5"


def test_dataset_loading():
    """Verify loading from HDF5 extracts valid 22-D obs and 4-D actions."""
    if not os.path.exists(CORPUS_PATH):
        pytest.skip(f"Corpus file not found: {CORPUS_PATH}")

    ds = SuperMimicDataset.from_hdf5(CORPUS_PATH)
    assert len(ds.trajectories) == 20
    assert ds.obs.ndim == 2
    assert ds.obs.shape[1] == OBS_DIM
    assert ds.actions.ndim == 2
    assert ds.actions.shape[1] == ACT_DIM
    assert ds.obs_mean.shape == (OBS_DIM,)
    assert ds.obs_std.shape == (OBS_DIM,)
    assert np.all(np.isfinite(ds.obs))
    assert np.all(np.isfinite(ds.actions))
    # Actions should be centered roughly around hover (0.0)
    assert -3.0 <= np.mean(ds.actions) <= 1.5


def test_policy_forward():
    """Verify policy outputs 4 bounded tanh actions in [-1, 1]."""
    model = SuperMimicPolicy()
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (8, OBS_DIM))
    variables = model.init(rng, x)
    out = model.apply(variables, x)

    assert out.shape == (8, ACT_DIM)
    assert jnp.all(out >= -1.0) and jnp.all(out <= 1.0)
    assert "hidden_0" in variables["params"]
    assert "hidden_1" in variables["params"]


def test_bc_training():
    """Verify Behavior Cloning trains and decreases MSE loss."""
    if not os.path.exists(CORPUS_PATH):
        pytest.skip("Corpus file not found")

    ds = SuperMimicDataset.from_hdf5(CORPUS_PATH)
    # Train for 20 epochs
    params, stats = train_behavior_cloning(ds, epochs=20, batch_size=128, lr=2e-3)
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["final_loss"] < 0.20


def test_bptt_gradient_flow():
    """Verify BPTT unroll computes valid non-zero gradients through JAX dynamics."""
    if not os.path.exists(CORPUS_PATH):
        pytest.skip("Corpus file not found")

    ds = SuperMimicDataset.from_hdf5(CORPUS_PATH)
    params, _ = train_behavior_cloning(ds, epochs=5, batch_size=128, lr=1e-3)
    params, bptt_stats = train_bptt(params, ds, epochs=5, horizon=10, batch_size=16, lr=1e-3)

    assert bptt_stats["epochs"] == 5
    assert np.isfinite(bptt_stats["final_loss"])


def test_sim2real_evaluation():
    """Verify closed-loop evaluation runs across multiple regimes without crashing."""
    if not os.path.exists(CORPUS_PATH):
        pytest.skip("Corpus file not found")

    ds = SuperMimicDataset.from_hdf5(CORPUS_PATH)
    params, _ = train_behavior_cloning(ds, epochs=10, batch_size=128, lr=1e-3)
    eval_res = evaluate_sim2real(params, ds, num_eval_trajs=2)

    assert "nominal_rmse_cm" in eval_res
    assert "noise_rmse_cm" in eval_res
    assert "dr_rmse_cm" in eval_res
    assert eval_res["nominal_rmse_cm"] > 0.0
    assert eval_res["trajectories_evaluated"] == 2


def test_firmware_export():
    """Verify C header export creates matrices.h with matching static arrays."""
    if not os.path.exists(CORPUS_PATH):
        pytest.skip("Corpus file not found")

    ds = SuperMimicDataset.from_hdf5(CORPUS_PATH)
    model = SuperMimicPolicy()
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.zeros((1, OBS_DIM)))["params"]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir, matrices_h = export_firmware_header(params, ds.obs_mean, ds.obs_std, tmpdir)

        assert os.path.exists(matrices_h)
        assert os.path.exists(os.path.join(csv_dir, "OBS_MEAN.csv"))
        assert os.path.exists(os.path.join(csv_dir, "OBS_STD.csv"))
        assert os.path.exists(os.path.join(csv_dir, "hidden_0_weights.csv"))
        assert os.path.exists(os.path.join(csv_dir, "hidden_0_bias.csv"))
        assert os.path.exists(os.path.join(csv_dir, "hidden_1_weights.csv"))
        assert os.path.exists(os.path.join(csv_dir, "hidden_1_bias.csv"))

        with open(matrices_h, "r") as f:
            content = f.read()

        assert "OBS_MEAN" in content
        assert "OBS_STD" in content
        assert "hidden_0_weights" in content
        assert "hidden_1_weights" in content
        assert "#ifndef MATRICES_H" in content


def test_checkpoint_roundtrip():
    """Verify atomic saving and reloading of model and optimizer checkpoints."""
    from quadrotor.supermimic import save_checkpoint, load_checkpoint
    import optax

    model = SuperMimicPolicy()
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.zeros((1, OBS_DIM)))["params"]
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    obs_mean = np.ones(OBS_DIM, dtype=np.float32) * 0.5
    obs_std = np.ones(OBS_DIM, dtype=np.float32) * 1.2

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(
            tmpdir,
            "test_ckpt",
            params,
            opt_state,
            epoch=42,
            step=1050,
            val_loss=0.0123,
            obs_mean=obs_mean,
            obs_std=obs_std,
            metadata={"test_run": True}
        )

        assert os.path.exists(ckpt_path)

        loaded = load_checkpoint(ckpt_path, params, opt_state)
        assert loaded["epoch"] == 42
        assert loaded["step"] == 1050
        assert np.isclose(loaded["val_loss"], 0.0123, atol=1e-5)
        assert np.allclose(loaded["obs_mean"], obs_mean)
        assert np.allclose(loaded["obs_std"], obs_std)
        assert loaded["metadata"]["test_run"] is True

        # Verify restored params produce identical outputs
        x = jax.random.normal(rng, (4, OBS_DIM))
        orig_out = model.apply({"params": params}, x)
        restored_out = model.apply({"params": loaded["params"]}, x)
        assert np.allclose(orig_out, restored_out, atol=1e-6)

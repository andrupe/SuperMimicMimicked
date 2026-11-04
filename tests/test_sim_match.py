"""Test consistency between quadrotor dynamics and crazyflie_nn JAX simulator."""

import numpy as np
import pytest
from quadrotor.params import NX, U_HOVER, MAX_THRUST
from quadrotor.quaternion import quat_exp
from quadrotor.sim_bridge import rollout_numpy, rollout_jax, validate_trajectory


def test_jax_sim_matching():
    """Verify that forward rollout in JAX sim matches OCP rollout within tolerance."""
    H = 50
    dt = 0.02 # 50 Hz OCP (4 substeps of 5ms each in sim)
    x0 = np.zeros(NX)
    x0[0:3] = np.array([0.0, 0.0, 1.0])
    x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    x0[13:17] = U_HOVER.copy()

    # Time-varying control sequence with yaw, roll and pitch excitation
    us = np.tile(U_HOVER, (H, 1))
    us[10:20, 0] += 0.04
    us[10:20, 1] -= 0.04
    us[25:35, 2] += 0.03
    us[25:35, 3] -= 0.03

    roll_np = rollout_numpy(x0, us, dt, substeps=4)
    roll_jax = rollout_jax(x0, us, dt)

    pos_err = np.linalg.norm(roll_np[:, 0:3] - roll_jax[:, 0:3], axis=1)
    max_pos_err = np.max(pos_err)
    final_pos_err = pos_err[-1]

    # Given float32 in JAX sim and float64 in NumPy, error over 50 steps (200 sim substeps)
    # should be within sub-millimeter level (< 1e-4 m)
    assert max_pos_err < 1e-4, f"Max pos discrepancy: {max_pos_err:.2e} m"
    assert final_pos_err < 1e-4, f"Final pos discrepancy: {final_pos_err:.2e} m"


def test_validation_gate():
    """Verify that validate_trajectory accurately passes good trajectories and fails bad ones."""
    H = 30
    dt = 0.02
    x0 = np.zeros(NX)
    x0[3] = 1.0
    x0[13:17] = U_HOVER.copy()
    us = np.tile(U_HOVER, (H, 1))
    xs = rollout_numpy(x0, us, dt)

    # Valid run
    res = validate_trajectory(xs, us, dt, solver_state_name="Success", check_jax=True)
    assert res["passed"] is True
    assert res["max_residual"] < 1e-6
    assert res["jax_pos_drift"] < 1e-3

    # Failed run due to solver state
    res_bad_solver = validate_trajectory(xs, us, dt, solver_state_name="MaxIterationsReached", check_jax=False)
    assert res_bad_solver["passed"] is False

    # Failed run due to out-of-bounds controls
    us_oob = us.copy()
    us_oob[5, 0] = 1.2
    res_bad_bounds = validate_trajectory(xs, us_oob, dt, solver_state_name="Success", check_jax=False)
    assert res_bad_bounds["passed"] is False

    # Failed run due to inconsistent states (dynamics residual)
    xs_bad = xs.copy()
    xs_bad[10, 0] += 0.5
    res_bad_residual = validate_trajectory(xs_bad, us, dt, solver_state_name="Success", check_jax=False)
    assert res_bad_residual["passed"] is False

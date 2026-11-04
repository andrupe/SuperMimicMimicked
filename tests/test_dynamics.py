"""Unit tests for quadrotor.dynamics."""

import numpy as np
import pytest
from quadrotor.params import (
    NX, NDX, NU, HOVER_RPM, MAX_THRUST, U_HOVER, G, M_NOMINAL
)
from quadrotor.quaternion import (
    quat_exp, state_right_plus, state_right_minus, quat_to_R
)
from quadrotor.dynamics import (
    continuous_dynamics_tangent, continuous_jacobian_tangent,
    rk4_step, rk4_step_jacobian, rk4_integrate, rk4_integrate_jacobian,
    CrazyflieDynamics, FixedStartCF
)


def test_hover_equilibrium():
    """At hover RPM and level orientation, acceleration and rate must be near zero."""
    x = np.zeros(NX)
    x[3] = 1.0 # level
    x[13:17] = U_HOVER
    u = U_HOVER.copy()

    x_dot = continuous_dynamics_tangent(x, u)
    p_dot = x_dot[0:3]
    theta_dot = x_dot[3:6]
    vb_dot = x_dot[6:9]
    wb_dot = x_dot[9:12]
    om_dot = x_dot[12:16]

    assert np.allclose(p_dot, 0.0, atol=1e-12)
    assert np.allclose(theta_dot, 0.0, atol=1e-12)
    assert np.allclose(vb_dot, 0.0, atol=1e-5) # hover thrust cancels gravity
    assert np.allclose(wb_dot, 0.0, atol=1e-12) # symmetric rotors cancel torques
    assert np.allclose(om_dot, 0.0, atol=1e-12)


def test_continuous_jacobian_accuracy():
    """Analytic continuous tangent Jacobian must match central differences to ~1e-6."""
    x = np.zeros(NX)
    x[0:3] = np.array([0.5, -0.3, 1.2])
    x[3:7] = quat_exp(np.array([0.15, -0.2, 0.25]))
    x[7:10] = np.array([0.6, -0.4, 0.2])
    x[10:13] = np.array([0.3, -0.2, 0.5])
    x[13:17] = np.array([0.72, 0.70, 0.74, 0.71])
    u = np.array([0.75, 0.69, 0.73, 0.72])

    Ac_ana, Bc_ana = continuous_jacobian_tangent(x, u)

    eps = 1e-7
    Ac_num = np.zeros((NDX, NDX))
    Bc_num = np.zeros((NDX, NU))

    for j in range(NDX):
        dx_p = np.zeros(NDX); dx_p[j] = eps
        dx_m = np.zeros(NDX); dx_m[j] = -eps
        f_p = continuous_dynamics_tangent(state_right_plus(x, dx_p), u)
        f_m = continuous_dynamics_tangent(state_right_plus(x, dx_m), u)
        Ac_num[:, j] = (f_p - f_m) / (2.0 * eps)

    for j in range(NU):
        du_p = np.zeros(NU); du_p[j] = eps
        du_m = np.zeros(NU); du_m[j] = -eps
        f_p = continuous_dynamics_tangent(x, u + du_p)
        f_m = continuous_dynamics_tangent(x, u + du_m)
        Bc_num[:, j] = (f_p - f_m) / (2.0 * eps)

    assert np.allclose(Ac_ana, Ac_num, atol=1e-5), f"Max diff Ac: {np.max(np.abs(Ac_ana - Ac_num))}"
    assert np.allclose(Bc_ana, Bc_num, atol=1e-5), f"Max diff Bc: {np.max(np.abs(Bc_ana - Bc_num))}"


def test_discrete_rk4_jacobian_accuracy():
    """Analytic discrete Jacobian for one RK4 step must match numerical differences."""
    dt = 0.01
    x = np.zeros(NX)
    x[0:3] = np.array([0.2, -0.1, 1.0])
    x[3:7] = quat_exp(np.array([0.1, -0.05, 0.15]))
    x[7:10] = np.array([0.4, -0.2, 0.1])
    x[10:13] = np.array([0.1, -0.1, 0.2])
    x[13:17] = U_HOVER.copy()
    u = U_HOVER.copy()

    Ad_ana, Bd_ana = rk4_step_jacobian(x, u, dt)

    eps = 1e-7
    Ad_num = np.zeros((NDX, NDX))
    Bd_num = np.zeros((NDX, NU))

    for j in range(NDX):
        dx_p = np.zeros(NDX); dx_p[j] = eps
        dx_m = np.zeros(NDX); dx_m[j] = -eps
        f_p = rk4_step(state_right_plus(x, dx_p), u, dt)
        f_m = rk4_step(state_right_plus(x, dx_m), u, dt)
        Ad_num[:, j] = state_right_minus(f_p, f_m) / (2.0 * eps)

    for j in range(NU):
        du_p = np.zeros(NU); du_p[j] = eps
        du_m = np.zeros(NU); du_m[j] = -eps
        f_p = rk4_step(x, u + du_p, dt)
        f_m = rk4_step(x, u + du_m, dt)
        Bd_num[:, j] = state_right_minus(f_p, f_m) / (2.0 * eps)

    assert np.allclose(Ad_ana, Ad_num, atol=0.02), f"Max diff Ad: {np.max(np.abs(Ad_ana - Ad_num))}"
    assert np.allclose(Bd_ana, Bd_num, atol=1e-4), f"Max diff Bd: {np.max(np.abs(Bd_ana - Bd_num))}"


def test_stage_dynamics_api():
    """Verify SSQP StageDynamics interface on CrazyflieDynamics."""
    dyn = CrazyflieDynamics(dt=0.02, substeps=4, use_analytic=True)
    assert dyn.output_size() == 17
    assert dyn.state_size() == 17
    assert dyn.control_size() == 4
    assert dyn.state_diff_size() == 16

    x0 = dyn.init_state()
    u0 = dyn.init_control()
    assert x0.shape == (17,)
    assert u0.shape == (4,)

    x_next = dyn.evaluate(x0, u0)
    assert x_next.shape == (17,)

    dX, dU = dyn.gradient(x0, u0)
    assert dX.shape == (16, 16)
    assert dU.shape == (16, 4)

    # FixedStart overrides
    fixed_x0 = np.ones(17)
    fixed_u0 = np.full(4, 0.5)
    dyn0 = FixedStartCF(dt=0.02, x0=fixed_x0, u0=fixed_u0)
    assert np.allclose(dyn0.init_state(), fixed_x0)
    assert np.allclose(dyn0.init_control(), fixed_u0)

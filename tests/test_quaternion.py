"""Unit tests for quadrotor.quaternion."""

import numpy as np
import pytest
from quadrotor.quaternion import (
    skew, quat_exp, quat_log, quat_mul, quat_conj, quat_to_R, R_to_quat,
    rpy_to_quat, quat_to_rpy, exp_so3, state_right_plus, state_right_minus,
    state17_to_22, state22_to_17
)
from quadrotor.params import NX, NDX, MAX_THRUST, HOVER_RPM


def test_skew():
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([4.0, 5.0, 6.0])
    assert np.allclose(skew(v) @ w, np.cross(v, w))


def test_quat_exp_log_roundtrip():
    # Small angle
    phi = np.array([1e-6, -2e-6, 1.5e-6])
    q = quat_exp(phi)
    phi_rec = quat_log(q)
    assert np.allclose(phi, phi_rec, atol=1e-12)

    # Moderate angle
    phi2 = np.array([0.3, -0.4, 0.5])
    q2 = quat_exp(phi2)
    phi2_rec = quat_log(q2)
    assert np.allclose(phi2, phi2_rec, atol=1e-12)


def test_quat_mul_and_conj():
    q = quat_exp(np.array([0.2, 0.4, -0.1]))
    q_inv = quat_conj(q)
    prod = quat_mul(q, q_inv)
    assert np.allclose(prod, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)


def test_quat_and_R_roundtrip():
    q = quat_exp(np.array([0.1, -0.2, 0.3]))
    R = quat_to_R(q)
    q_rec = R_to_quat(R)
    assert np.allclose(q, q_rec, atol=1e-12) or np.allclose(q, -q_rec, atol=1e-12)
    # Check orthogonal
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_rpy_roundtrip():
    roll, pitch, yaw = 0.1, -0.2, 0.5
    q = rpy_to_quat(roll, pitch, yaw)
    r_out, p_out, y_out = quat_to_rpy(q)
    assert np.allclose([roll, pitch, yaw], [r_out, p_out, y_out], atol=1e-12)


def test_exp_so3():
    w = np.array([0.1, -0.05, 0.2])
    R = exp_so3(w)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    q = quat_exp(w)
    R_from_q = quat_to_R(q)
    assert np.allclose(R, R_from_q, atol=1e-12)


def test_state_right_plus_minus():
    x = np.zeros(NX)
    x[0:3] = np.array([1.0, -2.0, 3.0])
    x[3:7] = quat_exp(np.array([0.1, 0.2, -0.1]))
    x[7:10] = np.array([0.5, -0.2, 0.1])
    x[10:13] = np.array([0.05, -0.02, 0.1])
    x[13:17] = np.full(4, HOVER_RPM / MAX_THRUST)

    dx = np.array([
        0.05, -0.03, 0.02,
        0.01, -0.02, 0.03,
        0.1, -0.05, 0.02,
        0.01, 0.02, -0.01,
        0.02, -0.01, 0.03, -0.02
    ])

    x_perturbed = state_right_plus(x, dx)
    dx_rec = state_right_minus(x_perturbed, x)
    assert np.allclose(dx, dx_rec, atol=1e-12)


def test_state17_22_roundtrip():
    x17 = np.zeros(NX)
    x17[0:3] = np.array([1.0, 2.0, 3.0])
    x17[3:7] = quat_exp(np.array([0.2, -0.1, 0.3]))
    x17[7:10] = np.array([0.4, 0.1, -0.2])
    x17[10:13] = np.array([0.02, -0.01, 0.05])
    x17[13:17] = np.full(4, 0.72)

    x22 = state17_to_22(x17)
    assert x22.shape == (22,)
    x17_rec = state22_to_17(x22)
    assert np.allclose(x17[0:3], x17_rec[0:3], atol=1e-12)
    # Quaternion up to sign
    assert np.allclose(x17[3:7], x17_rec[3:7], atol=1e-12) or np.allclose(x17[3:7], -x17_rec[3:7], atol=1e-12)
    assert np.allclose(x17[7:], x17_rec[7:], atol=1e-12)

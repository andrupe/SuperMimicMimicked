"""Quaternion and Lie-group operations for S^3 and SO(3) on the tangent bundle."""

import numpy as np
from quadrotor.params import MAX_THRUST


def skew(w: np.ndarray) -> np.ndarray:
    """Hat operator: skew-symmetric matrix from 3D vector."""
    return np.array([
        [0.0,   -w[2],  w[1]],
        [w[2],   0.0,  -w[0]],
        [-w[1],  w[0],  0.0]
    ], dtype=np.float64)


def quat_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector phi in R^3 -> unit quaternion [w, x, y, z]."""
    ang = np.linalg.norm(phi)
    if ang < 1e-12:
        return np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]], dtype=np.float64)
    half = 0.5 * ang
    s = np.sin(half) / ang
    return np.array([np.cos(half), s * phi[0], s * phi[1], s * phi[2]], dtype=np.float64)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: unit quaternion [w, x, y, z] -> rotation vector in R^3."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    q_norm = q / norm
    if q_norm[0] < 0.0:  # shortest arc antipodal identification
        q_norm = -q_norm
    v = q_norm[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return 2.0 * v
    ang = 2.0 * np.arctan2(s, q_norm[0])
    return v * (ang / s)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product a (x) b for quaternions formatted [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q^* = [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix R."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z] (Shepperd's algorithm)."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles (ZYX order) to quaternion [w, x, y, z]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] to Euler angles roll, pitch, yaw."""
    R = quat_to_R(q)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Lie group exponential on SO(3): Rodrigues formula."""
    theta_sq = float(np.dot(w, w))
    theta = np.sqrt(theta_sq)
    if theta_sq < 1e-6:
        A = 1.0 - theta_sq / 6.0
        B = 0.5 - theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta_sq
    K = skew(w)
    return np.eye(3, dtype=np.float64) + A * K + B * (K @ K)


def state_right_plus(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Manifold retraction x (+) dx.
    x (17):  p(3), q(4, wxyz), vb(3), wb(3), omega_bar(4)
    dx (16): dp(3), dtheta(3), dvb(3), dwb(3), domega_bar(4)
    """
    p = x[0:3] + dx[0:3]
    q = quat_mul(x[3:7], quat_exp(dx[3:6]))
    q = q / np.linalg.norm(q)
    vb = x[7:10] + dx[6:9]
    wb = x[10:13] + dx[9:12]
    om = x[13:17] + dx[12:16]
    return np.concatenate([p, q, vb, wb, om])


def state_right_minus(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manifold difference x (-) y returning tangent vector dx in R^16."""
    dp = x[0:3] - y[0:3]
    dth = quat_log(quat_mul(quat_conj(y[3:7]), x[3:7]))
    dvb = x[7:10] - y[7:10]
    dwb = x[10:13] - y[10:13]
    dom = x[13:17] - y[13:17]
    return np.concatenate([dp, dth, dvb, dwb, dom])


def state17_to_22(x17: np.ndarray) -> np.ndarray:
    """Map 17-state OCP vector to 22-state simulator vector."""
    p = x17[0:3]
    R = quat_to_R(x17[3:7]).flatten()
    vb = x17[7:10]
    wb = x17[10:13]
    om = x17[13:17] * MAX_THRUST
    return np.concatenate([p, R, vb, wb, om])


def state22_to_17(x22: np.ndarray) -> np.ndarray:
    """Map 22-state simulator vector to 17-state OCP vector."""
    p = x22[0:3]
    q = R_to_quat(x22[3:12].reshape((3, 3)))
    vb = x22[12:15]
    wb = x22[15:18]
    om_bar = x22[18:22] / MAX_THRUST
    return np.concatenate([p, q, vb, wb, om_bar])

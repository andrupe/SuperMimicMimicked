"""Batched Lie-algebra and rotation operations in JAX.

Provides:
  - skew: vector -> 3x3 skew-symmetric matrix
  - so3_exp: rotation vector in R^3 -> 3x3 rotation matrix
  - so3_log: 3x3 rotation matrix -> rotation vector in R^3
  - yaw_mat: extract heading/yaw-only 3x3 rotation matrix
  - mat_to_6d: first 2 columns flattened (6D rotation representation)
  - rot_angle_sq: squared geodesic rotation error
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def skew_jax(v: jax.Array) -> jax.Array:
    """Skew-symmetric matrix for 3D vector v (shape (..., 3))."""
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    o = jnp.zeros_like(x)
    row0 = jnp.stack([o, -z, y], axis=-1)
    row1 = jnp.stack([z, o, -x], axis=-1)
    row2 = jnp.stack([-y, x, o], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def so3_exp_jax(v: jax.Array) -> jax.Array:
    """Exponential map from tangent vector v in R^3 to SO(3) 3x3 matrix."""
    theta2 = jnp.sum(v * v, axis=-1, keepdims=True)
    theta = jnp.sqrt(jnp.maximum(theta2, 1e-12))
    small = theta2 < 1e-8

    a = jnp.where(small, 1.0 - theta2 / 6.0, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - theta2 / 24.0, (1.0 - jnp.cos(theta)) / (theta2 + 1e-12))

    K = skew_jax(v)
    eye = jnp.eye(3, dtype=v.dtype)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def mat_to_quat_jax(R: jax.Array) -> jax.Array:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z] in JAX."""
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    qw = jnp.sqrt(jnp.maximum(1.0 + r00 + r11 + r22, 1e-10)) / 2.0
    qx = jnp.sqrt(jnp.maximum(1.0 + r00 - r11 - r22, 1e-10)) / 2.0
    qy = jnp.sqrt(jnp.maximum(1.0 - r00 + r11 - r22, 1e-10)) / 2.0
    qz = jnp.sqrt(jnp.maximum(1.0 - r00 - r11 + r22, 1e-10)) / 2.0

    c0 = jnp.stack([qw, (r21 - r12) / (4.0 * qw), (r02 - r20) / (4.0 * qw), (r10 - r01) / (4.0 * qw)], axis=-1)
    c1 = jnp.stack([(r21 - r12) / (4.0 * qx), qx, (r01 + r10) / (4.0 * qx), (r02 + r20) / (4.0 * qx)], axis=-1)
    c2 = jnp.stack([(r02 - r20) / (4.0 * qy), (r01 + r10) / (4.0 * qy), qy, (r12 + r21) / (4.0 * qy)], axis=-1)
    c3 = jnp.stack([(r10 - r01) / (4.0 * qz), (r02 + r20) / (4.0 * qz), (r12 + r21) / (4.0 * qz), qz], axis=-1)

    cands = jnp.stack([c0, c1, c2, c3], axis=-2)  # (..., 4, 4)
    diag = jnp.stack([qw, qx, qy, qz], axis=-1)
    best_idx = jnp.argmax(diag, axis=-1, keepdims=True)  # (..., 1)

    q = jnp.take_along_axis(cands, best_idx[..., None], axis=-2).squeeze(-2)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    return jnp.where(q[..., :1] < 0.0, -q, q)


def so3_log_jax(R: jax.Array) -> jax.Array:
    """Logarithmic map from SO(3) 3x3 matrix to rotation vector in R^3."""
    q = mat_to_quat_jax(R)
    v = q[..., 1:]
    sin_half = jnp.linalg.norm(v, axis=-1, keepdims=True)
    half = jnp.arctan2(sin_half, q[..., :1])
    scale = jnp.where(sin_half < 1e-6, 2.0 / jnp.maximum(q[..., :1], 1e-6), 2.0 * half / (sin_half + 1e-12))
    return v * scale


def yaw_mat_jax(R: jax.Array) -> jax.Array:
    """Extract heading (yaw-only) rotation matrix from R."""
    fx = R[..., 0, 0]
    fy = R[..., 1, 0]
    n = jnp.sqrt(fx * fx + fy * fy + 1e-9)
    c = fx / n
    s = fy / n
    o = jnp.zeros_like(c)
    one = jnp.ones_like(c)

    row0 = jnp.stack([c, -s, o], axis=-1)
    row1 = jnp.stack([s, c, o], axis=-1)
    row2 = jnp.stack([o, o, one], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def mat_to_6d_jax(R: jax.Array) -> jax.Array:
    """Extract continuous 6D rotation representation (first two columns flattened)."""
    col0 = R[..., :, 0]
    col1 = R[..., :, 1]
    return jnp.concatenate([col0, col1], axis=-1)


def rot_angle_sq_jax(Ra: jax.Array, Rb: jax.Array) -> jax.Array:
    """Squared geodesic angular error between rotations Ra and Rb."""
    tr = jnp.sum(Ra * Rb, axis=(-1, -2))  # tr(Ra^T Rb)
    cos = jnp.clip((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    return jnp.square(jnp.arccos(cos))

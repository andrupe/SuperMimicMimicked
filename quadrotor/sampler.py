"""Task sampling and initial guess generation for quadrotor trajectories."""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from quadrotor.params import NX, U_HOVER, G
from quadrotor.quaternion import rpy_to_quat, quat_to_R, quat_mul, quat_conj, quat_log


class TaskType(Enum):
    POINT_TO_POINT = "point_to_point"
    FIGURE8 = "figure8"
    HELIX = "helix"
    WAYPOINTS = "waypoints"
    HOVER = "hover"


@dataclass
class Task:
    task_type: TaskType
    T: float
    H: int
    dt: float
    p_ref: np.ndarray        # (H + 1, 3)
    v_ref: np.ndarray        # (H + 1, 3)
    q_ref: np.ndarray        # (H + 1, 4)
    x0: np.ndarray           # (17,)
    init_states: np.ndarray  # (H + 1, 17)
    init_controls: np.ndarray# (H, 4)
    metadata: dict


def generate_initial_guess(p_ref: np.ndarray, v_ref: np.ndarray,
                           q_ref: Optional[np.ndarray] = None,
                           dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Generate physically plausible full (H+1, 17) state guess and (H, 4) control guess."""
    H = len(p_ref) - 1
    init_states = np.zeros((H + 1, NX), dtype=np.float64)
    init_states[:, 0:3] = p_ref

    if q_ref is None:
        init_states[:, 3] = 1.0 # level attitude [1, 0, 0, 0]
    else:
        init_states[:, 3:7] = q_ref

    # Body velocity approximation: vb = R^T v_ref
    for k in range(H + 1):
        R = quat_to_R(init_states[k, 3:7])
        init_states[k, 7:10] = R.T @ v_ref[k]

    # Kinematically consistent body angular rates: wb = quat_log(q_k^* (x) q_{k+1}) / dt
    if q_ref is not None and H > 0:
        for k in range(H):
            dq = quat_mul(quat_conj(q_ref[k]), q_ref[k + 1])
            init_states[k, 10:13] = quat_log(dq) / dt
        init_states[H, 10:13] = init_states[H - 1, 10:13]

    init_states[:, 13:17] = U_HOVER
    init_controls = np.tile(U_HOVER, (H, 1))
    return init_states, init_controls


def sample_point_to_point(rng: np.random.Generator,
                          T: Optional[float] = None,
                          dt: float = 0.02) -> Task:
    """Sample a point-to-point transfer with randomized initial attitude, position and velocity."""
    if T is None:
        T = float(rng.uniform(2.2, 3.8))
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)

    # Wide physical envelope for initial position and velocity
    p0 = rng.uniform([-1.5, -1.5, 0.5], [1.5, 1.5, 2.2])
    v0 = rng.uniform([-1.2, -1.2, -0.6], [1.2, 1.2, 0.8])  # Non-zero initial flight speed

    direction = rng.normal(size=3)
    direction[2] *= 0.4
    direction /= max(1e-4, np.linalg.norm(direction))
    dist = float(rng.uniform(0.6, 2.5))
    p1 = p0 + direction * dist
    p1[2] = np.clip(p1[2], 0.45, 2.3)
    v1 = rng.uniform([-0.3, -0.3, -0.2], [0.3, 0.3, 0.2])  # Terminal arrival velocity

    # Quintic polynomial satisfying p(0)=p0, p(1)=p1, v(0)=v0, v(1)=v1
    tau = t
    a0 = p0
    a1 = v0 * T
    a2 = np.zeros(3)
    a3 = 10.0 * (p1 - p0) - (6.0 * v0 + 4.0 * v1) * T
    a4 = -15.0 * (p1 - p0) + (8.0 * v0 + 7.0 * v1) * T
    a5 = 6.0 * (p1 - p0) - 3.0 * (v0 + v1) * T

    p_ref = (a0[None, :] +
             a1[None, :] * tau[:, None] +
             a2[None, :] * (tau**2)[:, None] +
             a3[None, :] * (tau**3)[:, None] +
             a4[None, :] * (tau**4)[:, None] +
             a5[None, :] * (tau**5)[:, None])

    v_ref = (a1[None, :] +
             2.0 * a2[None, :] * tau[:, None] +
             3.0 * a3[None, :] * (tau**2)[:, None] +
             4.0 * a4[None, :] * (tau**3)[:, None] +
             5.0 * a5[None, :] * (tau**4)[:, None]) / T

    # Randomized initial attitude (roll, pitch, yaw) within safe tilt limits (< 45 deg)
    roll0 = float(rng.uniform(-np.radians(35.0), np.radians(35.0)))
    pitch0 = float(rng.uniform(-np.radians(35.0), np.radians(35.0)))
    yaw0 = float(rng.uniform(-np.pi, np.pi))

    roll1 = float(rng.uniform(-np.radians(10.0), np.radians(10.0)))
    pitch1 = float(rng.uniform(-np.radians(10.0), np.radians(10.0)))
    yaw1 = yaw0 + float(rng.uniform(-np.pi * 0.5, np.pi * 0.5))

    rolls = (1.0 - tau) * roll0 + tau * roll1
    pitchs = (1.0 - tau) * pitch0 + tau * pitch1
    yaws = (1.0 - tau) * yaw0 + tau * yaw1
    q_ref = np.array([rpy_to_quat(rolls[k], pitchs[k], yaws[k]) for k in range(H + 1)])

    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref, dt=dt)
    x0 = init_states[0].copy()
    # Add physical angular rate perturbation within gyro envelope
    x0[10:13] += rng.uniform([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8])
    init_states[0] = x0

    return Task(
        task_type=TaskType.POINT_TO_POINT,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"p0": p0.tolist(), "p1": p1.tolist(), "v0": v0.tolist(), "roll0": roll0, "pitch0": pitch0, "yaw0": yaw0}
    )


def sample_figure8(rng: np.random.Generator,
                   T: Optional[float] = None,
                   dt: float = 0.02) -> Task:
    """Sample a 3D Lissajous figure-8 trajectory with randomized initial phase, velocity and attitude."""
    if T is None:
        T = float(rng.uniform(3.0, 4.5))
    H = int(round(T / dt))
    t0_phase = float(rng.uniform(0.0, T))  # Randomized initial phase along the figure-8
    t = np.linspace(t0_phase, t0_phase + T, H + 1)

    A = float(rng.uniform(0.5, 1.2))
    B = float(rng.uniform(0.3, 0.8))
    center_xy = rng.uniform([-0.8, -0.8], [0.8, 0.8])
    z0 = float(rng.uniform(0.7, 1.6))
    z_amp = float(rng.uniform(0.1, 0.35))
    w = 2.0 * np.pi / T

    p_ref = np.stack([
        center_xy[0] + A * np.sin(w * t),
        center_xy[1] + B * np.sin(2.0 * w * t),
        z0 + z_amp * np.sin(w * t)
    ], axis=1)

    v_ref = np.stack([
        A * w * np.cos(w * t),
        2.0 * B * w * np.cos(2.0 * w * t),
        z_amp * w * np.cos(w * t)
    ], axis=1)

    # Tangent-aligned attitude with banking roll into turns
    acc_ref = np.gradient(v_ref, dt, axis=0)
    yaws = np.unwrap(np.arctan2(v_ref[:, 1] + 1e-4, v_ref[:, 0] + 1e-4))
    pitchs = np.clip(-np.arctan2(acc_ref[:, 0], G), -np.radians(40.0), np.radians(40.0))
    rolls = np.clip(np.arctan2(acc_ref[:, 1], G), -np.radians(40.0), np.radians(40.0))

    q_ref = np.array([rpy_to_quat(rolls[k], pitchs[k], yaws[k]) for k in range(H + 1)])
    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref, dt=dt)
    x0 = init_states[0].copy()
    # Add physical angular rate perturbation within gyro envelope
    x0[10:13] += rng.uniform([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8])
    init_states[0] = x0

    return Task(
        task_type=TaskType.FIGURE8,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"A": A, "B": B, "z0": z0, "center_xy": center_xy.tolist(), "t0_phase": t0_phase}
    )


def sample_helix(rng: np.random.Generator,
                 T: Optional[float] = None,
                 dt: float = 0.02) -> Task:
    """Sample a 3D helical climb or descent with randomized start pose and velocity."""
    if T is None:
        T = float(rng.uniform(3.0, 4.5))
    H = int(round(T / dt))
    t0_phase = float(rng.uniform(0.0, T))
    t = np.linspace(t0_phase, t0_phase + T, H + 1)

    radius = float(rng.uniform(0.4, 1.0))
    center_xy = rng.uniform([-0.6, -0.6], [0.6, 0.6])
    z_start = float(rng.uniform(0.5, 1.1))
    z_end = float(rng.uniform(1.3, 2.2))
    climb_rate = (z_end - z_start) / T * rng.choice([-1.0, 1.0])
    w = 2.0 * np.pi / T * rng.choice([-1.0, 1.0])

    p_ref = np.stack([
        center_xy[0] + radius * np.cos(w * t),
        center_xy[1] + radius * np.sin(w * t),
        z_start + climb_rate * (t - t0_phase)
    ], axis=1)

    v_ref = np.stack([
        -radius * w * np.sin(w * t),
        radius * w * np.cos(w * t),
        np.full_like(t, climb_rate)
    ], axis=1)

    # Dynamic banking attitude for helical turns
    acc_ref = np.gradient(v_ref, dt, axis=0)
    yaws = np.unwrap(np.arctan2(v_ref[:, 1] + 1e-4, v_ref[:, 0] + 1e-4))
    pitchs = np.clip(-np.arctan2(acc_ref[:, 0], G), -np.radians(35.0), np.radians(35.0))
    rolls = np.clip(np.arctan2(acc_ref[:, 1], G), -np.radians(35.0), np.radians(35.0))

    q_ref = np.array([rpy_to_quat(rolls[k], pitchs[k], yaws[k]) for k in range(H + 1)])
    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref, dt=dt)
    x0 = init_states[0].copy()
    # Add physical angular rate perturbation within gyro envelope
    x0[10:13] += rng.uniform([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8])
    init_states[0] = x0

    return Task(
        task_type=TaskType.HELIX,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"radius": radius, "climb_rate": climb_rate, "center_xy": center_xy.tolist(), "t0_phase": t0_phase}
    )


def sample_hover(rng: np.random.Generator,
                 T: Optional[float] = None,
                 dt: float = 0.02) -> Task:
    """Sample station-keeping with randomized initial attitude recovery and micro-drift."""
    if T is None:
        T = float(rng.uniform(2.0, 3.5))
    H = int(round(T / dt))
    t = np.linspace(0.0, T, H + 1)

    target_pos = rng.uniform([-1.2, -1.2, 0.6], [1.2, 1.2, 2.0])

    # Initial attitude perturbation & position offset (testing recovery to hover)
    p_offset = rng.uniform([-0.6, -0.6, -0.35], [0.6, 0.6, 0.35])
    v_init = rng.uniform([-1.0, -1.0, -0.5], [1.0, 1.0, 0.5])
    decay = np.exp(-2.5 * t)[:, None]

    p_ref = target_pos[None, :] + p_offset[None, :] * decay
    v_ref = v_init[None, :] * decay

    roll0 = float(rng.uniform(-np.radians(35.0), np.radians(35.0)))
    pitch0 = float(rng.uniform(-np.radians(35.0), np.radians(35.0)))
    yaw0 = float(rng.uniform(-np.pi, np.pi))

    rolls = roll0 * np.exp(-3.0 * t)
    pitchs = pitch0 * np.exp(-3.0 * t)
    yaws = np.full(H + 1, yaw0)
    q_ref = np.array([rpy_to_quat(rolls[k], pitchs[k], yaws[k]) for k in range(H + 1)])

    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref, dt=dt)

    # Perturb initial state with wide physical envelope
    x0 = init_states[0].copy()
    x0[0:3] = p_ref[0]
    x0[3:7] = q_ref[0]
    R0 = quat_to_R(q_ref[0])
    x0[7:10] = R0.T @ v_ref[0]
    # Non-zero initial angular rates consistent with attitude tilt recovery
    x0[10] = float(rng.uniform(-2.0, 2.0))
    x0[11] = float(rng.uniform(-2.0, 2.0))
    x0[12] = float(rng.uniform(-1.5, 1.5))
    init_states[0] = x0

    return Task(
        task_type=TaskType.HOVER,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"target_pos": target_pos.tolist(), "p_offset": p_offset.tolist(), "v_init": v_init.tolist()}
    )


def sample_task(task_type: Optional[TaskType] = None,
                rng: Optional[np.random.Generator] = None,
                T: Optional[float] = None,
                dt: float = 0.02) -> Task:
    """Sample a task from specified or randomly chosen task family."""
    if rng is None:
        rng = np.random.default_rng()

    if task_type is None:
        types = [TaskType.POINT_TO_POINT, TaskType.FIGURE8, TaskType.HELIX, TaskType.HOVER]
        task_type = rng.choice(types)

    if task_type == TaskType.POINT_TO_POINT:
        return sample_point_to_point(rng, T=T, dt=dt)
    elif task_type == TaskType.FIGURE8:
        return sample_figure8(rng, T=T, dt=dt)
    elif task_type == TaskType.HELIX:
        return sample_helix(rng, T=T, dt=dt)
    elif task_type == TaskType.HOVER:
        return sample_hover(rng, T=T, dt=dt)
    else:
        return sample_point_to_point(rng, T=T, dt=dt)

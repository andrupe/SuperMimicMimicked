"""Task sampling and initial guess generation for quadrotor trajectories."""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from quadrotor.params import NX, U_HOVER, G
from quadrotor.quaternion import rpy_to_quat, quat_to_R


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
                           q_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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

    init_states[:, 13:17] = U_HOVER
    init_controls = np.tile(U_HOVER, (H, 1))
    return init_states, init_controls


def sample_point_to_point(rng: np.random.Generator,
                          T: Optional[float] = None,
                          dt: float = 0.02) -> Task:
    """Sample a point-to-point transfer with smooth quintic polynomial."""
    if T is None:
        T = float(rng.uniform(2.0, 3.5))
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)

    p0 = rng.uniform([-1.0, -1.0, 0.8], [1.0, 1.0, 1.8])
    # Feasible envelope: max speed ~ 1.0 m/s for Crazyflie thrust limits
    max_dist = min(1.0 * T, 2.5)
    direction = rng.normal(size=3)
    direction[2] *= 0.4  # moderate z changes
    direction /= np.linalg.norm(direction)
    dist = float(rng.uniform(0.5, max_dist))

    p1 = p0 + direction * dist
    p1[2] = np.clip(p1[2], 0.5, 2.2)

    # Quintic s-curve
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T

    p_ref = np.outer(1.0 - s, p0) + np.outer(s, p1)
    v_ref = np.outer(s_dot, p1 - p0)

    # Orientation: start level, optional small yaw
    yaw0 = float(rng.uniform(-np.pi, np.pi))
    yaw1 = yaw0 + float(rng.uniform(-np.pi * 0.4, np.pi * 0.4))
    yaws = (1.0 - s) * yaw0 + s * yaw1
    q_ref = np.array([rpy_to_quat(0.0, 0.0, y) for y in yaws])

    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref)
    x0 = init_states[0].copy()

    return Task(
        task_type=TaskType.POINT_TO_POINT,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"p0": p0.tolist(), "p1": p1.tolist(), "yaw0": yaw0, "yaw1": yaw1}
    )


def sample_figure8(rng: np.random.Generator,
                   T: Optional[float] = None,
                   dt: float = 0.02) -> Task:
    """Sample a 3D Lissajous figure-8 trajectory within achievable accelerations."""
    if T is None:
        T = float(rng.uniform(3.0, 4.5))
    H = int(round(T / dt))
    t = np.linspace(0.0, T, H + 1)

    A = float(rng.uniform(0.5, 1.0))
    B = float(rng.uniform(0.3, 0.6))
    z0 = float(rng.uniform(0.8, 1.5))
    z_amp = float(rng.uniform(0.0, 0.2))
    loops = 1.0

    w = 2.0 * np.pi * loops / T
    p_ref = np.stack([
        A * np.sin(w * t),
        B * np.sin(2.0 * w * t),
        z0 + z_amp * np.sin(w * t)
    ], axis=1)

    v_ref = np.stack([
        A * w * np.cos(w * t),
        2.0 * B * w * np.cos(2.0 * w * t),
        z_amp * w * np.cos(w * t)
    ], axis=1)

    # Attitude: aligned with velocity tangent or level
    q_ref = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (H + 1, 1))

    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref)
    x0 = init_states[0].copy()

    return Task(
        task_type=TaskType.FIGURE8,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"A": A, "B": B, "z0": z0, "z_amp": z_amp}
    )


def sample_helix(rng: np.random.Generator,
                 T: Optional[float] = None,
                 dt: float = 0.02) -> Task:
    """Sample a vertical helical climb or descent."""
    if T is None:
        T = float(rng.uniform(3.0, 4.5))
    H = int(round(T / dt))
    t = np.linspace(0.0, T, H + 1)

    radius = float(rng.uniform(0.4, 0.8))
    z_start = float(rng.uniform(0.6, 1.0))
    z_end = float(rng.uniform(1.4, 2.0))
    loops = 1.0
    w = 2.0 * np.pi * loops / T

    p_ref = np.stack([
        radius * np.cos(w * t),
        radius * np.sin(w * t),
        z_start + (z_end - z_start) * (t / T)
    ], axis=1)

    v_ref = np.stack([
        -radius * w * np.sin(w * t),
        radius * w * np.cos(w * t),
        np.full_like(t, (z_end - z_start) / T)
    ], axis=1)

    q_ref = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (H + 1, 1))
    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref)
    x0 = init_states[0].copy()

    return Task(
        task_type=TaskType.HELIX,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"radius": radius, "z_start": z_start, "z_end": z_end}
    )


def sample_hover(rng: np.random.Generator,
                 T: Optional[float] = None,
                 dt: float = 0.02) -> Task:
    """Sample station-keeping with initial offset perturbation."""
    if T is None:
        T = float(rng.uniform(1.5, 2.5))
    H = int(round(T / dt))

    target_pos = rng.uniform([-0.5, -0.5, 0.8], [0.5, 0.5, 1.8])
    p_ref = np.tile(target_pos, (H + 1, 1))
    v_ref = np.zeros((H + 1, 3))
    q_ref = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (H + 1, 1))

    init_states, init_controls = generate_initial_guess(p_ref, v_ref, q_ref)

    # Perturb initial state
    x0 = init_states[0].copy()
    x0[0:3] += rng.uniform(-0.3, 0.3, size=3)
    x0[7:10] += rng.uniform(-0.4, 0.4, size=3) # initial velocity offset
    init_states[0] = x0

    return Task(
        task_type=TaskType.HOVER,
        T=T, H=H, dt=dt,
        p_ref=p_ref, v_ref=v_ref, q_ref=q_ref,
        x0=x0, init_states=init_states, init_controls=init_controls,
        metadata={"target_pos": target_pos.tolist()}
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

"""quadrotor: Expert trajectory generation for Crazyflie 2.1 with SSQP."""

from quadrotor.params import (
    M_NOMINAL, L, KF, KM, G, GRAV, I_NOMINAL, I_INV,
    Kaero, TAU_M, MAX_THRUST, HOVER_RPM, U_HOVER,
    NX, NDX, NU,
)
from quadrotor.quaternion import (
    quat_exp, quat_log, quat_mul, quat_conj, quat_to_R, R_to_quat,
    skew, state_right_plus, state_right_minus, state17_to_22, state22_to_17,
)
from quadrotor.dynamics import CrazyflieDynamics, FixedStartCF
from quadrotor.costs import CFTrackingCost
from quadrotor.constraints import CFControlBounds, CFNoConstraints
from quadrotor.sim_bridge import rollout_numpy, rollout_jax, validate_trajectory
from quadrotor.sampler import Task, TaskType, sample_task, generate_initial_guess
from quadrotor.solver import build_cf_model, solve_task
from quadrotor.supermimic import (
    SuperMimicPolicy, SuperMimicDataset, compute_obs,
    train_behavior_cloning, train_bptt, evaluate_sim2real, export_firmware_header,
    save_checkpoint, load_checkpoint,
)

__all__ = [
    "M_NOMINAL", "L", "KF", "KM", "G", "GRAV", "I_NOMINAL", "I_INV",
    "Kaero", "TAU_M", "MAX_THRUST", "HOVER_RPM", "U_HOVER",
    "NX", "NDX", "NU",
    "quat_exp", "quat_log", "quat_mul", "quat_conj", "quat_to_R", "R_to_quat",
    "skew", "state_right_plus", "state_right_minus", "state17_to_22", "state22_to_17",
    "CrazyflieDynamics", "FixedStartCF",
    "CFTrackingCost", "CFControlBounds", "CFNoConstraints",
    "rollout_numpy", "rollout_jax", "validate_trajectory",
    "Task", "TaskType", "sample_task", "generate_initial_guess",
    "build_cf_model", "solve_task",
    "SuperMimicPolicy", "SuperMimicDataset", "compute_obs",
    "train_behavior_cloning", "train_bptt", "evaluate_sim2real", "export_firmware_header",
    "save_checkpoint", "load_checkpoint",
]

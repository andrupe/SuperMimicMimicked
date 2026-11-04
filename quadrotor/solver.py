"""SSQP model builder and trajectory solver for the Crazyflie."""

import time
from typing import Dict, Any, Optional
import numpy as np

import ssqpy
from quadrotor.params import NX, NU, U_HOVER
from quadrotor.dynamics import CrazyflieDynamics, FixedStartCF
from quadrotor.costs import CFTrackingCost
from quadrotor.constraints import CFControlBounds, CFNoConstraints
from quadrotor.sampler import Task
from quadrotor.sim_bridge import validate_trajectory


def build_cf_model(task: Task,
                   w_p: float = 200.0,
                   w_th: float = 0.5,
                   w_v: float = 20.0,
                   w_w: float = 0.5,
                   w_u: float = 0.1,
                   w_p_term: float = 1000.0,
                   w_th_term: float = 2.0,
                   w_v_term: float = 100.0,
                   w_w_term: float = 1.0,
                   use_analytic: bool = True) -> ssqpy.model.Model:
    """Construct and finalize an SSQP Model for the given task."""
    H = task.H
    dt = task.dt

    stage_costs = [
        CFTrackingCost(
            p_ref=task.p_ref[k],
            q_ref=task.q_ref[k],
            v_ref=task.v_ref[k],
            u_ref=U_HOVER,
            w_p=w_p, w_th=w_th, w_v=w_v, w_w=w_w, w_u=w_u,
            name=f"cost_{k}"
        )
        for k in range(H)
    ]

    terminal_cost = CFTrackingCost(
        p_ref=task.p_ref[H],
        q_ref=task.q_ref[H],
        v_ref=task.v_ref[H],
        u_ref=U_HOVER,
        w_p=w_p_term, w_th=w_th_term, w_v=w_v_term, w_w=w_w_term, w_u=0.0,
        name="cost_terminal"
    )

    dynamics = [CrazyflieDynamics(dt, use_analytic=use_analytic) for _ in range(H + 1)]
    dynamics[0] = FixedStartCF(dt, task.x0, U_HOVER, use_analytic=use_analytic)

    constraints = [CFControlBounds() for _ in range(H)] + [CFNoConstraints()]

    model = ssqpy.model.Model(H, dt)
    model.finalize(
        dynamics=dynamics,
        costs=stage_costs + [terminal_cost],
        terminal_cost=terminal_cost,
        constraints=constraints,
        init_states=task.init_states,
        init_controls=task.init_controls,
    )
    return model


def solve_task(task: Task,
               max_iter: int = 100,
               tolerance: float = 1e-5,
               use_analytic: bool = True,
               weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Solve an optimal control task using SSQP with HPIPM.

    Returns dict containing:
      - 'success': bool
      - 'xs': (H+1, 17) state trajectory
      - 'us': (H, 4) control trajectory
      - 'solve_time_s': float
      - 'iterations': int
      - 'solver_state': str
      - 'validation': dict from validate_trajectory
      - 'task': task object
    """
    w = {
        "w_p": 200.0, "w_th": 0.5, "w_v": 20.0, "w_w": 0.5, "w_u": 0.1,
        "w_p_term": 1000.0, "w_th_term": 2.0, "w_v_term": 100.0, "w_w_term": 1.0
    }
    if weights:
        w.update(weights)

    t_build0 = time.perf_counter()
    model = build_cf_model(task, use_analytic=use_analytic, **w)
    build_time = time.perf_counter() - t_build0

    params = ssqpy.solvers.ssqpParams()
    params.max_iter = max_iter
    params.tolerance = tolerance
    params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

    solver = ssqpy.solvers.ssqp(model, params)

    t0 = time.perf_counter()
    state, x, u = solver.solve()
    solve_time = time.perf_counter() - t0

    H = task.H
    xs = np.zeros((H + 1, NX), dtype=np.float64)
    us = np.zeros((H, NU), dtype=np.float64)

    for k in range(H + 1):
        xs[k] = np.asarray(x.stage(k))
    for k in range(H):
        us[k] = np.asarray(u.stage(k))

    val = validate_trajectory(
        xs=xs,
        us=us,
        dt=task.dt,
        solver_state_name=state.name,
        p_ref=task.p_ref
    )

    conv_data = solver.getConvergenceData()
    iterations = getattr(conv_data, "iterations", -1)

    return {
        "success": val["passed"],
        "xs": xs,
        "us": us,
        "dt": task.dt,
        "T": task.T,
        "H": H,
        "task_type": task.task_type.value,
        "metadata": task.metadata,
        "build_time_s": build_time,
        "solve_time_s": solve_time,
        "iterations": iterations,
        "solver_state": state.name,
        "validation": val,
    }

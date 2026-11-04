"""Generate ONE expert trajectory (figure-8) with SSQP, validate it, save it.

This is the template for every task family: build a reference, turn it into
per-stage costs + a seed, solve, then validate before it enters the corpus.

Run:  uv run --project . python gen_figure8.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ssqpy
from quadrotor_ssqp_proto import (
    NX, T_HOVER,
    ControlBounds, NoConstraints, QuadDynamics, TrackingCost,
    quat_to_R,
)


# ------------------------------------------------------------------ 1. task
def figure8(H, T, A=1.0, B=0.5, z=1.5, loops=1.0):
    """Lissajous figure-8 in a horizontal plane, plus its exact derivative."""
    t = np.linspace(0.0, T, H + 1)
    w = 2.0 * np.pi * loops / T
    p = np.stack([A * np.sin(w * t), B * np.sin(2.0 * w * t), np.full_like(t, z)], axis=1)
    v = np.stack([A * w * np.cos(w * t),
                  2.0 * B * w * np.cos(2.0 * w * t),
                  np.zeros_like(t)], axis=1)
    return p, v


# -------------------------------------------------------------- 2. model
class FixedStart(QuadDynamics):
    """Stage-0 dynamics: SSQP fixes x0 from `init_state()`, NOT from `init_states`."""

    def __init__(self, dt, x0, u0):
        super().__init__(dt)
        self._x0 = np.asarray(x0, dtype=np.float64)
        self._u0 = np.asarray(u0, dtype=np.float64)

    def init_state(self):
        return self._x0

    def init_control(self):
        return self._u0


def build_model(p_ref, v_ref, dt, w_th=0.5, w_p=200.0, w_v=20.0):
    """Costs are built per stage from the reference; one object per stage."""
    H = len(p_ref) - 1
    q_level = np.array([1.0, 0.0, 0.0, 0.0])
    u_ref = np.array([T_HOVER, 0.0, 0.0, 0.0])

    # attitude weight kept small: the vehicle must bank to follow the curve,
    # so we do NOT track level attitude strongly
    costs = [TrackingCost(p_ref[k], q_level, v_ref[k], u_ref,
                          w_p=w_p, w_th=w_th, w_v=w_v, w_w=0.1, w_u=0.05)
             for k in range(H)]
    terminal = TrackingCost(p_ref[H], q_level, v_ref[H], u_ref,
                            w_p=w_p, w_th=w_th, w_v=w_v, w_w=0.1, w_u=0.0)

    init_states = np.zeros((H + 1, NX))
    init_states[:, 0:3] = p_ref
    init_states[:, 3] = 1.0
    init_states[:, 7:10] = v_ref

    model = ssqpy.model.Model(H, dt)
    dynamics = [QuadDynamics(dt) for _ in range(H + 1)]
    dynamics[0] = FixedStart(dt, init_states[0], u_ref)
    model.finalize(
        dynamics=dynamics,
        costs=costs + [terminal],
        terminal_cost=terminal,
        constraints=[ControlBounds() for _ in range(H)] + [NoConstraints()],
        init_states=init_states,
        init_controls=np.tile(u_ref, (H, 1)),
    )
    return model


# ------------------------------------------------------------ 3. validate
def validate(x, u, p_ref, H, dt):
    xs = np.zeros((H + 1, NX))
    us = np.zeros((H, 4))
    for k in range(H + 1):
        xs[k] = np.asarray(x.stage(k))
    for k in range(H):
        us[k] = np.asarray(u.stage(k))

    pos_err = np.linalg.norm(xs[:, 0:3] - p_ref, axis=1)
    thrust = us[:, 0]
    # body tilt: angle between body z-axis and world z-axis
    tilts = np.degrees([np.arccos(np.clip(quat_to_R(xs[k, 3:7])[2, 2], -1, 1))
                        for k in range(H + 1)])

    # independent rollout: recompute the states from x0 and the controls
    dyn = QuadDynamics(dt)
    roll = xs[0].copy()
    residual = 0.0
    for k in range(H):
        roll = dyn.evaluate(roll, us[k])
        residual = max(residual, np.linalg.norm(dyn.state_right_minus(roll, xs[k + 1])))

    return dict(pos_err_max=pos_err.max(), pos_err_final=pos_err[-1],
                residual=residual,
                thrust_min=thrust.min(), thrust_max=thrust.max(),
                tilt_max=tilts.max(), xs=xs, us=us)


def main():
    H, T, DT = 80, 4.0, 4.0 / 80
    p_ref, v_ref = figure8(H, T, A=1.0, B=0.5, z=1.5)

    t0 = time.perf_counter()
    model = build_model(p_ref, v_ref, DT)
    solver = ssqpy.solvers.ssqp(model, _params())
    state, x, u = solver.solve()
    solve_s = time.perf_counter() - t0

    rep = validate(x, u, p_ref, H, DT)
    print(f"state        : {state}")
    print(f"solve        : {solve_s:.2f} s, {solver.getConvergenceData().iterations} iters")
    print(f"tracking err : max {rep['pos_err_max'] * 100:.2f} cm, final {rep['pos_err_final'] * 100:.2f} cm")
    print(f"residual     : {rep['residual']:.2e}")
    print(f"thrust       : [{rep['thrust_min']:.2f}, {rep['thrust_max']:.2f}] N (hover {T_HOVER:.2f})")
    print(f"max body tilt: {rep['tilt_max']:.1f} deg")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "figure8_A1.0_B0.5_T4.0.npz")
    np.savez_compressed(out,
                        x=rep["xs"], u=rep["us"], dt=DT,
                        task="figure8", A=1.0, B=0.5, T=T,
                        solver_state=state.name)
    print(f"saved        : {out}  ({(rep['xs'].size + rep['us'].size) * 4 / 1024:.1f} kB)")


def _params():
    p = ssqpy.solvers.ssqpParams()
    p.max_iter = 200
    p.tolerance = 1e-6
    p.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
    return p


if __name__ == "__main__":
    main()

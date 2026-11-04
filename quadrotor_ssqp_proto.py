"""Prototype: quadrotor trajectory generation with ssqp 0.1.2 (`ssqpy`).

SSQP has no quadrotor preset (presets: Go2, A1, Panda, G1, Talos, DoublePendulum) and no
rotor-thrust actuator model, so the system must be supplied as custom `StageDynamics`.

Model
-----
state  x (13):  p(3), q(4, wxyz), v(3), w(3)      -- quaternion lives on S^3, so
tangent dx(12): dp(3), dtheta(3), dv(3), dw(3)   -- state_size=13, state_diff_size=12
control u (4):  [T, tau_x, tau_y, tau_z]         -- collective thrust + body torques
                                                    (rotor mixing is a post-process)

Propagation is semi-implicit Euler on the tangent bundle:
    v' = v + dt*(R(q) e_z T/m + g)      w' = w + dt*I^-1(tau - w x I w)
    p' = p + dt*v'                      q' = q (x) Exp(dt/2 * w')

`gradient` uses tangent-space central differences (16 extra `evaluate` calls per stage).
Analytic Jacobians are the obvious speed-up; see notes at the bottom.
"""

import time

import numpy as np

import ssqpy
import ssqpy.model.constraints as constraints
import ssqpy.model.costs as costs

# ---------------------------------------------------------------- parameters
MASS = 1.0                 # kg
GRAV = np.array([0.0, 0.0, -9.81])
INERTIA = np.array([0.01, 0.01, 0.02])   # kg m^2, diagonal
INV_INERTIA = 1.0 / INERTIA

T_HOVER = MASS * 9.81
T_MAX = 3.0 * T_HOVER      # collective thrust bound
TAU_MAX = 1.0              # body torque bound

NX, NDX, NU = 13, 12, 4


# ------------------------------------------------------------- quat helpers
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_exp(phi):
    """Rotation vector (3) -> unit quaternion (wxyz)."""
    ang = np.linalg.norm(phi)
    if ang < 1e-12:
        return np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])
    half = 0.5 * ang
    s = np.sin(half) / ang
    return np.array([np.cos(half), s * phi[0], s * phi[1], s * phi[2]])


def quat_log(q):
    """Unit quaternion (wxyz) -> rotation vector (3)."""
    q = q / np.linalg.norm(q)
    if q[0] < 0:                      # shortest arc
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return 2.0 * v
    ang = 2.0 * np.arctan2(s, q[0])
    return v * (ang / s)


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ------------------------------------------------------------ stage dynamics
class QuadDynamics(ssqpy.model.dynamics.StageDynamics):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    # --- dimensions
    def output_size(self):
        return NX

    def state_size(self):
        return NX

    def control_size(self):
        return NU

    def state_diff_size(self):
        return NDX

    # --- defaults for the initial guess
    def init_state(self):
        return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    def init_control(self):
        return np.array([T_HOVER, 0, 0, 0], dtype=np.float64)

    # --- manifold operators (tangent-space retraction)
    def state_right_plus(self, x, dx):
        q = quat_mul(x[3:7], quat_exp(dx[3:6]))
        return np.asarray(np.concatenate([
            x[0:3] + dx[0:3],
            q / np.linalg.norm(q),
            x[7:10] + dx[6:9],
            x[10:13] + dx[9:12],
        ]), dtype=np.float64)

    def state_right_minus(self, x, y):
        dth = quat_log(quat_mul(quat_conj(y[3:7]), x[3:7]))
        return np.asarray(np.concatenate([
            x[0:3] - y[0:3], dth, x[7:10] - y[7:10], x[10:13] - y[10:13],
        ]), dtype=np.float64)

    # --- f(x, u)
    def evaluate(self, x, u):
        dt = self.dt
        q = x[3:7] / np.linalg.norm(x[3:7])
        v = x[7:10]
        w = x[10:13]

        thrust = quat_to_R(q) @ np.array([0.0, 0.0, u[0]]) / MASS
        v_new = v + dt * (thrust + GRAV)
        w_new = w + dt * INV_INERTIA * (u[1:4] - np.cross(w, INERTIA * w))
        p_new = x[0:3] + dt * v_new
        q_new = quat_mul(q, quat_exp(0.5 * dt * w_new))

        return np.asarray(np.concatenate([p_new, q_new / np.linalg.norm(q_new), v_new, w_new]),
                          dtype=np.float64)

    # --- tangent-space Jacobians by central differences
    def gradient(self, x, u):
        eps = 1e-6
        f0 = self.evaluate(x, u)
        dX = np.zeros((NDX, NDX))
        dU = np.zeros((NDX, NU))

        for j in range(NDX):
            e = np.zeros(NDX)
            e[j] = eps
            fp = self.evaluate(self.state_right_plus(x, e), u)
            fm = self.evaluate(self.state_right_plus(x, -e), u)
            dX[:, j] = self.state_right_minus(fp, fm) / (2 * eps)

        for j in range(NU):
            e = np.zeros(NU)
            e[j] = eps
            fp = self.evaluate(x, u + e)
            fm = self.evaluate(x, u - e)
            dU[:, j] = self.state_right_minus(fp, fm) / (2 * eps)

        return dX, dU

    # --- second-order terms (the Python trampoline requires them even though the docs
    #     call them optional); zeros = Gauss-Newton approximation of the dynamics
    def hessprod_xx(self, x, u, lam):
        return np.zeros((NDX, NDX))

    def hessprod_xu(self, x, u, lam):
        return np.zeros((NDX, NU))

    def hessprod_uu(self, x, u, lam):
        return np.zeros((NU, NU))


# ------------------------------------------------------------------- cost
class TrackingCost(ssqpy.model.costs.StageCost):
    """Quadratic tracking of a reference pose/velocity, plus control regularisation."""

    def __init__(self, p_ref, q_ref, v_ref, u_ref, w_p=100.0, w_th=10.0, w_v=10.0,
                 w_w=1.0, w_u=0.1, name="tracking"):
        super().__init__(name)
        self.p_ref = np.asarray(p_ref, dtype=np.float64)
        self.q_ref = np.asarray(q_ref, dtype=np.float64)
        self.v_ref = np.asarray(v_ref, dtype=np.float64)
        self.u_ref = np.asarray(u_ref, dtype=np.float64)
        self.w_p, self.w_th, self.w_v, self.w_w, self.w_u = w_p, w_th, w_v, w_w, w_u

    def evaluate(self, x, u):
        dp = x[0:3] - self.p_ref
        dth = quat_log(quat_mul(quat_conj(self.q_ref), x[3:7]))
        dv = x[7:10] - self.v_ref
        dw = x[10:13]
        # the terminal stage passes an empty control vector
        du = (u - self.u_ref) if u.size else np.zeros(NU)
        return np.asarray(np.array([
            self.w_p * dp @ dp
            + self.w_th * dth @ dth
            + self.w_v * dv @ dv
            + self.w_w * dw @ dw
            + self.w_u * du @ du
        ]), dtype=np.float64)

    def gradient(self, x, u):
        dp = x[0:3] - self.p_ref
        dth = quat_log(quat_mul(quat_conj(self.q_ref), x[3:7]))
        dv = x[7:10] - self.v_ref
        dw = x[10:13]
        du = (u - self.u_ref) if u.size else np.zeros(NU)
        gx = 2.0 * np.concatenate([self.w_p * dp, self.w_th * dth,
                                   self.w_v * dv, self.w_w * dw])
        gu = 2.0 * self.w_u * du
        return np.asarray(gx, dtype=np.float64), np.asarray(gu, dtype=np.float64)

    def hessian_xx(self, x, u):
        return np.diag(np.array(
            [2 * self.w_p] * 3 + [2 * self.w_th] * 3 + [2 * self.w_v] * 3 + [2 * self.w_w] * 3))

    def hessian_xu(self, x, u):
        return np.zeros((NDX, NU))

    def hessian_uu(self, x, u):
        return np.diag(np.full(NU, 2 * self.w_u))


# ------------------------------------------------------------- constraints
class ControlBounds(ssqpy.model.constraints.StageConstraints):
    """Box limits on [T, tau_x, tau_y, tau_z]."""

    def __init__(self, name="control_bounds"):
        super().__init__(name)

    def output_size(self):
        return NU

    def lower_bound(self):
        return np.array([0.0, -TAU_MAX, -TAU_MAX, -TAU_MAX])

    def upper_bound(self):
        return np.array([T_MAX, TAU_MAX, TAU_MAX, TAU_MAX])

    def evaluate(self, x, u):
        return np.asarray(u, dtype=np.float64)

    def gradient(self, x, u):
        return np.zeros((NU, NDX)), np.eye(NU)

    def hessprod_xx(self, x, u, mu):
        return np.zeros((NDX, NDX))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((NDX, NU))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((NU, NU))


class NoConstraints(ssqpy.model.constraints.StageConstraints):
    """Empty constraint set for the terminal stage (which has no controls)."""

    def output_size(self):
        return 0

    def lower_bound(self):
        return np.zeros(0)

    def upper_bound(self):
        return np.zeros(0)

    def evaluate(self, x, u):
        return np.zeros(0)

    def gradient(self, x, u):
        return np.zeros((0, NDX)), np.zeros((0, NU))

    def hessprod_xx(self, x, u, mu):
        return np.zeros((NDX, NDX))

    def hessprod_xu(self, x, u, mu):
        return np.zeros((NDX, NU))

    def hessprod_uu(self, x, u, mu):
        return np.zeros((NU, NU))


# ---------------------------------------------------------------- problem
def build_model(H, dt, p0, p1):
    """Reference: quintic-in/out straight line from p0 to p1, hover attitude."""
    t = np.linspace(0.0, 1.0, H + 1)
    # smooth-in/out reference (quintic), keeps the reference dynamically plausible
    ramp = 10 * t**3 - 15 * t**4 + 6 * t**5
    refs = np.outer(ramp, p1 - p0) + p0

    # NOTE: per-stage dynamics instances - reusing one object is flagged by ssqp
    dynamics = [QuadDynamics(dt) for _ in range(H + 1)]
    q_ref = np.array([1.0, 0, 0, 0])
    v_ref = np.zeros(3)
    u_ref = np.array([T_HOVER, 0, 0, 0])

    stage_costs = [TrackingCost(refs[k], q_ref, v_ref, u_ref) for k in range(H)]
    terminal = TrackingCost(refs[H], q_ref, v_ref, u_ref,
                            w_p=2000.0, w_th=50.0, w_v=500.0, w_w=10.0, w_u=0.0)
    # ssqp expects one cost per stage, terminal included
    all_costs = stage_costs + [terminal]

    # full 13-dim initial guess: reference position, hover attitude, zero velocity
    init_states = np.zeros((H + 1, NX))
    init_states[:, 0:3] = refs
    init_states[:, 3] = 1.0

    model = ssqpy.model.Model(H, dt)
    model.finalize(
        dynamics=dynamics,
        costs=all_costs,
        terminal_cost=terminal,
        # flat list, one entry per stage (terminal stage has no controls -> no-op)
        constraints=[ControlBounds() for _ in range(H)] + [NoConstraints()],
        init_states=init_states,
        init_controls=np.tile(u_ref, (H, 1)),
    )
    return model, refs


def main():
    H, DT = 60, 0.05          # 3 s horizon at 20 Hz
    p0 = np.zeros(3)
    p1 = np.array([2.0, 1.0, 1.5])

    t_build = time.perf_counter()
    model, refs = build_model(H, DT, p0, p1)
    t_build = time.perf_counter() - t_build

    params = ssqpy.solvers.ssqpParams()
    params.max_iter = 100
    params.tolerance = 1e-5
    params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

    solver = ssqpy.solvers.ssqp(model, params)

    t0 = time.perf_counter()
    state, x, u = solver.solve()
    solve_time = time.perf_counter() - t0

    xN = np.asarray(x.stage(H))
    err = np.linalg.norm(xN[0:3] - p1)

    print(f"build: {t_build * 1e3:.1f} ms   solve: {solve_time * 1e3:.1f} ms   state: {state}")
    print(f"final position {xN[0:3]}  |err| {err:.4f} m   final v {xN[7:10]}")
    print(f"thrust range [{min(u.stage(k)[0] for k in range(H)):.2f}, "
          f"{max(u.stage(k)[0] for k in range(H)):.2f}] N (hover {T_HOVER:.2f})")

    # independent forward-integration check: does the returned control sequence reproduce x?
    dyn = QuadDynamics(DT)
    xs = np.asarray(x.stage(0), dtype=np.float64)
    worst = 0.0
    for k in range(H):
        xs = dyn.evaluate(xs, np.asarray(u.stage(k)))
        worst = max(worst, np.linalg.norm(dyn.state_right_minus(xs, np.asarray(x.stage(k + 1)))))
    print(f"max dynamics residual over rollout: {worst:.2e}")
    assert err < 0.05, err
    assert worst < 1e-6, worst
    print("OK")


if __name__ == "__main__":
    main()

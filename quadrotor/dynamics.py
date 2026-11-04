"""Continuous and discrete dynamics for the Bitcraze Crazyflie 2.1 quadrotor with SSQP.

Provides:
  - Continuous-time dynamics and closed-form tangent-space Jacobians
  - Exact Lie-group RK4 integrator matching the JAX simulator
  - Chained analytic discrete Jacobians across substeps
  - StageDynamics subclass for SSQP with proper retraction and Gauss-Newton hessprods
"""

import numpy as np
import ssqpy.model.dynamics

from quadrotor.params import (
    M_NOMINAL, L, KF, KM, G, GRAV, I_NOMINAL, I_INV,
    Kaero, TAU_M, MAX_THRUST, HOVER_RPM, U_HOVER,
    NX, NDX, NU, SIM_DT,
)
from quadrotor.quaternion import (
    skew, quat_exp, quat_log, quat_mul, quat_conj, quat_to_R, R_to_quat, exp_so3,
    state_right_plus, state_right_minus,
)


def continuous_dynamics_tangent(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous-time rate of the state in tangent space R^16.
    Returns: [p_dot(3), theta_dot(3), vb_dot(3), wb_dot(3), om_bar_dot(4)]
    """
    q = x[3:7] / np.linalg.norm(x[3:7])
    R = quat_to_R(q)
    vb = x[7:10]
    wb = x[10:13]
    om_bar = x[13:17]
    om = om_bar * MAX_THRUST

    # 1. World position rate: p_dot = R_wb @ vb
    p_dot = R @ vb

    # 2. Body attitude rate (angular velocity in body frame)
    theta_dot = wb

    # 3. Body linear velocity rate:
    # F_b = R.T @ [0, 0, -M*G] + F_thrust - F_aero
    # vb_dot = (1/M) * F_b - wb x vb
    g_b = R.T @ GRAV
    thrust_b = np.array([0.0, 0.0, (KF * np.sum(om**2)) / M_NOMINAL], dtype=np.float64)
    drag_b = (np.sum(np.abs(om)) * (Kaero @ vb)) / M_NOMINAL
    coriolis_b = np.cross(wb, vb)
    vb_dot = g_b + thrust_b - drag_b - coriolis_b

    # 4. Body angular velocity rate:
    # tau = [tau_phi, tau_theta, tau_psi]
    sqrt2 = np.sqrt(2.0)
    tau_phi = L * (-KF*om[0]**2 - KF*om[1]**2 + KF*om[2]**2 + KF*om[3]**2) / sqrt2
    tau_theta = L * (-KF*om[0]**2 + KF*om[1]**2 + KF*om[2]**2 - KF*om[3]**2) / sqrt2
    tau_psi = KM * (-om[0]**2 + om[1]**2 - om[2]**2 + om[3]**2)
    tau = np.array([tau_phi, tau_theta, tau_psi], dtype=np.float64)

    gyro = np.cross(wb, I_NOMINAL @ wb)
    wb_dot = I_INV @ (tau - gyro)

    # 5. Rotor motor lag: om_bar_dot = (u_bar - om_bar) / tau_m
    om_bar_dot = (1.0 / TAU_M) * (u - om_bar)

    return np.concatenate([p_dot, theta_dot, vb_dot, wb_dot, om_bar_dot])


def continuous_jacobian_tangent(x: np.ndarray, u: np.ndarray):
    """Analytic tangent-space Jacobians of continuous dynamics.
    Ac = d(x_dot_tangent) / d(dx) in R^{16 x 16}
    Bc = d(x_dot_tangent) / d(u) in R^{16 x 4}
    """
    q = x[3:7] / np.linalg.norm(x[3:7])
    R = quat_to_R(q)
    vb = x[7:10]
    wb = x[10:13]
    om_bar = x[13:17]
    om = om_bar * MAX_THRUST

    Ac = np.zeros((NDX, NDX), dtype=np.float64)
    Bc = np.zeros((NDX, NU), dtype=np.float64)

    # 1. p_dot = R @ vb
    # d(p_dot) / dtheta = -R @ [vb]x
    Ac[0:3, 3:6] = -R @ skew(vb)
    # d(p_dot) / dvb = R
    Ac[0:3, 6:9] = R

    # 2. theta_dot = wb
    Ac[3:6, 9:12] = np.eye(3, dtype=np.float64)

    # 3. vb_dot = R^T g_w + thrust - drag - wb x vb
    g_b = R.T @ GRAV
    # d(g_b) / dtheta = [g_b]x
    Ac[6:9, 3:6] = skew(g_b)
    # d(vb_dot) / dvb = -[wb]x - (sum(|om|) * Kaero) / M
    Ac[6:9, 6:9] = -skew(wb) - (np.sum(np.abs(om)) / M_NOMINAL) * Kaero
    # d(vb_dot) / dwb = [vb]x
    Ac[6:9, 9:12] = skew(vb)
    # d(vb_dot) / d(om_bar)
    for i in range(4):
        dthrust = np.array([0.0, 0.0, (2.0 * KF * MAX_THRUST**2 * om_bar[i]) / M_NOMINAL])
        ddrag = (MAX_THRUST * np.sign(om_bar[i]) * (Kaero @ vb)) / M_NOMINAL
        Ac[6:9, 12 + i] = dthrust - ddrag

    # 4. wb_dot = I_inv @ (tau - wb x I wb)
    I_wb = I_NOMINAL @ wb
    dgyro = -skew(wb) @ I_NOMINAL + skew(I_wb)
    Ac[9:12, 9:12] = I_INV @ dgyro

    sqrt2 = np.sqrt(2.0)
    c_kf = (2.0 * L * KF * MAX_THRUST**2) / sqrt2
    c_km = 2.0 * KM * MAX_THRUST**2
    dtau_dom = np.array([
        [-c_kf * om_bar[0], -c_kf * om_bar[1],  c_kf * om_bar[2],  c_kf * om_bar[3]],
        [-c_kf * om_bar[0],  c_kf * om_bar[1],  c_kf * om_bar[2], -c_kf * om_bar[3]],
        [-c_km * om_bar[0],  c_km * om_bar[1], -c_km * om_bar[2],  c_km * om_bar[3]]
    ], dtype=np.float64)
    Ac[9:12, 12:16] = I_INV @ dtau_dom

    # 5. om_bar_dot = (u - om_bar) / tau_m
    Ac[12:16, 12:16] = -(1.0 / TAU_M) * np.eye(4, dtype=np.float64)
    Bc[12:16, 0:4] = (1.0 / TAU_M) * np.eye(4, dtype=np.float64)

    return Ac, Bc


def rk4_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Single Lie-group RK4 integration step matching the JAX simulator line-for-line."""
    rw_k = x[0:3]
    R_k = quat_to_R(x[3:7])
    vb_k = x[7:10]
    wb_k = x[10:13]
    omega_k = x[13:17] * MAX_THRUST
    u_rpm = u * MAX_THRUST

    def rates(rw, R, vb, wb, omega):
        omega_dot = (1.0 / TAU_M) * (u_rpm - omega)
        rw_dot = R @ vb
        F_gravity = R.T @ GRAV
        F_thrust = np.array([0.0, 0.0, (KF * np.sum(omega**2)) / M_NOMINAL])
        Faero = (np.sum(np.abs(omega)) * (Kaero @ vb)) / M_NOMINAL
        vb_dot = F_gravity + F_thrust - Faero - np.cross(wb, vb)

        sqrt2 = np.sqrt(2.0)
        tau_phi = L * (-KF*omega[0]**2 - KF*omega[1]**2 + KF*omega[2]**2 + KF*omega[3]**2) / sqrt2
        tau_theta = L * (-KF*omega[0]**2 + KF*omega[1]**2 + KF*omega[2]**2 - KF*omega[3]**2) / sqrt2
        tau_psi = KM * (-omega[0]**2 + omega[1]**2 - omega[2]**2 + omega[3]**2)
        tau = np.array([tau_phi, tau_theta, tau_psi], dtype=np.float64)
        wb_dot = I_INV @ (tau - np.cross(wb, I_NOMINAL @ wb))
        return rw_dot, vb_dot, wb_dot, omega_dot

    rw1, vb1, wb1, om1 = rates(rw_k, R_k, vb_k, wb_k, omega_k)

    wb_temp2 = wb_k + 0.5 * dt * wb1
    rw2, vb2, wb2, om2 = rates(
        rw_k + 0.5 * dt * rw1,
        R_k @ exp_so3(0.5 * dt * wb_k),
        vb_k + 0.5 * dt * vb1,
        wb_temp2,
        omega_k + 0.5 * dt * om1
    )

    wb_temp3 = wb_k + 0.5 * dt * wb2
    rw3, vb3, wb3, om3 = rates(
        rw_k + 0.5 * dt * rw2,
        R_k @ exp_so3(0.5 * dt * wb_temp2),
        vb_k + 0.5 * dt * vb2,
        wb_temp3,
        omega_k + 0.5 * dt * om2
    )

    wb_temp4 = wb_k + dt * wb3
    rw4, vb4, wb4, om4 = rates(
        rw_k + dt * rw3,
        R_k @ exp_so3(dt * wb_temp3),
        vb_k + dt * vb3,
        wb_temp4,
        omega_k + dt * om3
    )

    rw_next = rw_k + (dt / 6.0) * (rw1 + 2.0 * rw2 + 2.0 * rw3 + rw4)
    vb_next = vb_k + (dt / 6.0) * (vb1 + 2.0 * vb2 + 2.0 * vb3 + vb4)
    wb_next = wb_k + (dt / 6.0) * (wb1 + 2.0 * wb2 + 2.0 * wb3 + wb4)
    omega_next = omega_k + (dt / 6.0) * (om1 + 2.0 * om2 + 2.0 * om3 + om4)

    w_final = dt * wb_k + (dt**2 / 6.0) * (wb1 + wb2 + wb3)
    R_next = R_k @ exp_so3(w_final)
    q_next = R_to_quat(R_next)

    return np.concatenate([rw_next, q_next, vb_next, wb_next, omega_next / MAX_THRUST])


def rk4_step_jacobian(x: np.ndarray, u: np.ndarray, dt: float):
    """Analytic discrete Jacobian for a single Lie-group RK4 step."""
    A1, B1 = continuous_jacobian_tangent(x, u)
    k1 = continuous_dynamics_tangent(x, u)

    x2 = state_right_plus(x, 0.5 * dt * k1)
    A2_raw, B2_raw = continuous_jacobian_tangent(x2, u)
    k2 = continuous_dynamics_tangent(x2, u)
    Phi1 = np.eye(NDX) + 0.5 * dt * A1
    A2 = A2_raw @ Phi1
    B2 = B2_raw + A2_raw @ (0.5 * dt * B1)

    x3 = state_right_plus(x, 0.5 * dt * k2)
    A3_raw, B3_raw = continuous_jacobian_tangent(x3, u)
    k3 = continuous_dynamics_tangent(x3, u)
    Phi2 = np.eye(NDX) + 0.5 * dt * A2
    A3 = A3_raw @ Phi2
    B3 = B3_raw + A3_raw @ (0.5 * dt * B2)

    x4 = state_right_plus(x, dt * k3)
    A4_raw, B4_raw = continuous_jacobian_tangent(x4, u)
    Phi3 = np.eye(NDX) + dt * A3
    A4 = A4_raw @ Phi3
    B4 = B4_raw + A4_raw @ (dt * B3)

    Ad = np.eye(NDX) + (dt / 6.0) * (A1 + 2.0 * A2 + 2.0 * A3 + A4)
    Bd = (dt / 6.0) * (B1 + 2.0 * B2 + 2.0 * B3 + B4)
    return Ad, Bd


def rk4_integrate(x: np.ndarray, u: np.ndarray, dt: float, n_sub: int = 1) -> np.ndarray:
    """Integrate across interval dt using n_sub internal RK4 substeps."""
    h = dt / n_sub
    curr = x.copy()
    for _ in range(n_sub):
        curr = rk4_step(curr, u, h)
    return curr


def rk4_integrate_jacobian(x: np.ndarray, u: np.ndarray, dt: float, n_sub: int = 1):
    """Chained discrete Jacobian across n_sub internal RK4 substeps."""
    h = dt / n_sub
    curr = x.copy()
    A_total = np.eye(NDX, dtype=np.float64)
    B_total = np.zeros((NDX, NU), dtype=np.float64)

    for _ in range(n_sub):
        Ad_k, Bd_k = rk4_step_jacobian(curr, u, h)
        # Chain rule: x_{s+1} = f(x_s, u)
        # dx_{s+1} = Ad_k dx_s + Bd_k du
        # => dx_next = Ad_k (A_prev dx0 + B_prev du) + Bd_k du
        A_total = Ad_k @ A_total
        B_total = Ad_k @ B_total + Bd_k
        curr = rk4_step(curr, u, h)

    return A_total, B_total


class CrazyflieDynamics(ssqpy.model.dynamics.StageDynamics):
    """Crazyflie 2.1 StageDynamics with Lie-group RK4 and analytic Jacobians."""

    def __init__(self, dt: float, substeps: int = None, use_analytic: bool = True):
        super().__init__()
        self.dt = dt
        if substeps is None:
            # Default to 5ms substeps if dt >= 5ms, else 1
            self.substeps = max(1, int(round(dt / SIM_DT)))
        else:
            self.substeps = max(1, int(substeps))
        self.use_analytic = use_analytic

    def output_size(self) -> int:
        return NX

    def state_size(self) -> int:
        return NX

    def control_size(self) -> int:
        return NU

    def state_diff_size(self) -> int:
        return NDX

    def init_state(self) -> np.ndarray:
        s = np.zeros(NX, dtype=np.float64)
        s[3] = 1.0  # level orientation [1, 0, 0, 0]
        s[13:17] = U_HOVER
        return s

    def init_control(self) -> np.ndarray:
        return U_HOVER.copy()

    def state_right_plus(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return state_right_plus(x, dx)

    def state_right_minus(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return state_right_minus(x, y)

    def evaluate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return rk4_integrate(x, u, self.dt, self.substeps)

    def gradient(self, x: np.ndarray, u: np.ndarray):
        if self.use_analytic:
            return rk4_integrate_jacobian(x, u, self.dt, self.substeps)

        # Finite-difference fallback
        eps = 1e-6
        dX = np.zeros((NDX, NDX), dtype=np.float64)
        dU = np.zeros((NDX, NU), dtype=np.float64)

        for j in range(NDX):
            e = np.zeros(NDX, dtype=np.float64)
            e[j] = eps
            fp = self.evaluate(self.state_right_plus(x, e), u)
            fm = self.evaluate(self.state_right_plus(x, -e), u)
            dX[:, j] = self.state_right_minus(fp, fm) / (2.0 * eps)

        for j in range(NU):
            e = np.zeros(NU, dtype=np.float64)
            e[j] = eps
            fp = self.evaluate(x, u + e)
            fm = self.evaluate(x, u - e)
            dU[:, j] = self.state_right_minus(fp, fm) / (2.0 * eps)

        return dX, dU

    def hessprod_xx(self, x: np.ndarray, u: np.ndarray, lam: np.ndarray) -> np.ndarray:
        return np.zeros((NDX, NDX), dtype=np.float64)

    def hessprod_xu(self, x: np.ndarray, u: np.ndarray, lam: np.ndarray) -> np.ndarray:
        return np.zeros((NDX, NU), dtype=np.float64)

    def hessprod_uu(self, x: np.ndarray, u: np.ndarray, lam: np.ndarray) -> np.ndarray:
        return np.zeros((NU, NU), dtype=np.float64)


class FixedStartCF(CrazyflieDynamics):
    """Stage-0 dynamics to enforce initial condition x0 in SSQP."""

    def __init__(self, dt: float, x0: np.ndarray, u0: np.ndarray,
                 substeps: int = None, use_analytic: bool = True):
        super().__init__(dt, substeps=substeps, use_analytic=use_analytic)
        self._x0 = np.asarray(x0, dtype=np.float64)
        self._u0 = np.asarray(u0, dtype=np.float64)

    def init_state(self) -> np.ndarray:
        return self._x0

    def init_control(self) -> np.ndarray:
        return self._u0

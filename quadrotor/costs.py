"""Tracking cost for quadrotor optimal control."""

import numpy as np
import ssqpy.model.costs

from quadrotor.params import NX, NDX, NU, U_HOVER
from quadrotor.quaternion import quat_conj, quat_mul, quat_log


class CFTrackingCost(ssqpy.model.costs.StageCost):
    """Quadratic tracking cost for quadrotor reference trajectory.

    Penalizes:
      - Position error: w_p * ||p - p_ref||^2
      - Attitude error: w_th * ||Log(q_ref^* (x) q)||^2
      - Velocity error: w_v * ||vb - v_ref||^2
      - Angular rate:   w_w * ||wb||^2
      - Control effort: w_u * ||u - u_ref||^2
    """

    def __init__(self, p_ref: np.ndarray, q_ref: np.ndarray, v_ref: np.ndarray,
                 u_ref: np.ndarray = None,
                 w_p: float = 200.0, w_th: float = 10.0, w_v: float = 20.0,
                 w_w: float = 0.5, w_u: float = 0.1, name: str = "cf_tracking"):
        super().__init__(name)
        self.p_ref = np.asarray(p_ref, dtype=np.float64)
        self.q_ref = np.asarray(q_ref, dtype=np.float64)
        self.v_ref = np.asarray(v_ref, dtype=np.float64)
        if u_ref is None:
            self.u_ref = U_HOVER.copy()
        else:
            self.u_ref = np.asarray(u_ref, dtype=np.float64)

        self.w_p = float(w_p)
        self.w_th = float(w_th)
        self.w_v = float(w_v)
        self.w_w = float(w_w)
        self.w_u = float(w_u)

    def evaluate(self, x: np.ndarray, u: np.ndarray) -> float:
        dp = x[0:3] - self.p_ref
        dth = quat_log(quat_mul(quat_conj(self.q_ref), x[3:7]))
        dv = x[7:10] - self.v_ref
        dw = x[10:13]
        # Terminal stage passes empty u
        du = (u - self.u_ref) if u.size else np.zeros(NU, dtype=np.float64)

        cost = (self.w_p * np.dot(dp, dp)
                + self.w_th * np.dot(dth, dth)
                + self.w_v * np.dot(dv, dv)
                + self.w_w * np.dot(dw, dw)
                + self.w_u * np.dot(du, du))
        return float(cost)

    def gradient(self, x: np.ndarray, u: np.ndarray):
        dp = x[0:3] - self.p_ref
        dth = quat_log(quat_mul(quat_conj(self.q_ref), x[3:7]))
        dv = x[7:10] - self.v_ref
        dw = x[10:13]

        gx = 2.0 * np.concatenate([
            self.w_p * dp,
            self.w_th * dth,
            self.w_v * dv,
            self.w_w * dw,
            np.zeros(4, dtype=np.float64)  # No direct cost on motor state omega_bar
        ])

        if u.size:
            gu = 2.0 * self.w_u * (u - self.u_ref)
        else:
            gu = np.zeros(0, dtype=np.float64)

        return np.asarray(gx, dtype=np.float64), np.asarray(gu, dtype=np.float64)

    def hessian_xx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        diag = ([2.0 * self.w_p] * 3
                + [2.0 * self.w_th] * 3
                + [2.0 * self.w_v] * 3
                + [2.0 * self.w_w] * 3
                + [0.0] * 4)
        return np.diag(np.array(diag, dtype=np.float64))

    def hessian_xu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if u.size:
            return np.zeros((NDX, NU), dtype=np.float64)
        return np.zeros((NDX, 0), dtype=np.float64)

    def hessian_uu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if u.size:
            return np.diag(np.full(NU, 2.0 * self.w_u, dtype=np.float64))
        return np.zeros((0, 0), dtype=np.float64)

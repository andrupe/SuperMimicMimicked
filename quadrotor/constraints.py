"""Stage constraints for quadrotor optimal control."""

import numpy as np
import ssqpy.model.constraints

from quadrotor.params import NDX, NU, U_MIN, U_MAX


class CFControlBounds(ssqpy.model.constraints.StageConstraints):
    """Linear box constraints on normalized rotor commands: 0 <= u_i <= 1."""

    def __init__(self, u_min: np.ndarray = None, u_max: np.ndarray = None, name: str = "cf_control_bounds"):
        super().__init__(name)
        self._u_min = U_MIN if u_min is None else np.asarray(u_min, dtype=np.float64)
        self._u_max = U_MAX if u_max is None else np.asarray(u_max, dtype=np.float64)

    def output_size(self) -> int:
        return NU

    def lower_bound(self) -> np.ndarray:
        return self._u_min

    def upper_bound(self) -> np.ndarray:
        return self._u_max

    def evaluate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=np.float64)

    def gradient(self, x: np.ndarray, u: np.ndarray):
        return np.zeros((NU, NDX), dtype=np.float64), np.eye(NU, dtype=np.float64)

    def hessprod_xx(self, x: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.zeros((NDX, NDX), dtype=np.float64)

    def hessprod_xu(self, x: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.zeros((NDX, NU), dtype=np.float64)

    def hessprod_uu(self, x: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.zeros((NU, NU), dtype=np.float64)


class CFNoConstraints(ssqpy.model.constraints.StageConstraints):
    """Empty constraint set for the terminal stage (which has no controls)."""

    def __init__(self, name: str = "cf_no_constraints"):
        super().__init__(name)

    def output_size(self) -> int:
        return 0

    def lower_bound(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    def upper_bound(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    def evaluate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    def gradient(self, x: np.ndarray, u: np.ndarray):
        return np.zeros((0, NDX), dtype=np.float64), np.zeros((0, NU), dtype=np.float64)

    def hessprod_xx(self, x: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.zeros((NDX, NDX), dtype=np.float64)

    def hessprod_xu(self, x: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.zeros((NDX, NU), dtype=np.float64)

    def hessprod_uu(self, x: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.zeros((NU, NU), dtype=np.float64)

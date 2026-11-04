"""Single source of truth for Bitcraze Crazyflie 2.1 parameters.

Matches crazyflie_nn/dynamics/dynamics.py exactly:
  - Mass: 0.033 kg
  - Arm length L: 0.046 m
  - Motor constants: KF = 2.25e-8 N / (rad/s)^2, KM = 1.34e-10 N*m / (rad/s)^2
  - Inertia: off-diagonals included (~0.8e-6), diagonal [16.6e-6, 16.6e-6, 29.3e-6]
  - Motor time constant: tau_m = 0.1 s
  - Max rotor speed: omega_max = 2618 rad/s (~25,000 RPM)
  - Hover rotor speed: sqrt(M * g / (4 * KF)) ~ 1896.57 rad/s (~72% max)
  - Aerodynamic drag: Kaero = diag([9.1785e-7, 9.1785e-7, 10.311e-7])

State representation:
  - 17-dim OCP state: x = [p(3), q(4, wxyz), vb(3), wb(3), omega_bar(4)]
  - 16-dim tangent:   dx = [dp(3), dtheta(3), dvb(3), dwb(3), domega_bar(4)]
  - 4-dim control:    u = omega_bar_cmd(4) in [0, 1]
  - 22-dim sim state: [rw(3), Rwb_flat(9), vb(3), wb(3), omega(4)] (rad/s)
"""

import numpy as np

# Physical constants
M_NOMINAL = 0.033          # kg
L = 0.046                  # m (rotor arm length)
KF = 2.25e-8               # N / (rad/s)^2
KM = 1.34e-10              # N*m / (rad/s)^2
G = 9.81                   # m/s^2
GRAV = np.array([0.0, 0.0, -G])

# Rotor bounds
MAX_THRUST = 2618.0        # rad/s (often termed max RPM in firmware)
HOVER_RPM = float(np.sqrt(G * M_NOMINAL / (4.0 * KF))) # ~1896.57 rad/s
TAU_M = 0.1                # s (motor lag time constant)

# Aerodynamic drag matrix
Kaero = np.array([
    [9.1785e-7, 0.0,        0.0],
    [0.0,        9.1785e-7, 0.0],
    [0.0,        0.0,        10.311e-7]
], dtype=np.float64)

# Inertia tensor (kg * m^2)
I_NOMINAL = np.array([
    [16.6e-6, 0.83e-6, 0.72e-6],
    [0.83e-6, 16.6e-6, 1.8e-6],
    [0.72e-6, 1.8e-6,  29.3e-6]
], dtype=np.float64)

I_INV = np.linalg.inv(I_NOMINAL)

# Normalized controls and states
U_HOVER = np.full(4, HOVER_RPM / MAX_THRUST, dtype=np.float64)
U_MIN = np.zeros(4, dtype=np.float64)
U_MAX = np.ones(4, dtype=np.float64)

# Dimensions
NX = 17    # [p(3), q(4), vb(3), wb(3), omega_bar(4)]
NDX = 16   # [dp(3), dtheta(3), dvb(3), dwb(3), domega_bar(4)]
NU = 4     # [u_bar(4)]
SIM_STATE_DIM = 22 # [p(3), R(9), vb(3), wb(3), omega(4)]

# Nominal simulation integration step
SIM_DT = 0.005 # 5 ms (200 Hz)


def omega_bar_to_rad_s(omega_bar: np.ndarray) -> np.ndarray:
    """Convert normalized rotor speed [0, 1] to physical rad/s."""
    return np.asarray(omega_bar, dtype=np.float64) * MAX_THRUST


def rad_s_to_omega_bar(omega: np.ndarray) -> np.ndarray:
    """Convert physical rotor speed (rad/s) to normalized [0, 1]."""
    return np.asarray(omega, dtype=np.float64) / MAX_THRUST

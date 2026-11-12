"""Bitcraze Crazyflie 2.1 constants and entity configuration."""

from __future__ import annotations

from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF Model Path
##

CRAZYFLIE_XML: Path = (
    MJLAB_SRC_PATH
    / "asset_zoo"
    / "robots"
    / "bitcraze_crazyflie"
    / "xmls"
    / "crazyflie.xml"
)
assert CRAZYFLIE_XML.exists(), f"Crazyflie XML not found at {CRAZYFLIE_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(CRAZYFLIE_XML))


##
# Physical parameters (matching Bitcraze hardware & crazyflie_nn)
##

MASS_NOMINAL = 0.033          # kg (33 grams)
ARM_LENGTH = 0.046            # m (rotor arm length)
KF = 2.25e-8                  # N / (rad/s)^2
KM = 1.34e-10                 # N*m / (rad/s)^2
GRAVITY = 9.81                # m/s^2

# Rotor performance
MAX_OMEGA = 2618.0            # rad/s (~25,000 RPM)
MAX_THRUST_PER_ROTOR = KF * (MAX_OMEGA**2)  # ~0.1542 N
TOTAL_MAX_THRUST = 4.0 * MAX_THRUST_PER_ROTOR # ~0.617 N
WEIGHT = MASS_NOMINAL * GRAVITY # ~0.3237 N
TWR = TOTAL_MAX_THRUST / WEIGHT # ~1.91
HOVER_THRUST_PER_ROTOR = WEIGHT / 4.0 # ~0.0809 N

# Tau motor lag
TAU_M = 0.1                   # s

##
# Articulation & Entity Config
##

CRAZYFLIE_COLLISION = CollisionCfg(
    geom_names_expr=(r"^body_core$", r"^arm_.*$", r"^motor_.*$"),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.5,),
)

CRAZYFLIE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=("rotor_.*",),
            transmission_type=TransmissionType.SITE,
        ),
    ),
)

INIT_STATE_HOVER = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
)


def get_crazyflie_robot_cfg() -> EntityCfg:
    """Get a fresh Bitcraze Crazyflie 2.1 entity configuration."""
    return EntityCfg(
        spec_fn=get_spec,
        articulation=CRAZYFLIE_ARTICULATION,
        init_state=INIT_STATE_HOVER,
        collisions=(CRAZYFLIE_COLLISION,),
    )


CRAZYFLIE_ROTOR_NAMES = ("rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr")

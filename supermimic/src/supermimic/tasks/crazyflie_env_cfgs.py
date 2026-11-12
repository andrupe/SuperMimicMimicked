"""Environment configurations for SuperMimic Bitcraze Crazyflie tasks."""

from __future__ import annotations

import mjlab.envs.mdp as mdp
from mjlab.asset_zoo.robots import get_crazyflie_robot_cfg
from mjlab.envs.mdp.dr import body_com_offset
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import SiteEffortActionCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from supermimic.tasks.crazyflie_mdp import (
    BrownianVelocityCommandCfg,
    CorpusTrajectoryCommandCfg,
    quadcopter_action_rate_penalty,
    quadcopter_body_ang_vel,
    quadcopter_body_lin_vel,
    quadcopter_corpus_ori_tracking_reward,
    quadcopter_corpus_pos_tracking_reward,
    quadcopter_crash_termination,
    quadcopter_hover_effort_penalty,
    quadcopter_last_action,
    quadcopter_lin_vel_tracking_reward,
    quadcopter_projected_gravity,
    quadcopter_root_ori,
    quadcopter_root_pos,
    quadcopter_tilt_termination,
    quadcopter_upright_reward,
    quadcopter_velocity_command,
    quadcopter_yaw_rate_tracking_reward,
)


def crazyflie_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Bitcraze Crazyflie 3D Brownian Velocity tracking environment configuration."""

    # 1. Scene configuration
    scene = SceneCfg(
        num_envs=4096 if not play else 1,
        env_spacing=2.5,
        entities={"robot": get_crazyflie_robot_cfg()},
    )

    # 2. Physics simulation configuration (200 Hz sim, 50 Hz control via decimation=4)
    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.005, # 5 ms physics step (200 Hz)
            iterations=10,
            cone="elliptic",
        ),
    )

    # 3. Actions: 4 rotor thrust efforts centered at hover (0.0809325 N)
    # action = 0.0 -> 0.0809325 N (hover), +1.0 -> 0.1542 N (max), -1.0 -> 0.007665 N (idle)
    actions = {
        "thrust": SiteEffortActionCfg(
            entity_name="robot",
            actuator_names=("rotor_.*",),
            scale=0.0732675,
            offset=0.0809325,
            clip={"rotor_.*": (0.0, 0.1542)},
        ),
    }

    # 4. Commands: 3D Brownian velocity command [vx, vy, vz, wz]
    commands = {
        "velocity": BrownianVelocityCommandCfg(
            resampling_time_range=(3.0, 5.0),
            entity_name="robot",
            ranges=(
                (-1.5, 1.5),   # vx
                (-1.0, 1.0),   # vy
                (-0.8, 0.8),   # vz
                (-3.14, 3.14), # wz
            ),
            theta=0.5,
            sigma=0.35,
        ),
    }

    # 5. Observations:
    # - actor: [projected_gravity(3), body_lin_vel(3), body_ang_vel(3), actions(4), command(4)] = 17 dims
    # - full_state: [root_pos(3), root_ori(4), body_lin_vel(3), body_ang_vel(3)] = 13 dims
    # - critic: full state (13) + actions (4) + command (4) = 21 dims
    # - goal: [command(4)]
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "projected_gravity": ObservationTermCfg(func=quadcopter_projected_gravity),
                "body_lin_vel": ObservationTermCfg(func=quadcopter_body_lin_vel),
                "body_ang_vel": ObservationTermCfg(func=quadcopter_body_ang_vel),
                "actions": ObservationTermCfg(func=quadcopter_last_action),
                "command": ObservationTermCfg(func=quadcopter_velocity_command),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "full_state": ObservationGroupCfg(
            terms={
                "root_pos": ObservationTermCfg(func=quadcopter_root_pos),
                "root_ori": ObservationTermCfg(func=quadcopter_root_ori),
                "body_lin_vel": ObservationTermCfg(func=quadcopter_body_lin_vel),
                "body_ang_vel": ObservationTermCfg(func=quadcopter_body_ang_vel),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "root_pos": ObservationTermCfg(func=quadcopter_root_pos),
                "root_ori": ObservationTermCfg(func=quadcopter_root_ori),
                "body_lin_vel": ObservationTermCfg(func=quadcopter_body_lin_vel),
                "body_ang_vel": ObservationTermCfg(func=quadcopter_body_ang_vel),
                "actions": ObservationTermCfg(func=quadcopter_last_action),
                "command": ObservationTermCfg(func=quadcopter_velocity_command),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "command": ObservationTermCfg(func=quadcopter_velocity_command),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # 6. Rewards
    rewards = {
        "track_lin_vel": RewardTermCfg(func=quadcopter_lin_vel_tracking_reward, weight=2.5),
        "track_yaw_rate": RewardTermCfg(func=quadcopter_yaw_rate_tracking_reward, weight=1.2),
        "upright": RewardTermCfg(func=quadcopter_upright_reward, weight=0.6),
        "action_rate": RewardTermCfg(func=quadcopter_action_rate_penalty, weight=-0.05),
        "hover_effort": RewardTermCfg(func=quadcopter_hover_effort_penalty, weight=-0.01),
    }

    # 7. Terminations
    terminations = {
        "crash": TerminationTermCfg(func=quadcopter_crash_termination, params={"min_altitude": 0.05, "max_altitude": 4.5}),
        "tilt": TerminationTermCfg(func=quadcopter_tilt_termination, params={"max_tilt_angle_deg": 80.0}),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }

    # 8. Events (Domain Randomization & Reset)
    events = {
        "reset_scene_to_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "base_com": EventTermCfg(
            mode="startup",
            func=body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "operation": "add",
                "ranges": {
                    0: (-0.003, 0.003),
                    1: (-0.003, 0.003),
                    2: (-0.004, 0.004),
                },
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        sim=sim,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        decimation=4,         # 4 * 0.005s = 0.02s (50 Hz control rate)
        episode_length_s=6.0, # 6-second flight episodes
    )


def crazyflie_corpus_env_cfg(
    play: bool = False, rsi: bool = True, rsi_prob: float = 0.8
) -> ManagerBasedRlEnvCfg:
    """Create Bitcraze Crazyflie Expert Trajectory Tracking environment configuration."""

    # 1. Scene configuration
    scene = SceneCfg(
        num_envs=4096 if not play else 1,
        env_spacing=2.5,
        entities={"robot": get_crazyflie_robot_cfg()},
    )

    # 2. Physics simulation configuration (200 Hz sim, 50 Hz control via decimation=4)
    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.005,  # 5 ms physics step (200 Hz)
            iterations=10,
            cone="elliptic",
        ),
    )

    # 3. Actions: 4 rotor thrust efforts centered at hover (0.0809325 N)
    # action = 0.0 -> 0.0809325 N (hover), +1.0 -> 0.1542 N (max), -1.0 -> 0.007665 N (idle)
    actions = {
        "thrust": SiteEffortActionCfg(
            entity_name="robot",
            actuator_names=("rotor_.*",),
            scale=0.0732675,
            offset=0.0809325,
            clip={"rotor_.*": (0.0, 0.1542)},
        ),
    }

    # 4. Commands: SSQP Corpus Expert Trajectory Tracking
    commands = {
        "velocity": CorpusTrajectoryCommandCfg(
            entity_name="robot",
            corpus_path="corpus/crazyflie_corpus_50trajs.npz",
            rsi=rsi,
            rsi_prob=rsi_prob,
            loop_trajectories=True,
        ),
    }

    # 5. Observations
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "projected_gravity": ObservationTermCfg(func=quadcopter_projected_gravity),
                "body_lin_vel": ObservationTermCfg(func=quadcopter_body_lin_vel),
                "body_ang_vel": ObservationTermCfg(func=quadcopter_body_ang_vel),
                "actions": ObservationTermCfg(func=quadcopter_last_action),
                "command": ObservationTermCfg(func=quadcopter_velocity_command),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "full_state": ObservationGroupCfg(
            terms={
                "root_pos": ObservationTermCfg(func=quadcopter_root_pos),
                "root_ori": ObservationTermCfg(func=quadcopter_root_ori),
                "body_lin_vel": ObservationTermCfg(func=quadcopter_body_lin_vel),
                "body_ang_vel": ObservationTermCfg(func=quadcopter_body_ang_vel),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "root_pos": ObservationTermCfg(func=quadcopter_root_pos),
                "root_ori": ObservationTermCfg(func=quadcopter_root_ori),
                "body_lin_vel": ObservationTermCfg(func=quadcopter_body_lin_vel),
                "body_ang_vel": ObservationTermCfg(func=quadcopter_body_ang_vel),
                "actions": ObservationTermCfg(func=quadcopter_last_action),
                "command": ObservationTermCfg(func=quadcopter_velocity_command),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "command": ObservationTermCfg(func=quadcopter_velocity_command),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # 6. Rewards: Tracking expert velocity, yaw rate, attitude and regularization
    rewards = {
        "track_lin_vel": RewardTermCfg(func=quadcopter_lin_vel_tracking_reward, weight=2.5),
        "track_yaw_rate": RewardTermCfg(func=quadcopter_yaw_rate_tracking_reward, weight=1.2),
        "track_ori": RewardTermCfg(func=quadcopter_corpus_ori_tracking_reward, weight=1.0),
        "upright": RewardTermCfg(func=quadcopter_upright_reward, weight=0.5),
        "action_rate": RewardTermCfg(func=quadcopter_action_rate_penalty, weight=-0.05),
        "hover_effort": RewardTermCfg(func=quadcopter_hover_effort_penalty, weight=-0.01),
    }

    # 7. Terminations
    terminations = {
        "crash": TerminationTermCfg(
            func=quadcopter_crash_termination, params={"min_altitude": 0.05, "max_altitude": 4.5}
        ),
        "tilt": TerminationTermCfg(func=quadcopter_tilt_termination, params={"max_tilt_angle_deg": 80.0}),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }

    # 8. Events
    events = {
        "reset_scene_to_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "base_com": EventTermCfg(
            mode="startup",
            func=body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                "operation": "add",
                "ranges": {
                    0: (-0.003, 0.003),
                    1: (-0.003, 0.003),
                    2: (-0.004, 0.004),
                },
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        sim=sim,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        decimation=4,
        episode_length_s=6.0,
    )

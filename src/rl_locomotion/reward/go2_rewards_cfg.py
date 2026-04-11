"""Go2-specific reward configuration matching the project spec.

Maps the 6 reward terms onto Isaac Lab's reward manager system, plus
additional regularization terms from the default locomotion config that
improve training stability.

Weight sign convention: positive = encourage, negative = penalize.
The reward manager sums (weight * term_value) across all terms.
"""

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_locomotion.reward.foot_slip import foot_slip_penalty


@configclass
class Go2RewardsCfg:
    """Reward terms for Go2 locomotion.

    Organized into three tiers:
    - PRIMARY: velocity tracking (what we want the robot to do)
    - PENALTIES: behaviors we want to suppress
    - REGULARIZATION: soft shaping for smoother/more natural gaits
    """

    # ── PRIMARY: velocity tracking ──────────────────────────────────────

    # Spec term 1: linear velocity tracking (commanded vs actual, body frame)
    # exp(-error²/std²) kernel: reward is ~1.0 when tracking perfectly
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # Spec term 2: angular velocity tracking around yaw axis
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # ── PENALTIES: suppress bad behaviors ───────────────────────────────

    # Spec term 3a: energy penalty — torque²
    dof_torques_12 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)

    # Spec term 3b: energy penalty — joint velocity²
    dof_vel_12 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)

    # Spec term 4: foot slip penalty
    foot_slip = RewTerm(
        func=foot_slip_penalty,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    # Spec term 5: action smoothness -penalize rapid action changes
    action_rate_12 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Spec term 6: termination penalty (survival incentive)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ── REGULARIZATION: shaping for natural gaits ──────────────────────

    # Penalize vertical base velocity (bouncing)
    lin_vel_z_12 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # Penalize roll and pitch angular velocity (wobbling)
    ang_vel_xy_12 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # Penalize joint acceleration (jerk)
    dof_acc_12 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # Encourage feet to spend time in the air (promotes trotting gait)
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
        },
    )

    # Penalize non-flat orientation (tilt)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    # Penalize approaching joint position limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # Penalize undesired body contacts (thighs, base hitting ground)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0,
        },
    )

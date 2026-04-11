"""Foot slip penalty reward term.

Penalizes horizontal velocity of feet that are in contact with the ground.
This discourages the policy from learning gaits that slide the feet along
the surface, which wastes energy and transfers poorly to real hardware
where ground friction is imperfect.

Reference: Lee et al., "Learning Quadrupedal Locomotion over Challenging
Terrain", Science Robotics 2020 — uses a similar foot-velocity penalty
conditioned on contact.
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def foot_slip_penalty(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot velocity when feet are in contact.

    For each foot, if the contact force exceeds `threshold`, we add the
    squared horizontal velocity of that foot's body to the penalty. Feet
    that are in the air contribute zero.

    Args:
        env: The RL environment instance.
        asset_cfg: Config identifying the robot asset. Must have
            `body_names` set to the foot body names.
        sensor_cfg: Config identifying the contact sensor. Must have
            `body_ids` corresponding to the feet.
        threshold: Contact force threshold (N) above which a foot is
            considered to be in contact. Default 1.0 N.

    Returns:
        Per-env scalar penalty (shape: [num_envs]).
    """
    # Get the contact sensor and check which feet are in contact
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    # net_forces shape: [num_envs, history_len, num_bodies, 3]
    # We check the most recent step (index 0) for contact
    is_contact = torch.norm(net_forces[:, 0, sensor_cfg.body_ids, :], dim=-1) > threshold
    # is_contact shape: [num_envs, num_feet]

    # Get horizontal velocities of the foot bodies
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # foot_vel shape: [num_envs, num_feet, 2] (x, y only)

    # Squared horizontal speed per foot
    foot_speed_sq = torch.sum(torch.square(foot_vel), dim=-1)
    # foot_speed_sq shape: [num_envs, num_feet]

    # Only penalize feet that are in contact
    slip_penalty = torch.sum(foot_speed_sq * is_contact.float(), dim=-1)
    # slip_penalty shape: [num_envs]

    return slip_penalty

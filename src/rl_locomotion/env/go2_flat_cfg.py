"""Go2 flat-terrain environment configuration.

Inherits Isaac Lab's built-in Go2 flat config and pins settings we need
for the teacher-student pipeline. Future steps will add custom rewards
(Step 5), observation split (Step 7), and terrain curriculum (Step 8).
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from rl_locomotion.reward.go2_rewards_cfg import Go2RewardsCfg


@configclass
class Go2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2 on flat terrain with project-specific overrides.

    Progressing through the build roadmap, this class will accumulate:
    - Custom reward terms (Step 5)
    - Privileged vs proprioceptive observation split (Step 7)
    - Domain randomization events (Step 9)
    """

    def __post_init__(self) -> None:
        """Apply project-specific overrides after parent init."""
        super().__post_init__()

        # ── Environment sizing ──
        self.scene.num_envs = 16
        self.episode_length_s = 10.0

        # ── Reward shaping ──
        self.rewards = Go2RewardsCfg()

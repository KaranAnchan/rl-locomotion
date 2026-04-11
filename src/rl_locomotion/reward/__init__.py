"""Reward shaping terms for locomotion (velocity tracking, energy, foot slip, smoothness).

Custom reward functions live in this package. Built-in Isaac Lab reward functions
(from isaaclab.envs.mdp) are referenced directly in the config.
"""

from .foot_slip import foot_slip_penalty

__all__ = ["foot_slip_penalty"]

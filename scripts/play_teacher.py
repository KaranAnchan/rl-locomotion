"""Replay a trained teacher checkpoint with rendered viewport.

# Headless silent rollout (verifies policy loads and runs)
    python scripts/play_teacher.py --checkpoint <path> --headless

    # Headless rollout WITH MP4 video recording
    python scripts/play_teacher.py --checkpoint <path> --headless --video --video_length 500
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained Go2 teacher.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to .pt checkpoint.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Envs to render.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000,
    help="Steps to roll out.",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record an MP4 of the rollout.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length (steps) of the video to record.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from rl_locomotion.env.go2_flat_cfg import Go2FlatEnvCfg
from rl_locomotion.training.ppo.go2_ppo_cfg import Go2FlatPPORunerCfg


def main() -> None:
    """Load a teacher checkpoint and roll it out visually."""
    # --- Configure the env ---
    env_cfg = Go2FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 60.0  # Longer episodes for visualizing

    # --- Configure the agent ---
    env = ManagerBasedRLEnv(cfg=env_cfg)

    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    agent_cfg = Go2FlatPPORunerCfg()
    agent_cfg_dict = agent_cfg.to_dict()
    for role in ("actor", "critic"):
        for stale_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            agent_cfg_dict[role].pop(stale_key, None)

    runner = OnPolicyRunner(
        env,
        agent_cfg_dict,
        log_dir=None,
        device=env.device,
    )

    runner.load(args_cli.checkpoint)

    policy = runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    for i in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}/{args_cli.num_steps}")

    env.close()


if __name__ == "__main__":
    main()

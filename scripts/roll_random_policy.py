"""Roll a uniform-random policy on the Go2 flat environment.

It verifies that:
  1. Our Go2FlatEnvCfg loads and creates a valid ManagerBasedRLEnv.
  2. The env can be stepped with random actions.
  3. Rewards, terminations, and resets work correctly.

Usage (from the repo root, with venv activated):
    python scripts/roll_random_policy.py --headless
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

# ---------------------------------------------------------------------------
# PHASE 1: Omniverse app launch (MUST happen before any isaaclab imports)
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll a random policy on Go2 flat env.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_steps", type=int, default=100, help="Number of env steps to run.")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# PHASE 2: Imports (safe now that Omniverse is running)
# ---------------------------------------------------------------------------
import torch
from isaaclab.envs import ManagerBasedRLEnv

from rl_locomotion.env.go2_flat_cfg import Go2FlatEnvCfg


def main() -> None:
    """Run a random-action rollout and print summary statistics."""
    # --- Create the environment ---
    env_cfg = Go2FlatEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"\n{'='*60}")
    print("Environment created successfully!")
    print(f"    Num envs: {env.num_envs}")
    print(f"    Obs space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    print(f"{'='*60}\n")

    # --- Reset the environment ---
    obs, info = env.reset()
    print(f"Initial obs shape: {obs['policy'].shape}")

    # --- Rollout loop ---
    total_reward = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)

    for step in range(args_cli.num_steps):
        # Sample uniform-random actions in [-1, 1]
        actions = 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(actions)

        total_reward += reward
        episode_lengths += 1

        # Print progress every 25 steps
        if (step + 1) % 25 == 0:
            mean_reward = total_reward.mean().item()
            print(
                f"Step {step + 1:4d}/{args_cli.num_steps} | "
                f"Mean cumulative reward: {mean_reward:8.2f} | "
                f"Mean ep length: {episode_lengths.mean().item():6.1f}"
            )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Rollout complete: {args_cli.num_steps} steps x {env.num_envs} envs")
    print(f"Mean total reward: {total_reward.mean().item():.2f}")
    print(f"Min total reward: {total_reward.min().item():.2f}")
    print(f"Max total reward: {total_reward.max().item():.2f}")
    print(f"{'='*60}\n")

    # --- Cleanup ---
    env.close()


if __name__ == "__main__":
    main()

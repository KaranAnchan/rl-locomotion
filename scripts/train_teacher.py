"""Train the Go2 flat-terrain teacher policy with rsl_rl PPO.

Usage:
    python scripts/train_teacher.py --headless --num_envs 4096 --max_iterations 1500

The `--headless` flag is essential for training — rendering the viewport
at 4096 parallel envs kills GPU throughput. Use `--video` flag (passed to
the standard Isaac Lab recorder) only when you want periodic rollout videos.
"""
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

# ---------------------------------------------------------------------------
# PHASE 1: Omniverse launch (must precede any isaaclab import)
# ---------------------------------------------------------------------------

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Go2 flat-terrain teacher.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=1500,
    help="Maximum PPO iterations to train.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="go2_flat_teacher",
    help="Experiment name (subdir under logs/rsl_rl/).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# PHASE 2: Imports (safe now)
# ---------------------------------------------------------------------------
import os
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from rl_locomotion.env.go2_flat_cfg import Go2FlatEnvCfg
from rl_locomotion.training.ppo.go2_ppo_cfg import Go2FlatPPORunerCfg


def main() -> None:
    """Run PPO training on the Go2 flat-terrain env."""
    # --- Configure the env ---
    env_cfg = Go2FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # --- Configure the agent ---
    agent_cfg = Go2FlatPPORunerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg.seed = args_cli.seed

    # --- Log directory ---
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Training Go2 flat-terrain teacher")
    print(f"    Num envs: {env_cfg.scene.num_envs}")
    print(f"    Max iters: {agent_cfg.max_iterations}")
    print(f"    Log dir: {log_dir}")
    print(f"{'='*60}\n")

    # --- Create env and wrap for rsl_rl ---
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # --- Create runner and train ---
    agent_cfg_dict = agent_cfg.to_dict()

    # Strip kwargs that newer RslRlMLPModelCfg defines but older MLPModel rejects.
    # Schema mismatch between isaaclab_rl config layer and rsl_rl model layer in
    # the 4.0 transition. Will resolve naturally as both pin to the same version.
    for role in ("actor", "critic"):
        for stale_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            agent_cfg_dict[role].pop(stale_key, None)

    runner = OnPolicyRunner(
        env,
        agent_cfg_dict,
        log_dir=log_dir,
        device=env.device,
    )

    # Save the config alongside the checkpoints for reproducibility
    agent_cfg_path = os.path.join(log_dir, "agent_cfg.yaml")
    with open(agent_cfg_path, "w") as f:
        import yaml

        yaml.dump(agent_cfg.to_dict(), f)

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    # --- Final save ---
    final_path = os.path.join(log_dir, "teacher_final.pt")
    runner.save(final_path)
    print(f"\nFinal checkpoint saved: {final_path}")

    env.close()


if __name__ == "__main__":
    main()

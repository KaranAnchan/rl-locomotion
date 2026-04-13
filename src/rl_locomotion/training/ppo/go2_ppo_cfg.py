"""PPO agent configuration for the Go2 flat-terrain teacher.

Hyperparameters follow the spec §2.6 and ETH's published defaults for
quadruped locomotion. The teacher uses a 3x512 MLP actor-critic over
the concatenated observation.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_rl.rsl_rl.rl_cfg import RslRlMLPModelCfg


@configclass
class Go2FlatPPORunerCfg(RslRlOnPolicyRunnerCfg):
    """rsl_rl PPO runner config for the Go2 flat-terrain teacher."""

    # How many env steps per PPO update
    num_steps_per_env = 24

    # Maximum training iterations (tune per run; default short for dev)
    max_iterations = 1500

    # Save a checkpoint every N iterations
    save_interval = 100

    # Experiment naming — logs go to logs/rsl_rl/<experiment_name>/<run_date>
    experiment_name = "go2_flat_teacher"
    run_name = ""

    # Tell rsl_rl which obs group from the env feeds actor vs critic.
    # Both use "policy" until we add a privileged group in Step 7.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # Actor: 3x512 ELU MLP
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic: 3x512 ELU MLP
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )

    # PPO hyperparameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,  # minibatch = 4096 envs * 24 steps / 4 = 24576
        learning_rate=1.0e-3,
        schedule="adaptive",  # KL-adaptive learning rate decay
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

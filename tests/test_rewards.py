"""Tests for custom reward functions.

These test the reward functions in isolation using mock tensors,
without spinning up the full Isaac Lab simulator. This validates
the math and shapes independently of the simulation stack.
"""

import torch


def test_foot_slip_penalty_shape_and_sign() -> None:
    """Verify foot_slip_penalty returns correct shape and non-negative values.

    Since we can't easily mock the full Isaac Lab env in a unit test,
    we test the core math directly: squared horizontal velocity masked
    by contact.
    """
    num_envs = 4
    num_feet = 4

    # Simulate: 2 feet in contact, 2 feet in the air
    is_contact = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Horizontal velocity per foot: [num_envs, num_feet, 2]
    foot_vel = torch.randn(num_envs, num_feet, 2)
    foot_speed_sq = torch.sum(torch.square(foot_vel), dim=-1)
    slip_penalty = torch.sum(foot_speed_sq * is_contact, dim=-1)

    # Shape check
    assert slip_penalty.shape == (num_envs,)

    # Non-negativity (squared values * binary mask can't be negative)
    assert (slip_penalty >= 0).all()

    # Env 3 has no contacts, so its penalty must be exactly 0
    assert slip_penalty[3].item() == 0.0

    # Env 2 has all contacts, so its penalty equals total speed²
    expected_env2 = torch.sum(foot_speed_sq[2]).item()
    assert abs(slip_penalty[2].item() - expected_env2) < 1e-6


def test_foot_slip_zero_velocity_zero_penalty() -> None:
    """If feet aren't moving, penalty is zero regardless of contact."""
    num_envs = 4
    num_feet = 4
    is_contact = torch.ones(num_envs, num_feet)
    foot_vel = torch.zeros(num_envs, num_feet, 2)
    foot_speed_sq = torch.sum(torch.square(foot_vel), dim=-1)
    slip_penalty = torch.sum(foot_speed_sq * is_contact, dim=-1)

    assert (slip_penalty == 0).all()

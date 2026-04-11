"""Smoke test: every public submodule of rl_locomotion imports cleanly.

Modules that transitively import Isaac Lab require Omniverse to be running,
so they cannot be imported under plain pytest. They are exercised by the
rollout scripts instead, which launch AppLauncher first.
"""

import importlib

import pytest

# Pure-Python modules — safe to import without Isaac Lab / Omniverse running
PURE_SUBMODULES = [
    "rl_locomotion",
    "rl_locomotion.env",
    "rl_locomotion.terrain",
    "rl_locomotion.policy",
    "rl_locomotion.policy.teacher",
    "rl_locomotion.policy.student",
    "rl_locomotion.training",
    "rl_locomotion.training.ppo",
    "rl_locomotion.training.distill",
    "rl_locomotion.randomization",
    "rl_locomotion.eval",
    "rl_locomotion.eval.reality_gap",
    "rl_locomotion.deploy",
]

# Sim-dependent modules — require Omniverse, skipped at unit-test level
SIM_SUBMODULES = [
    "rl_locomotion.reward",
]


@pytest.mark.parametrize("module_name", PURE_SUBMODULES)
def test_pure_module_imports(module_name: str) -> None:
    """Importing the module must not raise."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", SIM_SUBMODULES)
def test_sim_module_skipped(module_name: str) -> None:
    """Sim-dependent modules cannot be imported under plain pytest.

    They are validated by the rollout scripts, which launch Omniverse
    via AppLauncher before any isaaclab imports.
    """
    pytest.skip(f"{module_name} requires Omniverse — exercised via rollout scripts")


def test_version_string_present() -> None:
    """The top-level package must expose a version string."""
    import rl_locomotion

    assert isinstance(rl_locomotion.__version__, str)
    assert len(rl_locomotion.__version__) > 0

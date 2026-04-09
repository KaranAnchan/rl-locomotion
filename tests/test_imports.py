"""Smoke test: every public submodule of rl_locomotion imports cleanly.

This is a structural test. It catches typos in package
names, missing __init__.py files, and circular imports.
"""

import importlib

import pytest

SUBMODULES = [
    "rl_locomotion",
    "rl_locomotion.env",
    "rl_locomotion.terrain",
    "rl_locomotion.reward",
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


@pytest.mark.parametrize("module_name", SUBMODULES)
def test_module_imports(module_name: str) -> None:
    """Importing the module must not raise."""
    importlib.import_module(module_name)


def test_version_string_present() -> None:
    """The top-level package must expose a version string."""
    import rl_locomotion

    assert isinstance(rl_locomotion.__version__, str)
    assert len(rl_locomotion.__version__) > 0

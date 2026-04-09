"""Smoke test script: print package version and confirm all submodules import."""

import rl_locomotion


def main() -> None:
    """Print the package version. Run via `python scripts/hello.py`."""
    print(f"rl-locomotion v{rl_locomotion.__version__}")
    print("Package skeleton is alive.")


if __name__ == "__main__":
    main()

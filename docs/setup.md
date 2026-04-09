# Development Environment Setup

This project targets **Windows 11 native** with an NVIDIA Blackwell GPU.
Linux setup should follow the [Isaac Lab Linux install docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html); the dependency pins below still apply.

## Verified working configuration

| Component               | Version                          |
|-------------------------|----------------------------------|
| OS                      | Windows 11 Pro (build 26200)    |
| CPU                     | AMD Ryzen 7 7700                |
| GPU                     | NVIDIA GeForce RTX 5070 (12 GB) |
| NVIDIA driver           | 595.97                          |
| CUDA runtime (per nvidia-smi) | 13.2 (driver-supported max) |
| Python                  | 3.11.9                          |
| PyTorch                 | 2.7.0+cu128                     |
| torchvision             | 0.22.0+cu128                    |
| torchaudio              | 2.7.0+cu128                     |
| tensordict              | 0.7.2 (pinned for torch 2.7 ABI)|
| Isaac Sim               | 5.1.0 (via isaaclab pip bundle) |
| Isaac Lab pip bundle    | 2.3.2.post1                     |
| Isaac Lab repo SHA      | 4df6560e187f2cc66685b41b21b259f4485d0c22           |
| Shell                   | PowerShell       |

## Install steps

> **Critical:** use PowerShell only. Git Bash mangles Windows paths and breaks `uv`.
> If conda is installed, ensure it does not auto-activate `base` in PowerShell:
> `conda config --set auto_activate_base false` then open a fresh shell.

1. **NVIDIA driver**: install Studio Driver ≥ 580.88 from the NVIDIA app, reboot, verify with `nvidia-smi`.
2. **Windows long paths**: enable via gpedit (`Computer Configuration → Administrative Templates → System → Filesystem → Enable Win32 long paths`) or registry, then reboot.
3. **Python 3.11**: install from python.org, check "Add to PATH". Verify with `py -3.11 --version`.
4. **uv**: install via PowerShell (not Git Bash):
```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
   Open a fresh shell after install so PATH updates take effect.
5. **PowerShell execution policy** (one-time):
```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
6. **Project venv**: from the repo root,
```powershell
   uv sync
   .\.venv\Scripts\Activate.ps1
```
   Verify `where.exe python` shows the venv path on the first line.
7. **Isaac Lab pip bundle** (the big one, ~10–15 GB):
```powershell
   uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```
8. **PyTorch for Blackwell** (must match CUDA 12.8):
```powershell
   uv pip install -U torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
9. **Pin tensordict for torch 2.7 ABI compatibility** (newer tensordict crashes on import):
```powershell
   uv pip install --force-reinstall --no-deps tensordict==0.7.2
```
10. **Verify GPU + imports**:
```powershell
    python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
    python -c "import isaacsim, tensordict; print('imports OK')"
```
    Expected: `True 12.8` and `imports OK`.
11. **Clone IsaacLab repo** (separate from this project, for the example scripts and `isaaclab.bat`):
```powershell
    cd C:\
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    .\isaaclab.bat --install
```
12. **GPU PhysX probe**:
```powershell
    .\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --headless --max_iterations 10
```
    Look for `Steps per second: <large number>` (target: ≥ 20,000; this machine sustains ~190,000).

## Probe result

See `docs/probe-output.log` for full output. Headline numbers from a successful run:

- `[INFO][AppLauncher]: Using device: cuda:0`
- `Number of environments: 4096`
- Iteration 1 throughput: **201,433 steps/second**
- 10 iterations completed in **13.02 seconds**
- Mean episode length grew from 19 → 232 (agent is learning, sanity check on full training pipeline)

## Known dependency hazards (record of pain)

| Pitfall                                          | Fix                                              |
|--------------------------------------------------|--------------------------------------------------|
| Default tensordict from PyPI built for newer torch ABI; segfaults on import | Pin to `tensordict==0.7.2` |
| `pip install -U torch ...` removes torchaudio    | Reinstall torchaudio explicitly in step 8        |
| Git Bash + `uv venv` → broken cygdrive paths     | Use PowerShell only                              |
| Conda auto-activates `base` and shadows venv `python` | `conda config --set auto_activate_base false` |
| `uv venv` creates env without pip                | Use `uv pip` instead of `python -m pip`          |
| `Activate.ps1` blocked by execution policy       | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## What to do if PhysX falls back to CPU on a different machine

If a future install on different hardware shows `Steps per second` in the hundreds rather than tens of thousands, that machine is hitting the Blackwell PhysX issue. Options in order of preference:

1. Try a different NVIDIA driver (both newer and the 570.195.03 fallback some users report success with).
2. Try PyTorch nightly cu128 build.
3. Switch to a Linux machine (dual-boot or cloud).

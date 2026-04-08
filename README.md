# rl-locomotion
Quadruped sim-to-real locomotion via teacher-student priveleged distillation.

Trains a Unitree Go2 control policy in NVIDIA Isaac Lab using the canonical ETH Robotic System Lab method: a privileged PPO teacher is distilled into a proprioception-only student via DAgger, then evaluated zero-shot on held-out physics in MuJoCo MJX and exported to ONNX for C++ deployment at 400 Hz control rate.

## Status

🚧 Under construction. See [BUILD_ROADMAP.md](BUILD_ROADMAP.md) for build phases

## Phases

1. **Teacher training** — PPO on 4096 parallel Isaac Lab envs with privileged observations (heightmap, friction, contact forces) and an auto-curriculum from flat terrain to random heightfields.
2. **Distillation** — DAgger-style transfer to a proprioception-only student with a temporal encoder over a 50-step history window.
3. **Reality-gap evaluation** — zero-shot MuJoCo MJX rollouts on physics parameters drawn outside the training randomization distribution.
4. **Deployment** — ONNX export and C++ inference harness targeting sub-millisecond p99 latency.

## License

MIT

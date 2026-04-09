# rl-locomotion — Build Roadmap

> **Purpose of this doc:** Single source of truth for building the quadruped sim-to-real project

---

## 1. Project one-liner

Train a Unitree Go2 quadruped policy via privileged teacher → proprioceptive student distillation
in Isaac Lab, with 14-D domain randomization, held-out MuJoCo MJX eval, and a C++ ONNX
deployment harness.

---

## 2. Locked conventions

### 2.1 Language & tooling
- **Python**: 3.14 (Isaac Lab's supported version as of build date)
- **Env manager**: `uv` (fast, modern) — fallback to `conda` only if Isaac Lab forces it
- **Formatter / linter**: `ruff` (replaces black + flake8 + isort in one tool)
- **Type checker**: `mypy` in `--strict` mode for new code, lenient for imported libs
- **Test runner**: `pytest`
- **Docstring style**: Google style
- **Pre-commit**: `pre-commit` framework running ruff + mypy + trailing-whitespace

### 2.2 Git workflow (trunk-based with short-lived branches)
- **Default branch**: `main`. Always buildable. Never push broken code here.
- **Feature branches**: `feat/<short-slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`.
  One branch per logical change.
- **Merge strategy**: squash-merge into `main` via PR.
- **Tags**: `v0.1.0-teacher-flatwalk`, `v0.2.0-teacher-curriculum`, `v0.3.0-student`,
  `v0.4.0-mjx-eval`, `v1.0.0-deploy`. Tag after each phase lands.

### 2.3 Commit message format — Conventional Commits
```
<type>(<scope>): <imperative, lowercase, no period, <= 72 chars>

<optional body — what and why, not how. wrap at 72 cols.>

<optional footer: refs, breaking changes>
```
**Types we'll use:** `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`.
**Scopes we'll use:** `env`, `terrain`, `reward`, `teacher`, `student`, `ppo`, `distill`,
`randomization`, `eval`, `deploy`, `repo` (for root-level stuff).

Examples:
- `chore(repo): initialize python project with uv and ruff`
- `feat(env): add proprioceptive observation mode to go2 wrapper`
- `fix(reward): correct foot-slip penalty sign`
- `docs(readme): add phase 1 training instructions`

**Commit cadence:** commit when a single logical thing works and tests (if any) pass.
Not per-file. Not per-hour. Per *idea*.

### 2.4 Repo layout (final target)
```
rl-locomotion/
├── pyproject.toml
├── README.md
├── BUILD_ROADMAP.md          # this file
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── src/rl_locomotion/
│   ├── __init__.py
│   ├── env/
│   ├── terrain/
│   ├── reward/
│   ├── policy/
│   │   ├── teacher/
│   │   └── student/
│   ├── training/
│   │   ├── ppo/
│   │   └── distill/
│   ├── randomization/
│   ├── eval/
│   │   └── reality_gap/
│   └── deploy/
├── scripts/                   # entrypoint CLI scripts: train_teacher.py, etc.
├── configs/                   # yaml hyperparameter files
├── tests/
└── assets/                    # URDFs, meshes (git-lfs if large)
```

### 2.5 Locked tech decisions
| Decision              | Choice                          | Why                                     |
|-----------------------|---------------------------------|-----------------------------------------|
| Robot                 | Unitree Go2                     | URDF freely distributed, Isaac Lab ex.  |
| Simulator (train)     | NVIDIA Isaac Lab                | 4096 parallel envs on one GPU           |
| Simulator (eval)      | MuJoCo MJX                      | Different physics → real reality gap    |
| PPO impl              | `rsl_rl`                        | Battle-tested, ETH's own, minimal code  |
| Student encoder       | 1D TCN (fallback: GRU)          | Spec says either; TCN trains faster     |
| Export format         | ONNX opset 17                   | CUDA EP supports it, future-proof       |
| Deploy runtime        | ONNX Runtime C++ (CUDA EP)      | Good enough; TensorRT later if needed   |
| Logging               | Weights & Biases                | Free academic tier, spec calls for it   |

---

## 3. Build order (15 steps across 4 phases)

Each step has: **Goal**, **Branch**, **Deliverables**, **Commit plan**, **Definition of done**, **Status**.

---

### Phase 0 — Foundations

#### Step 1 — Repository & environment bootstrap
- **Goal**: Empty but correct repo on GitHub, working Python env, pre-commit hooks live.
- **Branch**: work directly on `main` for the *very first* commit, then create
  `chore/repo-bootstrap` for everything after the first commit.
- **Deliverables**:
  - GitHub repo `rl-locomotion` (public, MIT license)
  - `pyproject.toml` with project metadata (no deps yet beyond dev tools)
  - `.gitignore` (Python + IDE + W&B + checkpoints)
  - `.pre-commit-config.yaml` (ruff + basic hooks)
  - `README.md` with 3-paragraph project pitch
  - `src/rl_locomotion/__init__.py` with `__version__ = "0.0.0"`
- **Commit plan**:
  1. `chore(repo): initial commit with readme and license` (on main)
  2. `chore(repo): add python project config with uv and ruff`
  3. `chore(repo): add pre-commit hooks`
  4. `docs(readme): add project pitch and build phases`
- **Definition of done**: `pre-commit run --all-files` passes. `python -c "import rl_locomotion"` works.
- **Status**: [✅]
- **Note**: folded ruff config into commit 1 instead of separate commit.

#### Step 2 — Isaac Lab installation & smoke test
- **Goal**: Isaac Lab installed, a built-in example runs headless with 16 parallel envs.
- **Branch**: `chore/isaac-lab-install`
- **Deliverables**: installation notes in `docs/setup.md`, a recorded terminal output showing the Isaac Lab cartpole example running.
- **Commit plan**: single commit `docs(setup): add isaac lab installation notes`.
- **Definition of done**: `./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py --headless` runs without error on your machine.
- **Status**: [ ]

#### Step 3 — Project package skeleton
- **Goal**: All the module directories from §2.4 exist as real Python packages with docstring-only `__init__.py` files; `scripts/hello.py` imports from the package and runs.
- **Branch**: `feat/repo-skeleton`
- **Commit plan**: one commit per subpackage is overkill — group into 2–3 commits by layer (core modules, training, eval+deploy).
- **Definition of done**: `pytest tests/test_imports.py` passes (a single test that imports every submodule).
- **Status**: [ ]

---

### Phase 1 — Teacher training

#### Step 4 — Minimal Go2 env wrapper (flat terrain, no observations split yet)
- **Goal**: A `GoTask` class that wraps Isaac Lab's Go2 example into a Gym-style interface. Flat terrain only. Observations = concat everything. Resetting and stepping work.
- **Branch**: `feat/env-go2-flat`
- **Definition of done**: a `scripts/roll_random_policy.py` script drives 16 envs with uniform-random actions for 100 steps and prints reward.
- **Status**: [ ]

#### Step 5 — Reward shaping module (spec §2.3)
- **Goal**: `src/rl_locomotion/reward/` implements all 6 reward terms with per-term logging.
- **Branch**: `feat/reward-shaping`
- **Definition of done**: unit tests verify each term in isolation; integrated reward logs to W&B when called from the random-policy rollout.
- **Status**: [ ]

#### Step 6 — Teacher network + PPO wiring (flat terrain only)
- **Goal**: rsl_rl PPO trains a teacher MLP on flat terrain to walk forward at a commanded velocity. **This is the spec's explicit build-order advice — nail this before touching curriculum.**
- **Branch**: `feat/teacher-ppo-flatwalk`
- **Definition of done**: a video rollout at 50M env steps shows Go2 walking forward coherently on flat ground. Tag `v0.1.0-teacher-flatwalk`.
- **Status**: [ ]

#### Step 7 — Observation split (privileged vs proprioceptive)
- **Goal**: env wrapper now exposes the two obs modes per spec §2.1. Teacher uses privileged.
- **Branch**: `feat/env-obs-split`
- **Status**: [ ]

#### Step 8 — Terrain curriculum
- **Goal**: `terrain/` module generates tiers flat→slopes→stairs→stones→rough and promotes/demotes envs.
- **Branch**: `feat/terrain-curriculum`
- **Status**: [ ]

#### Step 9 — Domain randomization layer (14-dim)
- **Goal**: `randomization/` applies all 14 parameters at reset. Ranges per spec §4.
- **Branch**: `feat/randomization-14d`
- **Status**: [ ]

#### Step 10 — Full teacher training run
- **Goal**: 500M env steps, 4096 envs, curriculum + randomization active. Target >94% success on hardest tier.
- **Branch**: `feat/teacher-full-training` (mostly config changes + the long run itself)
- **Definition of done**: `teacher.pt` checkpoint, W&B report, rollout videos. Tag `v0.2.0-teacher-curriculum`.
- **Status**: [ ]

---

### Phase 2 — Distillation

#### Step 11 — Student architecture
- **Goal**: `policy/student/` implements TCN temporal encoder → latent → MLP head. Wires into the env's proprioceptive obs mode.
- **Branch**: `feat/student-architecture`
- **Status**: [ ]

#### Step 12 — DAgger distillation trainer
- **Goal**: `training/distill/` runs the DAgger loop with action MSE + latent regression loss. Student acts, teacher labels.
- **Branch**: `feat/distill-dagger`
- **Definition of done**: student retains ≥85% of teacher success rate across tiers. `student.pt` checkpoint. Tag `v0.3.0-student`.
- **Status**: [ ]

---

### Phase 3 — Eval & deployment

#### Step 13 — MuJoCo MJX held-out evaluator
- **Goal**: `eval/reality_gap/` loads the student in MJX with out-of-distribution physics and reports success per tier.
- **Branch**: `feat/eval-mjx`
- **Definition of done**: headline number ≥80% success on held-out physics. Tag `v0.4.0-mjx-eval`.
- **Status**: [ ]

#### Step 14 — ONNX export
- **Goal**: `scripts/export_onnx.py` dumps `student.onnx` (opset 17) and a Python round-trip test confirms numerical equivalence with the PyTorch model.
- **Branch**: `feat/onnx-export`
- **Status**: [ ]

#### Step 15 — C++ inference harness
- **Goal**: `deploy/` C++ program loads `student.onnx`, runs 100k inference calls, reports p50/p95/p99 latency. Target p99 < 0.6 ms.
- **Branch**: `feat/deploy-cpp-harness`
- **Definition of done**: benchmark output in README. Tag `v1.0.0-deploy`. Project is résumé-ready.
- **Status**: [ ]

#### Step 16 (optional, gold-medal) — Real Go2 deployment
- Contact the Freiburg robotics group. 30-second corridor walk video. Goes at the top of README.
- **Status**: [ ]

---

## 4. Decision log (append-only)

| Date       | Decision                                         | Rationale                                 |
|------------|--------------------------------------------------|-------------------------------------------|
| 2026-04-08 | Pin Python 3.14, not 3.10                         | Isaac Sim 5.x requires 3.11; Step 1 pin was wrong |
| 2026-04-08 | Windows 11 native Isaac Lab              | User chose despite Blackwell friction; accepts risk |
| ...        | ...                                              | ...                                       |

---

## 5. Open questions to resolve before they bite

- [ ] GPU availability: confirm RTX 4090-class GPU access for the 24h training run. If not, plan for a shorter run or cloud rental.
- [ ] Isaac Lab version pin: lock to a specific git SHA once Step 2 works.
- [ ] W&B account set up?
- [ ] Freiburg robotics group contact for real-robot bench time (optional).

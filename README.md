<div align="center">
  <p>
    <a href="https://github.com/telekinesis-ai">
      <img width="100%" src="assets/telekinesis_banner.png" />
    </a>
  </p>

  <p align="center">
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/v/telekinesis-ai" />
    </a>
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/pyversions/telekinesis-ai" />
    </a>
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/l/telekinesis-ai" />
    </a>
    <a href="https://docs.telekinesis.ai">
      <img src="https://img.shields.io/badge/docs-telekinesis.ai-blue" />
    </a>
  </p>

  <h2>Any robot. Any task. One Physical AI platform.</h2>

  <p>
    <a href="https://docs.telekinesis.ai/">Telekinesis Docs</a>
    &nbsp;•&nbsp;
    <a href="https://discord.gg/S5v8bYAnc6">Discord</a>
    &nbsp;•&nbsp;
    <a href="https://www.linkedin.com/company/telekinesis-ai/">LinkedIn</a>
    &nbsp;•&nbsp;
    <a href="https://x.com/telekinesis_ai">X</a>
    &nbsp;•&nbsp;
    <a href="https://telekinesis.ai/">Website</a>
  </p>
</div>

# RLBotics

PPO for robot learning, in PyTorch. Train a policy on a simulator, watch the curves in TensorBoard,
export it to a single ONNX file, and run it on the robot with numpy alone.

Three simulators are supported, all through the same runner: **Gymnasium** (works everywhere, good
for getting started), **mjlab** (MuJoCo Warp on the GPU), and **Isaac Lab** (Isaac Sim).

```python
env = GymnasiumVecEnv("Hopper-v5", num_envs=32, device="auto")
runner = OnPolicyRunner(env=env, runner_cfg=cfg, device="auto")
runner.learn(num_learning_iterations=1500)
path = runner.export()                       # one self-contained policy.onnx
action = Policy(path).get_action(observation) # deployment: numpy in, actions out
```

## Requirements
- [Python 3.10–3.12](https://www.python.org/downloads/)
- PyTorch with CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA toolkit
```
## Install

```bash
git clone -b develop git@gitlab.com:telekinesis/rlbotics.git
cd rlbotics
```

Choose any of the following installations:
| Extra | Install | What it adds | Runs on |
| --- | --- | --- | --- |
| *(none)* | `pip install -e .` | The library: PPO, runner, models, logging, ONNX export | anywhere |
| `gym` | `pip install -e ".[gym]"` | Gymnasium classic-control and MuJoCo tasks | macOS, Linux, Windows |
| `mjlab` | `pip install -e ".[mjlab]"` then `pip install "mjlab[cu128]"` | mjlab tasks on MuJoCo Warp | Linux/Windows + NVIDIA GPU (macOS: `mjlab[cpu]`, evaluation only) |
| `isaaclab` | `pip install -e ".[isaaclab]" --extra-index-url https://pypi.nvidia.com` | Isaac Lab tasks on Isaac Sim | Linux/Windows + NVIDIA GPU, Python 3.11 |
| `examples` | `pip install -e ".[examples]"` | `gym` plus onnxruntime, needed to run the exported policy | anywhere |
| `dev` | `pip install -e ".[dev]"` | ruff, pylint, pytest | anywhere |

The Isaac Sim wheels live on NVIDIA's index, so that extra only resolves with
`--extra-index-url https://pypi.nvidia.com`. There is no macOS build of Isaac Sim.

## Quickstart

```bash
pip install -e ".[examples]"
python examples/gymnasium_example.py     # Humanoid-v5, roughly 20 minutes
tensorboard --logdir logs                # watch it learn
```

That teaches the MuJoCo humanoid — 17 actuators, a 348-dimensional observation — to stand and walk,
exports the policy to ONNX, then runs the exported file. Measured at the defaults, **mean reward goes
from 60 to 622 and mean episode length from 13 to 129 steps over 600 iterations**. Running is a
millions-of-steps problem, so the default 3000 iterations is a start, and `--resume` continues from
the newest checkpoint:

```bash
python examples/gymnasium_example.py --resume
```

Want a result in two minutes instead? Pendulum solves quickly, and the same script does it:

```bash
python examples/gymnasium_example.py -e Pendulum-v1 -n 8 -s 128 -i 120 \
    --learning-rate 3e-4 --epochs 10 --entropy-coef 0.01
```

Measured: mean reward from about **-1080 to -211**, a solved swing-up. Watch
`Episodes/mean_length` first on the humanoid — staying upright is what the return is built on — and
`Diagnostics/kl` with `Diagnostics/clip_fraction` to check the updates are sane.

## Examples

| Example | What it does | Needs |
| --- | --- | --- |
| [gymnasium_example.py](examples/gymnasium_example.py) | Train, export and deploy on any Gymnasium continuous control task | `[examples]` |
| [mjlab_example.py](examples/mjlab_example.py) | The same, on an mjlab task | `[mjlab]` |
| [isaaclab_example.py](examples/isaaclab_example.py) | The same, on an Isaac Lab task | `[isaaclab]` |
| [module_examples/](examples/module_examples/) | One library piece at a time: configs, MLP, CNN, distributions, normalization, rollout buffer, logger | *(none)* |
| [run_all_examples.py](examples/run_all_examples.py) | Runs every example as a smoke test; reports SKIP when an optional dependency is missing | *(none)* |

Every training example takes `--help`, and shares the same arguments: `-i` iterations, `-n`
environments, `-s` steps per environment, `-d` device, `--resume`, `--record-video`,
`--experiment-name`, `--log-dir`.

## Environments

### Gymnasium

Any continuous control task works from its id alone — the observation size, actuator count, action
bounds and episode limit are all read from the environment:

```bash
python examples/gymnasium_example.py -e Ant-v5 -n 32 -s 64 -i 2000 --epochs 5
```

| Task | Observation | Actions | Action range | Notes |
| --- | --- | --- | --- | --- |
| `Pendulum-v1` | 3 | 1 | ±2.0 | Quickest check; solved in ~120 iterations |
| `MountainCarContinuous-v0` | 2 | 1 | ±1.0 | Sparse reward, needs exploration |
| `InvertedPendulum-v5` | 4 | 1 | ±3.0 | Easiest MuJoCo task |
| `InvertedDoublePendulum-v5` | 9 | 1 | ±1.0 | |
| `Reacher-v5` | 10 | 2 | ±1.0 | 50-step episodes |
| `Swimmer-v5` | 8 | 2 | ±1.0 | |
| `Hopper-v5` | 11 | 3 | ±1.0 | Locomotion, terminates on falling |
| `Walker2d-v5` | 17 | 6 | ±1.0 | |
| `HalfCheetah-v5` | 17 | 6 | ±1.0 | Never terminates |
| `Pusher-v5` | 23 | 7 | ±2.0 | Manipulation |
| `Ant-v5` | 105 | 8 | ±1.0 | Use wider networks |
| `Humanoid-v5` | 348 | 17 | ±0.4 | **The default.** Hardest; millions of steps to run |

The MuJoCo tasks come with `[gym]`. Networks default to `(256, 256, 128)` above 100 observation
dimensions and `(64, 64)` below, so `Ant-v5` and `Humanoid-v5` get the capacity they need without
being told. The defaults elsewhere (learning rate 1e-4, 2 epochs, entropy 0.0011) suit a
many-actuator task; smaller ones want a larger rate, more epochs and more entropy, as in the recipes
at the top of [gymnasium_example.py](examples/gymnasium_example.py).

Discrete tasks (`CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`) are **rejected with a clear error**:
the policy is Gaussian, so it needs a continuous action space. Box2D tasks
(`LunarLanderContinuous-v3`, `BipedalWalker-v3`) would work, but `gymnasium[box2d]` is not in the
extra — `pip install "gymnasium[box2d]"` to use them.

### mjlab

GPU-parallel MuJoCo with Isaac Lab's manager-based API. Tasks publish an `actor` group and a
privileged `critic` group, which the example wires up automatically. All 12 registered tasks:

| Task | What it is |
| --- | --- |
| `Mjlab-Velocity-Flat-Unitree-G1` | Humanoid velocity tracking on flat ground (the default) |
| `Mjlab-Velocity-Rough-Unitree-G1` | The same on rough terrain |
| `Mjlab-Velocity-Flat-Unitree-Go1` | Quadruped velocity tracking, flat |
| `Mjlab-Velocity-Rough-Unitree-Go1` | Quadruped velocity tracking, rough |
| `Mjlab-Tracking-Flat-Unitree-G1` | Humanoid motion imitation |
| `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` | The same without state estimation |
| `Mjlab-Cartpole-Balance` | Cartpole balance — cheap enough to smoke-test on CPU |
| `Mjlab-Cartpole-Swingup` | Cartpole swing-up |
| `Mjlab-Lift-Cube-Yam` | Arm lifting a cube |
| `Mjlab-Lift-Cube-Yam-Rgb` | The same from RGB images |
| `Mjlab-Lift-Cube-Yam-Depth` | The same from depth images |
| `Mjlab-Multi-Cube-Seg-Yam` | Multi-cube manipulation with segmentation |

```bash
python examples/mjlab_example.py --list-tasks
python examples/mjlab_example.py -t Mjlab-Cartpole-Balance -n 8 -s 16 -i 3 -d cpu   # CPU smoke run
python examples/mjlab_example.py -t Mjlab-Velocity-Flat-Unitree-G1 -n 4096 -i 3000  # needs a GPU
```

mjlab task configs ship `num_envs=1`, so always pass `-n`. The vision tasks publish a `camera` group
the example ignores, since it trains an MLP.

### Isaac Lab

Isaac Sim, with the largest task library of the three. `--list-tasks` prints the authoritative list
for your install; a representative selection:

| Category | Tasks |
| --- | --- |
| Classic | `Isaac-Cartpole-v0`, `Isaac-Ant-v0`, `Isaac-Humanoid-v0` (each also `-Direct-v0`) |
| Quadruped locomotion | `Isaac-Velocity-Flat-Anymal-C-v0`, `Isaac-Velocity-Rough-Anymal-C-v0`, and the same for Anymal B/D, Unitree A1/Go1/Go2, Spot |
| Humanoid locomotion | `Isaac-Velocity-Flat-H1-v0`, `Isaac-Velocity-Flat-G1-v0`, Digit variants |
| Manipulation | `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0`, `Isaac-Stack-Cube-Franka-v0`, `Isaac-Open-Drawer-Franka-v0`, `Isaac-Repose-Cube-Allegro-v0` |

```bash
python examples/isaaclab_example.py --list-tasks
python examples/isaaclab_example.py -t Isaac-Cartpole-v0 -n 64 -i 50
python examples/isaaclab_example.py -t Isaac-Velocity-Flat-Anymal-C-v0 -n 4096 -i 1500
```

Isaac Lab tasks publish a `policy` group and, when asymmetric, a `critic` group. Isaac Sim is
launched by the adapter before the task is built, so nothing from `isaaclab_tasks` may be imported
before that.

### Your own environment

Nothing about the library is tied to a simulator. Implement
[`VecEnv`](src/telekinesis/rlbotics/envs/base.py) — observations as a `TensorDict` of named groups,
rewards and dones shaped `(num_envs,)`, truncations under `extras["time_outs"]` — and the runner
trains against it. The three adapters in [envs/](src/telekinesis/rlbotics/envs/) are each about 200
lines and are the reference.

## Training runs

Everything a run produces goes in one directory, `logs/<experiment>/<timestamp>/`:

```
logs/gymnasium_ppo/2026-08-06_18-08-47/
├── config.json        the full config the run used
├── events.out.*       TensorBoard scalars
├── model_100.pt       checkpoints, rotated to keep the last N
├── model_100.mp4      with --record-video, the rollout that produced that checkpoint
├── model_best.pt      the highest mean reward of the run, never rotated away
└── policy.onnx        the exported policy, which is the best checkpoint
```

`model_best.pt` is rewritten whenever the mean episode reward beats every earlier iteration's, and it
is checked every iteration rather than on the `save_interval` grid, so a peak that the reward later
falls back from is not lost. It records what it scored and when, which is how `--resume best`
compares runs.

`--record-video` writes an MP4 beside every checkpoint, rendered from that iteration's own rollout —
no second simulator, one render per step on the iterations that checkpoint. A reward curve says a
policy improved; the clip says whether it is walking or shuffling on one knee. The environment has to
be built to render, which the examples do when the flag is set, and rotation deletes a video with the
checkpoint it belongs to.

`--resume` continues from an earlier checkpoint and writes a fresh timestamped directory, so a
resumed run never overwrites the one it continued from. It takes one value, and searches every run of
the experiment:

| | |
| --- | --- |
| `--resume` | the most recently written checkpoint (same as `--resume last`) |
| `--resume best` | the highest-scoring checkpoint of the experiment, which may be in an older run |
| `--resume model_500.pt` | that file, found among the experiment's runs |
| `--resume path/to/model.pt` | exactly that file, no searching |

The best score travels inside the checkpoint, so a run that continues from a strong policy and then
does worse cannot demote it. Alongside the usual reward
and loss curves, `Diagnostics/kl`, `Diagnostics/clip_fraction` and `Diagnostics/explained_variance`
are what tell you whether an update was sane — a reward that climbs then collapses is almost always
KL far above target.

## Symmetry

A legged robot is left-right symmetric, so a policy that has learned to trot leading with one leg has
in principle learned the mirrored gait too. Telling PPO about that is worth real sample efficiency and
is what stops a policy settling into a limp: mirrored samples are appended to every mini-batch, and an
optional term penalizes the policy for disagreeing with itself on them. This follows
[Mittal et al., ICRA 2024](https://arxiv.org/abs/2403.04359), and the flow matches rsl_rl's.

It is off by default and configured in Python rather than from the command line, because the one piece
it needs cannot be shipped: a **mirror function** saying which observation entry mirrors which, which
only somebody who knows the robot's joint order can write.

```python
from telekinesis.rlbotics.config import PPOConfig, SymmetryConfig

PPOConfig(
    ...,
    symmetry_cfg=SymmetryConfig(
        # The function, or an import path to it so the config survives config.json
        data_augmentation_func="my_robot.symmetry:mirror",
        use_data_augmentation=True,   # mirrored samples in every mini-batch
        use_mirror_loss=True,         # the auxiliary term
        mirror_loss_coeff=1.0,
    ),
)
```

The function is called as `func(env=env, obs=obs, actions=actions)` — the environment is passed in, so
it can read the observation layout — and returns each argument as the originals stacked with their
mirrored copies along the batch dimension. Either argument may be `None`. Setting both flags to `False`
still computes and reports the loss, detached, which is a cheap way to watch how symmetric a policy is
without changing what it optimizes. `Diagnostics`-style reporting puts it in the loss table as
`symmetry`.

### Writing one

For a manager-based environment (mjlab, Isaac Lab) the layout is metadata you can read, not something
to guess:

```python
env.venv.observation_manager.active_terms        # term names per group
env.venv.observation_manager.group_obs_term_dim  # their dimensions, so you get each term's offsets
env.venv.scene["robot"].joint_names              # carries the side and the axis
```

For the Unitree G1's 99-dimensional actor group that yields `base_lin_vel[0:3]`,
`base_ang_vel[3:6]`, `projected_gravity[6:9]`, `joint_pos[9:38]`, `joint_vel[38:67]`,
`actions[67:96]`, `command[96:99]`. Each term then follows from what kind of quantity it is:

| Term | Under a left-right mirror |
| --- | --- |
| `base_lin_vel`, `projected_gravity` | polar vectors: `(+, −, +)`, the lateral component flips |
| `base_ang_vel` | pseudovector: `(−, +, −)`, the in-plane components flip, the normal one does not |
| `command` (vx, vy, yaw rate) | `(+, −, −)` |
| `joint_pos`, `joint_vel`, `actions` | swap left↔right joints, negate those whose axis flips |
| per-foot terms (height, air time, contact) | swap the two feet |
| contact forces | swap the feet, negate the lateral component |

The joint part is derivable rather than typed: names like `left_hip_roll_joint` give both the partner
(`left_`↔`right_`) and the sign (negate when the axis is `roll` or `yaw`). For the G1 that produces 26
swapped and 16 negated entries across 29 joints.

### Verifying one

**Do not trust a mirror you have not measured.** A wrong one trains happily and teaches an invariance
the robot does not have — nothing about the run looks wrong. Three checks, cheapest first:

1. **Involution**: `mirror(mirror(x)) == x`. Catches a bad permutation or a stray sign at once.
2. **Commutes with the dynamics**: mirror the simulation state as well, step both with mirrored
   actions, and confirm the next observation mirrors and the reward is unchanged. This is the real
   test. Note that observation noise and randomizing reset events have to be off for it to mean
   anything, and that rough-terrain tasks are *not* mirror-symmetric in their dynamics even when the
   observation map is right.
3. **During training**: the reported `symmetry` loss should start O(1) with a fresh policy and fall.
   Pinned at zero from the first iteration means your mirror is the identity somewhere.

Measured this way over 200 random states, three Gymnasium tasks have exact-enough maps — a sign flip
per entry, no permutation:

| Task | Observation error | Reward error |
| --- | --- | --- |
| `Pendulum-v1`, obs `(+, −, −)`, action `(−)` | exact | exact |
| `InvertedDoublePendulum-v5` | 1e-06 | 1e-10 |
| `InvertedPendulum-v5` | 4e-03, solver noise: it grows with derivative order | exact |

`Swimmer-v5` was measured too and is **not** symmetric under a sign flip (0.4 on observations, 1.0 on
the reward), which is a good illustration of why the measurement matters — it looks like it should be.
Humanoid and the walkers are left-right symmetric, but their mirror is a joint permutation over body
blocks (`cinert`, `cvel`, `cfrc_ext`), not a sign flip.

## Deployment

`runner.export()` writes **one self-contained `policy.onnx`** holding the run's **best** checkpoint,
not whichever policy training happened to end on — those differ whenever the reward peaked and fell
back, which for a long locomotion run is the normal case. Inside the graph are the observation
normalization the policy trained with, the deterministic action, and, for environments with action
bounds, the scaling onto them. Running it needs neither PyTorch nor this library:

```python
from telekinesis.rlbotics.policy import Policy

policy = Policy("logs/gymnasium_ppo/2026-08-06_18-08-47/policy.onnx")
action = policy.get_action(observation)   # (obs_dim,) -> (num_actions,), or batched
```

Only numpy and onnxruntime. The batch dimension is dynamic, so one observation or a batch both work.
`runner.export(from_best=False)` ships the final policy instead, for comparing where training ended
up against the best it found.

## Verification status

| Path | State |
| --- | --- |
| Library: PPO, runner, models, config, logging, checkpoints, video, ONNX export | 295 tests |
| Gymnasium adapter and example | Trained and deployed on Pendulum, Hopper, Reacher, MountainCarContinuous, Humanoid |
| mjlab adapter and example | Trained and deployed on CPU (`Mjlab-Cartpole-Balance`, `Mjlab-Cartpole-Swingup`) and on GPU across the velocity-tracking tasks |
| Isaac Lab adapter and example | Trained and deployed on GPU across the classic, quadruped and humanoid locomotion tasks |
| CUDA | Verified on an NVIDIA GPU across all three adapters |

Hyperparameters in `gymnasium_example.py` are measured. Those in the mjlab and Isaac Lab examples are
the usual starting points for legged locomotion, not measured recipes.

## Development

```bash
pip install -e ".[dev,examples]"
pytest                              # 295 tests
ruff check .
python examples/run_all_examples.py # smoke-test every example
```

## License

Apache 2.0, see [LICENSE](LICENSE). Copyright Telekinesis GmbH.

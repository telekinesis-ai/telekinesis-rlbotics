<div align="center">
  <p>
    <a align="center" href="https://telekinesis.ai/" target="_blank">
      <img
        width="100%"
        src="https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/Telekinesis+Banner.png"
      >
    </a>
  </p>

  <p align="center">
    <a href="https://pypi.org/project/telekinesis-rlbotics/">
      <img src="https://img.shields.io/pypi/v/telekinesis-rlbotics" />
    </a>
    <a href="https://pypi.org/project/telekinesis-rlbotics/">
      <img src="https://img.shields.io/pypi/pyversions/telekinesis-rlbotics" />
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%202.0-blue" />
    </a>
    <a href="https://docs.telekinesis.ai/skills/rlbotics/overview.html">
      <img src="https://img.shields.io/badge/docs-telekinesis.ai-blue" />
    </a>
  </p>


  <p>
    <a href="https://docs.telekinesis.ai/skills/rlbotics/overview.html">Docs</a>
    &nbsp;•&nbsp;
    <a href="https://github.com/telekinesis-ai/telekinesis-rlbotics">GitHub</a>
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

PPO for robot learning, in PyTorch. Train a policy on Gymnasium, mjlab or Isaac Lab through the same
runner, watch the curves in TensorBoard, export it to a single ONNX file, and run it on the robot with
numpy alone. Open source under [Apache 2.0](LICENSE).

```python
env = GymnasiumVecEnv("Hopper-v5", num_envs=32, device="auto")
runner = OnPolicyRunner(env=env, runner_cfg=cfg, device="auto")
runner.learn(num_learning_iterations=1500)
path = runner.export()                       # one self-contained policy.onnx
action = Policy(path).get_action(observation) # deployment: numpy in, actions out
```

## Requirements

- [Python 3.10–3.12](https://www.python.org/downloads/)
- PyTorch. To match a specific CUDA toolkit, install it first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Install

Bare minimum — enough to train against a config, export to ONNX, and run the module examples:

```bash
pip install telekinesis-rlbotics
```

Gymnasium (macOS, Linux, Windows):

```bash
pip install "telekinesis-rlbotics[gym]"
```

mjlab (Linux/Windows + NVIDIA GPU, macOS CPU-only for evaluation):

```bash
pip install "telekinesis-rlbotics[mjlab]"
pip install "mjlab[cu128]"  # or "mjlab[cpu]" for evaluation only on macOS
```

Isaac Lab (Linux/Windows + NVIDIA GPU, Python 3.11):

```bash
pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url https://pypi.nvidia.com
```

Development (ruff, pylint, pytest):

```bash
pip install "telekinesis-rlbotics[dev]"
```

## Quickstart

One command per simulator, each training a real task from a shipped configuration. See
[Configurations](#configurations) for the rest of the task list, and
[Training parameters](#training-parameters) for what else the command line takes.

Gymnasium — a MuJoCo humanoid, on any OS:

```bash
pip install "telekinesis-rlbotics[gym]"
python examples/training_example.py configs/gymnasium/Humanoid-v5.yaml
```

mjlab — a Unitree G1 on MuJoCo Warp, needs an NVIDIA GPU:

```bash
pip install "telekinesis-rlbotics[mjlab]" && pip install "mjlab[cu128]"
python examples/training_example.py configs/mjlab/Mjlab-Velocity-Flat-Unitree-G1.yaml
```

Isaac Lab — an Anymal C on Isaac Sim, needs an NVIDIA GPU:

```bash
pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url https://pypi.nvidia.com
python examples/training_example.py configs/isaaclab/Isaac-Velocity-Flat-Anymal-C-v0.yaml
```

Watch any of them learn with `tensorboard --logdir logs`. No simulator installed yet? Every piece of
the library — MLP/CNN models, distributions, the config classes, the logger — has a standalone demo
under [examples/module_examples/](examples/module_examples/) that needs nothing beyond the base
install.

## Training parameters

`training_example.py` takes exactly one required argument, the configuration path; everything else is
optional and named after the field it overrides, so an unset one leaves the file alone:

| Option | Overrides | What it does |
| --- | --- | --- |
| `-n, --num-envs` | `env.num_envs` | Parallel environments |
| `-d, --device` | `env.device` | Training device: `auto`, `cpu`, `mps`, or `cuda` |
| `-i, --num-learning-iterations` | `runner.num_learning_iterations` | Iterations to train for |
| `--log-dir` | `runner.logger.log_dir` | Where logs, checkpoints and videos go — useful for a throwaway test run |
| `--resume [WHICH]` | `runner.logger.resume` | Continue from `last`, `best`, or a checkpoint path |

```bash
python examples/training_example.py configs/gymnasium/Pendulum-v1.yaml \
    --num-envs 8 --num-learning-iterations 120 --device cpu --log-dir /tmp/rlbotics
```

The task and the simulator (`env.id`, `env.framework`) have no command-line option on purpose: they
are what the configuration is for, so training something else means pointing at a different file.
Every option and its default is also in `python examples/training_example.py --help`.

## Configurations

One YAML file per task, grouped by simulator, under [configs/](configs/) — there are more of them
than the Quickstart commands above show:

```
configs/
├── example.yaml                    every field commented, for reference (not a real task)
├── gymnasium/                      12 tasks — Pendulum-v1.yaml, Humanoid-v5.yaml, ...
├── mjlab/                          12 tasks — Mjlab-Cartpole-Balance.yaml, Mjlab-Velocity-Flat-Unitree-G1.yaml, ...
└── isaaclab/                       9 tasks  — Isaac-Cartpole-v0.yaml, Isaac-Velocity-Flat-Anymal-C-v0.yaml, ...
```

Start from [configs/example.yaml](configs/example.yaml) when writing one of your own — every field is
commented with what it does and what values it accepts. A real file describes a whole run: `env:`
names the simulator, the task, how many copies of it to step and what to step them on; everything
under `runner:` mirrors [`OnPolicyRunnerConfig`](src/telekinesis/rlbotics/config.py) field for field.

```yaml
env:
  framework: gymnasium
  id: Pendulum-v1
  num_envs: 8
  device: auto

runner:
  num_learning_iterations: 120
  num_steps_per_env: 128
  obs_groups:
    actor: [observation]
    critic: [observation]
  logger:
    experiment: gymnasium_ppo
    save_interval: 100
    log_video: true
  algorithm:
    class_name: PPO
    learning_rate: 3.0e-4
  actor:
    class_name: MLPModel
    hidden_dims: [64, 64]
    distribution_cfg:
      class_name: GaussianDistribution
  critic:
    class_name: MLPModel
```

The `class_name` entries are what makes a piece swappable: a custom algorithm or model is named
there, either by a registered name or as an import path such as `my_pkg.algorithms:MyPPO`. There is
no loader to learn — the `runner` block goes straight to the config classes, which do the validating:

```python
cfg = yaml.safe_load(path.read_text())
runner_cfg = OnPolicyRunnerConfig.from_dict(cfg["runner"])
```

### Gymnasium

Any continuous control task works — observation size, actuator count, action bounds and episode limit
are read from the environment. All 12 configured:

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
| `Humanoid-v5` | 348 | 17 | ±0.4 | Hardest; millions of steps to run |

Measured: `Humanoid-v5` goes from mean reward 60 to 622 over 600 iterations, and `Pendulum-v1` from
about -1080 to -211, a solved swing-up, in two minutes. `Hopper-v5`, `Reacher-v5` and
`MountainCarContinuous-v0` are trained and deployed; the rest are a reasonable starting point rather
than a measured result. Discrete tasks (`CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`) are **rejected
with a clear error**, since the policy is Gaussian and needs a continuous action space. Box2D tasks
(`LunarLanderContinuous-v3`, `BipedalWalker-v3`) work, but need `pip install "gymnasium[box2d]"`.

### mjlab

GPU-parallel MuJoCo with Isaac Lab's manager-based API. Tasks publish an `actor` group and a
privileged `critic` group, which each config in [configs/mjlab/](configs/mjlab/) names directly. All
12 configured:

| Task | What it is |
| --- | --- |
| `Mjlab-Velocity-Flat-Unitree-G1` | Humanoid velocity tracking on flat ground |
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

mjlab's own task configs ship `num_envs=1`; each file in [configs/mjlab/](configs/mjlab/) sets its own
instead. The vision tasks publish a `camera` group that `training_example.py` ignores, since it
trains an MLP. The authoritative list for your install:

```bash
python -c "from telekinesis.rlbotics.envs.mjlab_env import registered_tasks as r; print(r())"
```

### Isaac Lab

Isaac Sim, with the largest task library of the three. Tasks publish a `policy` group and, when
asymmetric, a `critic` group. A representative selection of what's configured:

| Category | Tasks |
| --- | --- |
| Classic | `Isaac-Cartpole-v0`, `Isaac-Ant-v0`, `Isaac-Humanoid-v0` |
| Quadruped locomotion | `Isaac-Velocity-Flat-Anymal-C-v0`, `Isaac-Velocity-Rough-Anymal-C-v0` |
| Humanoid locomotion | `Isaac-Velocity-Flat-H1-v0`, `Isaac-Velocity-Flat-G1-v0` |
| Manipulation | `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0` |

The adapter launches Isaac Sim before the task is built, so nothing from `isaaclab_tasks` may be
imported before that. The authoritative list of every registered task for your install:

```bash
python -c "from telekinesis.rlbotics.envs.isaaclab_env import registered_tasks as r; print(r())"
```

### Your own environment

Implement [`VecEnv`](src/telekinesis/rlbotics/envs/base.py) — observations as a `TensorDict` of named
groups, rewards and dones shaped `(num_envs,)`, truncations under `extras["time_outs"]` — and the
runner trains against it. The three adapters in [envs/](src/telekinesis/rlbotics/envs/) are the
reference.

## Symmetry

A legged robot is left-right symmetric, so a policy that has learned to trot leading with one leg has
in principle learned the mirrored gait too. Mirrored samples are appended to every mini-batch, and an
optional term penalizes the policy for disagreeing with itself on them. Follows
[Mittal et al., ICRA 2024](https://arxiv.org/abs/2403.04359).

Off by default, and configured in Python because the one piece it needs cannot be shipped: a **mirror
function** saying which observation entry mirrors which, which only somebody who knows the robot's
joint order can write.

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

It is called as `func(env=env, obs=obs, actions=actions)` and returns each argument as the originals
stacked with their mirrored copies along the batch dimension; either may be `None`. With both flags
`False` the loss is still computed and reported, detached — a cheap way to watch how symmetric a
policy is without changing what it optimizes.

For a manager-based environment the layout is readable metadata: the observation manager's
`active_terms` and `group_obs_term_dim` give each term's offsets, and `scene["robot"].joint_names`
carries both the partner (`left_`↔`right_`) and the sign. Vectors flip their lateral component;
pseudovectors like `base_ang_vel` flip the in-plane ones.

**Do not trust a mirror you have not measured** — a wrong one trains happily and teaches an invariance
the robot does not have. Check `mirror(mirror(x)) == x`, then that it commutes with the dynamics:
mirror the simulation state, step both with mirrored actions, and confirm the next observation mirrors
and the reward is unchanged, with observation noise and randomizing resets off. Rough terrain is *not*
mirror-symmetric in its dynamics even when the observation map is right. In training the `symmetry`
loss should start O(1) and fall; pinned at zero means the mirror is the identity somewhere. Measured
over 200 random states, `Pendulum-v1` and `InvertedDoublePendulum-v5` mirror exactly under a per-entry
sign flip while `Swimmer-v5` does not — which is why measuring matters.

## Training runs

Everything a run produces goes in one directory, `logs/<experiment>/<timestamp>/`:

```
logs/gymnasium_ppo/2026-08-06_18-08-47/
├── config.json        the full config the run used
├── events.out.*       TensorBoard scalars
├── model_100.pt       checkpoints, rotated to keep the last N
├── model_100.mp4      with logger.log_video, the rollout that produced that checkpoint
├── model_best.pt      the highest mean reward of the run, never rotated away
└── policy.onnx        the exported policy, which is the best checkpoint
```

- `model_best.pt` is rewritten whenever the mean episode reward beats every earlier iteration's, and
  is checked every iteration rather than on the `save_interval` grid, so a peak the reward later falls
  back from is not lost. The score travels inside the file, so a resumed run that does worse cannot
  demote it.
- `logger.log_video` writes an MP4 beside every checkpoint, rendered from that iteration's own rollout
  — one render per step, no second simulator. Rotation deletes a video with its checkpoint.
- `--resume` writes a fresh timestamped directory, so it never overwrites the run it continued from,
  and searches every run of the experiment:

| | |
| --- | --- |
| `--resume` | the most recently written checkpoint (same as `--resume last`) |
| `--resume best` | the highest-scoring checkpoint of the experiment, which may be in an older run |
| `--resume model_500.pt` | that file, found among the experiment's runs |
| `--resume path/to/model.pt` | exactly that file, no searching |

`Diagnostics/kl`, `Diagnostics/clip_fraction` and `Diagnostics/explained_variance` are what tell you
whether an update was sane — a reward that climbs then collapses is almost always KL far above target.
On a humanoid watch `Episodes/mean_length` first, since staying upright is what the return is built on.

## Deployment

`runner.export()` writes **one self-contained `policy.onnx`** holding the run's **best** checkpoint,
not whichever policy training happened to end on — those differ whenever the reward peaked and fell
back. The graph carries the observation normalization the policy trained with, the deterministic
action, and the scaling onto the environment's action bounds where it has them.

```python
from telekinesis.rlbotics.policy import Policy

policy = Policy("logs/gymnasium_ppo/2026-08-06_18-08-47/policy.onnx")
action = policy.get_action(observation)   # (obs_dim,) -> (num_actions,), or batched
```

Only numpy and onnxruntime, with a dynamic batch dimension. `runner.export(from_best=False)` ships the
final policy instead, for comparing where training ended up against the best it found.

## Configuring in Python

YAML is how the examples train, but a configuration is only ever an `OnPolicyRunnerConfig` — build one
directly when values are computed rather than fixed, such as a network sized from the observation, or
a sweep over learning rates:

```python
from telekinesis.rlbotics.config import (
    GaussianDistributionConfig, LoggerConfig, MLPConfig, OnPolicyRunnerConfig, PPOConfig,
)

def make_runner_config(obs_dim: int, log_dir: str) -> OnPolicyRunnerConfig:
    hidden_dims = (256, 256, 128) if obs_dim >= 100 else (64, 64)
    distribution_cfg = GaussianDistributionConfig(init_std=1.0, std_range=(0.2, 1.5))
    actor = MLPConfig(hidden_dims=hidden_dims, obs_normalization=True, distribution_cfg=distribution_cfg)
    critic = MLPConfig(hidden_dims=hidden_dims, obs_normalization=True)  # no distribution: one value out
    return OnPolicyRunnerConfig(
        obs_groups={"actor": ["observation"], "critic": ["observation"]},
        num_learning_iterations=3000,
        logger=LoggerConfig(log_dir=log_dir, log_video=True),
        algorithm=PPOConfig(learning_rate=3e-4),
        actor=actor,
        critic=critic,
    )
```

[examples/configuration_example.py](examples/configuration_example.py) is the runnable version of
this — it sizes the network from a real environment's observation, trains, exports, and writes the
resulting configuration back out as YAML so the same run is reproducible without the Python:

```bash
python examples/configuration_example.py
python examples/training_example.py <the file it just wrote>
```

## Documentation

The full reference — every config field, the algorithm internals, the environment adapters — is at
[docs.telekinesis.ai](https://docs.telekinesis.ai/skills/rlbotics/overview.html). This README covers
getting a policy trained and deployed; the docs are where to go for anything this doesn't answer.

## Citation

If RLBotics contributes to your research, please cite it:

```bibtex
@software{telekinesis_rlbotics,
  author = {Telekinesis GmbH},
  title  = {RLBotics: PPO for robot learning in PyTorch},
  year   = {2026},
  url    = {https://github.com/telekinesis-ai/telekinesis-rlbotics},
  note   = {Apache-2.0}
}
```

The symmetry extension follows [Mittal et al., ICRA 2024](https://arxiv.org/abs/2403.04359), which is
worth citing alongside it if you use that feature.

## Verification status

| Path | State |
| --- | --- |
| Library: PPO, runner, models, config, logging, checkpoints, video, ONNX export | 410 tests |
| Gymnasium adapter, via `training_example.py` | Trained and deployed on Pendulum, Hopper, Reacher, MountainCarContinuous, Humanoid |
| mjlab adapter, via `training_example.py` | Trained and deployed on CPU (`Mjlab-Cartpole-Balance`, `Mjlab-Cartpole-Swingup`) and on GPU across the velocity-tracking tasks |
| Isaac Lab adapter, via `training_example.py` | Trained and deployed on GPU across the classic, quadruped and humanoid locomotion tasks |
| CUDA | Verified on an NVIDIA GPU across all three adapters |

The Gymnasium configurations are measured where noted above; the mjlab and Isaac Lab ones are the
usual starting points for legged locomotion, not measured recipes.

## Development

```bash
git clone https://github.com/telekinesis-ai/telekinesis-rlbotics.git
cd telekinesis-rlbotics
pip install -e ".[dev,gym]"         # gym, so run_all_examples.py can smoke-test training_example.py
pytest                              # 410 tests
ruff check .
python examples/run_all_examples.py # smoke-test every example, including module_examples/
```

Issues and pull requests are welcome on
[GitHub](https://github.com/telekinesis-ai/telekinesis-rlbotics).

## License

Open source under [Apache 2.0](LICENSE). Copyright Telekinesis GmbH.

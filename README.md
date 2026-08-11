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
RLbotics is a GPU-accelerated, lightweight reinforcement learning library for robot learning. Its unified design supports training across multiple simulation environments, including Gymnasium, mjlab, and Isaac Lab, while keeping the training pipeline consistent across simulators. RLbotics supports multi-GPU training and provides common tools for robot learning, from experiment tracking and checkpointing to policy deployment.

Open source under [Apache 2.0](LICENSE).

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

```bash
pip install telekinesis-rlbotics
```

The base install already includes what the examples need to run — ONNX export and inference, video
recording, YAML configs. Only a simulator is an extra, since Gymnasium, mjlab and Isaac Lab are heavy
and most runs only need one:

| Extra | Install | What it adds | Runs on |
| --- | --- | --- | --- |
| *(none)* | `pip install telekinesis-rlbotics` | The library: PPO, runner, models, logging, ONNX export/inference, video | anywhere |
| `gym` | `pip install "telekinesis-rlbotics[gym]"` | Gymnasium classic-control and MuJoCo tasks | macOS, Linux, Windows |
| `mjlab` | `pip install "telekinesis-rlbotics[mjlab]"` then `pip install "mjlab[cu128]"` | mjlab tasks on MuJoCo Warp | Linux/Windows + NVIDIA GPU (macOS: `mjlab[cpu]`, evaluation only) |
| `isaaclab` | `pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url https://pypi.nvidia.com` | Isaac Lab tasks on Isaac Sim | Linux/Windows + NVIDIA GPU, Python 3.11 |
| `dev` | `pip install "telekinesis-rlbotics[dev]"` | ruff, pylint, pytest | anywhere |

Isaac Sim's wheels are on NVIDIA's index, so `[isaaclab]` only resolves with the `--extra-index-url`
above. There is no macOS build of Isaac Sim.

## Quickstart

Train a MuJoCo humanoid to walk in 20 minutes:

```bash
git clone https://github.com/telekinesis-ai/telekinesis-rlbotics.git
cd telekinesis-rlbotics
pip install -e ".[examples]"
python examples/training_example.py configs/gymnasium/Humanoid-v5.yaml
tensorboard --logdir logs                # watch it learn
```

Train a pendulum to swing up in 2 minutes:

```bash
python examples/training_example.py configs/gymnasium/Pendulum-v1.yaml
```

A configuration describes a whole run, so the path is the only argument needed. See
[Configuration](#configuration) to tune one.

## Examples

| Example | What it does | Needs |
| --- | --- | --- |
| [training_example.py](examples/training_example.py) | Train, export and deploy on any simulator, from a YAML config that names which one | `[examples]`, `[mjlab]` or `[isaaclab]` |
| [configuration_example.py](examples/configuration_example.py) | The same run with the config built in Python, then written back out as YAML | `[examples]` |
| [module_examples/](examples/module_examples/) | One library piece at a time: configs, MLP, CNN, distributions, normalization, rollout buffer, logger | *(none)* |
| [run_all_examples.py](examples/run_all_examples.py) | Runs every example as a smoke test; reports SKIP when an optional dependency is missing | *(none)* |
| [configs/](configs/) | One YAML per task, grouped by simulator, each describing a whole run | *(none)* |

Every training example takes `--help`.

## Configuration

One YAML per task, named after the task and grouped by simulator, in [configs/](configs/):

```
configs/
├── gymnasium/Pendulum-v1.yaml
├── mjlab/Mjlab-Velocity-Flat-Unitree-G1.yaml
└── isaaclab/Isaac-Velocity-Flat-Anymal-C-v0.yaml
```

A file describes a whole run. `env:` names the simulator, the task, how many copies of it to step and
what to step them on; everything under `runner:` mirrors
[`OnPolicyRunnerConfig`](src/telekinesis/rlbotics/config.py) field for field. `env.framework` is what
[training_example.py](examples/training_example.py) dispatches on, so one script covers all three
simulators:

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
    save_interval: 25
    log_video: false
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

To tune a task, edit its file — the training scripts hold no hyperparameters to contradict it. The
`class_name` entries are what makes a piece swappable: a custom algorithm or model is named there,
either by a registered name or as an import path such as `my_pkg.algorithms:MyPPO`. There is no loader
to learn, since the `runner` block goes straight to the config classes, which do the validating:

```python
cfg = yaml.safe_load(path.read_text())
runner_cfg = OnPolicyRunnerConfig.from_dict(cfg["runner"])
```

Command-line options only vary a run without editing the file, and each is named after the field it
overrides:

| Option | Overrides |
| --- | --- |
| `-n/--num-envs` | `env.num_envs` |
| `-d/--device` | `env.device` |
| `-i/--num-learning-iterations` | `runner.num_learning_iterations` |
| `--resume` | `runner.logger.resume` |

An unset option changes nothing. `env.framework` and `env.id` have no options on purpose: they are
what a configuration is for, so pointing at another file is how you train something else. To build
the same configuration in Python and write it back out as YAML, see
[configuration_example.py](examples/configuration_example.py).

## Environments

### Gymnasium

Any continuous control task works — observation size, actuator count, action bounds and episode limit
are all read from the environment, and the rest from the configuration:

```bash
python examples/training_example.py configs/gymnasium/Ant-v5.yaml
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
| `Humanoid-v5` | 348 | 17 | ±0.4 | Hardest; millions of steps to run |

- The MuJoCo tasks come with `[gym]`, and each of the twelve above has a config file, whose header
  links the task's Gymnasium documentation.
- Measured: `Humanoid-v5` goes from mean reward 60 to 622 over 600 iterations, and `Pendulum-v1` from
  about -1080 to -211, a solved swing-up, in two minutes. `Hopper-v5`, `Reacher-v5` and
  `MountainCarContinuous-v0` are trained and deployed. The rest are a reasonable starting point rather
  than a measured result.
- The many-actuator tasks use `(256, 256, 128)` networks with a low learning rate and few epochs; the
  small ones use `(64, 64)` with a higher rate, more epochs and more entropy. Compare
  [Humanoid-v5.yaml](configs/gymnasium/Humanoid-v5.yaml) against
  [Pendulum-v1.yaml](configs/gymnasium/Pendulum-v1.yaml).
- Discrete tasks (`CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`) are **rejected with a clear error**:
  the policy is Gaussian, so it needs a continuous action space.
- Box2D tasks (`LunarLanderContinuous-v3`, `BipedalWalker-v3`) work, but need
  `pip install "gymnasium[box2d]"`.

### mjlab

GPU-parallel MuJoCo with Isaac Lab's manager-based API. Tasks publish an `actor` group and a
privileged `critic` group, which each config in [configs/mjlab/](configs/mjlab/) names directly. All
12 registered tasks:

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
python examples/training_example.py configs/mjlab/Mjlab-Cartpole-Balance.yaml -n 8 -i 3 -d cpu
python examples/training_example.py configs/mjlab/Mjlab-Velocity-Flat-Unitree-G1.yaml  # needs a GPU
python -c "from telekinesis.rlbotics.envs.mjlab_env import registered_tasks as r; print(r())"
```

mjlab's own task configs ship `num_envs=1`; each file in [configs/mjlab/](configs/mjlab/) sets its own
instead. The vision tasks publish a `camera` group that training_example.py ignores, since it trains
an MLP.

### Isaac Lab

Isaac Sim, with the largest task library of the three. The call below prints the authoritative list
for your install; a representative selection:

| Category | Tasks |
| --- | --- |
| Classic | `Isaac-Cartpole-v0`, `Isaac-Ant-v0`, `Isaac-Humanoid-v0` (each also `-Direct-v0`) |
| Quadruped locomotion | `Isaac-Velocity-Flat-Anymal-C-v0`, `Isaac-Velocity-Rough-Anymal-C-v0`, and the same for Anymal B/D, Unitree A1/Go1/Go2, Spot |
| Humanoid locomotion | `Isaac-Velocity-Flat-H1-v0`, `Isaac-Velocity-Flat-G1-v0`, Digit variants |
| Manipulation | `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0`, `Isaac-Stack-Cube-Franka-v0`, `Isaac-Open-Drawer-Franka-v0`, `Isaac-Repose-Cube-Allegro-v0` |

```bash
python examples/training_example.py configs/isaaclab/Isaac-Cartpole-v0.yaml
python examples/training_example.py configs/isaaclab/Isaac-Velocity-Flat-Anymal-C-v0.yaml
python -c "from telekinesis.rlbotics.envs.isaaclab_env import registered_tasks as r; print(r())"
```

Tasks publish a `policy` group and, when asymmetric, a `critic` group. The adapter launches Isaac Sim
before the task is built, so nothing from `isaaclab_tasks` may be imported before that.

### Your own environment

Implement [`VecEnv`](src/telekinesis/rlbotics/envs/base.py) — observations as a `TensorDict` of named
groups, rewards and dones shaped `(num_envs,)`, truncations under `extras["time_outs"]` — and the
runner trains against it. The three adapters in [envs/](src/telekinesis/rlbotics/envs/) are the
reference.

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
- `logger.log_video` writes an MP4 beside every checkpoint, rendered from that iteration's own rollout —
  one render per step, no second simulator. Rotation deletes a video with its checkpoint.
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
pip install -e ".[dev,examples]"
pytest                              # 410 tests
ruff check .
python examples/run_all_examples.py # smoke-test every example
```

Issues and pull requests are welcome on
[GitHub](https://github.com/telekinesis-ai/telekinesis-rlbotics).

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

## License

Open source under [Apache 2.0](LICENSE). Copyright Telekinesis GmbH.

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

# Telekinesis - RLbotics

Telekinesis RLbotics is a lightweight, GPU-accelerated PyTorch library for Reinforcement Learning. It supports multi-environment training across Gymnasium, mjlab, and Isaac Lab, common learning algorithms, ONNX export, and deployment with NumPy alone.

Open source under [Apache 2.0](LICENSE). 

Full documentation: [Telekinesis Agentic OS: RLbotics](https://docs.telekinesis.ai/skills/rlbotics/overview.html).

## Requirements

- [Python 3.10–3.12](https://www.python.org/downloads/)
- PyTorch. To match a specific CUDA toolkit, install it first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Quickstart

### Gymnasium - (requirements: any OS, no GPU)

```bash
pip install "telekinesis-rlbotics[gym]"
python examples/training_example.py configs/gymnasium/Humanoid-v5.yaml
```

### mjlab - (requirements: Linux/Windows, NVIDIA GPU)

```bash
pip install "telekinesis-rlbotics[mjlab]" "mjlab[cu128]"
python examples/training_example.py configs/mjlab/Mjlab-Velocity-Flat-Unitree-G1.yaml
```

### Isaac Lab - (requirements: Linux/Windows, NVIDIA GPU, Python 3.11)

```bash
pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url https://pypi.nvidia.com
python examples/training_example.py configs/isaaclab/Isaac-Velocity-Flat-Anymal-C-v0.yaml
```

`tensorboard --logdir logs` to watch any of them learn. More tasks in `configs/<framework>/`.

## Options

| Option | What it does |
| --- | --- |
| `-n, --num-envs` | Parallel environments |
| `-d, --device` | `auto`, `cpu`, `mps`, or `cuda` |
| `-i, --num-learning-iterations` | Iterations to train for |
| `--log-dir` | Where logs, checkpoints and videos go |
| `--resume [WHICH]` | Continue from `last`, `best`, or a checkpoint path |

`python examples/training_example.py --help` for details.

## Deployment

Training exports the best checkpoint to one self-contained `policy.onnx`. Run it with just numpy and
onnxruntime:

```python
from telekinesis.rlbotics.policy import Policy

policy = Policy("logs/gymnasium_ppo/2026-08-06_18-08-47/policy.onnx")
action = policy.get_action(observation)   # (obs_dim,) -> (num_actions,)
```

## Documentation
Find the documentation for RLBotics at: [Telekinesis Agentic OS: RLbotics](https://docs.telekinesis.ai/skills/rlbotics/overview.html)

## Citation

```bibtex
@software{telekinesis_rlbotics,
  author = {Telekinesis GmbH},
  title  = {Telekinesis-Rlbotics: Reinforcement Learning for Robotics},
  year   = {2026},
  url    = {https://github.com/telekinesis-ai/telekinesis-rlbotics},
  note   = {Apache-2.0}
}
```

# MuJoCo LightWorld

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight, educational implementation of World Model assisted Reinforcement Learning (PPO) using Gymnasium MuJoCo environments. This project demonstrates how latent representation learning and auxiliary rewards can enhance RL agent performance.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (Recommended for fast setup)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mujoco-lightworld.git
   cd mujoco-lightworld
   ```

2. **Initialize environment with uv:**
   ```bash
   uv sync
   ```
   This will automatically create a virtual environment and install all dependencies (including PyTorch, Gymnasium, and MuJoCo).

### Usage

Run training experiments directly using `uv run`:

```bash
# Experiment 1: Baseline PPO
uv run lightworld-train --task Walker2d-v4 --use_wm false

# Experiment 2: PPO + World Model Features
uv run lightworld-train --task Walker2d-v4 --use_wm true --z_dim 32

# Experiment 3: PPO + WM Features + Auxiliary Reward
uv run lightworld-train --task Walker2d-v4 --use_wm true --use_aux_reward true --alpha 0.01
```

Or execute the script directly (if activated):
```bash
python -m mujoco_lightworld.train --help
```

## 📂 Project Structure

The project follows a modern `src-layout`:

```text
mujoco-lightworld/
├── src/mujoco_lightworld/    # Main package
│   ├── rl/                   # PPO Agent implementation (Actor-Critic, Buffer)
│   ├── wm/                   # World Model components (Encoder, Dynamics)
│   └── train.py              # Training entry point
├── tests/                    # Unit tests
├── docs/                     # Detailed documentation
├── results/                  # Experiment outputs (logs, models)
├── pyproject.toml            # Project configuration & dependencies
└── uv.lock                   # Dependency lock file
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- **Architecture**: [Overview](docs/architecture/Overview.md), [Data Flow](docs/architecture/DataFlow.md)
- **Basics**: [MuJoCo Setup](docs/basics/MuJoCo_Setup.md), [RL Concepts](docs/basics/RL_Basics.md)
- **Code Analysis**: [Modules](docs/code/Modules.md), [Training Loop](docs/code/TrainingLoop.md)
- **Runbook**: [End-to-End Guide](docs/runbook/EndToEnd.md), [UV Workflow](docs/runbook/UV_Workflow.md)
- **Experiments**: Guides for [E1](docs/experiments/E1_PPO_vs_WM.md) to [E4](docs/experiments/E4_Latent_Visualization.md)

## 🧪 Testing

Run unit tests to verify the installation:

```bash
uv run pytest tests
```

## 🤝 Contributing

Please read [Contributing Guide](docs/contributing/Contributing.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

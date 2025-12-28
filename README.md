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

# Visualization: Record Video & Snapshot
uv run lightworld-train --task Walker2d-v4 --viz_mode headless --save_video true --video_interval 50 --save_snapshot true
```

Or execute the script directly (if activated):
```bash
python -m mujoco_lightworld.train --help
```

### Advanced Training Features (High Performance)

The latest version includes critical features for achieving high scores (Reward > 3000) in MuJoCo tasks:

1.  **Normalization (归一化)**:
    *   **Observation**: Automatically normalizes inputs to mean 0, std 1. Critical for neural network stability.
    *   **Reward**: Normalizes rewards to stabilize PPO updates.
    *   **Clipping**: Removes outliers to prevent gradient explosions.
    *   **Statistics Persistence**: Saves running mean/std to `obs_rms.pkl`. **[Read the "Glasses" Guide](docs/runbook/BestPractices.md#the-glasses-of-the-agent-observation-normalization)** to understand why this is crucial.

2.  **Training Management**:
    *   **Resume (断点续训)**: Add `--resume` to continue training from the latest checkpoint.
    *   **Best Model Saving**: Automatically saves the model with the highest historical reward to `results/models/best_model.pt`.
    *   **LR Decay**: Linearly decays learning rate to ensure convergence in later stages.

### Monitoring & Operations

**1. Check Training Status (Real-time):**
View the latest logs (Reward, Best Score, Learning Rate):
```powershell
Get-Content results/logs/training_status.log -Tail 5
```

**2. Interrupt & Resume:**
*   **Stop**: Press `Ctrl+C` or close the terminal safely.
*   **Resume**:
    ```bash
    uv run python -m mujoco_lightworld.train --task Walker2d-v4 --total_steps 5000000 --resume
    ```

**3. View Final Results:**
*   **Metrics**: Check `results/logs/ppo.csv`.
*   **Visuals**: Run the video recorder to generate MP4 files:
    ```bash
    uv run python paper_assets/record_video.py
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
- **Runbook**: [End-to-End Guide](docs/runbook/EndToEnd.md), [UV Workflow](docs/runbook/UV_Workflow.md), [Monitoring & Visualization](docs/runbook/Monitoring_and_Visualization.md)
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

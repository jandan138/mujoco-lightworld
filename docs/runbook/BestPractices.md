# 🏆 Best Practices & Engineering Guide

This document outlines critical engineering practices and "gotchas" encountered during the development of MuJoCo LightWorld. It serves as a guide for reproducing high-performance results.

## 👓 The "Glasses" of the Agent: Observation Normalization

One of the most common pitfalls in Reinforcement Learning (especially with MuJoCo) is the **Observation Normalization Mismatch**.

### The Problem
Raw sensor data from MuJoCo environments can have vastly different scales:
- **Joint Velocities**: Can be large (e.g., ±10.0).
- **Joint Angles**: Usually small (e.g., ±1.0).
- **Contact Forces**: Can be huge (e.g., 100+).

Neural networks struggle to learn when input features have different magnitudes. To fix this, we use `NormalizeObservation` wrapper during training, which maintains a running mean ($\mu$) and variance ($\sigma^2$) of the inputs.

**The "Glasses" Analogy:**
- **Training**: The agent wears a pair of "prescription glasses" (the running statistics) that makes the world look standardized (mean 0, std 1).
- **Inference (Testing)**: If you load the trained model (the "brain") but forget to load the statistics (the "glasses"), the agent sees a distorted world. It might think a small tilt is a huge fall, leading to erratic behavior and immediate failure.

### The Solution: `obs_rms.pkl`

We explicitly save the running statistics alongside the model weights.

#### 1. Saving (in `train.py`)
We extract the `obs_rms` object from the environment wrapper and pickle it:

```python
# Save obs_rms if available
obs_rms = env.get_wrapper_attr("obs_rms")
if obs_rms is not None:
    with open("results/models/obs_rms.pkl", "wb") as f:
        pickle.dump(obs_rms, f)
```

#### 2. Loading (in Visualization Scripts)
When generating videos or plots, we must re-apply the "glasses":

```python
# 1. Apply the wrapper
env = NormalizeObservation(env)

# 2. Load the statistics
with open("results/models/obs_rms.pkl", "rb") as f:
    obs_rms = pickle.load(f)

# 3. Inject statistics into the env
env.obs_rms = obs_rms
```

---

## 📉 Learning Rate Decay

For tasks like Walker2d-v4, the policy often oscillates near the optimal solution if the learning rate remains high.

- **Strategy**: Linear Decay.
- **Implementation**: Start at `3e-4` and linearly decrease to `0` over the total training steps.
- **Effect**: Allows the agent to make large updates early on and fine-tune its behavior in later stages without destabilizing the policy.

## 💾 Best Model Checkpointing

PPO is an on-policy algorithm, and performance can fluctuate. Saving only the latest checkpoint is risky.

- **Mechanism**: We track `mean_reward` after every epoch.
- **Action**: If `current_reward > best_reward`, we save a separate copy to `results/models/best_model.pt`.
- **Usage**: Always use `best_model.pt` for final evaluation and video generation.

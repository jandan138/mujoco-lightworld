import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import os
import numpy as np

# Reusing definitions from generate_gait_strip.py for consistency
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        return mu

class Encoder(nn.Module):
    def __init__(self, obs_dim, z_dim=32, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], z_dim),
        )

    def forward(self, obs):
        return self.net(obs)

def record_demo_video(task="Walker2d-v4", model_dir="results/models", output_dir="results/media/demo", duration_sec=10):
    print(f"Recording demo video for {task}...")
    
    # Setup Env with Video Recorder
    # Force rgb_array for headless recording
    env = gym.make(task, render_mode="rgb_array")

    # -----------------------------------------------------------
    from gymnasium.wrappers import NormalizeObservation, TransformObservation
    import pickle
    
    # Normalize observations to mean 0, std 1
    env = NormalizeObservation(env)
    
    # Load running mean/std from training if available
    rms_path = os.path.join(model_dir, "obs_rms.pkl")
    if os.path.exists(rms_path):
        print(f"Loading obs_rms from {rms_path}...")
        with open(rms_path, "rb") as f:
            obs_rms = pickle.load(f)
        # Iterate wrappers to find NormalizeObservation
        current_env = env
        while hasattr(current_env, "env"):
            if hasattr(current_env, "obs_rms"):
                current_env.obs_rms = obs_rms
                print("Loaded obs_rms statistics.")
                break
            current_env = current_env.env
    else:
        print(f"[WARN] obs_rms.pkl not found at {rms_path}. The policy might fail if input statistics mismatch.")

    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    # -----------------------------------------------------------

    env = RecordVideo(
        env, 
        video_folder=output_dir,
        name_prefix="demo_walker",
        episode_trigger=lambda x: True # Record all episodes
    )
    
    # Load Models
    actor_path = os.path.join(model_dir, "actor.pt")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Since we trained with Pure PPO (use_wm=False), input dim is obs_dim (17), not z_dim
    actor = Actor(obs_dim, act_dim)
    
    try:
        actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        actor.eval()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Run Loop
    obs, _ = env.reset(seed=100)
    done = False
    
    total_reward = 0
    step = 0
    
    while not done:
        with torch.no_grad():
            o_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            # Pure PPO: Direct input
            action = actor(o_tensor).squeeze(0).numpy()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}, Reward: {reward:.2f}")

    env.close()
    print(f"Video saved to {output_dir}")
    print(f"Episode finished. Steps: {step}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    record_demo_video()

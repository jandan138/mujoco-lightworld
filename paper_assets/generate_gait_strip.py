import gymnasium as gym
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from typing import Optional

# Define minimal Actor class to load the model
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        return mu  # Just return mean for visualization

# Define minimal Encoder class to load the WM
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

def generate_gait_strip(task="Walker2d-v4", output_path="paper_assets/gait_strip.png", num_frames=8, step_interval=3, model_dir="results/models", use_wm=True, z_dim=32):
    """
    Runs the environment with a trained model (and optional WM) and captures a filmstrip.
    """
    print(f"Generating gait strip for {task} (WM={use_wm})...")
    env = gym.make(task, render_mode="rgb_array")
    
    # -----------------------------------------------------------
    from gymnasium.wrappers import NormalizeObservation, TransformObservation
    import numpy as np
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

    actor_path = os.path.join(model_dir, "actor.pt")
    encoder_path = os.path.join(model_dir, "wm_encoder.pt")
    
    actor = None
    encoder = None
    
    # Check model existence
    if os.path.exists(actor_path):
        print(f"Loading actor from {actor_path}...")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # If using WM, Actor input is z_dim; else obs_dim
        actor_input_dim = z_dim if use_wm else obs_dim
        
        actor = Actor(actor_input_dim, act_dim)
        try:
            actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
            actor.eval()
        except Exception as e:
            print(f"Failed to load Actor: {e}")
            actor = None
            
        if use_wm and actor:
            if os.path.exists(encoder_path):
                print(f"Loading encoder from {encoder_path}...")
                encoder = Encoder(obs_dim, z_dim)
                try:
                    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
                    encoder.eval()
                except Exception as e:
                    print(f"Failed to load Encoder: {e}")
                    encoder = None
            else:
                print(f"Encoder not found at {encoder_path}, but use_wm=True. Cannot run inference.")
                actor = None # Disable actor if missing encoder
    else:
        print(f"Actor not found at {actor_path}. Using random actions.")
        
    obs, _ = env.reset(seed=42)
    
    # Helper to get action
    def get_action(o):
        if actor:
            with torch.no_grad():
                o_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0) # Add batch dim
                if use_wm and encoder:
                    z = encoder(o_tensor)
                    a = actor(z)
                else:
                    a = actor(o_tensor)
                return a.squeeze(0).numpy()
        else:
            return env.action_space.sample()

    # Run warmup
    for _ in range(30):
        action = get_action(obs)
        obs, _, done, _, _ = env.step(action)
        if done: obs, _ = env.reset()
        
    frames = []
    
    # Capture loop
    for i in range(num_frames):
        frame = env.render()
        frames.append(frame)
        
        for _ in range(step_interval):
            action = get_action(obs)
            obs, _, done, _, _ = env.step(action)
            if done: obs, _ = env.reset()
            
    env.close()
    
    # Stitch frames
    resized_frames = []
    for f in frames:
        h, w, _ = f.shape
        new_h = 240
        new_w = int(w * (new_h / h))
        rf = cv2.resize(f, (new_w, new_h))
        resized_frames.append(rf)
        
    if not resized_frames:
        print("No frames captured.")
        return

    filmstrip = np.hstack(resized_frames)
    filmstrip = cv2.copyMakeBorder(filmstrip, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    filmstrip_bgr = cv2.cvtColor(filmstrip, cv2.COLOR_RGB2BGR)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, filmstrip_bgr)
    print(f"Saved gait strip to {output_path}")

if __name__ == "__main__":
    # Since we trained with use_wm=False (Pure PPO), we must set use_wm=False here
    generate_gait_strip(use_wm=False)

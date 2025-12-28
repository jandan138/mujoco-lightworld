"""
Visualization utilities for MuJoCo environments.
Includes wrappers for video recording and periodic snapshot capturing.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


@dataclass
class VisualizationConfig:
    """Configuration for environment visualization and recording."""
    viz_mode: str = "headless"  # "headless" or "headed"
    save_video: bool = False
    save_snapshot: bool = False
    video_interval: int = 50      # Record video every N episodes
    snapshot_interval: int = 1000 # Save snapshot every N steps
    output_dir: str = "results/media"


class SnapshotWrapper(gym.Wrapper):
    """Wrapper to save environment snapshots (images) and metadata periodically."""
    
    def __init__(self, env: gym.Env, cfg: VisualizationConfig):
        super().__init__(env)
        self.cfg = cfg
        self.step_count = 0
        self.snapshot_dir = Path(cfg.output_dir) / "snapshots"
        self.metadata_dir = Path(cfg.output_dir) / "metadata"
        
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        if self.step_count % self.cfg.snapshot_interval == 0:
            self._save_snapshot(action, reward, info)
            
        return obs, reward, terminated, truncated, info

    def _save_snapshot(self, action, reward, info):
        """Capture render frame and save as PNG with metadata JSON."""
        try:
            # Render frame (force rgb_array for snapshot regardless of mode)
            # Note: The underlying env must support rgb_array. 
            # If render_mode is human, we might need to grab the window content, 
            # but gym usually allows switching or separate calls if configured.
            # Here we assume the env was initialized to support rgb_array access if needed.
            # For simplicity, if mode is human, this might fail or return None depending on backend.
            # A robust way is ensuring 'rgb_array' is available.
            
            # If current render_mode is human, grabbing frame might be tricky without closing window.
            # We skip snapshot if mode is strictly human and not configured for dual rendering.
            # But for headless, it works fine.
            frame = self.env.render()
            if frame is None and self.env.render_mode == "human":
                # In human mode, render() typically returns None. 
                # We skip snapshot to avoid interrupting the GUI loop or complexity.
                return

            if frame is not None:
                # Convert RGB to BGR for OpenCV
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                timestamp = int(time.time())
                filename = f"step_{self.step_count}_{timestamp}"
                
                # Save Image
                img_path = self.snapshot_dir / f"{filename}.png"
                cv2.imwrite(str(img_path), frame)
                
                # Save Metadata
                meta = {
                    "step": self.step_count,
                    "timestamp": timestamp,
                    "reward": float(reward),
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "info_keys": list(info.keys())
                }
                meta_path = self.metadata_dir / f"{filename}.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                    
        except Exception as e:
            print(f"[WARN] Snapshot failed at step {self.step_count}: {e}")


def make_visualized_env(
    task_id: str, 
    seed: int, 
    cfg: VisualizationConfig
) -> gym.Env:
    """Factory to create and wrap the environment based on visualization config."""
    
    # Determine render mode
    # For recording, we usually need rgb_array. 
    # If mode is 'headed', we use 'human'.
    # Note: 'human' mode usually doesn't support VideoRecorder seamlessly in all gym versions 
    # without window conflict. Priority:
    # 1. If save_video=True -> prefer rgb_array (headless) to ensure recording works.
    #    If user wants 'headed' AND 'video', gym might struggle. We prioritize recording if requested.
    # 2. If only 'headed' -> use human.
    
    render_mode = "rgb_array"
    if cfg.viz_mode == "headed" and not cfg.save_video:
        render_mode = "human"
    
    # Create env
    env = gym.make(task_id, render_mode=render_mode)
    
    # -----------------------------------------------------------
    # CRITICAL: Apply Observation and Reward Normalization
    # This is standard practice for achieving high scores (>3000)
    # in MuJoCo tasks like Walker2d.
    # -----------------------------------------------------------
    from gymnasium.wrappers import NormalizeObservation, NormalizeReward, TransformObservation, TransformReward
    import numpy as np
    
    # Normalize observations to mean 0, std 1
    env = NormalizeObservation(env)
    
    # Clip observations to [-10, 10] to remove outliers
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    
    # Normalize rewards (important for PPO stability)
    env = NormalizeReward(env)
    
    # Clip rewards to [-10, 10] (standard practice)
    env = TransformReward(env, lambda r: np.clip(r, -10, 10))
    # -----------------------------------------------------------

    env.reset(seed=seed)
    
    # 1. Video Recording Wrapper
    if cfg.save_video and render_mode == "rgb_array":
        video_folder = os.path.join(cfg.output_dir, "videos")
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda ep_id: ep_id % cfg.video_interval == 0,
            name_prefix=f"{task_id}-seed{seed}"
        )
    
    # 2. Snapshot Wrapper
    if cfg.save_snapshot:
        env = SnapshotWrapper(env, cfg)
        
    return env

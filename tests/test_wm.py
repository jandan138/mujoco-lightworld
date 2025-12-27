import torch
import pytest
from mujoco_lightworld.wm.encoder import Encoder
from mujoco_lightworld.wm.dynamics import Dynamics

def test_encoder_output_shape():
    """Test if encoder outputs correct latent dimension."""
    batch_size = 4
    obs_dim = 17  # Walker2d default
    z_dim = 32
    
    encoder = Encoder(obs_dim=obs_dim, z_dim=z_dim)
    obs = torch.randn(batch_size, obs_dim)
    z = encoder(obs)
    
    assert z.shape == (batch_size, z_dim)

def test_dynamics_prediction_shape():
    """Test if dynamics model predicts next latent state correctly."""
    batch_size = 4
    z_dim = 32
    
    dynamics = Dynamics(z_dim=z_dim)
    z_t = torch.randn(batch_size, z_dim)
    z_next_pred = dynamics(z_t)
    
    assert z_next_pred.shape == (batch_size, z_dim)

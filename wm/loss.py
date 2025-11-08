"""
世界模型损失与辅助奖励函数。

1) world_model_loss：
   - 计算基于潜空间的一步预测 MSE：Dynamics(Encoder(obs_t)) 与 Encoder(obs_{t+1}) 之间的误差。
   - 返回 (loss, metrics) 以便记录与打印。

2) prediction_error_reward：
   - 推理模式下计算一步预测误差，作为辅助奖励项（一般为惩罚，alpha 控制权重）。
   - 用于在 PPO 的环境奖励基础上添加额外项，提高训练信号质量。
"""

import torch
import torch.nn.functional as F


def world_model_loss(encoder, dynamics, obs_t: torch.Tensor, obs_tp1: torch.Tensor):
    """世界模型的 MSE 损失。

    参数：
    - encoder: 观测编码器，obs → z。
    - dynamics: 潜空间动力学模型，z_t → z_{t+1}。
    - obs_t/obs_tp1: 连续两步的原始观测张量，形状通常为 [B, obs_dim]。
    返回：
    - loss: 标量 MSE 损失。
    - metrics: 便于日志记录的字典（例如 {'wm_mse': float}）。
    """
    z_t = encoder(obs_t)
    z_pred = dynamics(z_t)
    z_tp1 = encoder(obs_tp1)
    loss = F.mse_loss(z_pred, z_tp1)
    return loss, {
        "wm_mse": loss.detach().item()
    }


@torch.no_grad()
def prediction_error_reward(wm: dict, obs_t: torch.Tensor, obs_tp1: torch.Tensor) -> float:
    """计算一步预测误差，作为辅助奖励（数值越大，误差越大）。

    - wm: 包含 {'encoder', 'dynamics'} 的组件字典。
    - obs_t/obs_tp1: 当前与下一步观测（单样本）。
    返回：标量误差，通常用于 `reward += alpha * err`。
    """
    encoder = wm["encoder"]; dynamics = wm["dynamics"]
    z_t = encoder(obs_t.unsqueeze(0))
    z_pred = dynamics(z_t)
    z_tp1 = encoder(obs_tp1.unsqueeze(0))
    err = F.mse_loss(z_pred, z_tp1, reduction="none").mean().item()
    return err
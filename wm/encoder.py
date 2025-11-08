"""
Encoder：将原始观测映射到潜空间特征 z。

设计：
- 轻量两层 MLP，激活使用 ReLU，输出维度为 z_dim。
- 该特征既用于 PPO 输入（Feature 模式），也用于 Dynamics 的监督目标（预测下一步 z）。
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """观测编码器：obs → z。"""

    def __init__(self, obs_dim: int, z_dim: int = 32, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], z_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """返回潜特征 z（形状 [B, z_dim]）。"""
        return self.net(obs)
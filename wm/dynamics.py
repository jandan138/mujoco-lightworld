"""
Dynamics：在潜空间 z 上进行一步预测（z_t → z_{t+1}）。

设计：
- 轻量两层 MLP；损失使用与 Encoder 输出 z 的 MSE。
- 在线训练：每步用 (obs_t, obs_{t+1}) 的 z 表示进行监督训练。
"""

import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """潜空间动力学模型：预测下一步潜特征。"""

    def __init__(self, z_dim: int = 32, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], z_dim),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """返回预测的 z_{t+1}（形状 [B, z_dim]）。"""
        return self.net(z_t)
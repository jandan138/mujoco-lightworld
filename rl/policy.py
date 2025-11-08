"""
PPO 策略网络与价值网络实现（Actor / Critic）。

设计原则：
- 使用两层 MLP + Tanh 激活，结构轻量但对 MuJoCo 连续控制任务足够稳定。
- Actor 输出动作分布的均值 mu 与对数标准差 log_std（可学习参数），采用正态分布。
- Critic 输出状态价值 V(s)。

张量约定：
- obs: 形状 [B, obs_dim] 的 float32 张量。
- action: 形状 [B, act_dim] 的 float32 张量。
- 所有前向计算在同一设备（CPU/GPU）上进行。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """策略网络（Actor）：输出高斯分布参数并进行重参数采样。

    - obs_dim: 观测维度，或使用 WM 特征时的潜特征维度。
    - act_dim: 动作维度（MuJoCo 环境通常为连续动作）。
    - hidden_sizes: 两层 MLP 的隐藏层大小，默认 (256, 256)。
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        # 主干网络：提取状态或特征的高阶表示
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh(),
        )
        # 输出动作均值；对数标准差作为可学习参数（每个维度一个）
        self.mu = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算：得到高斯分布的均值与对数标准差。

        返回：
        - mu: [B, act_dim]，动作分布均值。
        - log_std: [act_dim]，按维度共享的对数标准差（裁剪防止过大/过小）。
        """
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std.clamp(-5, 2)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于当前策略进行重参数采样，并返回动作与对数概率。

        - 使用 `rsample()` 支持梯度传播（对 PPO 的策略更新无影响，但保持一致风格）。
        返回：
        - action: 采样动作 [B, act_dim]
        - logp: 动作的对数概率 [B]
        - mu: 均值（便于调试或记录）
        """
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, mu


class Critic(nn.Module):
    """价值网络（Critic）：估计状态价值 V(s)。"""

    def __init__(self, obs_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """返回 V(s)，形状 [B]。"""
        return self.net(obs).squeeze(-1)
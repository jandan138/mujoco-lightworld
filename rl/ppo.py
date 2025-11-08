"""
PPO 训练器实现，支持轻量世界模型特征与辅助奖励的接入。

核心职责：
- 构建 Actor/Critic 与优化器；管理 RolloutBuffer；
- 每个 epoch 采样 `steps_per_epoch` 条数据，路径结束时计算 GAE 优势与回报；
- 使用剪切概率比（clip ratio）与 KL 早停策略进行参数更新；
- 可选：
  - 使用 Encoder 的潜特征作为 PPO 的输入（Feature 模式）。
  - 使用预测误差构造辅助奖励（Reward 模式），通过系数 alpha 控制权重。

数值稳定性：
- 对优势进行标准化；
- KL 早停避免策略更新幅度过大；
- 对环境动作进行裁剪，确保在动作空间范围内。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from rl.policy import Actor, Critic
from rl.buffer import RolloutBuffer, RolloutBufferConfig


@dataclass
class PPOConfig:
    """PPO 超参数配置。"""
    pi_lr: float = 3e-4
    vf_lr: float = 3e-4
    train_iters: int = 80
    clip_ratio: float = 0.2
    target_kl: float = 0.03
    steps_per_epoch: int = 4096


class PPOTrainer:
    """PPO 训练器：封装采样、更新与指标统计。"""

    def __init__(self, env, device: torch.device, log_dir: str, model_dir: str, cfg: Optional[PPOConfig] = None, input_dim: Optional[int] = None):
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.cfg = cfg or PPOConfig()

        # 若使用 WM 特征，输入维度为 z_dim；否则为原始观测维度
        obs_dim = input_dim or env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.pi_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.pi_lr)
        self.vf_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.vf_lr)

        self.buf = RolloutBuffer(RolloutBufferConfig(obs_dim, act_dim, self.cfg.steps_per_epoch), device)

    @torch.no_grad()
    def select_action(self, obs_t):
        """给定单步观测（或特征），采样动作并估计价值。"""
        action, logp, _ = self.actor.sample(obs_t)
        value = self.critic(obs_t)
        return action, logp, value

    def update(self, obs, act, logp_old, adv, ret):
        """执行 PPO 更新：
        - 重新计算当前策略下的 logp，与旧 logp 比较得到比率 ratio。
        - 剪切目标：min(ratio * adv, clip(ratio) * adv)，减少过大更新带来的不稳定。
        - 价值网络用 MSE 逼近回报。
        - 当近似 KL 超过阈值时提前停止，保护策略不被过度更新。
        """
        for _ in range(self.cfg.train_iters):
            mu, log_std = self.actor(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(act).sum(-1)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()

            v = self.critic(obs)
            v_loss = ((v - ret) ** 2).mean()

            self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
            self.vf_opt.zero_grad(); v_loss.backward(); self.vf_opt.step()

            approx_kl = (logp_old - logp).mean().item()
            if approx_kl > 1.5 * self.cfg.target_kl:
                break

    def train_epoch(self, use_wm=False, wm=None, use_aux_reward=False, alpha=0.01, wm_opt=None, wm_loss_fn=None):
        """执行一个训练周期（epoch）：
        - 采样 steps_per_epoch 条数据（可能跨多个 episode）。
        - 可选：
          * 使用 WM 的 Encoder 将 obs 映射为特征作为 Actor/Critic 的输入（Feature 模式）。
          * 使用 WM 预测误差作为辅助奖励（Reward 模式），权重为 alpha。
          * 在线训练 WM 的 Encoder/Dynamics（MSE 监督）。
        - 完成采样后进行 PPO 更新，并统计平均回报与平均 WM 损失。
        返回：字典 {"mean_reward": float, "wm_loss": float}
        """
        obs, _ = self.env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        ep_return = 0.0
        ep_returns = []
        wm_losses = []
        for t in range(self.cfg.steps_per_epoch):
            # Use world model feature as input if enabled
            if use_wm and wm is not None:
                obs_feat = wm["encoder"](obs_t.unsqueeze(0)).squeeze(0)
            else:
                obs_feat = obs_t
            act_t, logp_t, val_t = self.select_action(obs_feat.unsqueeze(0))
            act = act_t.squeeze(0).cpu().numpy()
            # Ensure action within bounds for MuJoCo
            if hasattr(self.env.action_space, "low") and hasattr(self.env.action_space, "high"):
                import numpy as np
                act = np.clip(act, self.env.action_space.low, self.env.action_space.high)
            next_obs, rew, terminated, truncated, _ = self.env.step(act)

            # Auxiliary reward from WM prediction error (if enabled)
            if use_wm and use_aux_reward and wm is not None:
                with torch.no_grad():
                    from wm.loss import prediction_error_reward
                    aux = prediction_error_reward(wm, obs_t, torch.tensor(next_obs, dtype=torch.float32, device=self.device))
                rew = float(rew + alpha * aux)

            self.buf.store(obs_feat, act_t.squeeze(0), torch.tensor(rew, device=self.device), val_t.squeeze(0), logp_t.squeeze(0))
            ep_return += float(rew)

            # Train world model online (feature prediction)
            if use_wm and wm is not None and wm_opt is not None and wm_loss_fn is not None:
                loss, m = wm_loss_fn(wm["encoder"], wm["dynamics"], obs_t.unsqueeze(0), torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                wm_opt.zero_grad(); loss.backward(); wm_opt.step()
                wm_losses.append(m.get("wm_mse", float(loss.detach().item())))

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            timeout = (t == self.cfg.steps_per_epoch - 1)
            terminal = terminated or truncated
            if terminal or timeout:
                if terminal:
                    last_val = 0.0
                else:
                    if use_wm and wm is not None:
                        next_feat = wm["encoder"](obs_t.unsqueeze(0))
                        last_val = self.critic(next_feat).item()
                    else:
                        last_val = self.critic(obs_t.unsqueeze(0)).item()
                self.buf.finish_path(last_val)
                if terminal:
                    ep_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = self.env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        obs, act, logp, adv, ret = self.buf.get()
        self.update(obs, act, logp, adv, ret)
        mean_ep_ret = sum(ep_returns) / max(1, len(ep_returns))
        mean_wm_loss = sum(wm_losses) / max(1, len(wm_losses))
        return {"mean_reward": mean_ep_ret, "wm_loss": mean_wm_loss}
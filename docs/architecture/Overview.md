# 整体框架概览

模块关系：
- 环境（Gymnasium MuJoCo）提供连续控制任务与奖励信号。
- PPO（Actor/Critic + Buffer + Trainer）负责策略优化与价值估计。
- 世界模型（Encoder + Dynamics）提供：
  - Feature：更好的状态表征作为 PPO 输入。
  - Reward：预测误差作为辅助奖励，增强训练信号。

目标：
- 提升样本效率、收敛速度与稳定性，同时保持实现轻量可复现。
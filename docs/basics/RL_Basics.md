# 强化学习与 PPO 基础

关键概念：
- 策略 `π(a|s)`：给定状态选择动作的分布。
- 价值 `V(s)`：从状态 s 开始的期望回报。
- 优势 `A(s,a)`：动作相对平均价值的好坏程度。

PPO（Proximal Policy Optimization）：
- 目标：稳定地提升策略性能，避免过大的更新幅度。
- 做法：用剪切比率 `clip(ratio)` 限制策略更新（`ratio = exp(logp_new - logp_old)`）。
- 辅助：使用 GAE-Lambda 估计优势，减少方差并保持较好偏差。

GAE-Lambda：
- 递推公式：`adv_t = δ_t + γλ adv_{t+1}`，其中 `δ_t = r_t + γ V(s_{t+1}) - V(s_t)`。
- 兼顾偏差与方差的折中，提升训练稳定性。
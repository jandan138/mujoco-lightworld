# 数据流与调用链

一次交互的主要数据流：
1. 读取当前观测 `obs_t`。
2. 若启用 WM Feature：`z_t = Encoder(obs_t)`，作为 PPO 的输入；否则使用原始 `obs_t`。
3. Actor 采样动作 `a_t` 并计算 `logp_t`；Critic 估计价值 `V(s_t)`。
4. 环境执行 `a_t`，得到 `next_obs, rew, done`。
5. 若启用 WM Reward：计算预测误差 `err` 并更新 `rew += alpha * err`。
6. 存入缓冲：`(input_feat, a_t, rew, V(s_t), logp_t)`。
7. 路径结束：计算 GAE 优势与回报；`get()` 返回训练批次。
8. PPO 更新：剪切比率、KL 早停、价值 MSE。
9. WM 在线训练：`MSE(Dynamics(z_t), Encoder(next_obs))`。
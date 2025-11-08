# 训练循环详解（train.py）

流程分解：
1. 解析参数、创建目录、保存 `run_config.json`。
2. 设置随机种子（Python/numpy/torch），创建环境并 `reset(seed)`。
3. 根据 `use_wm` 确定 PPO 输入维度（原始观测或 `z_dim`）。
4. 启用 WM 时，实例化 Encoder/Dynamics 与优化器；定义 `world_model_loss`。
5. 写入 CSV 表头（`step,mean_reward,wm_loss`）。
6. 计算 `total_epochs = total_steps // steps_per_epoch`。
7. 每个 epoch：
   - `ppo.train_epoch(...)` 采样与更新，返回平均回报与 WM 损失。
   - 追加 CSV 一行，定期打印进度。
8. 保存模型权重：`actor.pt`、`critic.pt`、`wm_encoder.pt`、`wm_dynamics.pt`。

日志命名规则：
- 纯 PPO：`results/logs/ppo.csv`
- +WM(Feature)：`results/logs/ppo_wm_feat.csv`
- +WM(Feature+Reward)：`results/logs/ppo_wm_feat_reward.csv`
# 源码模块职责映射

顶层：
- `train.py`：训练脚本，参数解析、训练循环、日志与模型保存。

RL 包：
- `rl/policy.py`：Actor/Critic 网络（两层 MLP）。
- `rl/buffer.py`：RolloutBuffer 与 GAE-Lambda 优势/回报计算。
- `rl/ppo.py`：PPOTrainer（采样、更新、KL 早停、WM 接入）。

WM 包：
- `wm/encoder.py`：观测→潜特征 `z` 的编码器。
- `wm/dynamics.py`：潜空间一步预测 `z_t → z_{t+1}`。
- `wm/loss.py`：世界模型 MSE 损失与预测误差辅助奖励。

配置与依赖：
- `config.yaml`：默认超参与路径。
- `requirements.txt`：依赖清单。
# E1：PPO vs PPO + World Model

目的：比较基础 PPO 与加入世界模型后的性能与样本效率。

配置与命令：
- 纯 PPO：`python train.py --task Walker2d-v4 --use_wm False`
- +WM(Feature)：`python train.py --task Walker2d-v4 --use_wm True --use_aux_reward False`
- +WM(Feature+Reward)：`python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.01`

输出：
- CSV：`results/logs/ppo*.csv`，包含 `step,mean_reward,wm_loss`
- 曲线：参考 `runbook/Plotting.md` 生成 Fig.2；表格可基于最终均值/方差整理。
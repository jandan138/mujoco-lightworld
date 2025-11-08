# E2：潜空间维度 z_dim 消融

目的：评估不同 `z_dim` 对性能与稳定性的影响（HalfCheetah）。

命令：
- `python train.py --task HalfCheetah-v4 --use_wm True --z_dim 16`
- `python train.py --task HalfCheetah-v4 --use_wm True --z_dim 32`
- `python train.py --task HalfCheetah-v4 --use_wm True --z_dim 64`

输出与可视化：
- 汇总最终平均奖励与标准差，绘制柱状图（Fig.3）。
- 参考 `runbook/Plotting.md` 或在 notebook 中绘制。
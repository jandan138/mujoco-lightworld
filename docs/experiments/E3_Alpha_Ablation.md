# E3：辅助奖励权重 α 消融

目的：分析不同 `alpha` 下辅助奖励对总体性能的影响（Walker2d）。

命令：
- `python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.0`
- `python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.01`
- `python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.05`

输出与结论：
- 记录最终平均回报与备注（过强惩罚可能降低性能）。
- 在 Fig.4 中展示不同 α 的效果对比。
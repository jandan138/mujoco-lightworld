# 跑通手册（E1–E4）

准备：
- 参考 `basics/MuJoCo_Setup.md` 安装并验证环境。

E1：PPO vs PPO + World Model（Walker2d）
- 纯 PPO：`python train.py --task Walker2d-v4 --use_wm False`
- +WM(Feature)：`python train.py --task Walker2d-v4 --use_wm True --use_aux_reward False`
- +WM(Feature+Reward)：`python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.01`

E2：z_dim 消融（HalfCheetah）
- `python train.py --task HalfCheetah-v4 --use_wm True --z_dim 16`
- `python train.py --task HalfCheetah-v4 --use_wm True --z_dim 32`
- `python train.py --task HalfCheetah-v4 --use_wm True --z_dim 64`

E3：alpha 消融（Walker2d）
- `python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.0`
- `python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.01`
- `python train.py --task Walker2d-v4 --use_wm True --use_aux_reward True --alpha 0.05`

E4：潜空间可视化
- 参见 `experiments/E4_Latent_Visualization.md`
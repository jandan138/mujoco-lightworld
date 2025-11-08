# 配置与参数说明

`config.yaml` 字段：
- `experiment.task/seed/total_steps/device`
- `ppo.pi_lr/vf_lr/train_iters/clip_ratio/target_kl/steps_per_epoch/gamma/lam`
- `wm.use_wm/use_aux_reward/alpha/z_dim`
- `paths.log_dir/model_dir/fig_dir`

命令行覆盖：
- 所有关键参数均可通过 `train.py` 的命令行参数覆盖，例如：
  - `python train.py --task Walker2d-v4 --use_wm True --z_dim 32 --alpha 0.01`

配置优先级：
- 命令行 > `config.yaml` 默认值。
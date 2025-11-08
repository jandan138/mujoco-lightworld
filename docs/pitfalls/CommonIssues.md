# 踩坑与排障手册

环境安装：
- MuJoCo 下载失败：检查网络或代理；重试或使用镜像。
- Windows GPU：更新 NVIDIA 驱动与 CUDA 运行库，确保与 PyTorch 版本匹配。

训练稳定性：
- 动作越界：已在代码中裁剪到 `action_space` 范围，若仍报错检查环境版本。
- 优势波动大：降低 `steps_per_epoch` 或增大 batch；检查 `gamma/lam` 设置。

速度/内存：
- OOM：降低 `steps_per_epoch`、网络规模（隐藏层或 z_dim），或使用 CPU 试跑。
- 训练慢：确认 GPU 已启用；减少 `total_steps` 做快速验证。

日志与绘图：
- CSV 未写入：检查 `results/logs` 目录权限与路径；确保 Windows 下没有中文路径问题。
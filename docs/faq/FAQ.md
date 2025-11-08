# 常见问答（FAQ）

Q：GPU 不可用怎么办？
- A：指定 `--device cpu` 并减少 `total_steps` 做快速验证；确认 CUDA 与驱动后再切回 GPU。

Q：日志没有生成？
- A：检查 `results/logs` 是否存在与有权限；Windows 中文路径可能导致权限问题，建议英文路径。

Q：如何复现实验图表？
- A：使用 `runbook/Plotting.md` 的代码片段，或将其改写成脚本统一生成 Fig.2–Fig.5。
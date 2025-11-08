# 贡献指南

代码风格：
- 保持模块职责清晰，不在训练器中混入可视化逻辑。
- 注释采用中文简洁说明，关键函数提供 docstring。

新功能建议：
- 例如增加 Replay-based WM 或更复杂的 Dynamics，请在 `wm/` 下新增模块并在 `train.py` 增加参数开关。

实验扩展：
- 新环境（如 Hopper/Ant），直接通过 `--task` 指定，必要时更新 `docs/experiments/`。

问题反馈：
- 在 `docs/pitfalls/` 中补充新问题与解决方案，维护 `changelog/CHANGELOG.md`。
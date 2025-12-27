# 使用 uv 管理与运行项目指南

本文档介绍如何使用现代化的 Python 包管理工具 [uv](https://github.com/astral-sh/uv) 来管理本项目环境并运行实验。

## 1. 为什么选择 uv？
本项目已迁移至 `pyproject.toml` 标准架构。相比传统的 `pip`，`uv` 提供了：
- **极速安装**：依赖解析与安装速度比 pip 快 10-100 倍。
- **自动环境管理**：无需手动创建/激活虚拟环境，`uv` 会自动处理。
- **严格的版本锁定**：通过 `uv.lock` 确保所有开发者使用完全一致的依赖树。

## 2. 安装 uv

推荐使用官方脚本安装（支持 Windows/macOS/Linux）：

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux
```bash
curl -lsSf https://astral.sh/uv/install.sh | sh
```

安装完成后，请重启终端或重新加载 Shell 配置文件。

## 3. 项目初始化与依赖同步

在项目根目录下，执行以下命令即可一键创建虚拟环境并安装所有依赖：

```bash
uv sync
```

该命令会：
1. 读取 `pyproject.toml`。
2. 生成或更新 `uv.lock` 锁定文件。
3. 自动创建 `.venv` 虚拟环境。
4. 安装所有依赖包。

## 4. 运行训练脚本

使用 `uv run` 命令可以在隔离的环境中直接执行脚本，无需手动激活环境：

### 基础运行
```bash
uv run train.py
```

### 带参数运行（覆盖默认配置）
例如，运行 Walker2d 任务并启用世界模型：

```bash
uv run train.py --task Walker2d-v4 --use_wm true --total_steps 500000
```

### 常用实验命令示例

**实验 1：纯 PPO 基线**
```bash
uv run train.py --task Walker2d-v4 --use_wm false
```

**实验 2：PPO + 世界模型特征 (Feature Mode)**
```bash
uv run train.py --task Walker2d-v4 --use_wm true --z_dim 64
```

**实验 3：PPO + WM 特征 + 辅助奖励 (Reward Mode)**
```bash
uv run train.py --task Walker2d-v4 --use_wm true --use_aux_reward true --alpha 0.05
```

## 5. 管理依赖

如果需要添加新包（例如 ipython），请使用：
```bash
uv add ipython
```
这会自动更新 `pyproject.toml` 和 `uv.lock`。

如果需要移除包：
```bash
uv remove some-package
```

## 6. 常见问题

**Q: 之前的 `requirements.txt` 还有用吗？**
A: 本项目已全面转向 `pyproject.toml`。`requirements.txt` 仅作为历史备份保留，建议忽略。请始终以 `pyproject.toml` 为准。

**Q: 我习惯用 `conda`，还能用吗？**
A: 可以，但建议尝试 `uv`。如果必须使用 conda，可以手动安装 `requirements.txt` 中的依赖，但可能会错过版本锁定的好处。

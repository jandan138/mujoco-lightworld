# 项目深度分析报告 (Project Analysis Report)

本文件是对 `mujoco-lightworld` 项目代码、架构与逻辑的全面分析总结。

## 1. 项目结构与模块划分

项目采用模块化设计，将强化学习算法 (RL)、世界模型 (World Model) 与实验基础设施分离。

### 目录结构说明

```text
mujoco-lightworld/
├── config.yaml          # [配置] 全局实验参数（环境、超参、路径）
├── train.py             # [入口] 训练脚本，负责组装各模块并执行主循环
├── requirements.txt     # [依赖] 项目依赖清单
├── rl/                  # [RL 模块] PPO 算法实现
│   ├── ppo.py           # PPOTrainer: 采样、更新、WM集成逻辑
│   ├── policy.py        # Actor/Critic 网络结构
│   └── buffer.py        # RolloutBuffer: 轨迹存储与 GAE 计算
├── wm/                  # [WM 模块] 世界模型组件
│   ├── encoder.py       # Encoder: 观测 -> 潜变量 (Obs -> Latent)
│   ├── dynamics.py      # Dynamics: 潜变量动力学 (z_t -> z_{t+1})
│   └── loss.py          # 损失函数与辅助奖励计算
├── docs/                # [文档] 详细的架构与使用说明
└── results/             # [产物] 自动生成的日志、模型与图表
```

## 2. 核心架构与数据流

项目采用了 **Actor-Critic (PPO)** 结合 **World Model (Latent Dynamics)** 的架构。

### 架构图 (Architecture Diagram)

```mermaid
graph TD
    subgraph Environment
        Env[Gymnasium MuJoCo]
    end

    subgraph World_Model
        Enc[Encoder]
        Dyn[Dynamics Model]
    end

    subgraph PPO_Agent
        Actor[Actor Policy]
        Critic[Critic Value]
        Buf[Rollout Buffer]
    end

    %% Data Flow
    Env -->|Obs| Enc
    Enc -->|Latent z| Actor
    Enc -->|Latent z| Critic
    
    Actor -->|Action| Env
    
    %% World Model Training flow
    Env -->|Next Obs| Enc
    Enc -->|z_t| Dyn
    Dyn -->|Pred z_{t+1}| LossCal[MSE Loss]
    Enc -->|z_{t+1}| LossCal
    
    %% PPO Update
    Env -->|Reward| Buf
    Enc -->|z| Buf
    Actor -->|Action| Buf
    Critic -->|Value| Buf
    
    %% Auxiliary Reward
    LossCal -.->|Prediction Error| AuxRew[Auxiliary Reward]
    AuxRew -.->|Add to| Buf
```

### 关键数据流 (Data Flow)

1.  **观测处理**: 环境产生原始观测 `obs`。
    *   **Feature 模式**: `obs` -> `Encoder` -> `z` (32维)。`z` 代替 `obs` 成为 Agent 的状态输入。
    *   **普通模式**: 直接使用 `obs`。
2.  **决策生成**: `z` (或 `obs`) -> `Actor` -> `Action`。
3.  **环境交互**: `Action` -> `Env` -> `Next Obs`, `Reward`。
4.  **世界模型预测**:
    *   `Dynamics` 根据当前 `z_t` 预测下一时刻 `pred_z_{t+1}`。
    *   `Encoder` 将 `Next Obs` 编码为真实 `z_{t+1}`。
5.  **辅助奖励 (Aux Reward)**:
    *   计算预测误差 `||pred_z_{t+1} - z_{t+1}||^2`。
    *   `Total Reward = Env Reward + alpha * Prediction Error` (用于鼓励探索或新颖性)。
6.  **学习更新**:
    *   **WM 更新**: 每步交互后，立即通过 MSE Loss 更新 Encoder 和 Dynamics。
    *   **PPO 更新**: 收集一定步数 (Epoch) 后，使用 Buffer 中的数据进行多次 PPO 迭代更新。

## 3. 核心代码逻辑审查

### 3.1 训练入口 (`train.py`)
*   **职责**: 参数解析、环境初始化、模块组装、主循环控制。
*   **关键点**:
    *   使用 `argparse` 允许命令行覆盖 `config.yaml`。
    *   `use_wm` 参数控制是否实例化 `wm` 组件。
    *   循环调用 `ppo.train_epoch()` 并记录日志到 CSV。

### 3.2 PPO 训练器 (`rl/ppo.py`)
*   **职责**: 执行具体的交互与训练逻辑。
*   **关键函数 `train_epoch`**:
    *   **采样循环**: 执行 `steps_per_epoch` 次 `env.step()`。
    *   **WM 集成**: 显式调用 `wm['encoder']` 转换状态，调用 `wm_loss_fn` 计算损失并反向传播 (Online Training)。
    *   **Buffer 管理**: 调用 `buf.store()` 存入数据，路径结束时调用 `buf.finish_path()` 计算 GAE。
*   **关键函数 `update`**:
    *   标准的 PPO Loss 实现：`min(surr1, surr2)`。
    *   包含 `clip_ratio` (0.2) 和 `target_kl` (0.03) 早停机制，保证训练稳定性。

### 3.3 世界模型 (`wm/`)
*   **Encoder (`wm/encoder.py`)**: 简单的 MLP (ObsDim -> 256 -> 256 -> 32)，将高维或原始观测映射到紧凑的潜空间。
*   **Dynamics (`wm/dynamics.py`)**: 潜空间动力学模型 (32 -> 256 -> 256 -> 32)，学习状态转移规律。
*   **Loss (`wm/loss.py`)**: 定义了 MSE 损失函数和辅助奖励计算逻辑。

## 4. 技术栈与设计评估

### 技术选型
*   **框架**: PyTorch (深度学习), Gymnasium (强化学习接口)。
*   **环境**: MuJoCo (高效、精确的连续控制物理引擎)。
*   **配置**: YAML (可读性好，易于管理实验配置)。

### 质量与扩展性
*   **优点**:
    *   **模块化**: RL 与 WM 解耦，便于单独测试或替换模型。
    *   **清晰性**: 代码逻辑直观，无过度封装，非常适合教学和科研。
    *   **可复现性**: 严格管理随机种子。
*   **潜在优化点**:
    *   **在线训练效率**: 目前 WM 是逐步更新 (Per-step update)，在 GPU 上效率较低。若规模扩大，建议改为从 Buffer 中采样进行 Batch 更新。
    *   **网络结构**: 目前均为 MLP，对于图像输入任务需扩展为 CNN。

## 5. 总结
该项目实现了一个结构清晰、易于扩展的 **World Model 辅助强化学习** 基线。它不仅实现了标准的 PPO 算法，还通过模块化设计展示了如何通过表征学习 (Representation Learning) 和辅助奖励 (Auxiliary Reward) 来增强 RL Agent 的能力。

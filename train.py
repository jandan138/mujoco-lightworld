"""
轻量级世界模型辅助 PPO 训练脚本（train.py）

功能概览：
- 参数解析：支持论文指南中的 E1–E4 各实验参数（任务、是否使用世界模型、潜空间维度、辅助奖励权重等）。
- 环境与随机种子：统一设置 Python/numpy/torch 的随机种子，保证可复现；创建 Gymnasium MuJoCo 环境。
- 世界模型组件（可选）：包含 Encoder（观测→潜特征 z）与 Dynamics（预测下一步潜特征），并在训练过程中进行轻量在线训练。
- PPO 训练循环：按 steps_per_epoch 采样、GAE 计算优势、剪切概率比与 KL 早停；记录 CSV 日志；保存模型参数。
- 输出约定：日志 CSV 写入 results/logs；模型权重写入 results/models；图表由外部脚本生成至 results/figures。

与实验指南一致：
- 支持三种组别：纯 PPO、PPO+WM(Feature)、PPO+WM(Feature+Reward)。
- 支持 z_dim、alpha 的消融实验与不同任务环境（如 Walker2d-v4、HalfCheetah-v4）。
"""

import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym


def parse_args():
    """解析命令行参数。

    主要参数说明：
    - task: 环境名称（如 Walker2d-v4 / HalfCheetah-v4）。
    - seed: 随机种子，用于控制实验可复现性。
    - total_steps: 总交互步数（将按 steps_per_epoch 计算训练轮数）。
    - device: 训练设备，默认自动选择 CUDA；可显式指定为 'cpu'。
    - use_wm: 是否使用世界模型的特征作为 PPO 的输入（Feature 模式）。
    - use_aux_reward: 是否启用辅助奖励（预测误差作为奖励惩罚项）。
    - alpha: 辅助奖励系数（论文中默认 0.01，可做消融）。
    - z_dim: 潜空间维度（Encoder 输出的特征维度）。
    - log_dir/model_dir/fig_dir: 日志、模型与图表输出目录。
    """
    parser = argparse.ArgumentParser(description="Lightweight WorldModel-assisted PPO Trainer")
    parser.add_argument("--task", type=str, default="Walker2d-v4", help="Gymnasium MuJoCo task name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total_steps", type=int, default=1_000_000, help="Total environment steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--use_wm", type=lambda x: str(x).lower() == "true", default=False, help="Use world model features")
    parser.add_argument("--use_aux_reward", type=lambda x: str(x).lower() == "true", default=False, help="Use auxiliary reward from WM prediction error")
    parser.add_argument("--alpha", type=float, default=0.01, help="Auxiliary reward coefficient")
    parser.add_argument("--z_dim", type=int, default=32, help="Latent dimension of encoder")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="Directory to save logs")
    parser.add_argument("--model_dir", type=str, default="results/models", help="Directory to save models")
    parser.add_argument("--fig_dir", type=str, default="results/figures", help="Directory to save figures")
    return parser.parse_args()


def ensure_dirs(*dirs):
    """确保若干目录存在，不存在则递归创建。

    参数：
    - *dirs: 变长目录列表，例如 log_dir/model_dir/fig_dir。
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def make_env(task: str, seed: int):
    """创建并设置指定 MuJoCo 任务的 Gymnasium 环境。

    - 使用 `env.reset(seed=seed)` 设置环境随机种子。
    - 注意：MuJoCo 需要正确安装依赖与驱动，首次运行会下载二进制。
    """
    env = gym.make(task)
    env.reset(seed=seed)
    return env


def main():
    """训练主入口。

    流程：
    1) 解析参数并创建输出目录。
    2) 统一设置随机种子（Python/numpy/torch）。
    3) 创建环境与 PPO 训练器（根据 use_wm 切换输入维度）。
    4) 若启用 WM，则实例化 Encoder/Dynamics 与损失，并构建优化器。
    5) 运行训练循环：按 epoch（steps_per_epoch）采样与更新，记录 CSV。
    6) 定期打印训练进度，完成后保存 Actor/Critic 以及 WM 参数。
    """
    args = parse_args()
    ensure_dirs(args.log_dir, args.model_dir, args.fig_dir)

    # Persist run config for reproducibility
    run_cfg_path = Path(args.log_dir) / "run_config.json"
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 统一设置随机种子，保证实验可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = make_env(args.task, args.seed)

    # Lazily import trainers to allow environment setup first
    from rl.ppo import PPOTrainer
    # 如果使用世界模型特征（Feature 模式），则 PPO 的输入维度设为 z_dim
    input_dim = env.observation_space.shape[0]
    if args.use_wm:
        input_dim = args.z_dim
    ppo = PPOTrainer(
        env=env,
        device=torch.device(args.device),
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        input_dim=input_dim,
    )

    wm_components = None
    if args.use_wm:
        try:
            from wm.encoder import Encoder
            from wm.dynamics import Dynamics
            from wm.loss import world_model_loss
            # 世界模型组件：Encoder 将原始观测压缩为潜特征 z；Dynamics 预测下一时刻潜特征
            wm_components = {
                "encoder": Encoder(obs_dim=env.observation_space.shape[0], z_dim=args.z_dim).to(args.device),
                "dynamics": Dynamics(z_dim=args.z_dim).to(args.device),
                "loss_fn": world_model_loss,
            }
        except Exception as e:
            print(f"[WARN] World Model components unavailable yet: {e}")

    # 训练循环：根据模式选择输出 CSV 文件名
    def resolve_csv_name():
        base = "ppo.csv"
        if args.use_wm and not args.use_aux_reward:
            base = "ppo_wm_feat.csv"
        elif args.use_wm and args.use_aux_reward:
            base = "ppo_wm_feat_reward.csv"
        return str(Path(args.log_dir) / base)

    csv_path = resolve_csv_name()
    # 写入 CSV 表头，格式与指南一致：step,mean_reward,wm_loss
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,mean_reward,wm_loss\n")

    wm_opt = None
    if wm_components is not None:
        # 将 Encoder 与 Dynamics 的参数合并，使用同一个 Adam 优化器进行在线训练
        params = list(wm_components["encoder"].parameters()) + list(wm_components["dynamics"].parameters())
        wm_opt = torch.optim.Adam(params, lr=3e-4)

    steps_per_epoch = ppo.cfg.steps_per_epoch
    total_epochs = max(1, args.total_steps // steps_per_epoch)
    print("Start training:")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    for epoch in range(1, total_epochs + 1):
        # 每个 epoch：先收集 steps_per_epoch 条样本（采样 + 缓冲 + 结束路径），再进行若干次 PPO 参数更新
        metrics = ppo.train_epoch(
            use_wm=bool(wm_components),
            wm=wm_components,
            use_aux_reward=args.use_aux_reward,
            alpha=args.alpha,
            wm_opt=wm_opt,
            wm_loss_fn=wm_components["loss_fn"] if wm_components else None,
        )
        step = epoch * steps_per_epoch
        # 将当前 epoch 的平均回报与 WM 损失写入 CSV，供绘制学习曲线
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{step},{metrics['mean_reward']:.6f},{metrics['wm_loss']:.6f}\n")
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{total_epochs} | step={step} | mean_reward={metrics['mean_reward']:.2f} | wm_loss={metrics['wm_loss']:.4f}")

    # Save models
    # 训练完成后保存 PPO 的 Actor/Critic 参数，以及（若启用）WM 的 Encoder/Dynamics 参数
    torch.save(ppo.actor.state_dict(), str(Path(args.model_dir) / "actor.pt"))
    torch.save(ppo.critic.state_dict(), str(Path(args.model_dir) / "critic.pt"))
    if wm_components is not None:
        torch.save(wm_components["encoder"].state_dict(), str(Path(args.model_dir) / "wm_encoder.pt"))
        torch.save(wm_components["dynamics"].state_dict(), str(Path(args.model_dir) / "wm_dynamics.pt"))

    print("Training complete. Logs:", csv_path)
    print("Models saved in:", args.model_dir)


if __name__ == "__main__":
    main()
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
import time
from datetime import datetime
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
    - resume: 是否尝试从最新的 checkpoint 恢复训练。
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
    # Visualization args
    parser.add_argument("--viz_mode", type=str, default="headless", choices=["headless", "headed"], help="Visualization mode")
    parser.add_argument("--save_video", type=lambda x: str(x).lower() == "true", default=False, help="Enable video recording")
    parser.add_argument("--save_snapshot", type=lambda x: str(x).lower() == "true", default=False, help="Enable periodic snapshot")
    parser.add_argument("--video_interval", type=int, default=50, help="Record video every N episodes")
    parser.add_argument("--snapshot_interval", type=int, default=1000, help="Save snapshot every N steps")
    parser.add_argument("--media_dir", type=str, default="results/media", help="Directory to save media")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
    
    return parser.parse_args()


def ensure_dirs(*dirs):
    """确保若干目录存在，不存在则递归创建。"""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def log_status(msg, log_file):
    """写入精简状态日志，同时打印到控制台。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


import pickle

def save_checkpoint(path, epoch, ppo, wm_components, wm_opt, env=None):
    """保存训练检查点，包含模型参数与优化器状态，以及环境的归一化统计量。"""
    checkpoint = {
        "epoch": epoch,
        "actor_state": ppo.actor.state_dict(),
        "critic_state": ppo.critic.state_dict(),
        "pi_opt_state": ppo.pi_opt.state_dict(),
        "vf_opt_state": ppo.vf_opt.state_dict(),
    }
    if wm_components:
        checkpoint.update({
            "encoder_state": wm_components["encoder"].state_dict(),
            "dynamics_state": wm_components["dynamics"].state_dict(),
            "wm_opt_state": wm_opt.state_dict() if wm_opt else None,
        })
    torch.save(checkpoint, path)

    # Save obs_rms if available
    if env is not None:
        try:
            # NormalizeObservation wrapper stores stats in `obs_rms`
            # We need to access it. Depending on wrapper depth, we use get_wrapper_attr
            obs_rms = env.get_wrapper_attr("obs_rms")
            if obs_rms is not None:
                rms_path = Path(path).parent / "obs_rms.pkl"
                with open(rms_path, "wb") as f:
                    pickle.dump(obs_rms, f)
        except Exception as e:
            print(f"[WARN] Failed to save obs_rms: {e}")


def load_checkpoint(path, ppo, wm_components, wm_opt, device, env=None):
    """加载训练检查点。"""
    if not os.path.exists(path):
        return 0
    
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    ppo.actor.load_state_dict(checkpoint["actor_state"])
    ppo.critic.load_state_dict(checkpoint["critic_state"])
    ppo.pi_opt.load_state_dict(checkpoint["pi_opt_state"])
    ppo.vf_opt.load_state_dict(checkpoint["vf_opt_state"])
    
    if wm_components:
        wm_components["encoder"].load_state_dict(checkpoint["encoder_state"])
        wm_components["dynamics"].load_state_dict(checkpoint["dynamics_state"])
        if wm_opt and checkpoint.get("wm_opt_state"):
            wm_opt.load_state_dict(checkpoint["wm_opt_state"])
    
    # Load obs_rms if available
    if env is not None:
        rms_path = Path(path).parent / "obs_rms.pkl"
        if rms_path.exists():
            try:
                with open(rms_path, "rb") as f:
                    obs_rms = pickle.load(f)
                # Set obs_rms in the wrapper
                # We assume the env has NormalizeObservation wrapper
                # Accessing the wrapper attribute directly might be tricky if it's property-based in newer gym
                # But typically obs_rms is an attribute of NormalizeObservation
                # We iterate wrappers to find it
                current_env = env
                while hasattr(current_env, "env"):
                    if hasattr(current_env, "obs_rms"):
                        current_env.obs_rms = obs_rms
                        print("Loaded obs_rms statistics.")
                        break
                    current_env = current_env.env
            except Exception as e:
                print(f"[WARN] Failed to load obs_rms: {e}")

    return checkpoint["epoch"]


def main():
    """训练主入口。"""
    args = parse_args()
    ensure_dirs(args.log_dir, args.model_dir, args.fig_dir, args.media_dir)
    
    status_log_path = Path(args.log_dir) / "training_status.log"
    checkpoint_path = Path(args.model_dir) / "checkpoint.pt"

    # Persist run config for reproducibility
    run_cfg_path = Path(args.log_dir) / "run_config.json"
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 统一设置随机种子，保证实验可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup Environment with Visualization
    from mujoco_lightworld.common.visualizer import make_visualized_env, VisualizationConfig
    viz_cfg = VisualizationConfig(
        viz_mode=args.viz_mode,
        save_video=args.save_video,
        save_snapshot=args.save_snapshot,
        video_interval=args.video_interval,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.media_dir
    )
    env = make_visualized_env(args.task, args.seed, viz_cfg)

    # Lazily import trainers to allow environment setup first
    from mujoco_lightworld.rl.ppo import PPOTrainer
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
            from mujoco_lightworld.wm.encoder import Encoder
            from mujoco_lightworld.wm.dynamics import Dynamics
            from mujoco_lightworld.wm.loss import world_model_loss
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
    
    wm_opt = None
    if wm_components is not None:
        # 将 Encoder 与 Dynamics 的参数合并，使用同一个 Adam 优化器进行在线训练
        params = list(wm_components["encoder"].parameters()) + list(wm_components["dynamics"].parameters())
        wm_opt = torch.optim.Adam(params, lr=3e-4)

    steps_per_epoch = ppo.cfg.steps_per_epoch
    total_epochs = max(1, args.total_steps // steps_per_epoch)
    
    start_epoch = 1
    if args.resume and os.path.exists(checkpoint_path):
        loaded_epoch = load_checkpoint(checkpoint_path, ppo, wm_components, wm_opt, args.device, env=env)
        if loaded_epoch > 0:
            start_epoch = loaded_epoch + 1
            log_status(f"Resumed training from epoch {loaded_epoch}", status_log_path)
    
    if start_epoch == 1:
        # 写入 CSV 表头，格式与指南一致：step,mean_reward,wm_loss
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,mean_reward,wm_loss\n")
        log_status("Started new training session", status_log_path)
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    # 初始化最优模型保存变量
    best_reward = -float('inf')
    best_model_path = Path(args.model_dir) / "best_model.pt"

    for epoch in range(start_epoch, total_epochs + 1):
        # Linear Learning Rate Decay
        # 简单实现：随着 epoch 增加，线性降低 PPO 的 Actor 和 Critic 的学习率
        current_lr = 3e-4 * (1.0 - (epoch - 1) / total_epochs)
        current_lr = max(current_lr, 0.0) # 保证不为负
        
        for param_group in ppo.pi_opt.param_groups:
            param_group['lr'] = current_lr
        for param_group in ppo.vf_opt.param_groups:
            param_group['lr'] = current_lr

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
            
        # 保存历史最优模型
        if metrics['mean_reward'] > best_reward:
            best_reward = metrics['mean_reward']
            # 使用与 checkpoint 相同的保存函数，但路径不同
            save_checkpoint(best_model_path, epoch, ppo, wm_components, wm_opt, env=env)
            # 可以在日志中记录一下（可选，避免刷屏只在状态日志体现）
            # log_status(f"New best reward: {best_reward:.2f} at epoch {epoch}", status_log_path)
            
        # Log status periodically
        if epoch % 10 == 0 or epoch == 1 or epoch == total_epochs:
            msg = f"Epoch {epoch}/{total_epochs} | Step {step} | Reward: {metrics['mean_reward']:.2f} | Best: {best_reward:.2f} | LR: {current_lr:.2e}"
            log_status(msg, status_log_path)
            
            # Save checkpoint
            save_checkpoint(checkpoint_path, epoch, ppo, wm_components, wm_opt, env=env)

    # Save final models
    # 训练完成后保存 PPO 的 Actor/Critic 参数，以及（若启用）WM 的 Encoder/Dynamics 参数
    torch.save(ppo.actor.state_dict(), str(Path(args.model_dir) / "actor.pt"))
    torch.save(ppo.critic.state_dict(), str(Path(args.model_dir) / "critic.pt"))
    if wm_components is not None:
        torch.save(wm_components["encoder"].state_dict(), str(Path(args.model_dir) / "wm_encoder.pt"))
        torch.save(wm_components["dynamics"].state_dict(), str(Path(args.model_dir) / "wm_dynamics.pt"))

    log_status(f"Training complete. Models saved to {args.model_dir}", status_log_path)


if __name__ == "__main__":
    main()
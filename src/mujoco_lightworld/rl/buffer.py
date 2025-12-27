"""
PPO 采样缓冲区与 GAE-Lambda 优势计算。

职责：
- 按时间顺序存储 (obs, act, rew, val, logp)。
- 路径结束时计算：
  1) Generalized Advantage Estimation (GAE-Lambda) 的优势 adv。
  2) Rewards-to-go（折扣回报）ret。
- `get()` 时进行优势归一化，并清空指针以便下一轮收集。

实现要点：
- 使用张量在指定设备上存储，避免频繁 CPU/GPU 迁移。
- 指针与切片管理路径边界，确保计算时不越界。
"""

from dataclasses import dataclass
import torch


@dataclass
class RolloutBufferConfig:
    """缓冲区配置。

    - obs_dim: 观测或特征维度（使用 WM 时为 z_dim）。
    - act_dim: 动作维度。
    - size: 每个 epoch 收集的步数（steps_per_epoch）。
    - gamma: 折扣因子。
    - lam: GAE 的 lambda 系数。
    """
    obs_dim: int
    act_dim: int
    size: int
    gamma: float = 0.99
    lam: float = 0.95


class RolloutBuffer:
    """按时间顺序存储采样数据，并在路径结束时计算优势与回报。"""

    def __init__(self, cfg: RolloutBufferConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        # 指针（当前写入位置）、当前路径起点索引、最大容量（steps_per_epoch）
        self.ptr, self.path_start_idx, self.max_size = 0, 0, cfg.size
        # 预先分配缓冲区张量，避免在采样过程中重复分配内存
        self.obs_buf = torch.zeros((cfg.size, cfg.obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((cfg.size, cfg.act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(cfg.size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(cfg.size, dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros(cfg.size, dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(cfg.size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(cfg.size, dtype=torch.float32, device=device)

    def store(self, obs, act, rew, val, logp):
        """存储单步采样数据。"""
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """在路径终点调用，计算 GAE 优势与折扣回报。

        参数：
        - last_val: 若路径因中途截断（非真正终止）而结束，则使用价值网络估计的下一个状态价值；如果是终止（done），则为 0。
        步骤：
        1) 构造 rews 与 vals，并在末尾拼接 last_val，便于统一处理边界。
        2) 使用 GAE-Lambda 递推计算优势。
        3) 反向递推计算 rewards-to-go（折扣回报）。
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat([self.rew_buf[path_slice], torch.tensor([last_val], device=self.device)])
        vals = torch.cat([self.val_buf[path_slice], torch.tensor([last_val], device=self.device)])

        # 计算 TD 残差（delta_t）并进行 GAE 递推：adv_t = delta_t + gamma*lambda*adv_{t+1}
        deltas = rews[:-1] + self.cfg.gamma * vals[1:] - vals[:-1]
        adv = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.cfg.gamma * self.cfg.lam * gae
            adv[t] = gae
        self.adv_buf[path_slice] = adv

        # Rewards-to-go：ret_t = r_t + gamma * ret_{t+1}
        ret = torch.zeros_like(rews[:-1])
        g = 0.0
        for t in reversed(range(len(rews) - 1)):
            g = rews[t] + self.cfg.gamma * g
            ret[t] = g
        self.ret_buf[path_slice] = ret

        # 将路径起点更新为当前指针，准备下一条路径
        self.path_start_idx = self.ptr

    def get(self):
        """在收集满一个 epoch 的数据后调用，返回训练所需张量并重置指针。"""
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # 对优势做标准化，有助于稳定训练与提升数值尺度一致性
        adv_mean = self.adv_buf.mean()
        adv_std = self.adv_buf.std() + 1e-8
        adv = (self.adv_buf - adv_mean) / adv_std
        return self.obs_buf, self.act_buf, self.logp_buf, adv, self.ret_buf
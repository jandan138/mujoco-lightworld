# MuJoCo 环境安装与验证

推荐：
- 使用 Conda + Python 3.10。
- 按 PyTorch 官网选择 GPU 版本（CUDA 12.1）。

安装步骤：
- 创建环境与激活：`conda create -n mujoco_wm python=3.10 -y && conda activate mujoco_wm`
- 安装 PyTorch（GPU）：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- 安装依赖：`pip install -r requirements.txt`

验证：
- 运行 `python -c "import gymnasium as gym; env=gym.make('Walker2d-v4'); env.reset(); print('ok')"`

常见问题：
- MuJoCo 二进制下载失败：检查网络或更换镜像。
- Windows GPU 驱动不匹配：更新 NVIDIA 驱动与 CUDA 运行时。
- ImportError：确保 `gymnasium[mujoco]` 安装成功。
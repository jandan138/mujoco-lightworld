# 训练监控与可视化系统指南

MuJoCo LightWorld 提供了一套完整的训练监控系统，支持实时可视化、视频录制和关键帧截图。无论是本地调试还是服务器端的大规模训练，都能满足您的观测需求。

## 1. 核心模式 (Visualization Modes)

通过 `--viz_mode` 参数选择运行模式：

| 模式 | 参数值 | 适用场景 | 描述 |
| :--- | :--- | :--- | :--- |
| **无头模式** (默认) | `headless` | 服务器/云端训练 | 后台静默运行，不依赖物理显示器。支持通过离屏渲染（Off-screen Rendering）进行视频和截图录制。 |
| **有头模式** | `headed` | 本地调试 | 弹出 MuJoCo 原生渲染窗口，实时显示机器人动作。**注意：此模式下通常不建议同时开启视频录制，可能会有窗口冲突。** |

## 2. 媒体记录功能

系统支持自动记录训练过程中的多媒体数据，所有产物均保存在 `results/media/` 目录下。

### 2.1 视频录制
使用 Gymnasium 的 `RecordVideo` 包装器实现。
*   **启用**: `--save_video true`
*   **频率**: `--video_interval N` (每 N 个 Episode 录制一次)
*   **产物**: MP4 文件，存放在 `results/media/videos/`。

### 2.2 关键帧截图与元数据
使用自定义的 `SnapshotWrapper` 实现。
*   **启用**: `--save_snapshot true`
*   **频率**: `--snapshot_interval N` (每 N 个 Step 截取一帧)
*   **产物**:
    *   **图片**: PNG 格式，存放在 `results/media/snapshots/`。
    *   **元数据**: 对应的 JSON 文件，存放在 `results/media/metadata/`。

**JSON 元数据示例**:
```json
{
  "step": 1000,
  "timestamp": 1735286960,
  "reward": 1.25,
  "action": [0.1, -0.5, 0.3, ...],
  "info_keys": ["x_position", "x_velocity"]
}
```

## 3. 使用示例

### 场景 A：服务器端训练并录制视频
在远程服务器上运行，每 50 个 Episode 录制一次视频，每 1000 步保存一张截图：

```bash
uv run lightworld-train \
    --task Walker2d-v4 \
    --viz_mode headless \
    --save_video true \
    --video_interval 50 \
    --save_snapshot true \
    --snapshot_interval 1000
```

### 场景 B：本地实时观察
在本地机器上直接观察训练效果（不录制）：

```bash
uv run lightworld-train \
    --task Walker2d-v4 \
    --viz_mode headed
```

## 4. 常见问题

### 1. 无头模式下报错 `OpenGL` 或 `GLFW`
确保服务器已安装基本的图形库（即使没有显示器）。
*   **Ubuntu**: `apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev`
*   **Python 依赖**: `opencv-python-headless` (本项目默认安装的是完整版 `opencv-python`，在大多数环境下也能工作，若报错可尝试替换)。

### 2. 视频文件无法播放
Gymnasium 默认使用 `mp4v` 或 `h264` 编码。如果生成的视频在某些播放器无法打开，尝试使用 VLC 播放器，或检查系统是否安装了 `ffmpeg`。

### 3. 有头模式下录制视频失败
这是 Gym 的已知限制。在 `human` 渲染模式下，窗口上下文由 GLFW 接管，`RecordVideo` 难以同时抓取帧。建议**录制视频时始终使用 `headless` 模式**。

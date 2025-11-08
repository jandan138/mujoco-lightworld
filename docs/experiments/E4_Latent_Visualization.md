# E4：潜空间可视化（t-SNE）

目的：检验 Encoder 的潜特征是否有结构性（轨迹聚类或环状分布）。

步骤：
1. 载入 `wm_encoder.pt`。
2. 在环境中随机采样多条轨迹，收集 `z`。
3. 使用 t-SNE 降维到 2D 并绘制散点图（按时间着色）。

参考代码：见 `Experiment_Guide_WorldModelPPO.md` 或将其改造成脚本。

判读：
- 出现簇状/环状结构 → Encoder 学到动力学规律。
- 随机分布 → 世界模型训练不足，增大训练步数或调整 z_dim。
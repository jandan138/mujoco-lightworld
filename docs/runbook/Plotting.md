# 绘图与结果生成

学习曲线（Fig.2）：
```python
import pandas as pd, matplotlib.pyplot as plt
df1 = pd.read_csv("results/logs/ppo.csv")
df2 = pd.read_csv("results/logs/ppo_wm_feat.csv")
df3 = pd.read_csv("results/logs/ppo_wm_feat_reward.csv")
plt.plot(df1['step'], df1['mean_reward'], label='PPO')
plt.plot(df2['step'], df2['mean_reward'], label='PPO+WM(Feature)')
plt.plot(df3['step'], df3['mean_reward'], label='PPO+WM(Feature+Reward)')
plt.xlabel('环境步数'); plt.ylabel('平均奖励')
plt.legend(); plt.grid(); plt.savefig('results/figures/learning_curves.pdf')
```

柱状图（Fig.3）：
参考 `Experiment_Guide_WorldModelPPO.md` 中的示例代码。

潜空间 t-SNE（Fig.5）：
参考 `experiments/E4_Latent_Visualization.md`。
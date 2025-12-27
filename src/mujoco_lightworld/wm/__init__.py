"""
世界模型（WM）包：提供 Encoder、Dynamics 与损失函数。

模块说明：
- encoder：将原始观测压缩为潜空间特征 z。
- dynamics：在潜空间上进行一步预测，得到 z_{t+1}。
- loss：包含世界模型的 MSE 损失与预测误差辅助奖励。
"""
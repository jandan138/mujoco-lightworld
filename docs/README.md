# 文档总览（docs）

本目录收录项目的说明文档，包括基础知识、代码详解、整体框架、运行手册、踩坑与更新日志等，帮助你快速上手并顺利复现论文实验。

建议阅读顺序：
- basics/：强化学习与世界模型的基础概念、环境安装
- architecture/：整体框架与数据流，把握模块关系
- code/：各源码文件的职责与训练循环细节
- runbook/：从命令到结果的完整跑通手册与绘图
- experiments/：对应 E1–E4 的实验指导
- pitfalls/：常见问题与解决方案
- config/：配置项说明与覆盖方法
- changelog/：更新日志

快速开始：
- 安装环境：参考 `basics/MuJoCo_Setup.md`
- 运行 E1：参考 `runbook/EndToEnd.md`
- 绘制学习曲线：参考 `runbook/Plotting.md`
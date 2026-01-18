# 期刊投稿操作指南 (Submission Guide)

## 目标期刊：Advanced Robotics / Robotica / SIMULATION

本指南协助您手动完成在线投稿流程。所有必要的投稿材料已为您准备在 `submission/` 文件夹中。

### 1. 准备材料清单
请确保您手头有以下文件（已生成）：
*   **Manuscript.tex / PDF**: 论文正文。如果期刊要求 PDF，请先将 .tex 编译为 .pdf（建议使用 Overleaf 或本地 LaTeX 环境）。
*   **Cover_Letter.txt**: 给编辑的投稿信。
*   **Suggested_Reviewers.txt**: 推荐的审稿人名单。
*   **Figures**:
    *   `paper_assets/gait_strip.png` (Fig 7)
    *   (可选) `paper_assets/fig1_system_arch.png` (如已生成)
    *   (可选) `paper_assets/fig2_network_arch.png` (如已生成)

### 2. 投稿步骤 (以 Advanced Robotics 为例)

1.  **访问投稿系统**: 
    *   网址: [Advanced Robotics Submission Site](https://www.rsj.or.jp/advanced_robotics/submission/) 或 Taylor & Francis 的 Submission Portal。
2.  **注册/登录**: 使用您的学术邮箱注册账号。
3.  **Start New Submission**:
    *   **Article Type**: 选择 "Full Paper" 或 "Short Paper" (根据您的篇幅，本文适合 Full Paper)。
    *   **Title**: 填入 "A Lightweight World Model–Assisted PPO Framework for Efficient Reinforcement Learning in MuJoCo Environments"。
    *   **Abstract**: 复制 `PAPER_DRAFT.md` 中的 Abstract 内容。
    *   **Keywords**: 填入 "Reinforcement Learning", "World Models", "MuJoCo", "PPO", "Robotic Control"。
4.  **Upload Files**:
    *   上传 `Manuscript` (PDF 或 Word/LaTeX)。
    *   上传 `Cover Letter`。
    *   单独上传图片文件 (Figures)，如果系统要求。
5.  **Authors & Institutions**:
    *   按顺序添加所有作者信息。
6.  **Reviewers**:
    *   复制 `submission/Suggested_Reviewers.txt` 中的信息填入推荐审稿人部分。
7.  **Review & Submit**:
    *   生成 PDF 用于预览。
    *   仔细检查无误后点击 "Submit"。

### 3. 常见问题 (FAQ)

*   **Q: 期刊要求提供 "Conflict of Interest" 声明？**
    *   A: 请在系统中勾选 "No conflict of interest"，或在 Manuscript 最后添加一段声明。
*   **Q: 图片清晰度不够？**
    *   A: 当前 `gait_strip.png` 是位图。如果期刊要求矢量图 (EPS/PDF)，请告知我，我可以尝试用 Matplotlib 重新生成架构图。
*   **Q: 需要提供代码链接吗？**
    *   A: 建议将代码上传至 GitHub，并在 Manuscript 中附上链接（"Code is available at: https://github.com/..."），这能增加论文被录用的概率。

祝您投稿顺利！

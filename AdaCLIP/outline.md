# 大纲 · AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L30 摘要
- L37–65 1 Introduction
- L66–86 2 Related Work
  - L69–76 2.1 Traditional Anomaly Detection
  - L77–79 2.2 Zero-shot Anomaly Detection
  - L80–86 2.3 Prompt Learning
- L87–101 3 Problem Formulation
- L102–154 4 AdaCLIP
  - L104–107 4.1 Overview
  - L108–118 4.2 Prompting Layers
  - L119–131 4.3 Hybrid Learnable Prompts
  - L132–134 4.4 Projection Layer
  - L135–141 4.5 Pixel-Level Anomaly Localization
  - L142–150 4.6 Hybrid Semantic Fusion Module
  - L151–154 4.7 Image-Level Anomaly Detection
- L155–690 5 Experiments
  - L158–170 5.1 Experimental Setup
  - L171–566 5.2 Main Experimental Results
  - L567–658 5.3 Ablation Study
  - L659–690 5.4 Analysis
- L691–697 6 Conclusion
- L698–702 Acknowledgements

## 公式（3 个，按出现顺序；L 后面是所在小节）

- L112 §4.2 Prompting Layers：[\mathbf{T}_{j+1}, \_] = L^P_j([\mathbf{T}_j, \mathbf{P}_j]), \quad j \leq J, \\ [\mathbf{T}_{j+1}, \mathbf{P}…
- L125 §4.3 Hybrid Learnable Prompts：\mathbf{P}^D = \text{DPG}(\mathbf{I}).
- L137 §4.5 Pixel-Level Anomaly Localization `eq:AL`：~ \mathbf{M}_i = \phi\left(\frac{\exp(\cos(\mathbf{F}^P_i, \mathbf{F}^T_A))}{\exp(\cos(\mathbf{F}^P_i, \mathbf…

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

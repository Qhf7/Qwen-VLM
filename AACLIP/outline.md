# 大纲 · AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L12 摘要
- L16–42 1 Introduction
- L43–58 2 Related Work
- L59–190 3 Method
  - L60–115 3.1 Overview
    - L61–64 3.1.1 Problem Formulation
    - L65–100 3.1.2 Current Challenges
    - L101–115 3.1.3 Overview of Our Solution
  - L116–190 3.2 AA-CLIP with Two-Stage Adaptation Strategy
    - L117–137 3.2.1 Residual Adapter
    - L138–190 3.2.2 Two-Stage Training Strategy
- L191–348 4 Experiments
  - L192–203 4.1 Experiment Setups
  - L204–286 4.2 Comparison with SOTA Methods
  - L287–300 4.3 Visualization
  - L301–348 4.4 Ablations Analysis
- L349–355 5 Conclusion and Discussion
- L356–363 Acknowledgement

## 公式（7 个，按出现顺序；L 后面是所在小节）

- L121 §3.2.1 Residual Adapter `eq:adapter`：\begin{aligned} x_{residual}^i&=Norm(Act(W^ix^i)), \end{aligned}
- L129 §3.2.1 Residual Adapter `eq:residual`：\begin{aligned} x^i_{enhanced}=\lambda\ x_{residual}^i + (1-\lambda)\ x^i, \end{aligned}
- L144 §3.2.2 Two-Stage Training Strategy `eq:cos`：\begin{aligned} p_{cls}&=CosSim(V_{image}, [T_N,T_A]), \\ p_{seg}^o &=CosSim(V_{patch}, [T_N,T_A]), \end{align…
- L152 §3.2.2 Two-Stage Training Strategy `eq:align`：\begin{aligned} &\mathcal{L}_{cls} =\text{BCE} (p_{cls}, y), \\ &\mathcal{L}_{seg} =\text{Dice} (p_{seg}, S)+\…
- L162 §3.2.2 Two-Stage Training Strategy `eq:dis`：\mathcal{L}_{dis}=|<T_N,T_A>|^2.
- L167 §3.2.2 Two-Stage Training Strategy `eq:total`：\mathcal{L}_{total}=\mathcal{L}_{align}+\gamma \mathcal{L}_{dis}.
- L181 §3.2.2 Two-Stage Training Strategy `eq:agg`：\begin{aligned} V_{patch}^i&=Proj_i(F^i),\ i\in \{1,2,3,4\} \\ V_{patch}&=\sum^4_{i=1}V_{patch}^i. \end{aligne…

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

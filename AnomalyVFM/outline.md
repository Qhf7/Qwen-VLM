# 大纲 · AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L6 摘要
- L10–35 1 Introduction
- L36–49 2 Related Work
- L50–99 3 Dataset Generation Scheme
- L100–128 4 AnomalyVFM
- L129–330 5 Experiments
  - L240–244 5.1 Datasets
  - L245–249 5.2 Implementation Details
  - L250–253 5.3 Generalisation of the proposed framework
  - L254–272 5.4 Comparison to zero-shot methods
  - L273–330 5.5 Comparison to few-shot methods
- L331–364 6 Ablation study
- L365–399 7 Conclusion
- L386 —— 附录 ——
- L400–447 A Limitations
- L448–454 B Discussion about dataset generation phase
- L455–507 C Training Competing methods with the proposed synthetic dataset
- L508–512 D Synthetic Dataset Details
- L513–565 E Additional Ablation Studies
- L566–578 F Additional Qualitative Examples
- L579–718 G Image Generation Data

## 公式（4 个，按出现顺序；L 后面是所在小节）

- L73 §3 Dataset Generation Scheme：I = G(p).
- L83 §3 Dataset Generation Scheme：w \sim \text{U}(w_{min}, w_{max}), h \sim \text{U}(h_{min}, h_{max}).
- L116 §4 AnomalyVFM `eq:conf`：\mathcal{L}_{base} = \mathcal{L}_{1}(M_{o}, M_{GT}) + \beta * \mathcal{L}_{focal}(M_{o}, M_{GT}),
- L121 §4 AnomalyVFM `eq:conf`：\mathcal{L}_{seg} = \mathcal{L}_{base}(M_{o}, M_{GT}) * C - \alpha log(C),

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

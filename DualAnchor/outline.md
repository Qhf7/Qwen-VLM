# 大纲 · Dual Anchors, Do It Better: Hierarchical Group Merging for Zero-Shot Anomaly Detection

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L24 摘要
- L28–55 1 Introduction
- L56–87 2 Related Works
  - L57–77 2.1 Zero-Shot Anomaly Detection
  - L78–87 2.2 Hierarchical Grouping
- L88–399 3 Method
  - L93–95 3.1 Problem Setting
  - L96–97 3.2 Feature Extraction.
  - L98–194 3.3 Hierarchical Group Merging
  - L195–228 3.4 Dual-Anchor Contrastive Learning
  - L229–399 3.5 Optimization
- L400–503 4 Experiment
  - L412–421 4.1 Experimental Setup
  - L422–462 4.2 Experimental Results
  - L463–503 4.3 Ablations Analysis
- L504–510 5 Conclusion
- L511–519 Acknowledgements

## 公式（19 个，按出现顺序；L 后面是所在小节）

- L109 §3.3 Hierarchical Group Merging：F_{i}'^{(l)} = \mathrm{CrossAttn}\!\left(F_{i}^{(l)},\, \mathbf{G}'^{(h)}\right),
- L115 §3.3 Hierarchical Group Merging：\pi_i = \frac{\exp\!\left((\log \alpha_i + g_i)/\tau\right)} {\sum_{j=1}^{K} \exp\!\left((\log \alpha_j + g_j)…
- L122 §3.3 Hierarchical Group Merging：\begin{aligned} \boldsymbol{\Pi} = \mathrm{Softmax}\!\left( \frac{\mathbf{F}_{i}'^{(l)}W_Q(\mathbf{G}^{(h)}W_K…
- L130 §3.3 Hierarchical Group Merging：\mathbf{G}'^{(h)} = \mathbf{G}^{(h)} + \mathrm{Linear}\!\big(\boldsymbol{\Pi}(\mathbf{G}^{(h)}W_V)\big)
- L142 §3.3 Hierarchical Group Merging：\mathbf{S} = \cos\!\left( \mathbf{G}'_{\mathrm{src}},\, \mathbf{G}'_{\mathrm{dst}} \right),
- L149 §3.3 Hierarchical Group Merging：v'(u) = \arg\max_{v} \mathbf{S}_{u,v}.
- L156 §3.3 Hierarchical Group Merging：\tilde{\mathbf{g}}_{u} = \frac{1}{2}\left( \mathbf{g}_u + \mathbf{g}_{v'(u)} \right).
- L162 §3.3 Hierarchical Group Merging：\mathbf{G}'^{(h+1)} = \mathrm{Concat}\!\left( \tilde{\mathbf{G}}',\, \mathbf{G}'_{\mathrm{remain}} \right),
- L183 §3.3 Hierarchical Group Merging：\begin{aligned} \{ \mathbf{G}_N,\, \mathbf{G}_A \},\, F_{assign} &= \sum_{h=1}^{H} \textit{MergeBlock}\!\left(…
- L200 §3.4 Dual-Anchor Contrastive Learning：w_n &= \frac{t_{\mathrm{cls}}^\top G_N}{\|t_{\mathrm{cls}}\|\|G_N\|}, \quad w_a = \frac{t_{\mathrm{cls}}^\top …
- L206 §3.4 Dual-Anchor Contrastive Learning：\begin{aligned} \hat{w} &= \mathrm{Concat}(w_n, w_a), \\ t_{\mathrm{cls}}^{\text{refined}} &= t_{\mathrm{cls}}…
- L219 §3.4 Dual-Anchor Contrastive Learning：\begin{aligned} p_n &= [V_1][V_2]\cdots[V_E][W_1][G_N][\texttt{class}], \\ p_a &= [V'_1][V'_2]\cdots[V'_E][W'_…
- L352 §3.5 Optimization：\mathcal{A}^{(L+1)} \;=\; \mathrm{Softmax}\!\big(\mathrm{Up}([\frac{1}{L}\sum_{l=1}^{L}F_{i}^{(l)},\, F_{\text…
- L357 §3.5 Optimization：\mathcal{M} = \frac{1}{L+1}\sum_{l=1}^{L+1}\frac{\mathcal{A}^{(l)}_{a} + (1 - \mathcal{A}^{(l)}_{n})}{2},
- L364 §3.5 Optimization：p_{\text{cls}} = \frac{t_{\mathrm{cls}}^{\text{refined}} \cdot \mathcal{T}} {t_{\mathrm{cls}}^{\text{refined}}…
- L373 §3.5 Optimization：\begin{aligned} \mathcal{L}_{\mathrm{cls}} &= \mathrm{BCE}(p_{\mathrm{cls}}, y), \\ \mathcal{L}_{\mathrm{seg}}…
- L382 §3.5 Optimization：\mathcal{L}_{\mathrm{ortho}} \;=\; \big| \langle \mathbf{G}_{N},\, \mathbf{G}_{A} \rangle \big|^{2}.
- L388 §3.5 Optimization：\mathcal{L}_{\mathrm{group}} \;=\; 1 - \tfrac{1}{2}\!\left( \cos(\mathbf{G}_N, \mathbf{Z}_N) + \cos(\mathbf{G}…
- L395 §3.5 Optimization：\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\mathrm{cls}} + \mathcal{L}_{\mathrm{seg}} + \mathcal{L}_{\mathr…

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

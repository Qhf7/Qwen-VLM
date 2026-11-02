# 大纲 · WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L28 摘要
- L33–74 1 Introduction
- L75–91 2 Related work
- L92–116 3 Background
- L117–202 4 WinCLIP and WinCLIP+
  - L122–149 4.1 Language-driven zero-shot AC
  - L150–178 4.2 WinCLIP for zero-shot AS
  - L179–202 4.3 WinCLIP+ with few-normal-shots
- L203–493 5 Experiments
  - L312–321 5.1 Zero-/few-shot anomaly classification
  - L322–328 5.2 Zero-/few-shot anomaly segmentation
  - L329–470 5.3 Comparison with many-shot methods
  - L471–493 5.4 Ablation study
- L494–516 6 Conclusion
- L503 —— 附录 ——
- L517–605 A Experimental details
- L606–678 B Additional results on ablation study
- L679–716 C Additional qualitative results
- L717–1087 D Detailed quantitative results

## 公式（6 个，按出现顺序；L 后面是所在小节）

- L109 §3 Background `eq:clip_0shot`：p(\rvs = s_i|\rvx; \rvs \in S) := \frac{\exp(\langle f(\rvx), g(s_i) \rangle / \tau) }{\sum_{s \in S} \exp(\la…
- L157 §4.2 WinCLIP for zero-shot AS `eq:winclip`：\mathbf{F}^{\tt W}_{ij} := f(\rvx \odot \mathbf{w}_{ij}),
- L171 §4.2 WinCLIP for zero-shot AS `eq:harmonic`：\bar{\mathbf{M}}^{\tt W}_{0,ij} := \left(\frac{1}{\sum_{u,v} {(\mathbf{w}_{uv})_{ij}}} \sum_{u,v} \frac{(\math…
- L184 §4.3 WinCLIP+ with few-normal-shots `eq:memory_al`：\mathbf{M}_{ij} := \min_{r \in \mathbf{R}} \tfrac{1}{2} (1 - \langle \mathbf{F}_{ij}, r \rangle).
- L192 §4.3 WinCLIP+ with few-normal-shots `eq:winplus_al`：\mathbf{M}^{\tt W} := \tfrac{1}{3}(\mathbf{M}^{\tt P} + \mathbf{M}^{\tt W}_{\tt s} + \mathbf{M}^{\tt W}_{\tt m…
- L199 §4.3 WinCLIP+ with few-normal-shots `eq:winplus_ad`：\mathrm{ascore}_{\tt W}(\rvx) := \tfrac{1}{2}\left(\mathrm{ascore}_0(f(\rvx)) + \max_{ij} \mathbf{M}^{\tt W}_{…

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

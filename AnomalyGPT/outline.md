# 大纲 · AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L7 摘要
- L36–55 1 Introduction
- L56–80 2 Related Work
- L81–354 3 Method
  - L84–86 3.1 Model Architecture
  - L87–107 3.2 Decoder and Prompt Learner
  - L108–134 3.3 Data for Image-Text Alignment
  - L135–354 3.4 Loss Functions
- L355–392 4 Experiments
  - L372–386 4.1 Quantitative Results
  - L387–389 4.2 Qualitative Examples
  - L390–392 4.3 Ablation Studies
- L393–411 5 Conclusion
- L401 —— 附录 ——
- L412–435 A More Experimental Results of Existing IAD Methods
- L436–512 B Normal and Abnormal Texts
- L513–600 C Detailed Image Description
- L601–646 D More Qualitative Examples

## 公式（6 个，按出现顺序；L 后面是所在小节）

- L93 §3.2 Decoder and Prompt Learner `eq:localization result`：M = Upsample\left(\sum_{i=1}^{4}{softmax(\tilde{F}^{i}_{patch}F_{text}^{T})}\right).
- L99 §3.2 Decoder and Prompt Learner `eq:few-shot localization result`：M = Upsample\left(\sum_{i=1}^{4}{\left(1 - max(F^{i}_{patch} \cdot {B^i}^T)\right)}\right).
- L296 §3.4 Loss Functions `eq:ce`：L_{ce} = -\sum_{i=1}^{n}{y_i log(p_i)},
- L303 §3.4 Loss Functions `eq:focal`：L_{focal} = -\frac{1}{n}\sum_{i=1}^{n}{(1-p_i)^{\gamma}log(p_i)},
- L310 §3.4 Loss Functions `eq:dice`：L_{dice} = -\frac{\sum_{i=1}^{n}{y_i\hat{y}_i}}{\sum_{i=1}^{n}{y_i^2}+\sum_{i=1}^{n}{\hat{y}^2_i}},
- L316 §3.4 Loss Functions `eq:loss`：L = \alpha L_{ce}+\beta L_{focal}+\delta L_{dice},

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

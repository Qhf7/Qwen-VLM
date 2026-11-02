# 大纲 · AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L11 摘要
- L18–60 1 Introduction
- L61–82 2 Preliminary
- L83–167 3 AnomalyCLIP: object-agnostic prompt learning
  - L84–86 3.1 Approach overview
  - L87–105 3.2 Object-agnostic text prompt design
  - L106–167 3.3 Learning generic abnormality and normality prompts
- L168–383 4 Experiments
  - L170–242 4.1 Experiment setup
  - L243–308 4.2 Main results
  - L309–383 4.3 Ablation study
- L384–392 5 Related Work
- L393–404 6 Conclusion
    - L396–398 Acknowledgments
    - L399–404 Reproducibility Statement
- L403 —— 附录 ——
- L405–419 A Implementation details and baselines
  - L407–408 A.1 Implementation details
  - L409–419 A.2 Baselines
- L420–460 B Dataset
- L461–475 C Detailed Analysis of DPAM
- L476–600 D Additional results and ablations
- L601–828 E Visualization
- L829–1113 F Fine-grained ZSAD performance

## 公式（8 个，按出现顺序；L 后面是所在小节）

- L67 §2 Preliminary `equ: softmax`：p(y = c | x_i) = P(g_c, f_i) = {\frac{exp(<g_c, f_i>/\tau)}{\sum_{c\in \mathcal{C}}exp(<g_c, f_i>)/\tau)}},
- L93 §3.2 Object-agnostic text prompt design：g_n &= [V_1][V_2]\dots[V_E][cls] \nonumber\\ g_a &= [W_1][W_2]\dots[W_E] [damaged][cls],\nonumber
- L100 §3.2 Object-agnostic text prompt design：g_n&= [V_1][V_2]\dots[V_E][object] \nonumber\\ g_a &= [W_1][W_2]\dots[W_E][damaged][object].\nonumber
- L112 §3.3 Learning generic abnormality and normali `eq:total_loss`：L_{total} = L_{global} + \lambda \textstyle \sum_{M_l \in \mathcal{M}}L_{local}^{M_l},
- L117 §3.3 Learning generic abnormality and normali：\color{black} S^{(j,k)}_{n, M_l} = P(g_n, f^{m(j, k)}_{i, M_l}), \enspace \color{black} S^{(j,k)}_{a, M_l} = \…
- L128 §3.3 Learning generic abnormality and normali：t_2 &= T_1(t_1) \nonumber\\ &\ldots \nonumber\\ [r^i_{m+1}|_1^Q, t^i_{m+1}|_{Q+1}^P] &= T_m([q^i_m|_1^Q, t^i_m…
- L466 §C Detailed Analysis of DPAM `eq:std-att-2`：Attention( {Q}, {K}, {V})_i = softmax\left(\frac{q_iK^{\top}}{\sqrt{D}}\right)V =\frac{\sum\limits_{j=1}^n a_{…
- L469 §C Detailed Analysis of DPAM：a^{qq}_{ij} = e^{ \frac{q_i q_j^{\top}}{\sqrt{D}}}, \quad a^{kk}_{ij} = e^{ \frac{k_i k_j^{\top}}{\sqrt{D}}}, …

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

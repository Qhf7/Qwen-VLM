# 大纲 · MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L6 摘要
- L10–34 1 Introduction
- L35–40 2 Related Work
  - L36–38 2.1 Zero-Shot Anomaly Detection (ZSAD)
  - L39–40 2.2 Mixture of Experts (MoE)
- L41–76 3 Preliminaries
  - L42–45 3.1 Low-Rank Adaptation (LoRA)
  - L46–64 3.2 Mixture of Experts (MoE)
  - L65–76 3.3 Simplex Equiangular Tight Frame (ETF)
- L77–278 4 Methods
  - L78–91 4.1 Problem Definition
  - L92–94 4.2 Overview
  - L95–108 4.3 MoE-based Feature Adaptation
  - L109–152 4.4 Expert Specialization
  - L153–162 4.5 Patch Average Aggregation (PAA)
  - L163–197 4.6 Anomaly Map and Anomaly Score
  - L198–278 4.7 Loss Functions
- L279–392 5 Experiments
  - L280–318 5.1 Experimental Setup
  - L319–323 5.2 Comparison with State-of-the-art methods
  - L324–365 5.3 Visualization
  - L366–392 5.4 Ablation Study
- L393–396 6 Conclusion
- L397–423 Acknowledgement
- L424–428 Contents
- L429–479 7 Algorithm Implementation
- L480–673 8 Additional Information
  - L482–559 8.1 Data Descriptions
  - L560–582 8.2 Comparison Method Details
  - L583–586 8.3 Detailed Experimental Setup
  - L587–626 8.4 Text Prompt
  - L627–652 8.5 Details of Balance Loss
  - L653–673 8.6 Details of Focal Loss and Dice Loss
- L674–980 9 Additional Experimental Analysis
  - L677–904 9.1 Analysis of Model Configurations
  - L905–937 9.2 Analysis of Training on Medical Data
  - L938–980 9.3 Analysis of Computation Overhead
- L981–1052 10 Theoretical Grounding for MoE in ZSAD
- L1053–1097 11 Analysis of Expert Specialization
  - L1054–1069 11.1 Expert Utilization
  - L1070–1097 11.2 Expert Interpretability
- L1098–1109 12 Detailed Quantitative Results
  - L1101–1104 12.1 Performance Results for All Categories
  - L1105–1109 12.2 Inter-Expert Similarity
- L1110–1124 13 More Visualization Results
  - L1113–1116 13.1 Multi-layer and Multi-Scale Maps
  - L1117–1120 13.2 Grad-CAM and Patch Selection Map
  - L1121–1124 13.3 Final Anomaly Map
- L1125–1422 14 Limitations and Future Work

## 公式（22 个，按出现顺序；L 后面是所在小节）

- L50 §3.2 Mixture of Experts (MoE) `eq:Router`：\hat{R}_{n} = \begin{cases} \dfrac{{R}_{n}(x)} {\sum_{m \in {top}({R(x)},k)} {R}_{m}(x)}, & \text{if } n \in \…
- L60 §3.2 Mixture of Experts (MoE) `eq:moe_output`：\mathrm{x}' = \sum_{n=1}^{K} \hat{R}_{n}(x) {E}_n({x})
- L68 §3.3 Simplex Equiangular Tight Frame (ETF) `eq:ideal_etf`：{G}_{n,m}^\text{ideal} = {w}_n^\top {w}_m = \begin{cases} 1, & \text{if } n = m, \\ -\frac{1}{K-1}, & \text{if…
- L99 §4.3 MoE-based Feature Adaptation `eq:norm_expert_output`：F^l_{i,\text{norm}} = F^l_{i, \text{expert}} \cdot \frac{\|F^l_i\|_2}{\|F^l_{i, \text{expert}}\|_2 + \epsilon}
- L104 §4.3 MoE-based Feature Adaptation `eq:final_moe_output`：F^l_{i, \text{MoE}} = \lambda_{\text{MoE}} \cdot F^l_{i,\text{norm}} + (1 - \lambda_{\text{MoE}}) \cdot F^l_i
- L115 §4.4 Expert Specialization `eq:fofs`：& \hspace{0.8em} \begin{aligned} & {Q}_n = (\mathrm{qr}_{{Q}}(C_n))^T, \quad \text{where } C_n \sim \mathcal{N…
- L128 §4.4 Expert Specialization `eq:gram`：(G^l_i)_{n,m} = (\hat{e}^l_{i,n})^\top \hat{e}^l_{i,m}
- L134 §4.4 Expert Specialization `eq:etf_loss`：\begin{split} \mathcal{L}_{etf} &= \sum_{l} \left( \frac{1}{L} \sum_{i=1}^{L} \| G_i^{l} - G^{\mathrm{ideal}} …
- L147 §4.4 Expert Specialization `eq:balance_loss`：\mathcal{L}_{bal} = \sum_{l} \left[ \text{CV}^2\left(\sum_{i=1}^{L} R(F^l_i)\right) \right]
- L157 §4.5 Patch Average Aggregation (PAA) `eq:paa`：\hat{p}^l_{h,w} = \frac{1}{s^2} \sum_{u=-\lfloor s/2 \rfloor}^{\lfloor s/2 \rfloor} \sum_{v=-\lfloor s/2 \rflo…
- L169 §4.6 Anomaly Map and Anomaly Score `eq:proj_layer`：V^{l,s} = Proj_l(\hat{F}^{l,s}_{{PAA}})
- L174 §4.6 Anomaly Map and Anomaly Score `eq:anomaly_map_lr`：\hat{M}_{N/A}^{l,s} = \text{softmax}(\phi(\mathrm{cos}(V^{l,s}, T_{{N/A}})))
- L183 §4.6 Anomaly Map and Anomaly Score：\begin{aligned} &\hat{F}_{\mathrm{dw}} = GELU({DwConv1d}({LN}(\hat{F}^{f}_{\mathrm{PAA}}))) \\ &V_{\mathrm{ima…
- L191 §4.6 Anomaly Map and Anomaly Score `eq:anomaly_score`：\hat{S}_{N/A} = \mathrm{cos}\left(V_{\mathrm{image}}, [T_A, T_N]\right)
- L200 §4.7 Loss Functions `eq:loss_ac_seg`：\begin{aligned} \mathcal{L}_{seg} &= \sum_{l} \sum_{s} \Big[ \text{Focal}(\hat{M}^{l,s}_{N/A}, M) + \text{Dice…
- L210 §4.7 Loss Functions `eq:total_loss`：\mathcal{L}_{total} = \mathcal{L}_{seg} + \mathcal{L}_{ac} + \lambda_{etf} \mathcal{L}_{etf} + \lambda_{bal} \…
- L630 §8.5 Details of Balance Loss `eq:cv_calculation`：\text{CV}^2(B^l) = \frac{\sigma(B^l)^2}{\mu(B^l)^2 + \epsilon}, \quad \text{where } B^l = \sum_{i=1}^{L} R(F^l…
- L636 §8.5 Details of Balance Loss `eq:total_balance_loss`：\mathcal{L}_{bal} = \sum_{l} \left[ \text{CV}^2\left(\sum_{i=1}^{L} R(F^l_i)\right) \right] = \sum_{l} \left[ …
- L659 §8.6 Details of Focal Loss and Dice Loss `eq:focal_loss`：\mathcal{L}_{Focal} = -\frac{1}{N} \sum_{i=1}^{N} (1 - p_i)^\gamma \log(p_i)
- L668 §8.6 Details of Focal Loss and Dice Loss `eq:dice_loss`：\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i=1}^{N} y_i \hat{y}_i}{\sum_{i=1}^{N} y_i + \sum_{i=1}^{N} \hat{y}_i}
- L994 §10 Theoretical Grounding for MoE in ZSAD：\cos\!\left(\nabla_{\theta}\ell_{i},\,\nabla_{\theta}\ell_{j}\right) < 0.
- L1008 §10 Theoretical Grounding for MoE in ZSAD：\nabla_{B_{n}}\mathcal{L} = \sum_{i} \hat{R}_{n}(F_{i}^{l})\, \frac{\partial\ell_{i}}{\partial\mathbf{y}_{i,n}…

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

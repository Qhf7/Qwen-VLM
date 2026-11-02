# 大纲 · Learning Transferable Visual Models From Natural Language Supervision

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L36 摘要
- L41–72 1 Introduction and Motivating Work
- L73–122 2 Approach
  - L75–82 2.1 Natural Language Supervision
  - L83–88 2.2 Creating a Sufficiently Large Dataset
  - L89–110 2.3 Selecting an Efficient Pre-Training Method
  - L111–118 2.4 Choosing and Scaling a Model
  - L119–122 2.5 Training
- L123–340 3 Experiments
  - L125–257 3.1 Zero-Shot Transfer
    - L128–135 3.1.1 Motivation
    - L136–139 3.1.2 Using CLIP for Zero-Shot Transfer
    - L140–162 3.1.3 Initial Comparison to Visual N-Grams
    - L163–183 3.1.4 Prompt Engineering and Ensembling
    - L184–257 3.1.5 Analysis of Zero-Shot CLIP Performance
  - L258–287 3.2 Representation Learning
  - L288–340 3.3 Robustness to Natural Distribution Shift
- L341–391 4 Comparison to Human Performance
- L392–409 5 Data Overlap Analysis
- L410–429 6 Limitations
- L430–685 7 Broader Impacts
  - L569–617 7.1 Bias
  - L618–670 7.2 Surveillance
  - L671–685 7.3 Future Work
- L686–702 8 Related Work
- L703–713 9 Conclusion
    - L707–713 Acknowledgments
- L712 —— 附录 ——
- L714–1038 A Linear-probe evaluation
  - L718–774 A.1 Datasets
  - L775–791 A.2 Models
  - L792–797 A.3 Evaluation
  - L798–1038 A.4 Results
- L1039–1043 B Zero-Shot Prediction
- L1044–1052 C Duplicate Detector
- L1053–1127 D Dataset Ablation on YFCC100M
- L1128–1260 E Selected Task and Dataset Results
  - L1133–1162 E.1 Image and Text Retrieval
  - L1163–1198 E.2 Optical Character Recognition
  - L1199–1250 E.3 Action Recognition in Videos
  - L1251–1254 E.4 Geolocalization
  - L1255–1260 E.5 Robustness to Distribution Shift
- L1261–1313 F Model Hyperparameters

图表的页码、caption、在哪节被引用：见 `figures/index.md`。

# 固定经验 FPR 独立复核

结论：PASS，1038 项检查通过，0 项失败。全部统计在指定 AutoDL 实例 `autodl-container-132444827f-9c594038`、指定 Python 上完成；独立实现仅用 Python 标准库，未导入或执行原 runner。

只读取六个明确 allowlisted 的 stage12 旧输入和本次 fixedFPR 冻结产物。未运行 API、GPU 推理、训练、重拟合或调阈值；本审计未读取 stage13/confirmation。原协议、代码、阈值、结果、逐样本决策、原报告和 SHA manifest 均保持原字节。

| α | 分数 | normal降序rank | 阈值 | 校准 TP/FN/FP/TN | heldout TP/FN/FP/TN | heldout FPR（Wilson95%） | heldout Recall（Wilson95%） |
|---:|---|---:|---:|---|---|---|---|
| 5.00% | raw | 4 | 0.63 | 38/22/3/57 | 45/15/2/58 | 3.33% [0.92%, 11.36%] | 75.00% [62.77%, 84.22%] |
| 5.00% | calibrated | 4 | 0.6632409877622698 | 38/22/3/57 | 45/15/2/58 | 3.33% [0.92%, 11.36%] | 75.00% [62.77%, 84.22%] |
| 10.00% | raw | 7 | 0.58 | 40/20/5/55 | 49/11/3/57 | 5.00% [1.71%, 13.70%] | 81.67% [70.08%, 89.44%] |
| 10.00% | calibrated | 7 | 0.586503596776197 | 40/20/5/55 | 49/11/3/57 | 5.00% [1.71%, 13.70%] | 81.67% [70.08%, 89.44%] |

两个 split 各120来源（60 normal、60 anomaly），交集0；预测240唯一来源完全对齐。冻结的四个运行点均重新得到第4/7高 normal 分数，严格 `score > threshold`，等于阈值均判正常。α=5%校准 FP=3/60（5%）；α=10%校准 FP=5/60（8.33%），后者因阈值上有2个 normal 并列而低于6/60。

逐条核完480条 heldout 决策（120来源×2方法×2α，每α240条），标签、映射分数、阈值和布尔决策全部一致；math标量实现与原存储分数最大绝对差 `1.11e-16`，容差 `1e-14`。两个 split 的 raw/calibrated 决策在两α均0差异，5%检测集合均为10%检测集合子集。

calibration 的7140对中 ties=93，heldout 的7140对中 ties=87；两侧各62个不同分数，raw/calibrated 的顺序或并列差异均0，clip数均0。合并240来源的28680对也全部保持顺序/并列。

阈值冻结顺序：原 runner 第150行持久化 `thresholds.json`（独占创建、flush、fsync）后，第153行读取 heldout 标签；账本记录 `2026-11-10T20:25:17.923733+00:00` 冻结，`2026-11-10T20:25:17.939908+00:00` 保存，`2026-11-10T20:25:17.940025+00:00` 读取标签。此处仅审查静态代码及已记录时间的一致性；事后审计无法逆推出或独立证明不可见的实际读取时序。

解释限度：这是已使用过的开发 heldout 上的运行点诊断；normal n=60，一次误报对应1.67个百分点。经验FPR不保证总体FPR；Wilson95%是边际、未校正的描述性区间，校准侧区间不包括阈值选择不确定性。正单调校准仅改变分数尺度，本结果没有固定FPR召回率或排名提升。

独立审计 JSON SHA256：`874c7b38a54d34bd5ea9e6b940950edf31bd08686ebdaeb333c73fc3ab3d99bc`。
独立审计代码 SHA256：`3690a426b2f057f2346d16dca2cd48cb2ea06bd9ca49246f9651450b6bc8eebb`。
原 runner SHA256：`29d4a3f37a9464dbe35310bc75eeb4836ac3e4032124fe70ff34edacf2014ca9`。
原 manifest SHA256：`89f6cc15b687605aae2d90beba3ca5ebb1162d332128ff163fc26fd524efcae3`。
原冻结文件 SHA256：

- `PROTOCOL.md`：`851c59fa71f79997ee033effe1164c89743124c8814e81cd796599b25b0f2fb1`
- `thresholds.json`：`ffbc25d10d69ef8ef3516b0cde7441aebf92301008ba4996f732d5383300b6ce`
- `results.json`：`db603bbeafbbd471cdb62c3b6076a1446b6c4edc3c4f6bad51e059ff3618cc78`
- `heldout_decisions.json`：`8f649f62b1ace78419bcbfb0dd28eda04ef9d495022e3acd459c4365e1af1933`
- `REPORT.md`：`6a765be89c667d83abee3a8730bbcb8f3fe7ced9447f4c3ca00b45c003a94435`

# 可见缺陷空间命题支持诊断（v2）

状态：服务器执行完成；8个固定target全部首轮成功，started=8、failed=0、unknown=0。

## 分母与可行性

固定来源24张：12正常、12异常；严格几何规则仅2个异常来源合格，发送2×2条声明×有图/无图=8请求。独立来源只有2，不是8；未补挑其他图片、未放松任何规则。

完整排除原因（可多选，不能相加当来源数）：正常仅描述12；多连通6；无空且不相邻false格7；dominant<80%3；其他格>10%5；2像素margin失败8。margin检查实际要求全部GT像素在单格内，80/10为冗余核验。

## 逐来源结果

| source后8位 | 有图true | 有图false | 无图true | 无图false | 有图分离度 | 无图分离度 | 配对增量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| d62c0c01 | 0.590000 | 0.330000 | 0.340000 | 0.290000 | 0.260000 | 0.050000 | 0.210000 |
| 45ba6c4a | 0.770000 | 0.150000 | 0.310000 | 0.250000 | 0.620000 | 0.060000 | 0.560000 |

有图对mask空间proxy的固定0.5 accuracy=100.00%；平均有图分离度=0.440000；无图措辞/位置prior分离度=0.055000；平均配对增量=0.385000。

## 结论边界

- 这是mask-derived spatial proxy支持诊断；没有人工视觉事实gold，未验证缺陷类型、解释质量或成因／因果推理。
- 无图条件缺少视觉证据；true/false为mask proxy映射。无图分离度用于揭示措辞、位置或缺失证据的prior，不能称为视觉准确率。
- 只有2个此前看过的开发来源，经严格空间条件筛选；不计算看似稳定的泛化置信区间，不据此声称部署提升或普遍视觉语义能力。
- query复用online448 letterbox PNG；state中content_bounds明确原图内容，网格排除白padding。未向API提供mask、proxy label、GT统计、teacher map或旧分数。
- 原v1设计完整保留；v2修订、proxy标签、输入、请求hash和预测均有独立冻结链。输入与结果已独立重算核验。

## 成本与追溯

- 接受8／started8／失败0／unknown0；usage可见字段汇总：{"input_tokens": 6550, "output_tokens": 160}。
- 金额账单不可用，未估费用；完成上述8次请求后，未追加API或训练。
- contract / feasibility / proxy_labels / request_hashes / attempts / predictions / prediction_freeze / metrics / independent_input_audit / independent_result_audit均保留在本目录。

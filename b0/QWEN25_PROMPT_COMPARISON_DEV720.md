# Qwen2.5 提示词对照：P0 vs P3

## 结论

对 Qwen2.5-VL-3B，原来的 P0 提示词明显优于“先识别零件、再判断缺陷”的 P3。P3 虽然加入了零件类别描述，但在首 token 概率排序上造成退化，因此不应替换冻结的 B0+ P0。

## 主指标对比

两组实验使用同一 fold1 Dev720 清单：18 类、720 组、3,600 视角；同一 `Dataset448`、同一 direct 448 预处理、同一五视角 max 聚合。P0 的异常分数是 `P(Yes)`；P3 的异常分数是 `P(No)`，因为 P3 定义 `Yes=正常、No=缺陷`。

| Qwen2.5 提示词 | 异常分数 | Overall AUROC | Overall I-AUPR | Overall F1max | Macro AUROC | Macro I-AUPR | Macro F1max |
|---|---|---:|---:|---:|---:|---:|---:|
| P0，原冻结提示词 | `P(Yes)` | 0.6674 | 0.6697 | 0.6781 | 0.7208 | **0.7495** | 0.7380 |
| P3，先识别零件 | `P(No)` | 0.5651 | 0.5446 | 0.6744 | 0.6274 | **0.6517** | 0.7273 |
| P3 - P0 | | -0.1023 | -0.1251 | -0.0037 | -0.0934 | **-0.0979** | -0.0108 |

P3 的主指标宏 I-AUPR 比 P0 低约 9.79 个百分点；P3 的 bootstrap 95% CI 为
`[0.6198, 0.6888]`，P0 的对应区间为 `[0.7277, 0.7924]`，区间基本分离。

## 视角聚合对比

| 提示词 | 聚合 | Macro I-AUPR | Macro F1max | Macro AUROC |
|---|---|---:|---:|---:|
| P0 | max | 0.7495 | 0.7380 | 0.7208 |
| P0 | top2_mean | 0.7648 | 0.7565 | 0.7400 |
| P0 | mean | 0.7734 | 0.7623 | 0.7483 |
| P3 | max | 0.6517 | 0.7273 | 0.6274 |
| P3 | top2_mean | 0.7007 | 0.7260 | 0.6810 |
| P3 | mean | 0.7553 | 0.7506 | 0.7386 |

P3 只有在非主的 mean 聚合下接近 P0，但仍略低于 P0 的 mean；按照冻结协议，主结论必须使用 max，不能用 mean 替换。

## P3 生成回答审计

36 个平衡样本（每类一个正常和一个缺陷）均输出了 `component` 和 `region` 字段。

| 真实标签 | 首 token No | 首 token Yes | Markdown 开头 |
|---|---:|---:|---:|
| 正常，18 个 | 8 | 9 | 1 |
| 缺陷，18 个 | 13 | 4 | 1 |

与 Qwen3 P3 不同，Qwen2.5 P3 的贪心首 token 已有一定区分能力，但仍不完美；更重要的是，完整生成会偶尔输出 Markdown，而评分协议只使用首 token logits，因此不能以文本回答准确率替代概率指标。

## 为什么 P3 对 Qwen2.5 变差

1. P0 直接把任务压缩为“可见缺陷二分类”，模型只需在 Yes/No 之间比较；P3 加入零件识别、正常结构建模、类别和区域输出，增加了任务负担。
2. P3 把 `Yes` 定义为正常、`No` 定义为缺陷，改变了模型原来已适应的回答方向。即使我们正确使用 `P(No)`，模型的 token 概率仍会受到这个新语言上下文的影响。
3. P3 要求输出 component 描述，但数据中的类别是细粒度工业零件；模型容易输出泛化词，如 `industrial component`，这部分文本没有帮助缺陷排序，反而占用了提示词上下文。
4. P0 的异常分数分布虽然校准不完美，但正常/缺陷的排序已经被模型和提示词共同建立；P3 的分数集中在约 0.5-0.6，正常与缺陷仅小幅分离，导致整体 AUROC 和 I-AUPR 下降。

## 与 Qwen3 的参考

同一 P3 在 Qwen3-VL-2B 上的主 max 宏 I-AUPR 为 `0.8019`，高于 Qwen2.5 P3 的 `0.6517`。这不能说明 P3 对所有模型都更好，只说明提示词和模型的语言先验存在交互：P3 恰好缓解了 Qwen3 的 `P(Yes)≈1` 饱和，却破坏了 Qwen2.5 P0 已有的二分类排序。

## 文件

* 原 P0：`artifacts/b0plus_dev720/aggregation_metrics.csv`
* Qwen2.5 P3 原始视角：`artifacts/b0_cross/qwen25_vl_3b_component_first_p3_dev720/P3_COMPONENT_FIRST_views.csv`
* Qwen2.5 P3 分析：`artifacts/b0_cross/qwen25_vl_3b_component_first_p3_dev720/analysis/`
* Qwen3 P3：`artifacts/b0_cross/qwen3_vl_2b_component_first_p3_dev720/analysis/`

# 真实 VisA 与匹配语言骨干对照

真实 VisA 统一来源核查通过 8/16 项评测；未完成项继续标记待完成。
三臂三 seed 的训练、真实评测与初始化／顺序／缓存／checkpoint关联核查仍待全部完成，暂不据部分结果决定语言骨干优劣或扩大训练。
本轮是 16-step、小规模新增训练和已有 VisA decoder/PromptLearner 的迁移诊断；不支持从零训练、未见类别零样本或统计显著优越性结论。
发布的 PromptLearner 原本与 Vicuna 配套；共享权重仅控制初值，不能保证 Qwen 已获得同等语言接口适配。Qwen 较差时优先检查 bridge alignment 与软提示利用，再判断扩训；该 pilot 不代表骨干总体能力排名。
这 2,005 张官方测试来源经本轮查看后属于开发诊断集。后续配方选择不能再把它称为未见的最终确认。

## 数据与初始化

冻结协议状态：verified。真实评测 2005 张，正常 925、真实异常 1080，类别数 12。协议记录排除原生封存确认来源 133 张及既有 synthetic test 来源 24 张。
图像全图 bicubic 缩放到 224²；真实类别索引掩码先 >0 二值化，再 nearest 缩放。无正常参考图。此评测与旧 synthetic cache 的软 mask >0.3 分开报告。

初始化使用发布的 train_visa decoder 与 PromptLearner，SHA256 `a7e858586d9118c3322ff8ca3b60a8c7c521859ec8c1825e733df6666acab7d6`；本地下载与服务器字节已核对一致。它们不是随机头。语言投影在匹配实验中按 seed 单独初始化，同 seed 三臂共享。完整预训练来源清单尚未核验。
shared_initial_qwen42 使用保存的 seed42 投影；历史语言投影不保证能完整重建，因此该初始点对历史回答性能的解释有限；像素基线对应同一发布 decoder。

## 逐模型真实结果

指标单位为百分比。生成 Macro-F1 是正常／异常二标签宏平均；类别宏指标对 12 种产品等权。未完成或来源核查失败的行不填分数。

| 模型 | 状态/进度 | 生成 F1 | 异常召回 | 正常 FPR | 无效回答 | 热图 I-AUROC | 回答 I-AUROC | 类宏热图 I-AUROC | 类宏 P-AUROC | 类宏 P-AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_frozen42 | 完成 | 33.32 | 1.11 | 0.00 | 6.68 | 73.49 | 60.67 | 77.27 | 90.35 | 18.34 |
| historical_lora42 | 完成 | 42.91 | 13.06 | 5.84 | 0.00 | 70.76 | 58.45 | 76.30 | 90.31 | 18.39 |
| historical_frozen43 | 完成 | 32.77 | 1.11 | 0.11 | 0.00 | 73.16 | 59.33 | 78.23 | 91.62 | 19.94 |
| historical_lora43 | 完成 | 39.32 | 7.41 | 0.11 | 0.00 | 74.56 | 60.40 | 77.27 | 90.18 | 20.04 |
| historical_frozen44 | 完成 | 39.85 | 11.85 | 12.32 | 0.00 | 75.97 | 51.36 | 80.10 | 91.15 | 20.54 |
| historical_lora44 | 完成 | 31.57 | 0.00 | 0.00 | 0.00 | 74.18 | 46.05 | 78.41 | 91.39 | 20.28 |
| shared_initial_qwen42 | 完成 | 26.52 | 7.78 | 10.38 | 40.95 | 80.08 | 37.66 | 80.55 | 90.50 | 20.99 |
| matched_qwen_frozen42 | 完成 | 31.42 | 8.15 | 9.41 | 22.99 | 74.68 | 43.72 | 77.42 | 90.13 | 19.11 |
| matched_qwen_lora42 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_vicuna42 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_qwen_frozen43 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_qwen_lora43 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_vicuna43 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_qwen_frozen44 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_qwen_lora44 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_vicuna44 | missing（—/2005） | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

## 三 seed 汇总

均值 ± 样本标准差（ddof=1）；三 seed 均完整通过来源核查后才展示正式均值。初始模型为单次结果，见上表。

| 组别 | 生成 F1 | 异常召回 | 正常 FPR | 无效回答 | 热图 I-AUROC | 回答 I-AUROC | 类宏热图 I-AUROC | 类宏 P-AUROC | 类宏 P-AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_frozen | 35.32 ± 3.94 | 4.69 ± 6.20 | 4.14 ± 7.08 | 2.23 ± 3.86 | 74.21 ± 1.54 | 57.12 ± 5.04 | 78.53 ± 1.44 | 91.04 ± 0.64 | 19.61 ± 1.14 |
| historical_lora | 37.93 ± 5.79 | 6.82 ± 6.55 | 1.98 ± 3.34 | 0.00 ± 0.00 | 73.17 ± 2.09 | 54.97 ± 7.79 | 77.32 ± 1.06 | 90.63 ± 0.67 | 19.57 ± 1.03 |
| matched_qwen_frozen | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） | 待完成（已有1/3） |
| matched_qwen_lora | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_vicuna | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

历史组使用旧 max_tgt_len=64；新匹配组为128。两组不作为同一控制实验求训练提升。旧 teacher-forcing token accuracy 不等于自由回答准确率。

## 匹配预算与实际覆盖

训练缓存有48个source、96个variant，但16 steps×batch2只曝光32个variant。表中报告实际被用到的来源；不同 seed 的抽样覆盖可不同。相同 seed 的三臂共享显式顺序与完整 bridge，tokenizer及实际监督 token 数不同。

| 训练臂/seed | 状态 | steps | 实际source | 独立variant | 曝光 | max_tgt_len | 截断回答数 | 监督token总数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen_frozen_42 | complete | 16 | 27 | 32 | 32 | 128 | 0 | 504 |
| qwen_lora_42 | complete | 16 | 27 | 32 | 32 | 128 | 0 | 504 |
| vicuna_42 | complete | 16 | 27 | 32 | 32 | 128 | 0 | 608 |
| qwen_frozen_43 | pending | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| qwen_lora_43 | pending | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| vicuna_43 | pending | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| qwen_frozen_44 | pending | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| qwen_lora_44 | pending | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| vicuna_44 | pending | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

| seed | 三臂训练完成 | 顺序hash相同 | 初始文件hash相同 | 实际tensor指纹相同 | 缓存相同 | 控制核查通过 |
|---|---|---|---|---|---|---|
| 42 | True | True | True | True | None | False |
| 43 | False | None | None | None | None | False |
| 44 | False | None | None | None | None | False |

Vicuna 使用可用的 Vicuna-7B-v1.1，非原论文 v0。NF4精度、参考禁用、损失及图片曝光预算统一；相同图片／步数不代表相同 token 数或计算量。发布 PromptLearner 原本与 Vicuna 配套，共享4096维 bridge 初值不保证各语言模型得到相同程度的语义适配。当前省略正常参考图及 PandaGPT 混训，是简化迁移对照。

## 配对差异

下表是前者减后者的百分点差，按同 seed 及相同 source_id 配对；仅为描述统计，没有显著性检验。正常FPR和无效回答差值为负表示下降。匹配因果解释还需上表控制核查全部通过；历史三seed不保证同投影初值与顺序。

| 比较 | 生成F1差 | 异常召回差 | 正常FPR差 | 类宏P-AUROC差 | 类宏P-AP差 |
|---|---:|---:|---:|---:|---:|
| historical_lora_minus_frozen | 2.62 ± 9.56 | 2.13 ± 12.43 | -2.16 ± 9.27 | -0.41 ± 0.90 | -0.03 ± 0.20 |
| matched_qwen_lora_minus_frozen | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_qwen_frozen_minus_vicuna | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| matched_qwen_lora_minus_vicuna | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

## Synthetic 指标修订

Historical pixel AUROC/AP pooled layer maps as extra samples. Corrected metrics score the final four-layer mean map once per variant; not a retraining improvement.
v2 uses explicit image labels, otherwise conversation Yes/No, and only then threshold-mask fallback. Pixel masks remain >0.3; a semantically positive image can have no pixel above that threshold. v1 image metrics are superseded.
旧 Pixel AUROC/AP 不能继续标作最终热图成绩；GT随每层同步追加，问题是层间混合排序。四层平均后的修订值独立列出，不将口径改变写成训练收益。

| 模型 | 状态 | pooled P-AUROC | pooled P-AP | 类宏 P-AUROC | 类宏 P-AP |
|---|---|---:|---:|---:|---:|
| initial | complete | 99.42 | 82.68 | 96.86 | 68.12 |
| frozen42 | complete | 99.36 | 79.65 | 96.63 | 66.04 |
| lora42 | complete | 99.42 | 81.93 | 97.07 | 66.80 |
| frozen43 | complete | 99.38 | 80.46 | 96.34 | 64.65 |
| lora43 | complete | 99.42 | 83.54 | 96.94 | 68.73 |
| frozen44 | complete | 99.42 | 82.43 | 96.56 | 67.72 |
| lora44 | complete | 99.41 | 82.11 | 96.64 | 66.27 |
| frozen 三seed | complete | 99.39 ± 0.03 | 80.85 ± 1.43 | 96.51 ± 0.15 | 66.14 ± 1.54 |
| lora 三seed | complete | 99.42 ± 0.01 | 82.53 ± 0.88 | 96.88 ± 0.22 | 67.26 ± 1.30 |

## 来源与缩放审计

每个完整模型均核对2,005个逐图 source_id、标签、类别与冻结清单一致，并重算生成F1、图像AUROC/AP及其类别宏平均。像素指标核对逐类值与宏平均的一致性，本汇总器未重新读取 NPZ 计算像素排名。掩码缩放后前景为空的真实异常仍保留官方异常标签。

| 模型 | 逐图数 | 来源集合一致 | 异常空224mask | 指标重算最大误差 |
|---|---:|---|---:|---:|
| historical_frozen42 | 2005 | True | 0 | 2.22e-16 |
| historical_lora42 | 2005 | True | 0 | 3.33e-16 |
| historical_frozen43 | 2005 | True | 0 | 4.44e-16 |
| historical_lora43 | 2005 | True | 0 | 2.22e-16 |
| historical_frozen44 | 2005 | True | 0 | 3.33e-16 |
| historical_lora44 | 2005 | True | 0 | 2.22e-16 |
| shared_initial_qwen42 | 2005 | True | 0 | 2.22e-16 |
| matched_qwen_frozen42 | 2005 | True | 0 | 3.33e-16 |
| matched_qwen_lora42 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_vicuna42 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_qwen_frozen43 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_qwen_lora43 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_vicuna43 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_qwen_frozen44 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_qwen_lora44 | 待完成 | 待核验 | 待核验 | 待核验 |
| matched_vicuna44 | 待完成 | 待核验 | 待核验 | 待核验 |

## 后续选择

- 真实 VisA 统一来源核查通过 8/16 项评测；未完成项继续标记待完成。
- 三臂三 seed 的训练、真实评测与初始化／顺序／缓存／checkpoint关联核查仍待全部完成，暂不据部分结果决定语言骨干优劣或扩大训练。
- 本轮是 16-step、小规模新增训练和已有 VisA decoder/PromptLearner 的迁移诊断；不支持从零训练、未见类别零样本或统计显著优越性结论。
- 发布的 PromptLearner 原本与 Vicuna 配套；共享权重仅控制初值，不能保证 Qwen 已获得同等语言接口适配。Qwen 较差时优先检查 bridge alignment 与软提示利用，再判断扩训；该 pilot 不代表骨干总体能力排名。
- 这 2,005 张官方测试来源经本轮查看后属于开发诊断集。后续配方选择不能再把它称为未见的最终确认。

若匹配结果支持保留某个候选，下一轮先预先固定训练／验证协议，再扩大正常来源与更新预算；保留初始化基线、真实像素指标和自由回答指标。只有文本拟合变好时，应先检查定位软提示被利用的证据。未完成任务在 summary.json 中保留状态与缺失字段。

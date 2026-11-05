# 第二阶段：原生Qwen视觉与异常图接口

状态：complete。冻结官方Qwen3-VL-8B-Instruct BF16；map臂训练18 tokens的PromptLearner，1500 updates，microbatch1×accum4。实际为9个shared base与9个map-adaptive tokens。两臂使用同一原生图像输入、模板与val380；原生RGB先square224再由官方processor至256，teacher map224。选型仅用val AUROC，最优step500。

| 方案 | Selected step | AUROC | AP | 生成macro F1 | 生成invalid |
|---|---:|---:|---:|---:|---:|
| Frozen native Qwen | 0 | 0.626342 | 0.703170 | 0.218069 | 0.702632 |
| Native Qwen + map adapter | 500 | 0.866391 | 0.907728 | 0.812690 | 0.000000 |

## 输入干预

| 条件 | AUROC |
|---|---:|
| real_map | 0.866391 |
| zero_map | 0.514898 |
| donor_opposite_label_map | 0.146048 |
| spatial_shuffle_map | 0.849969 |

## 实现核验与解释

- 正式训练前与best重载后：官方/manual相同形状prefill、cached step、full replay logits及语言输入exact；MRoPE、DeepStack、mask与cache长度检查通过。保存重载最大首答案分数差0。
- Baseline与map的val380分数和自由生成完整，全部干预完整；包含最终complete.json。Smoke仅val24/gen8，不并入此表。
- Map内容影响明显，空间shuffle影响较小；donor使用val标签构造诊断，不能部署。zero_map是同一模型推理干预，不是独立训练的无map匹配臂。
- Baseline原prompt没有要求Yes/No首词，267条invalid均为非空解释性开头；生成改善同时包含canonical格式监督收益。详见NATIVE_PROTOCOL_AUDIT.md。
- 这是有监督custom60/20/20开发验证证据，单seed42；不做统计显著或未知test泛化声明。
- 第一阶段Qwen s42 AUROC0.958659，高于本路线0.866391；两者输入、参数、序列位置、模板和batch聚合不同，不能据此单独否定Qwen视觉表征。下一阶段用pre/post特征和独立定位头检验。

指标来源：native_baseline/val_metrics.json、native_map_s42/val_metrics.json；核验来源为各目录的run.json、train_complete.json、selected_auxiliary_cache_check.json与complete.json。完整输入文件SHA见all_summary.json。

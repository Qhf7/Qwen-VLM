# 原生Qwen ROI工具结果

状态：complete

只纳入推理complete且后验审计passed、输入报告SHA匹配的结果。未完成阶段仅列状态，不报告部分效能。

- tool_preflight：complete
- tool_val：complete

下表为完整开发验证；AUROC/AP是首答案logit排序，F1/recall/FPR是实际生成。

| 条件 | AUROC | AP | 生成F1 | 召回 | FPR | 实际调用率 | planner无效率 | 部署秒/图 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_tool | 0.751705 | 0.803359 | 0.583752 | 0.317308 | 0.029070 | 0.000000 | 0.000000 | 0.674717 |
| duplicate_original | 0.741531 | 0.781569 | 0.343056 | 0.028846 | 0.000000 | 0.000000 | 0.000000 | 0.999283 |
| center_crop | 0.719603 | 0.796224 | 0.631566 | 0.389423 | 0.034884 | 1.000000 | 0.000000 | 1.089189 |
| random_crop | 0.715955 | 0.772426 | 0.595174 | 0.379808 | 0.110465 | 1.000000 | 0.000000 | 1.095113 |
| model_tool | 0.741531 | 0.781569 | 0.343056 | 0.028846 | 0.000000 | 0.000000 | 0.000000 | 1.240340 |

| model_tool减控制（百分点） | ΔAUROC | Δ生成F1 | Δ召回 | ΔFPR |
|---|---:|---:|---:|---:|
| duplicate_original | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| center_crop | 2.192811 | -28.850982 | -36.057692 | -3.488372 |
| random_crop | 2.557581 | -25.211864 | -35.096154 | -11.046512 |
| no_tool | -1.017442 | -24.069679 | -28.846154 | -2.906977 |

模型真实裁剪次数：0/380；独特执行bbox：0。

planner action计数：{"answer": 380}。
planner error计数：{"valid": 380}。

No valid model crop was executed; the complete result does not demonstrate autonomous cropping.

解释边界：
- Original RGB only at inference; no category/label/mask/map inputs.
- Final classifier is a fresh prompt with returned crop evidence; this is a bounded one-call workflow.
- Four two-image arms match visual tokens, not text tokens or wall time; no_tool has fewer visual tokens.
- Planner is reused across controls; deployment_seconds excludes unnecessary planner cost for fixed controls.
- All-answer or all-invalid planners remain legitimate complete negative results.
- GT geometry was read post-inference and does not prove detail visibility or language attention.
- Native vision experiment is distinct from compressed bridge C/D LoRA; validation is development data.

全部配对逐图正确/错误迁移、正常/异常调用率、几何覆盖与成本分布详见JSON。

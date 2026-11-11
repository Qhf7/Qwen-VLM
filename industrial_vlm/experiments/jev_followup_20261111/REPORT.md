# Jev 固定候选区域复核结果

评价时间（UTC）：2026-11-10T17:51:30.638384+00:00；服务器：`autodl-container-132444827f-9c594038`。

## 评价边界

- 正式240个目标完整，预测/请求/图像/依赖一致性通过并冻结后，才读取当前120张清单内60张异常图的GT。
- 主比较为同一服务器处理图、同两资产预算和同区域问题的 ROI 与 duplicate 配对；阈值固定为 p≥0.5。
- 全部120张是已查看开发集；Teacher训练暴露按逐图split披露，未见子集也不是新的独立盲测。
- 历史0/1/4-shot是全图判断，渲染器、问题语义和输入预算存在差异，只作历史门控对照。
- 新服务器图与历史Jev图均非像素完全相同，不能把跨阶段分数变化归因于裁片；Qwen原生max_pixels524288也不同于Jev448。
- ROI概率的目标是候选区域异常，不能直接解释为整图概率校准，故不报告对应Brier/ECE。

## 完整120张开发集

| 门控 | 全60异常平均IoU | 全60异常IoU≥0.5 | 异常保留召回 | 全60正常出框率 |
|---|---:|---:|---:|---:|
| unconditional_teacher | 0.089529 | 6.67% | 100.00% | 100.00% |
| historical_jev_0shot | 0.076257 | 5.00% | 76.67% | 33.33% |
| historical_jev_1shot | 0.083513 | 6.67% | 86.67% | 36.67% |
| historical_jev_4shot | 0.078013 | 5.00% | 78.33% | 23.33% |
| roi | 0.069719 | 5.00% | 71.67% | 21.67% |
| duplicate | 0.068542 | 5.00% | 65.00% | 18.33% |
| historical_qwen_base | 0.058214 | 5.00% | 30.00% | 0.00% |
| historical_qwen_decision_decision | 0.080662 | 6.67% | 81.67% | 11.67% |

## 配对主差异：ROI − duplicate

| 指标 | 差值 | 配对bootstrap 95%区间 |
|---|---:|---:|
| anomaly_bbox_iou_mean_all | 0.001176 | [-0.000681, 0.003213] |
| anomaly_iou_at_05_all | 0.000000 | [0.000000, 0.000000] |
| anomaly_retained_recall | 0.066667 | [0.000000, 0.150000] |
| normal_positive_box_rate | 0.033333 | [-0.033333, 0.100000] |

差值采用原单位；率指标乘100后为百分点。固定类别×图像标签分层、2000次、seed42；未做多重比较校正。

## 分组与退化候选

| 分组 | 图数 | 异常 | 正常 | ROI平均IoU | duplicate平均IoU | ROI正常出框率 |
|---|---:|---:|---:|---:|---:|---:|
| all120 | 120 | 60 | 60 | 0.069719 | 0.068542 | 21.67% |
| teacher_unseen | 24 | 22 | 2 | 0.108560 | 0.109828 | 100.00% |
| teacher_training_exposed | 96 | 38 | 58 | 0.047232 | 0.044640 | 18.97% |
| full_image_candidate | 60 | 24 | 36 | 0.008540 | 0.008466 | 19.44% |
| nonfull_image_candidate | 60 | 36 | 24 | 0.110505 | 0.108593 | 25.00% |
| query_crop_bytes_identical | 60 | 24 | 36 | 0.008540 | 0.008466 | 19.44% |
| query_crop_bytes_different | 60 | 36 | 24 | 0.110505 | 0.108593 | 25.00% |

Teacher未见子集只有少量正常图时，误报率估计非常不稳定；配对bootstrap不能创造额外正常样本。
各组正常出框率与异常召回的描述性Wilson95区间保存在JSON；分层bootstrap若仅有单样本stratum可能退化，不能据此声称精确。
full_image_candidate与query_crop_bytes_identical分别报告；后者两臂像素相同，其差异用于检查服务随机性，不能宣称裁剪带来收益。
详细逐类、宏平均、各分组配对区间、候选面积和逐图错误归因见05_evaluation.json。

## 结果分析

ROI相对duplicate：全异常平均IoU变化 0.001176，异常保留召回变化 6.67%，正常出框率变化 3.33%。应共同判断定位收益、漏检和误报的取舍。
- anomaly_bbox_iou_mean_all：区间跨零，当前开发集证据不足以排除零差异。
- anomaly_iou_at_05_all：区间跨零，当前开发集证据不足以排除零差异。
- anomaly_retained_recall：区间跨零，当前开发集证据不足以排除零差异。
- normal_positive_box_rate：区间跨零，当前开发集证据不足以排除零差异。

## 执行与成本

接受判断 240 个；started 278 次；显式失败 38 次。
仅累加实际返回usage，失败或缺失usage保留未知；不估造货币费用。请求时延不是模型固有速度比较。

## 下一阶段依据

先审查nonfullbox的配对净收益及完整召回/误报权衡，再冻结工具路由和预算协议。
若未证明ROI净收益，保留该负结果，不通过调阈值或挑子集改写当前结论。校准/拒答须另立独立数据协议，confirmation仍保持隔离。

Qwen同source对照：audited_same_source_historical_controls；使用默认严格log_odds>0，不使用同120张调过的FPR5阈值。

## 完整性记录

- 正式预测SHA256：`76b3a9521aa3f46e875b2a14456406e7a280293cf164ed1090e1b8b34cbd8abb`
- GT目标记录SHA256：`9782d4d1e8d50f1e14fe983041bea31fd52811caf0278279cd35b0db56f4b548`
- 评价源码SHA256：`aad082c77ee00f9cc1fcd214fa50a06ee9cf7560469e8891fb6fd05274c18496`
- confirmation未读取；GT仅用于冻结后的评价，不进入候选、请求或阈值选择。

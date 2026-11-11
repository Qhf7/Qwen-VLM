# Jev 第一阶段独立最终审计

审计状态：**通过**。本审计未修改模型协议、原始预测、评测脚本或已有指标。

## 完整性与隔离

- 120张query、48张训练正常支持图；三臂各120，360条唯一成功结果。
- 168张原图重新核对历史SHA与大小，168张处理PNG重新核hash；来源、路径和字节交集均为0。
- 全360个payload按旧contract重新构造，model、UUID、图片、问题和请求SHA一致；未传入标签、GT框、面积、答案或含标签路径。
- 原始manifest与分离标签逐条一致；固定SHA排序的每类4图选择重新计算一致。
- runner与两次独立executor SHA符合冻结/修订记录，旧预测和attempt前缀未改写。

## 独立指标复算

AUROC采用正负样本两两比较并对同分计0.5；AP采用阈值分组非插值定义，独立于sklearn实现。与primary及supplement逐项一致（容差1e-12）。

| 支持图 | AUROC | AP | TP | FP | TN | FN | Macro-F1 | Recall | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 82.5417% | 85.9369% | 46 | 20 | 40 | 14 | 71.5957% | 76.6667% | 33.3333% |
| 1 | 87.1944% | 89.3371% | 52 | 22 | 38 | 8 | 74.6550% | 86.6667% | 36.6667% |
| 4 | 88.8472% | 90.8045% | 47 | 14 | 46 | 13 | 77.4984% | 78.3333% | 23.3333% |

## 调用与重试账目

494次started = 360条接受成功 + 133条显式失败 + 1条中断未知尝试；每目标最多5次started。没有接受成功落盘后再次提交该目标的记录。
未知中断和部分超时/传输失败可能已在上游完成，因此不能声称上游计算或计费绝无重复。并发/传输参数的两次修订保留完整链；模型、图像、支持、问题、请求身份未变。

## 可辩护结论

- 该固定开发协议下，正常参考图提高pooled排序点估计：1-shot相对0-shot AUROC +4.6528pp；4-shot相对0-shot +6.3056pp。
- 4-shot在阈值0.5下误报更低（14/60对20/60），但召回低于1-shot（47/60对52/60）；不能称所有指标单调随参考数提升。
- 排序改善不等于概率已充分校准；Brier改善但ECE并未单调改善。
- 类宏排序与pooled排序有区别，类别分数偏移可能影响统一阈值；每类仅10图。
- 结论限于这组120张已查看过的同域开发图、单一正常支持集和网关接受响应。

## 凭证检查

对实验及Qwen-VLM/docs文本产物扫描，live-key或长Bearer模式匹配文件数为0；不输出或记录匹配值。

## 限制

- 120 previously-viewed same-domain development images, not a new blind test.
- Fixed single support set and one accepted response per target; no support-selection or provider-repeat uncertainty.
- ModelGo logical response model echo is not independent verification of upstream weights/version.
- Only image-level decisions; no localization or pixel segmentation evaluated.
- Raw latency across shots/phases includes gateway, upload, concurrency, timeout and retry effects; not model compute benchmark.
- Probability calibration is empirical on this small balanced development sample, not a guarantee for deployment.
- Paired bootstrap resamples fixed query strata and does not include service/support variability or multiple-comparison correction.
- Jev image support is experimental; existing native Qwen results used different inputs/processing/training and are historical references, not strict controlled head-to-head.

逐项hash、tied-pair数量、独立Brier/ECE及attempt分布见final_audit.json。

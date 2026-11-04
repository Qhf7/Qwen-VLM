# 学生定位头接入ROI：执行进度

2026-11-04（Asia/Shanghai）。GPU实验、单图验证及CPU漏检诊断均已完成。

- map导出complete：380次Qwen视觉提取、GT/KD各380map，来源/输入/数组SHA与实际几何核验通过。
- 六例旧teacher ROI重放exact：Yes/No logits、score、生成IDs/文本及token预算完全相同。
- 两学生完整val380均complete；五臂metrics独立CPU精确重算、路径顺序/标签/预算/源码核验全部通过。
- GT：AUROC0.836161、生成macro F10.743272、TP118/208、FP6/172；KD：0.817993/0.743272、TP118/208、FP6/172。teacher：0.837265/0.765826、TP126/208、FP6/172。
- 独立CPU post-export像素评测：两head四指标完全等于原selected reload；GT仅评测读取。
- 任意单图Qwen-only CLI真实GPU两例complete，地图数组/ROI/logits/score/IDs/文本与批量exact；异常candle010保持漏检。
- 本轮没有新增test推理、没有额外GPU训练；开发val/单seed、分阶段监督系统，不能推断等价或泛化。

正式结果：STUDENT_ROI_RESULTS.md、summary.json。单图命令：INFERENCE_USAGE.md。一致性验证：demo/verification.json。

后验224几何诊断已complete：GT漏检90＝零覆盖30＋部分17＋全部43；KD为30＋21＋39；teacher为13＋18＋51。三臂共同漏检74。报告与逐图CSV/JSONL见failure_analysis_224/。缺陷保留不能等同于细节可见或语言理解。

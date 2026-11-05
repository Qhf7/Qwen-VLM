# 第四阶段：原生 Qwen ROI 与正常参考结果

更新日期：2026-11-04。状态：**complete**。`roi_full/run.json.status=complete`，五个条件均完成同一完整 val380（正常172、异常208）；pilot48 与单 query preflight 也已完成。

在本次固定协议下，监督 teacher 定位的 ROI 相比同两图预算的中心/随机裁剪，提高 image AUROC、AP、生成 F1 和召回；其误报率也有所上升。再加入正常参考图降低 FPR，但 AUROC、AP、生成 F1 和召回均下降。结果支持本协议中的 teacher 定位帮助，不能解释为纯 Qwen 零样本定位能力或参考图的普遍增益。

## 协议与指标

- 使用完整原生 Qwen3-VL-8B-Instruct，BF16、SDPA、冻结/eval、`torch.inference_mode()`；本阶段只推理，无新增训练或按结果调整阈值。硬件为 RTX 4090 D，seed42。
- 所有视图均为 448×448 RGB，processor `do_resize=False`；实际每图 grid `[1,28,28]`，patch16、merge2，对应196个语言模型 visual tokens。统一题干说明原图、近景裁剪和可选正常参考图的角色，并要求先回答 Yes/No。
- 裁剪面积固定为原图归一化面积1/9，边界处平移保持面积；从原 RGB 以浮点边界 bicubic 采样。center、random、teacher 三臂均为原图＋裁剪两图，分辨率、面积、题干及视觉预算相同。random 由 seed＋图片路径确定。
- image AUROC/AP 来自首答案 token 的原始 `logit(Yes)−logit(No)`，不使用生成后的文字。生成采用 greedy、单 beam、KV cache、最多32 tokens；生成指标按初始 Yes/No 独立计算，invalid 保留在分母。
- 下表数值均为0–1。AUROC/AP 为 `metrics.overall.scores`，生成 F1/recall/FPR/invalid 为 `metrics.overall.generated`。生成 F1 是汇总380张后的二分类 macro F1，**不是类别等权平均 F1**；其余类别等权指标另列。

## 完整 val380 结果

| 条件 | 图数 / visual tokens | 总 prompt tokens | Image AUROC | Image AP | 生成 macro F1 | 生成 recall | 生成 FPR | 生成 invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_only` | 1 / 196 | 272 | 0.753480 | 0.819075 | 0.446888 | 0.134615 | 0.000000 | 0.000000 |
| `original_center_crop` | 2 / 392 | 470 | 0.713174 | 0.800782 | 0.639874 | 0.384615 | 0.005814 | 0.000000 |
| `original_random_crop` | 2 / 392 | 470 | 0.726996 | 0.787239 | 0.548301 | 0.269231 | 0.029070 | 0.000000 |
| `original_teacher_roi` | 2 / 392 | 470 | **0.837265** | **0.884576** | **0.765826** | **0.605769** | 0.034884 | 0.000000 |
| `original_teacher_roi_reference` | 3 / 588 | 668 | 0.830417 | 0.875620 | 0.731000 | 0.538462 | 0.023256 | 0.000000 |

图数与 token 数均由实际 processor 输出记录，非按配置推测。五臂的 `visual_tokens_total_min_max` 与 `prompt_tokens_total_min_max` 分别为表中单一固定值；`actual_image_grids` 均为 `[1,28,28]`。所有生成均有可解析的初始 Yes/No；零 invalid 不等同于后续解释文字正确或有定位依据。

## 同预算 ROI 增量与参考图权衡

下表 Δ 为前者减后者，单位为**百分点**（原始0–1差值×100）；FPR 越低越好。

| 比较 | Δ AUROC | Δ AP | Δ 生成 F1 | Δ recall | Δ FPR | 预算变化 |
|---|---:|---:|---:|---:|---:|---|
| teacher ROI − center | +12.409 | +8.379 | +12.595 | +22.115 | +2.907 | 相同两图 / 392 visual tokens |
| teacher ROI − random | +11.027 | +9.734 | +21.753 | +33.654 | +0.581 | 相同两图 / 392 visual tokens |
| reference − teacher ROI | −0.685 | −0.896 | −3.483 | −6.731 | −1.163 | 增加一图 / 196 visual tokens；总 prompt +198 |

teacher ROI 对 center/random 的比较控制了裁剪面积、图数和视觉 token 预算：这里的主要改变是裁剪位置由外部 teacher 概率图选择。相对 center，异常检出数净增46、正常误报数净增5；相对 random，异常检出数净增70、误报净增1。提升同时包含更高召回与一定误报代价，不能只报告 F1/AUROC 而省略 FPR。

| 条件 | TP / 208异常 | FN / 208异常 | FP / 172正常 | TN / 172正常 |
|---|---:|---:|---:|---:|
| 原图 | 28 | 180 | 0 | 172 |
| 原图＋中心裁剪 | 80 | 128 | 1 | 171 |
| 原图＋随机裁剪 | 56 | 152 | 5 | 167 |
| 原图＋teacher ROI | 126 | 82 | 6 | 166 |
| 原图＋teacher ROI＋正常参考 | 112 | 96 | 4 | 168 |

上述计数由生成 recall×208、FPR×172复算并核对为整数。参考条件相对 teacher ROI 的异常检出数净减14、误报净减2，体现更保守的决策；这不是逐图片配对翻转分析，也不能解释为“减少两张误报必然导致十四张异常漏检”的因果关系。它在增加50% visual tokens和额外训练正常图信息后仍未提高本次汇总 AUROC/AP/F1，不能据此宣称参考图无条件有效。

中心裁剪相对原图的生成 F1 上升而 pooled AUROC 下降，说明固定 Yes/No 决策与连续分数的全局排序衡量不同性质，不应互相替代。

## 类别等权核对

以下为12类等权 `metrics.category_macro`，与主表 pooled 指标区分。

| 条件 | 类别宏平均 image AUROC | 类别宏平均 image AP | 类别宏平均生成二分类 F1 |
|---|---:|---:|---:|
| 原图 | 0.813004 | 0.852484 | 0.412334 |
| 原图＋中心裁剪 | 0.824377 | 0.861468 | 0.606352 |
| 原图＋随机裁剪 | 0.796791 | 0.830805 | 0.513756 |
| 原图＋teacher ROI | 0.886279 | 0.925596 | 0.759223 |
| 原图＋teacher ROI＋正常参考 | 0.868188 | 0.905509 | 0.720370 |

类别等权方向与完整集主结论一致：teacher ROI 在这三项中优于中心/随机裁剪，加入参考后均下降。未执行置信区间、显著性检验或跨随机 seed 重复，因此这里报告观察到的差值，不声称统计显著性或普遍性。

## 完成与执行证据

- preflight：`roi_preflight/run.json` 为 `complete`、`passed=true`，14项固定 checks 全通过，实际三图588 visual tokens、668 prompt tokens，观察到16个 cached generation steps，`performance_metrics_computed=false`。这是执行检查，不计入效果样本。
- 完整 val 五臂均为380张，score与生成均覆盖全部380。各臂耗时依次为217.08、281.68、266.06、272.72、316.50秒；五臂合计约22.57分钟。这些为该次执行时间，未做等计算量匹配。
- 完整 val 的 allocated peak 依次约16.425、16.479、16.479、16.479、16.536 GiB；三图参考条件 reserved peak 为16.656 GiB。单 query preflight allocated peak 约16.538 GiB，实际完成 score＋cached free generation。MiB/GiB 均按2的幂换算。
- GPU reserved allocator pool会跨条件延续；reserved peak不是各臂独立增量显存。上述实测说明该固定448、最多三图、32-token协议在本次设备上完成，不能外推任意分辨率、batch或输出长度。

## 解释边界

1. **pilot 不构成独立 holdout。** pilot48是完整val380的严格子集（全部48路径重叠），每类/标签轮转选样，正常24/异常24；full为正常172/异常208。两者同seed42、同题干、同448/1⁄9/32-token设置，不能合并为428个独立样本或将full称为独立确认集。pilot中reference pooled AUROC为0.838542、teacher ROI为0.835938，但full排序相反；完整val是本报告主结果。所有五臂在pilot与full均已运行，不从pilot挑最好一臂代替完整对照。
2. **native baseline 的输入协议不同。** `native_baseline/run.json.geometry_audit` 为requested224、processor实际256、64 visual tokens；`migration_native_train.py:286–304` 使用system消息和类别描述题干。ROI `original_only` 为448、196 visual tokens，使用 `migration_roi_eval.py:30–35,252–258` 的统一多图角色说明题干。两者同时改变prompt/system/category提示、分辨率与token预算，不能把跨两阶段差值仅归因ROI、原生适配或模型能力变化。本阶段的原图控制是表中的 `original_only`。
3. **ROI 使用监督 teacher。** teacher ROI 对验证图使用缓存的 BigDec 概率图argmax，未读取当前图的GT mask；但teacher来自 `supervised_custom_60_20_20` 的已训练 `dec_big_fixed_s42/best.pt`，不是纯Qwen或无外部监督的零样本定位。原始像素重新裁剪后仍由原生Qwen判断。
4. **参考图同时引入信息与预算。** 参考只从同类训练正常RGB中，由缓存ImageBind CLS cosine最近邻选择，并明确告知语言模型其为正常参考；不选验证/测试参考。第三图增加196 visual tokens，总prompt由470升668；这不是等预算参考消融，不能把其差值单独归因检索质量或Qwen原生表征。
5. **本报告只含验证集结论。** `select_validation_indices` 仅选择val；缓存完整性检查包含train/val/test条目，不表示这些test图被本阶段查询、评分或用作参考。当前代码不使用GT定位ROI；pilot标签只用于固定平衡选样及最终指标。历史测试集不应另称未查看确认集。
6. **尚未替换地图来源。** 本ROI路径读取共享BigDec teacher map；独立Qwen GT/KD SpatialHead没有接入这里。不能将本阶段ROI增量当作GT/KD head端到端语言收益或完全移除ImageBind/BigDec后的系统收益。

## 可追溯来源

本报告只写入当前文件；未运行SSH/GPU、未读取或修改checkpoint、未改动其他脚本或报告。主表与计数来自JSON汇总，不依赖训练日志或预测文本重评。

| 来源 | 主要字段 | SHA256 |
|---|---|---|
| `roi_full/run.json` | `status/conditions/n/results.*.metrics.overall/results.*.metrics.category_macro`、实际grid/token/内存 | `10e02b111c120ddc5b1cff5a0826b957ff37b4980dfb06d5e80b66be24250495` |
| `roi_pilot/run.json` | `status/config/selected_paths/results` | `d32b3ccd849de3f8625491ad1e469506585e118abb9a6a0a77ca87c26c16155a` |
| `roi_preflight/run.json` | `passed/checks/forward_trace/generation_cached_steps/memory` | `d322a2b590aaeb3a7a96973033138382893eb27ad4ef3fd36c93962afa1cc2f3` |
| `native_baseline/run.json` | `geometry_audit`，跨阶段输入边界 | `3c2785143b67367f212d1184c8022f3d19fff182acfc925013a06987d8c589e8` |
| `teacher_cache.json` | `training_protocol/decoder/decoder_sha256/geometry` | `700ceec1dac513b68f52e8daaf768631bd9a2787e3e1cbe1fb284849aa82cc42` |

共享teacher decoder SHA256：`cddfa2df9ad63659f65aa3a798566e52d608ace43a1b37704484db5bbfc5617d`。ROI固定验证路径清单SHA256：`73b294cd66e143719cf263a8142d984085d15215bbf9688a293418aa507e1af2`。源代码协议锚点为 `/Users/qiaohongfei/anomalygpt_remote/files/migration_roi_eval.py`；更细的预执行审查见 `ROI_PROTOCOL_AUDIT.md`，其当时尚未实测的项目以本报告中的完整执行证据更新理解。

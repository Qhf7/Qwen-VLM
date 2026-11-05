# 学生地图替换 ROI 选择器：验证结果

日期：2026-11-03T17:04:25.996583+00:00。状态：**complete**。

同一val380：正常172、异常208、12类，seed42。五臂均为原图＋ROI两图、448/图、196×2=392 visual tokens、总prompt470、normalized面积1/9；同QUESTION、BF16原生Qwen/SDPA、greedy最多32 tokens。

## 完整集指标

AUROC/AP来自LM首答案raw logit(Yes)−logit(No)，其他指标来自独立free generation。生成macro F1是完整集二分类macro F1，类别等权另列。pending不以0补齐。

| 条件 | 状态 / 已见行数 | Image AUROC | Image AP | 生成 F1 | recall | FPR | invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧中心裁剪 | complete / 380 | 0.713174 | 0.800782 | 0.639874 | 0.384615 | 0.005814 | 0.000000 |
| 旧随机裁剪 | complete / 380 | 0.726996 | 0.787239 | 0.548301 | 0.269231 | 0.029070 | 0.000000 |
| 旧BigDec teacher ROI | complete / 380 | 0.837265 | 0.884576 | 0.765826 | 0.605769 | 0.034884 | 0.000000 |
| Qwen pre448 GT ROI | complete / 380 | 0.836161 | 0.881745 | 0.743272 | 0.567308 | 0.034884 | 0.000000 |
| Qwen pre448 GT＋KD ROI | complete / 380 | 0.817993 | 0.870490 | 0.743272 | 0.567308 | 0.034884 | 0.000000 |

## 类别等权

| 条件 | Image AUROC | Image AP | 生成 F1 | recall | FPR | invalid |
|---|---:|---:|---:|---:|---:|---:|
| 旧中心裁剪 | 0.824377 | 0.861468 | 0.606352 | 0.388226 | 0.004630 | 0.000000 |
| 旧随机裁剪 | 0.796791 | 0.830805 | 0.513756 | 0.265027 | 0.023148 | 0.000000 |
| 旧BigDec teacher ROI | 0.886279 | 0.925596 | 0.759223 | 0.609759 | 0.030093 | 0.000000 |
| Qwen pre448 GT ROI | 0.881553 | 0.909170 | 0.733869 | 0.574141 | 0.027515 | 0.000000 |
| Qwen pre448 GT＋KD ROI | 0.861338 | 0.904421 | 0.732130 | 0.570886 | 0.032407 | 0.000000 |

## 差值与取舍

Δ=前者−后者，百分点；FPR/invalid越低越好。summary另保存类别等权差值。

| 比较 | Δ AUROC | Δ AP | Δ F1 | Δ recall | Δ FPR | Δ invalid |
|---|---:|---:|---:|---:|---:|---:|
| pre_gt_minus_teacher | -0.110409 | -0.283100 | -2.255418 | -3.846154 | +0.000000 | +0.000000 |
| pre_gt_minus_center | +12.298748 | +8.096310 | +10.339822 | +18.269231 | +2.906977 | +0.000000 |
| pre_gt_minus_random | +10.916536 | +9.450626 | +19.497152 | +29.807692 | +0.581395 | +0.000000 |
| pre_kd_minus_teacher | -1.927270 | -1.408581 | -2.255418 | -3.846154 | +0.000000 | +0.000000 |
| pre_kd_minus_center | +10.481887 | +6.970829 | +10.339822 | +18.269231 | +2.906977 | +0.000000 |
| pre_kd_minus_random | +9.099676 | +8.325145 | +19.497152 | +29.807692 | +0.581395 | +0.000000 |
| pre_kd_minus_pre_gt | -1.816860 | -1.125481 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

KD−GT的AUROC/AP/F1为-1.817/-1.125/+0.000个百分点；recall/FPR变化+0.000/+0.000个百分点。排序、召回和误报需共同解释，不能将单项变化概括为普遍改善。

## 生成计数

FN含异常invalid；正常invalid单列而不充当TN。

| 条件 | TP / 208 | FN / 208 | FP / 172 | TN / 172 | 正常invalid | 异常invalid |
|---|---:|---:|---:|---:|---:|---:|
| 旧中心裁剪 | 80 | 128 | 1 | 171 | 0 | 0 |
| 旧随机裁剪 | 56 | 152 | 5 | 167 | 0 | 0 |
| 旧BigDec teacher ROI | 126 | 82 | 6 | 166 | 0 | 0 |
| Qwen pre448 GT ROI | 118 | 90 | 6 | 166 | 0 | 0 |
| Qwen pre448 GT＋KD ROI | 118 | 90 | 6 | 166 | 0 | 0 |

## 真正逐图配对变化

按完整path匹配teacher与student生成决策，正常/异常分别统计。same_decision含both-invalid；invalid另列属于重叠描述，不能与各列相加当互斥分组。完整0/1/invalid转移矩阵保存在summary，以下正确/错误变化来自逐图片配对，非仅净TP/FP差。

| Student vs teacher | 真实标签 | n | old正确→new错 | old错→new正确 | 相同决策 | old invalid | new invalid | 双invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pre_gt_vs_teacher | normal | 172 | 4 | 4 | 164 | 0 | 0 | 0 |
| pre_gt_vs_teacher | anomaly | 208 | 16 | 8 | 184 | 0 | 0 | 0 |
| pre_kd_vs_teacher | normal | 172 | 3 | 3 | 166 | 0 | 0 | 0 |
| pre_kd_vs_teacher | anomaly | 208 | 14 | 6 | 188 | 0 | 0 | 0 |

## 地图与crop离线诊断

GT mask仅在预测完成后的独立CPU评估脚本中使用，不进入student infer或ROI中心选择；地图指标不是LM分数。

| Head | Pixel AP macro | Pixel AUROC macro | Weighted pixel AUROC | Map-p99 image AUROC | Peak hit / 208 | Mean GT前景保留比例 | Crop含任意GT / 208 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre_gt | 0.336404 | 0.952411 | 0.948290 | 0.804590 | 110 | 0.775447 | 177 |
| pre_kd | 0.327643 | 0.953943 | 0.949534 | 0.816805 | 108 | 0.758584 | 176 |

导出地图四项指标与原selected checkpoint reload记录完全相等。crop覆盖定义为224二值GT像素中心落入normalized框，属于后验诊断，不能当作推理使用GT或语言模型确实关注缺陷的证据。

## 计时与地图成本

秒。condition wall和prepare/score/generate不含预计算地图；score与generate各有独立prefill。旧teacher图也已预计算，其离线地图成本本次未测。

| 条件 | condition wall | prepare合计 | score合计 | generate合计 | allocated peak GiB |
|---|---:|---:|---:|---:|---:|
| 旧中心裁剪 | 281.678601 | 17.693965 | 36.699716 | 226.958660 | 16.478703 |
| 旧随机裁剪 | 266.064301 | 17.765713 | 36.718150 | 211.236264 | 16.478703 |
| 旧BigDec teacher ROI | 272.723140 | 17.506475 | 36.690088 | 218.187608 | 16.478703 |
| Qwen pre448 GT ROI | 282.591613 | 19.344165 | 49.584953 | 213.343281 | 16.479070 |
| Qwen pre448 GT＋KD ROI | 287.335443 | 18.855412 | 48.648876 | 219.484441 | 16.479070 |

380次visual提取由两个head共享，成本只计一次；不是两臂各自独立部署延迟。

| 地图阶段 | 秒 |
|---|---:|
| elapsed_seconds | 50.823646 |
| visual_load_seconds | 5.714847 |
| export_map_seconds | 37.617782 |
| inference_seconds_total | 32.075369 |
| serialization_seconds_total | 5.260806 |

两图/token预算不匹配地图产生的额外计算。map job wall含CPU审计/hash/加载；export/inference/serialization互有包含，不能全部相加。地图读取校验在ROI条件计时前完成；未记录的在线端到端延迟不作推算。

## 完成与证据

五臂各380条path唯一且原序一致，label/category/split/cache_index与旧manifest逐项一致，actual tokens/geometry一致；独立CPU重算metrics与run完整metric对象exact对照，另验map payload/head/input/source身份及执行源码SHA。

固定旧teacher框replay：6/6通过，logits/score/generated IDs/text/tokens exact；只覆盖三类×正常/异常各一例。

## 解释边界

- 学生组合推理用Qwen视觉塔＋SpatialHead定位，再把原RGB和ROI交给完整Qwen。学生infer不加载ImageBind/BigDec或teacher概率图，旧teacher框只在独立连续性replay使用；历史teacher对照仍使用BigDec地图。
- GT head训练用过异常GT mask；KD head还用过GT与weight0.5的训练teacher概率监督。推理移除外部teacher依赖不等于无teacher训练、无监督或纯原生Qwen零样本定位。
- 两head由同开发val选择best1200，这里再用val380报告ROI；无新增test查询、无独立holdout、单seed42，无显著性或跨seed稳定性证明。
- 汇总器只读JSON/JSONL及源码字节，不重开原图/.npy/checkpoint，不SSH/GPU。原图/数组/权重字节核验由exporter/evaluator执行并记录，此处核验其manifest身份链。

## 来源 SHA256

| 本地来源 | SHA256 |
|---|---|
| roi_full/run.json | 10e02b111c120ddc5b1cff5a0826b957ff37b4980dfb06d5e80b66be24250495 |
| student_roi_20261104/val_manifest.json | f703048d89ad5de71681aeed7e6a0b97fb7658705895ed01dbbe667dc4d14c46 |
| student_roi_20261104/maps/manifest.json | f6a0495670ea828b7130659a8009a407a8b07bc713946857f4ea736b61e94a8f |
| student_roi_20261104/roi_full/run.json | 859fac3bbdf9e6d7525b64235ba3f4529b99c518d40d0e561d367eea2dad47fe |
| student_roi_20261104/center_predictions.jsonl | 130384aaff731172cf1ccff92acde3aaba7345e36ae25981b483d3ff5b6baee1 |
| student_roi_20261104/random_predictions.jsonl | 7c232368b2d0ea053787859e04f5d92eabdae953415da81e781f21b11e55a3a4 |
| student_roi_20261104/teacher_roi_predictions.jsonl | 821f6a467bd53fce7389e66ca387348e17b4e0e7a1faf29c676022d46e4a7f77 |
| student_roi_20261104/roi_full/pre_gt_predictions.jsonl | c4f53074834ea0a6a6916f3c737f939d35b8a5a291d659503b7102900f1e117c |
| student_roi_20261104/roi_full/pre_kd_predictions.jsonl | a3aa21a17193f925393a5ed5d7ab51a419254167b3698e8d33c45dce9bb9e235 |
| student_roi_20261104/code/source_manifest.json | 580e107f9866c1c45ad0e536f949f559557ff13dd382e8dc7522d13d6de001a8 |
| student_roi_20261104/code/migration_dense_infer.py | 9cd9411b5353be646a36c86be9c0f6608aae36aaf07e359046101b6ca9175de1 |
| student_roi_20261104/code/migration_student_roi.py | 563eae5654166b0aec3ded2cd517ad517df63cf66759aacfe32bafb62df3b600 |
| student_roi_20261104/code/migration_student_map_metrics.py | d6fd323fc72150dc2d25204c1970da4f90f59b587d59669fcf1fc4585688d81f |
| student_roi_20261104/code/migration_infer_student_roi.py | f500c9c7cbf27c2ce28e923babf7e9c50eeed90401f331c101a25b6137b91ee4 |
| student_roi_20261104/code/migration_summarize_student_roi.py | b1f308f26b8c5b6a60e1aa3c69791112067d94e103e88f8ecbb7d241d30d1933 |
| student_roi_20261104/code/migration_plot_student_roi.py | fd4b945fc4b34a266c49a82918b25f5ee65dabae9e2886246aaa93e378a2c5d0 |
| student_roi_20261104/code/migration_verify_student_demo.py | e51585ea3ad8608882f8ecfe915725e1830fd4049b85c484e9de26715816755b |
| student_roi_20261104/code/migration_student_failure_analysis.py | 35e36dd0084457063e505534e9180984f4ae3ac6be2ef7eeee78ba312751918b |
| code/migration_metrics.py | 18780dda38a176fbe438f54c65d5f25bd6879f74153ac1eebd7e1816d00f08d0 |
| student_roi_20261104/map_metrics.json | 294d5ca39a8b355e1f75f204ec04e75ff1333968a965a4483d3dfe73f7e49b55 |
| dense_pre_gt_s42/best_validation_reloaded.json | 7e816c777d4ca19b36efaf4be7fd16d7a4e99c92a513d4b0ececb84334c710e9 |
| dense_pre_kd_s42/best_validation_reloaded.json | b540a778df5ea687debb64b0d0dcb33251cc98a465c4d5502b069017ad8af248 |
| summarizer | b1f308f26b8c5b6a60e1aa3c69791112067d94e103e88f8ecbb7d241d30d1933 |
| metrics_implementation | 18780dda38a176fbe438f54c65d5f25bd6879f74153ac1eebd7e1816d00f08d0 |

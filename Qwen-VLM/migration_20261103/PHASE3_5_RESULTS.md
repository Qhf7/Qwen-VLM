# 第三、五阶段：冻结视觉 memory 与监督 SpatialHead / KD 结果

更新日期：2026-11-04。状态：**complete**。本报告覆盖同一验证集上的 normal-memory 12格、GT SpatialHead 5格与GT＋KD 2格；均为seed42探索性矩阵。当前 `all_summary.json.status=complete`，第三、五阶段结果均已落盘。

核心结果：冻结Qwen normal-memory的pixel AP仍低；监督GT head明显提升地图质量，但跨特征源比较同时存在输入宽度、参数量、精度和预处理差异。对同feature的448配对，KD使pre pixel AP下降、post pixel AP上升；pre的image-map AUROC上升、post下降。当前证据不足以称为稳定蒸馏成功。

## 指标与共同数据边界

- 主表pixel AP/AUROC均为12个类别等权平均；weighted pixel AUROC另列。image-map AUROC是每图地图第99百分位数作为异常分数后的完整val图像AUROC，不是语言模型Yes/No分数。
- 完整val为380张、12类，正常172、异常208；统一地图评估为224×224，共19,066,880像素，其中前景105,436（0.552980%），背景18,961,444。pixel类别平均与完整图像池化AUROC不可互相替代。
- GT训练属于 `supervised_custom_60_20_20`；val用于checkpoint选择，属于开发验证结果。本报告没有未查看测试集结果、跨seed重复或显著性检验。

## 冻结 normal-memory：12格

同类别训练正常图为support，seed42固定选择，16-shot为64-shot的前缀子集；正常support共12类×64条不重复路径，head参数为0，teacher未使用。Qwen视觉encoder BF16，bank与cosine FP32。RGB直接bicubic缩放到指定正方形，processor `do_resize=False`，实际patch grid必须与请求分辨率一致。

| 输入尺寸 | 特征 | 正常support/类 | Pixel AP macro | Pixel AUROC macro | Pixel AUROC weighted | Image-map AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 224 | pre | 16 | 0.028558 | 0.679571 | 0.626720 | 0.652784 |
| 224 | pre | 64 | 0.032635 | 0.689270 | 0.636008 | 0.653986 |
| 224 | post | 16 | 0.022756 | 0.722113 | 0.698662 | 0.701504 |
| 224 | post | 64 | 0.025178 | 0.751966 | 0.730667 | 0.703181 |
| 448 | pre | 16 | 0.018271 | 0.714047 | 0.645723 | 0.684537 |
| 448 | pre | 64 | 0.023452 | 0.726718 | 0.656181 | 0.699799 |
| 448 | post | 16 | 0.035138 | 0.790202 | 0.721451 | 0.737561 |
| 448 | post | 64 | 0.039597 | 0.809481 | 0.742374 | 0.740385 |
| 672 | pre | 16 | 0.014272 | 0.714658 | 0.645144 | 0.678472 |
| 672 | pre | 64 | 0.017092 | 0.728787 | 0.657617 | 0.690463 |
| 672 | post | 16 | 0.032227 | 0.799706 | 0.732600 | 0.727722 |
| 672 | post | 64 | 0.039527 | 0.814005 | 0.749379 | 0.762774 |

64-shot在本矩阵各同尺寸/同feature配对的上述四项指标均高于16-shot。最高pixel AP是post448/64（0.039597），最高pixel AUROC与image-map AUROC是post672/64（0.814005 / 0.762774）；三项最优并不在同一格。post从448升至672时，AP略降而AUROC上升；增大输入并未带来所有指标单调提升。输入升高增加query patch数量和计算；bank按每图最多14×14位置取样。pre三种尺寸均保留196 patches/图，post224为49、post448/672均为196，因此bank容量在448到672之间保持不变，不能把全部差值解释为bank容量。

实际几何核对如下；DeepStack分支虽被extractor观察记录，本次pre/post指标并未融合其多层输出。

| 请求/实际RGB | Pre patch grid / 通道 | Post grid / 通道 |
|---|---|---|
| 224 / 224 | 14×14 / 1152 | 7×7 / 4096 |
| 448 / 448 | 28×28 / 1152 | 14×14 / 4096 |
| 672 / 672 | 42×42 / 1152 | 21×21 / 4096 |

## GT SpatialHead：5格

以下结果均来自各run的 `best_validation_reloaded.json.validation_reloaded`，按完整val的macro pixel AP选best。

| 特征 | 输入尺寸 | Head参数 | Best step / 1200 | Pixel AP macro | Pixel AUROC macro | Pixel AUROC weighted | Image-map AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| imagebind | 224 | 697,345 | 1000 | 0.390244 | 0.974284 | 0.971631 | 0.839893 |
| pre | 224 | 664,577 | 1200 | 0.217631 | 0.907415 | 0.906863 | 0.793269 |
| post | 224 | 1,418,241 | 1200 | 0.181908 | 0.935581 | 0.953129 | 0.809034 |
| pre | 448 | 664,577 | 1200 | 0.336404 | 0.952411 | 0.948290 | 0.804590 |
| post | 448 | 1,418,241 | 1200 | 0.295317 | 0.966730 | 0.961398 | 0.810040 |

同为GT，Qwen从224升至448时，pre/post的pixel AP分别由0.217631升至0.336404、0.181908升至0.295317；对应pixel AUROC也提高。两个尺寸的pre AP都高于post，而post的pixel AUROC和image-map AUROC都高于pre。ImageBind单层224在本预算的pixel AP/AUROC仍高于四个Qwen GT格，但这不是严格等容量、等精度的纯encoder比较。

## 同feature GT＋KD：2格

KD真实权重为0.5，损失为GT focal(gamma=2)＋per-image smoothed Dice，再加 `0.5 × L1(sigmoid(logits), train_teacher_probability)`。两项KD均保留GT监督。

| 448特征 | 训练臂 | Best step / 1200 | Pixel AP macro | Pixel AUROC macro | Pixel AUROC weighted | Image-map AUROC |
|---|---|---:|---:|---:|---:|---:|
| pre | GT | 1200 | 0.336404 | 0.952411 | 0.948290 | 0.804590 |
| pre | GT＋KD(0.5) | 1200 | 0.327643 | 0.953943 | 0.949534 | 0.816805 |
| post | GT | 1200 | 0.295317 | 0.966730 | 0.961398 | 0.810040 |
| post | GT＋KD(0.5) | 1200 | 0.302921 | 0.967149 | 0.964455 | 0.801068 |

Δ为KD减GT，单位为百分点（0–1差值×100）：

| 448配对 | Δ Pixel AP | Δ Pixel AUROC | Δ weighted Pixel AUROC | Δ Image-map AUROC |
|---|---:|---:|---:|---:|
| pre | -0.876133 | +0.153170 | +0.124376 | +1.221489 |
| post | +0.760332 | +0.041858 | +0.305654 | -0.897250 |

pre的pixel AP下降约0.876个百分点，虽pixel AUROC与image-map AUROC上升，也不能按AP选择协议称其改善。post的pixel AP上升约0.760个百分点，但image-map AUROC下降约0.897个百分点。两项来自单seed、同开发val，尚无跨seed稳定性或独立确认集证据；结论是指标/feature依赖的收益与代价。按12类逐项比较，pre和post各只有4类pixel AP上升、8类下降；post的macro AP小幅上升也不能解释为多数类别改善。pre pixel AUROC为6类升/6类降，post为8类升/4类降。

## 独立配对、best与reload核验

本报告重新读取七个dense run的计划、配置、六次validation记录、completion与reloaded JSON，并逐格对照summary。CPU JSON核验全部通过：

1. **计划内容与两种SHA。** 七份 `training_plan.json` 的完整JSON内容逐项相等，1200个batch、每batch4张，合计4800次样本暴露，2799条唯一图片路径；正常2386次、异常2414次。采样以0.5概率选正常/异常池再有放回抽图，是近似平衡，不能称恰好各2400。对 `batches` 用排序key、无空格JSON重新求SHA256，与全部run记录的 `train_plan_sha256` 和计划内 `sha256` 相同：`c057c9cf45a2fc1048ca042742da08d145a705ac83dc66b47507a96fd717821b`。七份计划文件的原始字节SHA另为 `2ba005ad91a71e65cbaaeb7363e7baaa57dc90bf6e8b5b9f3011fa50f8cc15bd`，不能混淆内容摘要与文件字节摘要。
2. **训练预算。** 七run均seed42、1200 optimizer steps、batch4、lr0.001、每200步完整val，无梯度累积；AdamW weight_decay=1e-4，cosine T_max1200、eta_min=lr×1e-3=1e-6，clip_grad_norm=1。训练计划、数据暴露、训练步数相同；KD额外读teacher图并增加损失，不能称总计算量严格相同。蒸馏项在GT臂权重0、KD臂0.5；GT focal＋Dice两臂均保留。预算单位是optimizer step，不是完整训练集epoch。
3. **matched_run_key独立重算。** 根据run中的完整compatibility、seed/steps/batch/lr/plan SHA/val间隔及源码的optimizer/schedule构造同一对象，重新求SHA，七项均等于各run和completion的记录。同feature/同448的GT/KD `compatibility`、head参数、数值精度、trainer脚本SHA均完全一致。源码在实例化head之前重置同seed；head参数还按各卷积weight/bias独立重算一致。配对key如下；不同feature使用不同key。

| 同448配对 | 完整 matched_run_key | Head参数 / 配置 | 结果 |
|---|---|---|---|
| pre GT ↔ pre KD | `e79f550e88f58e8a51829608f716fb0049e057cea8beadbb73f6b5ccb827dacf` | 664,577；in1152 / width256 / LN无affine / [56,112,224] | 内容、配置、预算均匹配 |
| post GT ↔ post KD | `6074c2f308100ca8b327c7bdf6e3ea6763a29e8f95a07e317ace8fb647f346ff` | 1,418,241；in4096 / width256 / LN无affine / [56,112,224] | 内容、配置、预算均匹配 |

4. **完整验证与选择。** 每run `validation.jsonl` 都恰好包含step200/400/600/800/1000/1200，所有42条记录与七份original/reloaded验证均为380张、12/12类、208异常、无undefined pixel AUROC类别；重新按各类指标平均复算macro AP/AUROC一致。按六个AP取最大，得到ImageBind1000、全部Qwen1200，与completion及reload的selected step一致；源码严格 `>` 更新best，平局保留最早。
5. **reload。** 七run均 `passed=true`，四项指标逐项 `metric_abs_errors=0`，`reload_max_abs_metric_error=0`，与best历史记录和original/reloaded值精确相等（执行门限1e-10）。这是重新载入checkpoint后的指标一致性证据；本报告不把指标完全相等升级为逐像素tensor逐位相等，亦未重新执行模型。

## Teacher train-only消费审计

两个KD的 `run.json.distillation.consumer_audit` 完全一致：`exported_manifest=true`、`payload_verified=true`、`required_images=2799`、`planned_map_files_verified=2799`，training-plan SHA与上面的计划相同。teacher是seed42、`supervised_custom_60_20_20` 下的 `dec_big_fixed_s42/best.pt`；decoder SHA为 `cddfa2df9ad63659f65aa3a798566e52d608ace43a1b37704484db5bbfc5617d`，manifest文件SHA记录为 `d3aaf18bdd8997e990e0934150d63f68a5ab9934e69f8840ca4718cc9a338612`。

对应源码快照 `code/migration_dense_train.py` 的 `TeacherCache` 检查manifest为train split、224直接正方形坐标、当前pipeline，maps key全部属于训练集且与validation不重叠；要求显式 `test_or_validation_maps_exported=false`，校验export payload SHA、计划SHA、decoder与源cache provenance、map元数据，并在建GPU模型前逐个读取2799个计划map，检查文件字节/大小、float32 [224,224]、数组SHA、有限且处于[0,1]。`evaluate`不接收TeacherCache，输出 `teacher_used=false`；七run的全部validation与reload记录也均为false。

源teacher cache SHA为 `90b003550655617f93837e4c0336819b8c6dc0d341d215e9cd3949930eee59f4`，cache manifest SHA为 `349942973b9207c575be98790d074226845c4631b9f3c5a410adfd1165cd22e2`。二者与本地 `teacher_cache.json` 的记录一致。原始teacher cache包含多个split，KD实际消费的是按训练计划导出的train-only子集，不能因源cache有val/test就直接认定KD消费了它们，也不能把train-only消费等同于teacher从未用开发val选择权重。

本地结果包没有 `teacher_train_subset/manifest.json` 及2799个`.npy`；以上是已落盘consumer审计＋匹配SHA源码的独立一致性检查，没有在本地重新hash这些概率文件或独立检查其数值内容。此限制不影响已完成状态，也不应描述为本次审计重新逐文件复验了远端payload。

## 解释边界

- 三种特征都只用单个最终层级，未做多层融合。pre是最终visual block送入main merger前的1152维patch特征；post是main merger后的4096维特征；ImageBind是block31输出、post_transformer_layer之前的1280维patch特征，去除CLS。
- SpatialHead共享总体结构（无affine的per-patch LayerNorm，1×1/C→256、3×3/256→128、3×3/128→64、3×3/64→1，GELU及56→112→224空间上采样），但首层输入不同使参数量分别为ImageBind697,345、pre664,577、post1,418,241。ImageBind encoder/cache FP16，Qwen encoder/cache BF16，head均FP32；预处理与原始网格也不同。
- 所有Qwen GT/KD的best均在1200步预算末点，不能称为已收敛或已充分搜索；ImageBind GT选best1000。验证选择与验证报告复用同一val，不能当作独立测试表现。
- 历史BigDec teacher验证pixel AP0.435522、pixel AUROC0.975934、image-map AUROC0.814317来自独立四层解码器，不是本报告的单层ImageBind SpatialHead。它不能充当严格配对的单层baseline。
- Qwen GT/KD dense推理只需冻结Qwen视觉encoder＋SpatialHead，不需要BigDec teacher在线推理；本次没有把这些地图接入LM map tokens或ROI。既有LM/ROI仍使用其各自记录的BigDec地图，不能把dense指标增益解释为端到端语言收益或完整系统已替换。

## 可追溯来源与审计范围

本报告只新增当前文件，读取已有本地JSON与源码快照；未SSH、未运行GPU、未重训/重评、未读取checkpoint内容、未修改训练代码或其他报告。`all_summary.json` 的本次引用快照为 `2026-11-03T16:08:38.240761+00:00`（北京时间2026-11-04 00:08:38），其完成状态与本报告所核原始结果一致。

- memory：`visual_probe/val.json` 的 `resolutions[*].metrics`、实际几何、memory、data.support_paths、complete；12格均逐项对照summary，且复算类别macro均值。
- dense：七run目录的 `run.json`、`training_plan.json`、`validation.jsonl`、`train_complete.json`、`best_validation_reloaded.json`；主表以reload后的结果为准。
- KD消费：两个run的 `distillation.consumer_audit`，teacher来源另对照 `teacher_cache.json`；执行逻辑来自同SHA的 `code/migration_dense_train.py`。
- 特征与bank：`code/migration_visual_probe.py`；ROI未接入边界另见 `PHASE4_RESULTS.md`，本报告没有借用ROI效果作为dense收益。

| 关键来源 | SHA256（原始文件字节） |
|---|---|
| `visual_probe/val.json` | `fece1c62f5ff4535e60c1f24018d00b94c3442c74669757154924e129faa5694` |
| `code/migration_dense_train.py` | `a05336a7e94aea1fff50bec009eebc1ee1fbba327c6da5da3a140415df5a7a3d` |
| `code/migration_visual_probe.py` | `af80a1b9d9fd585cacb25b1e7a63c483f73a15bb1811ac3ed3492fb19c33aea6` |
| `teacher_cache.json` | `700ceec1dac513b68f52e8daaf768631bd9a2787e3e1cbe1fb284849aa82cc42` |
| `teacher_val.json` | `bb3c19a1767fb013c148794a7835fc8e39a0865f27c2d0154360154ff8981b57` |
| `dense_pre_gt_s42/run.json` | `c96a259aa96bdc6b989ccd867a2e4ff393ff3ebf84b45f13e72506a01cb7ca53` |
| `dense_pre_kd_s42/run.json` | `02228eb13bb6a2a47554789da6dbc84f16221c7f5263819e00f6cb9b4fc4212c` |
| `dense_post_gt_s42/run.json` | `151c8c7c00cbd6ae1e94dd590f1715c354429f5741e8464e156e5b65129a09e9` |
| `dense_post_kd_s42/run.json` | `313c61997a8db23a77eeaed01f3eddac82c94a8402f4682f1cbcc0f36fd5e715` |

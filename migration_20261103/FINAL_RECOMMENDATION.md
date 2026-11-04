# 本轮迁移结论与后续主线

更新：2026-11-04（Asia/Shanghai）。原五路线矩阵已完成；其后又完成Qwen学生定位图→native Qwen ROI回答的完整val380匹配集成，以及无需teacher/cache/类别输入的任意单图入口。续篇完整结果见[STUDENT_ROI_RESULTS.md](student_roi_20261104/STUDENT_ROI_RESULTS.md)，命令和真实GPU验证见[单图使用说明](student_roi_20261104/INFERENCE_USAGE.md)。原all_summary.json只表示第一阶段预定矩阵，续篇另以student_roi_20261104/summary.json/status.json记录。

**当前同时保留两条明确路线。** 追求现有检测表现及较充分重复证据时，以“ImageBind＋BigDec＋Qwen语言骨干”为基线；追求推理时完整移除ImageBind/BigDec时，以“Qwen pre448 GT SpatialHead→预测ROI→native Qwen”为已跑通基线。后一条AUROC0.836161，与同协议teacher ROI0.837265接近，但生成F1为0.743272，低于teacher0.765826；单seed开发集结果不能证明性能等价。两个路线使用不同输入接口和预算，不直接排总榜。

## 按原定顺序得到什么

所有表值是0–1。图像级与像素级指标分开；不同路线的输入、prompt和预算不同，表格不构成编码器的单变量总排名。

| 路线 | 已完成范围 | 关键结果 | 当前判断 |
|---|---|---|---|
| 1. 只替换语言骨干 | Vicuna/Qwen × seed42/43/44；每组1500 updates、完整val380和生成 | Image AUROC均值±样本SD：0.889684±0.013285 → 0.925779±0.032698；配对均值增量+0.036095 | 当前支持最充分的迁移方向；保留外部视觉与异常图 |
| 2. Qwen原生视觉＋异常图token | baseline与18map-token适配分支；官方接口exact、训练和重载通过 | baseline AUROC0.626342，map分支0.866391；map生成F1=0.812690、invalid0 | 原生接口已通，外部map有用；尚未超过路线1的完整接口方案 |
| 3. Qwen特征＋独立定位 | normal-memory12格、监督GT定位5格 | Qwen pre448 Pixel AP0.336404、post448 0.295317；ImageBind224同类单层head对照0.390244 | 监督head能读出异常定位信息；当前纯Qwen定位仍落后该IB对照 |
| 4. 原图＋ROI／正常参考 | preflight、pilot48、full380五条件全部完成 | Image AUROC：原图0.753480、中心0.713174、随机0.726996、teacher ROI0.837265、ROI＋ref0.830417 | teacher定位裁剪有效；本协议下正常参考使总体检测更保守且未改善主指标 |
| 5. 教师蒸馏 | pre448/post448，GT与GT＋0.5KD两对匹配；每组1200updates | pre AP0.336404→0.327643；post AP0.295317→0.302921 | 固定L1蒸馏收益不一致；没有形成稳定替代teacher的证据 |

路线1的AUROC在三个seed都提高，但生成F1在seed43略低；Qwen的三seed波动也更大。因此可说“本轮配对检测排序改善”，不能说“各项始终更好或更稳定”。三seed是描述性证据，未做显著性声明。

## 对“替换LLM”和“整个替换”的回答

### 替换LLM：本轮已经实现并验证

实际路径是：RGB square224 → 冻结ImageBind的CLS与四层patch特征 → 修复BigDec异常概率图 → 已训练CLS投影和PromptLearner → 冻结BF16 Qwen语言模型。Qwen在此不使用自身视觉tower；训练更新的是桥接模块，未训练Qwen骨干。两臂的初始化、teacher/cache、数据计划与预算在同seed内匹配，故语言替换结论的控制较清楚。

新单图CLI直接接收图片路径和类别，重算teacher特征，再strict加载所选桥接权重。没有要求label、GT mask、split或预计算cache。正常candle样例分数−1.75，生成“No, there is no anomaly in the image.”；按已有val顺序固定选择的第一张异常candle样例分数2.625，生成“Yes, there is an anomaly in the image.”。两张都成功写出原概率图和固定标尺热图。这只证明入口可运行，不是额外泛化评价。命令和实际产物见INFERENCE_USAGE.md与route1_demo/。

### 整个替换：已跑通不依赖外部teacher的推理链

2026-11-04续篇把Qwen pre448 GT与GT＋KD两个head接为native Qwen ROI选择器，完成同val380、同prompt、同2×448图像／392 visual tokens／470 prompt tokens／1⁄9裁剪面积的匹配对照。GT的AUROC/AP/生成F1为0.836161/0.881745/0.743272；KD为0.817993/0.870490/0.743272。两者异常召回118/208、误报6/172，teacher为126/208与6/172。KD没有改善这些总体生成指标，且排序指标下降；相同总体计数也不表示逐图预测相同。

学生推理只加载Qwen视觉、独立SpatialHead和完整原生Qwen，不读取ImageBind、BigDec、teacher缓存或GT mask；独立旧teacher框六例重放仅核验协议连续性，不进入学生ROI来源。任意单图CLI不需label/category/split或历史control，已在固定正常0784与异常010真实GPU验证：地图数组/ROI/logits/score/token IDs/文本与批量路径精确一致。异常010仍被回答No，已保留漏检。

这已经连接了移除外部teacher所需的推理部件，但仍是分阶段组合：定位head用GT监督训练；KD版本还用过teacher训练概率；原生Qwen未进行定位—语言联合训练。当前不是无监督或零样本AnomalyGPT，也没有证明与原系统性能等价。定位图导出使用Qwen视觉一次，语言路径又处理原图和裁剪；同语言token预算不等于整条管线计算成本相同。两次加载的单图验证约20.2–20.4秒、allocated峰值16.479GiB，不能当作常驻服务吞吐。

保存结果的CPU后验诊断已经完成：GT组90个FN中30个ROI零覆盖、17个部分覆盖、43个全部覆盖GT224前景；三臂共同漏74例。不能把所有失败解释为裁剪完全偏离，也不能把有GT覆盖当作可见/理解的证明。详见[后验失败诊断](student_roi_20261104/failure_analysis_224/REPORT.md)。下一轮先固定裁剪位置/尺度和识别接口的控制实验，再决定多区域ROI或map-token桥接；不直接扩大固定KD训练。

## 为何暂不扩大另外几个模块

- **不先扩normal-memory。** 当前最高Pixel AP仅0.039597，说明所测最终单层余弦记忆库不够强；监督Qwen head的AP明显更高。该结果不等于Qwen所有层都没有异常信息。
- **不默认启用reference。** full380中加入第三图后AUROC下降0.006848、生成F1下降0.034826、召回下降0.067308；FPR改善0.011628。它是一个保守性权衡，还增加196 visual tokens与训练正常图信息。
- **不把固定KD当默认增强。** pre的AP下降0.008761、post上升0.007603；pre的map图像AUROC上升0.012215、post反降0.008972。像素与图像指标方向不同，且只有seed42。
- **不把最后步best当作收敛。** 四格Qwen GT与两格KD的best均在1200步边界。若继续定位主线，先做预先规定预算的同feature多seed/训练时长对照，再判断多层融合或视觉LoRA；本轮未额外启动这些训练。

## 完成检查与证据边界

- 统一修复后的visa-square-binary-sklearn-v1：mask二值化、图像/mask几何、ImageBind patch布局、strict decoder和sklearn指标。教师val380 Pixel AUROC0.975934、Pixel AP0.435522，属于四层BigDec参考，不与单层IB head当同配置。
- native初始和最佳重载权重均通过官方/manual相同形状exact gate；MRoPE、DeepStack、mask、cache长度已检查。官方自身BF16 cached/full形状变化的数值差仅作诊断。
- 七格dense完成1200 updates、六次val380，best按macro Pixel AP选；四个重载指标最大误差均0。两对GT/KD计划、matched_run_key、compatibility、head与预算匹配。2799个独立训练路径、4800次曝光，实际正常2386/异常2414。
- ROI center/random/teacher同两图、同1/9面积、同448、同392 visual tokens；teacher框来自预测峰值，不用GT mask。reference第三图增加计算量并使用同类train-normal检索。native baseline和ROI original的prompt、system、分辨率、token数不同。
- 数据采用含真实异常训练的自定义监督60/20/20：train9985、val380；历史dev test456已用于早期开发。本轮所有新性能判断来自val，未新增test推理，不能称未知测试集泛化或官方normal-only基准。
- baseline生成invalid主要来自非空解释性前缀，题干未强制Yes/No首词；learned map的生成收益包含格式监督。invalid不等同于视觉判断完全失败；空间shuffle降幅较小也不能证明精确空间理解。

全部主要数字、状态与输入SHA见[ALL_RESULTS.md](ALL_RESULTS.md)及[all_summary.json](all_summary.json)。详细路线报告见[PHASE1_RESULTS.md](PHASE1_RESULTS.md)、[PHASE2_RESULTS.md](PHASE2_RESULTS.md)、[PHASE3_5_RESULTS.md](PHASE3_5_RESULTS.md)、[PHASE4_RESULTS.md](PHASE4_RESULTS.md)。实际代码快照和SHA见code/source_manifest.json。

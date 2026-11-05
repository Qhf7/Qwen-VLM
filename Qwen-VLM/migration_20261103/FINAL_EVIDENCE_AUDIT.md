# 最终证据一致性审计（本地快照）

## 最终完成核验（2026-11-04）

远程预定序列于2026-11-04 00:07:35 +0800全部complete；本地all_summary也为complete。ROI full380五臂、dense GT5格和KD2格均已验收，两对GT/KD合同匹配且best重载四指标误差0。KD的详细独立检查见PHASE3_5_RESULTS.md。路线1正常/异常两张单图入口已实际运行并保存结果，属于执行验证，不增加泛化证据。下列11月3日/4日较早快照保留其审计时间，不能当当前队列状态。

审计日期：2026-11-03。基准汇总快照：`all_summary.json.created_at_utc=2026-11-03T15:53:20.586627+00:00`。范围为已完成路线1/2、GT定位头5格及ROI协议/当前状态；只读现有本地JSON和源码，不SSH、不载入checkpoint、不运行GPU、不重跑实验。其他汇总文件由父代理维护，本文不修改它们。

**2026-11-04后续纠正：**父代理已将`PROGRESS.md`重写为与真实JSON一致的自动快照，并在summary中按相同内容去重parity报告、用`mirrored_paths`保留镜像来源；`NATIVE_PROTOCOL_AUDIT.md`也已更新为承认native两臂完整完成并引用`PHASE2_RESULTS.md`。最新核对的`all_summary.json.created_at_utc=2026-11-03T16:00:30.176403+00:00`（北京时间2026-11-04 00:00:30），`roi_full/run.json.status=complete`，full val380五条件已齐；KD仍在运行，当前结果合同继续为ongoing。下文列出的旧PROGRESS状态、重复parity行及ROI full未齐等问题仅描述11月3日原审计快照，已被本段纠正，不应再当作当前未解决问题；其余协议与证据边界保留。

**原审计快照结论（2026-11-03）：路线1/2和GT5格的主要数值、选择步数与已记录重载检查互相一致，未发现影响这些已完成数值的矛盾。当时最明显的问题是PROGRESS混合旧快照与新正文，以及重复列出同一native parity报告；ROI全量与KD在该快照中尚不能标为完成。**

## 1. 已完成结果核验

### 路线1：语言骨干替换

六组 `lm_{vicuna,qwen}_s{42,43,44}` 都是val380，四种map条件各380分数、自由生成380；best均step1500，`train_complete.json.reload_max_abs_score_error=0`。每seed的初始化、teacher/cache/manifest、采样顺序、参数量、训练配置通过`phase1_summary.json.pairs`匹配；`cross_seed_contract_mismatches=[]`。直接用六个`val_metrics.json`重算mean及sample SD，AUROC/AP/generated macro F1均与汇总一致：

| 指标 | Vicuna mean ± sample SD | Qwen mean ± sample SD |
|---|---:|---:|
| image AUROC | 0.889684 ± 0.013285 | 0.925779 ± 0.032698 |
| image AP | 0.919143 ± 0.012122 | 0.941595 ± 0.026385 |
| generated macro F1 | 0.816122 ± 0.005694 | 0.845672 ± 0.041649 |

这是三个预定seed在共同监督缓存/桥接方案下的开发验证结果；不是原生视觉比较，也不是统计显著性或未知来源泛化证明。三seed标准差仅描述这三个seed的波动。证据：六组`run.json`、`train_complete.json`、`val_metrics.json`，及`phase1_summary.json.aggregate/pairs`。

### 路线2：native baseline与learned map

| 方案 | best step | val AUROC | val AP | generated macro F1 | invalid |
|---|---:|---:|---:|---:|---:|
| native baseline | 0 | 0.626342 | 0.703170 | 0.218069 | 267/380 |
| native map s42 | 500 | 0.866391 | 0.907728 | 0.812690 | 0/380 |

两组`complete.json.passed=true`，分数与生成覆盖380，重载误差0；初始与selected的12项exact检查全通过，prefill/cached/full-replay官方与manual的同形状logits最大差均0。baseline和map的teacher/cache/manifest、官方model manifest/source及native helper SHA一致。map正式6000图像曝光的采样顺序SHA也与Phase1 seed42一致（`69cd5e37…be641f`），但微批次聚合不同。

map条件AUROC依次为real 0.866391、zero 0.514898、反标签donor 0.146048、shuffle 0.849969，与`PHASE2_RESULTS.md`一致。不能把内容敏感性、仅0.016422的shuffle降幅解释成精确空间定位能力；zero是训练后输入干预，donor用val标签构造。gate检查仅一张当前分支样本（cache index9985），不是对380图每张执行exact gate；独立aux报告另有4例。

关键边界保留于`NATIVE_PROTOCOL_AUDIT.md`：9 shared base +9 adaptive；107,416,504参数的监督适配加map相对于0参数baseline是整体分支增量。baseline题干未要求Yes/No首词，267条invalid均是非空说明性输出，其中78条首段明确否定异常；生成指标的改善包含canonical格式适配，invalid不能等同视觉检测完全失败。当前Qwen两类canonical答案均11 tokens，模板CE下降不等于首词排序改善。

### GT独立定位头5格

| 特征 / 实际输入 | selected step | Pixel AUROC macro | Pixel AP macro | map-p99 image AUROC |
|---|---:|---:|---:|---:|
| ImageBind /224 | 1000 | 0.974284 | 0.390244 | 0.839893 |
| Qwen pre /224 | 1200 | 0.907415 | 0.217631 | 0.793269 |
| Qwen post /224 | 1200 | 0.935581 | 0.181908 | 0.809034 |
| Qwen pre /448 | 1200 | 0.952411 | 0.336404 | 0.804590 |
| Qwen post /448 | 1200 | 0.966730 | 0.295317 | 0.810040 |

逐格核对`run.json`、`training_plan.json`、`validation.jsonl`、`best_validation_reloaded.json`和`train_complete.json`：1200updates/B4/lr1e-3/每200步val380/seed42/GT-only；五组训练plan内容完全相同，重算plan SHA等于报告`c057c9cf…7821b`。4800次曝光包含2799个独立训练路径，实际正常2386/异常2414，属于随机近似平衡而非强制各2400。每格selected为已记录六次评估的最大macro pixel AP，重载四项指标误差均0；validation无teacher，`test_set_used=false`，12类别/380图完整。

这组结果支持“在既定单层特征和该监督SpatialHead/预算下，Qwen特征可读出定位信息，448改善两种Qwen特征的AP；同分辨率pre的AP高于post，但post的AUROC更高”。不能笼统称post全面更好，也不能用memory弱结果否定视觉信息。四格Qwen best均在最后1200步，说明当前预算内最佳点在边界，不等于已收敛。

跨源不是等参数/等精度：ImageBind/pre/post head分别697,345/664,577/1,418,241参数，tower/cache分别FP16/BF16/BF16，head均FP32；ImageBind最后block、Qwen pre/post具有不同网格/通道/预训练来源。共享head结构与训练plan不消除这些差异。共同teacher的pixel AP0.435522来自另一个四层融合BigDec及其独立训练/val选择，不能拿它与本轮单层ImageBind AP0.390244当同配置复现误差。代码：`migration_dense_train.py:181–208,218–254,829–858,895–978`；`dec_big.py:24–57,271–279`。

## 2. 文档与真实JSON的差异

| 位置 | 问题 | 以什么为准 |
|---|---|---|
| `PROGRESS.md:10–12` | route2/3仍ongoing、ROI pilot仍ongoing；同文19–21行又已说native/memory完成 | native两目录complete、`visual_probe/val.json.complete=true`、`roi_pilot/run.json.status=complete` |
| `PROGRESS.md:21` | 仍说GT“正在运行，先ImageBind224” | 当前GT五目录均有passed best重载和train_complete |
| `PROGRESS.md:64` | “当前本地快照尚无native main完整val380结果”已过时 | native baseline/map均完整val380/gen380 |
| `ALL_RESULTS.md:32–34` | native-v1出现两行，容易误读成两次独立parity实验 | `native_parity.json`与`native_parity/report.json`逐字节同SHA：`e9757dfd…24aec`，为同一3例报告镜像 |
| ROI full表已有数值但总状态ongoing | 这是阶段部分条件完成，不是错误；本次快照仅original/center/random三项 | `roi_full/run.json.status=running`，仍缺teacher与reference两项；不能把pilot五项抄成full |
| `NATIVE_PROTOCOL_AUDIT.md`开头 | 旧审计的“map主运行结果尚待汇总”已滞后 | 本文及`PHASE2_RESULTS.md`已核对完成值；其协议分析仍有效 |

`ALL_RESULTS.md`主要已完成表值在六位小数精度与源JSON一致，整体ongoing也合理（ROI full/KD未齐）。本次快照`all_summary.json.input_files`所有仍存在本地文件的SHA均匹配；未发现“报告值被后续改写”证据。最终交付前刷新一次父代理维护的ALL_RESULTS/PROGRESS即可，历史快照不要混用为当前状态。

## 3. ROI协议及待完成边界

`roi_preflight/run.json`是单query接口验证：14项checks全true，三张448图、每图196视觉token、总588，实际cached步骤16；不包含效果指标。`roi_pilot/run.json`五条件complete、48图（24正常/24异常）；48路径是full380的严格子集，不能合并算428，也不是独立确认集。full为172正常/208异常，本次快照status仍running、仅三条件有完整380结果。

ROI所有条件统一题干明确要求Yes/No first、user-only、448且processor `do_resize=False`。它不同于native baseline的system/category题干、未要求首词、square224→实际256/64token。因此ROI original_only AUROC0.753480相对native baseline0.626342的差异，同时包含prompt/system/类别信息/分辨率/token预算变化，不能直接解释成ROI收益或修复了视觉编码器。代码：`migration_roi_eval.py:30–35,249–281`；`migration_native_train.py:286–324`。

最公平的定位选择比较在center/random/teacher三种**双图**条件内：同题干、每个crop面积1/9、实际448、392视觉token/470总prompt tokens。original_only为196/272；增加reference后是588/668，第三张同类训练正常图额外增加预算。reference按缓存ImageBind CLS余弦选图，不是Qwen原生检索。teacher ROI由监督外部teacher概率图argmax决定，非GT mask但也非纯Qwen零样本；normal train reference不使用val/test参考图片。代码：`migration_roi_eval.py:92–192,218–281`；JSON：`roi_definition/reference_definition/token_budget_comparison/results.*`。

pilot上teacher→reference的AUROC 0.835938→0.838542略升，但generated macro F1 0.785714→0.762483下降，不能预先称reference全面改善。full及KD最终数值由父代理在真实完成后补齐；KD应只与同feature/size GT比较，必须核对相同`matched_run_key`、training plan、head、seed、优化器/调度/预算和实际weight0.5；缺失结果不填0。

## 4. 最后报告必须保留的局限

1. 全部表是自定义监督60/20/20开发验证结果，含真实异常训练；历史test456已用于早期开发，本次无test效果结论。同一val被teacher/adapter/head选best，不能当未见确认集。
2. 路线1三seed，native/dense/ROI当前单seed；不把单seed差异写成显著提升，不挑最好seed或把pilot与full混合。
3. 候选Yes−No、greedy解析、pixel AP/AUROC、map-p99图像分数与memory距离是不同输出面；raw logit差不是校准异常概率。invalid须与内容判断分开报告。
4. Phase1→native同时变化输入信息、参数、模板、位置及梯度聚合；native→ROI又变化prompt/分辨率/图数。跨路线排序不能独立归因某个组件。
5. GT跨特征共享预算但参数/精度/网格不同；teacher为四层融合，当前GT为单层head；pre/post对AP和AUROC的排序不同。
6. 外部teacher/CLS检索是ROI方法输入的一部分；多图增加token与计算量，pilot不证明full收益，reference不保证所有指标改善。
7. exact gate与重载日志验证实现/已记录输出一致，不替代检测效果；本次未独立反序列化checkpoint，completion依赖可追溯的运行审计JSON。

## 关键证据路径

- `phase1_summary.json`与六组`lm_*/{run,train_complete,val_metrics}.json`：路线1配对与统计。
- `native_{baseline,map_s42}/{run,val_metrics,train_complete,complete}.json`及各`selected_auxiliary_cache_check.json`：路线2数值/当前权重gate。
- 五组`dense_*_gt_s42/{run,training_plan,best_validation_reloaded,train_complete}.json`与`validation.jsonl`：GT5格及选择依据。
- `roi_{preflight,pilot,full}/run.json`：接口/子集/当前full状态、真实grids/token预算与protocol。
- `teacher_cache.json`、`teacher_val.json`：共同teacher来源与参考指标，非新dense输出。
- `ALL_RESULTS.md`、`PROGRESS.md`、`all_summary.json`：派生汇总；以各自快照时间和输入SHA判定新旧。

本目录绝对根：`/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/migration_20261103`。源码只读根：`/Users/qiaohongfei/anomalygpt_remote/files`。

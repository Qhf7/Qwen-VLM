# Jev与Qwen工业异常检测后续研究总结

## 结论与数据边界

全部研究固定既有120张VisA开发query（12类各5正常/5异常），未读取confirmation。后续实验与统计均在指定AutoDL服务器完成。
ROI复核尚未证明定位净收益；共同448输入下DD相对量化底座有明确开发集排序改善；支持与分辨率结论限定于本次固定query和服务调用。

## 共同448图像的全图判断

| 系统 | AUROC | AP | Recall | FPR | 工作点 |
|---|---:|---:|---:|---:|---|
| base | 82.44% | 85.97% | 23.33% | 0.00% | logodds>0 |
| decision_decision | 93.49% | 94.33% | 71.67% | 5.00% | logodds>0 |
| jev_shared0 | 83.46% | 86.84% | 78.33% | 35.00% | p≥0.5 |

- DD−base AUROC：+11.04pp [+7.40,+14.97]；这对应新共同448输入，不能与历史原图82.39%→94.56%的差值混用。
- 代码`code/qwen_matched_input.py`；证据`stage09_qwen_matched/scores.jsonl`、`metrics.json`、`final_audit.json`及`stage08_rescue/metrics.json`。
- Qwen两路都使用语言NF4 double quant/BF16 compute、视觉BF16、SDPA；DD为r16/alpha32/qkvo LoRA。240图像判断=480候选forward。
- 两路逐query native像素tensor相同，实际grid=[1,28,28]与196个merged视觉token；Jev共享同PNG字节。
- Qwen JSON布尔候选评分与Jev Noul问题协议不同；Jev内部token/算力不可见，只匹配PNG/图数。跨系统比较AUROC/AP排名，不宣称相同校准或相同算力。
- DD−base逐类别宏平均AUROC：+1.00pp [-4.34,+6.50]。宏平均区间跨零，pooled改善不能解释为每类都改善，pcb3和capsules仍有弱项。

## 正常参考图的种子敏感性

| 新采集条件 | AUROC | AP | Recall | FPR |
|---|---:|---:|---:|---:|
| shared0 | 83.46% | 86.84% | 78.33% | 35.00% |
| s43_1 | 89.21% | 90.97% | 86.67% | 30.00% |
| s43_4 | 91.33% | 92.84% | 85.00% | 23.33% |
| s44_1 | 89.14% | 90.48% | 88.33% | 25.00% |
| s44_4 | 91.12% | 92.78% | 85.00% | 23.33% |
- 1-shot seeds43/44 AUROC均值89.17%，范围[89.14%,89.21%]；这是两组点估计描述，不是独立query或总体支持分布区间。
- 4-shot seeds43/44 AUROC均值91.23%，范围[91.12%,91.33%]；这是两组点估计描述，不是独立query或总体支持分布区间。
- s43_4-minus-shared0 AUROC：+7.87pp [+4.62,+11.74]。
- s44_4-minus-shared0 AUROC：+7.67pp [+3.96,+11.63]。
- s44_4-minus-s43_4 AUROC：-0.21pp [-2.11,+1.53]。
- 384张训练正常候选每类32张，query来源/字节无交集；1-shot嵌套4-shot首张，固定种子哈希选择，不按结果重选。
- 新shared0单独采集120，两seed1/4各120，总600判断；旧seed42因renderer差异仅为历史对照。
- 这里zero/few-shot指推理时不给/给正常参考图，不更新Jev参数；Qwen DD则已做768训练来源的监督LoRA，不能将它与参考图数量当等训练预算比较。
- 代码`code/support_robustness.py`；支持结果证据`stage08_rescue/metrics.json`，逐文件路径与SHA见汇总JSON。下一门槛是更多bank和服务重复，而非把两seed当大样本。

## 第一阶段保留的历史零少样本基线

| 旧seed42条件 | AUROC | Recall(p≥0.5) | FPR(p≥0.5) |
|---|---:|---:|---:|
| jev_0shot | 82.54% | 76.67% | 33.33% |
| jev_1shot | 87.19% | 86.67% | 36.67% |
| jev_4shot | 88.85% | 78.33% | 23.33% |

历史360条完整响应保留不覆盖；本轮后续API/GPU均改由服务器执行。旧renderer与新renderer不同，旧seed42与新43/44的差异不能纯归因为支持种子。

## 320与448零参考分辨率

| 条件 | AUROC | AP | Recall | FPR |
|---|---:|---:|---:|---:|
| size320_0shot | 79.04% | 83.63% | 66.67% | 21.67% |
| shared0_448 | 83.46% | 86.84% | 78.33% | 35.00% |
- 320−448 AUROC：-4.42pp [-9.18,+0.08]。
- 320/448画布像素比0.5102；这同时改变可见细节与输入预算，不等同纯算力因果或精确token比例。
- 新320接受120判断，仅复用stage08 shared0的120个448响应；其余480支持响应不混入。服务采集时间未交错，无法独立排除provider变化。
- 采集代码`code/resolution_ablation.py`；救援参考评价代码`code/resolution_amended_reference.py`；证据`stage10_resolution/metrics.json`与`evaluation_contract_amended_reference.json`。下一门槛是独立数据及交错重复的质量/成本测量。

## 固定候选区域复核与瓶颈

| 方法 | 全60异常平均IoU | IoU≥0.5 | 异常保留召回 | 全60正常出框率 |
|---|---:|---:|---:|---:|
| unconditional_teacher | 0.089529 | 6.67% | 100.00% | 100.00% |
| roi | 0.069719 | 5.00% | 71.67% | 21.67% |
| duplicate | 0.068542 | 5.00% | 65.00% | 18.33% |
- ROI−duplicate anomaly_bbox_iou_mean_all：+0.12pp [-0.07,+0.32]。
- ROI−duplicate anomaly_iou_at_05_all：+0.00pp [+0.00,+0.00]。
- ROI−duplicate anomaly_retained_recall：+6.67pp [+0.00,+15.00]。
- ROI−duplicate normal_positive_box_rate：+3.33pp [-3.33,+10.00]。

Teacher query split={'train': 96, 'val': 9, 'test': 15}；96/120在teacher训练划分中，未见24张仅22异常/2正常；未见子集不是新盲测，FPR分母极小。
候选中60/120为整图框。当前teacher无条件平均IoU低，门控只能保留或删除现有框，无法修正候选几何，候选质量是主要科学瓶颈。
ROI−duplicate的定位/保留召回区间跨或触及0，误报也有代价，保留负结果。历史全图0/1/4与原生Qwen只作输入/语义预算不同的历史门控。
代码`code/extract_teacher.py`、`code/evaluate_roi.py`；证据`stages/02_teacher_alignment.json`、`03_candidates.json`、`05_evaluation.json`。下一步先冻结train-only候选改进，再用新确认方案验证。

## 离线顺序路由与成本

| 策略 | 逻辑成功请求 | ROI升级 | Recall | 正常出框率 |
|---|---:|---:|---:|---:|
| duplicate | 120 | 0 | 65.00% | 18.33% |
| roi | 120 | 0 | 71.67% | 21.67% |
| sequential_uncertain | 162 | 42 | 70.00% | 23.33% |
| geometry_small | 153 | 33 | 68.33% | 23.33% |
| matched_random_s42 | 162 | 42 | 68.33% | 23.33% |
| matched_random_s43 | 162 | 42 | 68.33% | 23.33% |
| matched_random_s44 | 162 | 42 | 70.00% | 23.33% |
- 不确定性路由仅根据duplicate概率、几何及固定中性哈希，策略冻结后回放，无oracle选择；并非在线Agent部署。
- 实际已采集ROI/duplicate共240成功目标及重试；逻辑162请求不抵消已发生采集成本，未估造货币账单。
- 不确定性路由−duplicate Recall：+5.00pp [-1.67,+11.67]；正常出框率：+5.00pp [+1.67,+10.00]。
- 代码`code/routing_replay.py`；证据`stages/06_routing_results.json`。下一门槛是独立数据上预声明误报/漏检效用及在线时延/账单测量。

## 开发集OOF校准与拒答诊断

| 历史全图分数 | raw Brier | OOF Brier | raw ECE10 | OOF ECE10 |
|---|---:|---:|---:|---:|
| jev_0shot | 0.174259 | 0.167706 | 0.100917 | 0.060568 |
| jev_1shot | 0.157224 | 0.147661 | 0.136583 | 0.061343 |
| jev_4shot | 0.141269 | 0.134716 | 0.114917 | 0.072830 |
| qwen_base | 0.298766 | 0.165891 | 0.331814 | 0.087381 |
| qwen_decision_decision | 0.093157 | 0.094009 | 0.054049 | 0.037947 |

拒答代表例使用历史全图分数；这些诊断采用其既有p≥0.5协议，不替换stage09原生严格logodds>0工作点。

| 历史分数 | 版本 | 覆盖率 | 接收异常/正常 | 接收准确率 | Selective risk | 全异常自动检出率 |
|---|---|---:|---:|---:|---:|---:|
| jev_4shot | raw | 50.00% | 37/23 | 96.67% | 3.33% | 58.33% |
| jev_4shot | raw | 75.00% | 48/42 | 86.67% | 13.33% | 70.00% |
| jev_4shot | raw | 100.00% | 60/60 | 77.50% | 22.50% | 78.33% |
| jev_4shot | oof_logistic | 50.00% | 37/23 | 93.33% | 6.67% | 55.00% |
| jev_4shot | oof_logistic | 75.00% | 44/46 | 86.67% | 13.33% | 60.00% |
| jev_4shot | oof_logistic | 100.00% | 60/60 | 78.33% | 21.67% | 78.33% |
| qwen_decision_decision | raw | 50.00% | 38/22 | 100.00% | 0.00% | 63.33% |
| qwen_decision_decision | raw | 75.00% | 46/44 | 94.44% | 5.56% | 71.67% |
| qwen_decision_decision | raw | 100.00% | 60/60 | 85.83% | 14.17% | 85.00% |
| qwen_decision_decision | oof_logistic | 50.00% | 37/23 | 100.00% | 0.00% | 61.67% |
| qwen_decision_decision | oof_logistic | 75.00% | 47/43 | 95.56% | 4.44% | 71.67% |
| qwen_decision_decision | oof_logistic | 100.00% | 60/60 | 85.00% | 15.00% | 81.67% |
- 固定5折category×label分层，每折96拟合/24验证；25个既有Logistic校准器，不在汇总中重拟合或调超参。
- 这是已查看120开发图上的交叉拟合诊断，协议冻结时ROI结果已存在；独立校准/confirmation仍未完成，也未拟合生产校准器。
- ECE对分箱和小样本敏感；拒答准确率应同时报告覆盖率、接收异常/正常分母及全异常自动检出率。区域Noul未错误套用整图校准标签。
- 代码`code/calibration_diagnostics.py`；证据`stages/07_calibration_results.json`、`CALIBRATION_REPORT.md`。下一门槛是独立校准数据与固定最终方案后的确认。

## 完整执行与审计

| 实验 | 接受目标 | Started | 显式失败 | 未决 |
|---|---:|---:|---:|---:|
| roi | 240 | 278 | 38 | 0 |
| support | 600 | 699 | 99 | 0 |
| resolution | 120 | 120 | 0 | 0 |

支持完整结果采用显式HTTP502运维修订：原接受597，救援接受3，救援额外started=3；旧stage08/status仍为`incomplete_requires_explicit_retry`。
原contract、支持seed、PNG、问题、请求UUID/body、已接受响应及旧日志原样保留；`stage08_rescue/completion_certificate.json`明确记录合并完整状态，未把原阶段伪装成complete。
仅对已三次明确HTTP502且无未决请求的目标增加最多两次显式单并发重试；重试时间变化属于额外解释边界。320评价新增冻结合同显式绑定该证书与combined预测，只复用shared0。

Qwen单独240个图像判断/480个候选forward。新增API成功目标合计960（ROI240+支持600+320分辨率120），448共同对照被复用一次，未重复算作新增调用。
费用只依据实际返回usage；失败或缺失usage保持未知，不补0。逐run文件SHA、计数、配对区间与下一门槛保存于`stages/11_summary.json`。
阶段1 source审计、阶段2 teacher对齐、阶段3固定候选、阶段4预测冻结、阶段5区域评价、阶段6路由回放、阶段7开发OOF、阶段8支持种子、阶段9共同输入Qwen、阶段10分辨率均完整留存。

## 下一科学门槛

1. Improve candidate generator before another ROI gain claim。原因：Teacher mean IoU is low and half candidates cover whole image。门槛：Freeze a train-only candidate rule; report all-anomaly IoU, IoU>=.5, fullbox fraction and teacher exposure without tuning on confirmation。对应：`code/extract_teacher.py; code/evaluate_roi.py`。
2. Separate calibration data and final confirmation。原因：OOF reused viewed development120; independent calibration and confirmation remain incomplete。门槛：Fix system/prompt/threshold using training and independent calibration; open confirmation only at final approved evaluation。对应：`code/calibration_diagnostics.py; source_baseline/frozen/plan.json`。
3. Broader supports and provider repeat controls。原因：Seeds43/44 are only two fixed banks;0shot is shared once。门槛：Predeclare more support banks and interleaved repeated requests; paired source evaluation plus support/provider variance。对应：`code/support_robustness.py`。
4. Evaluate online routing after predictive benefit is supported。原因：Current uncertain replay increases logical calls and normal positive boxes。门槛：Predeclare escalation budget and FP/FN utility; compare online wall time and real billed usage on independent data。对应：`code/routing_replay.py`。

所有区间固定当前query与条件，未纳入服务重复/训练seed/支持总体不确定性，未校正多重比较。此文档不把开发结果表述为跨域、盲测、生产概率保证或部署收益。

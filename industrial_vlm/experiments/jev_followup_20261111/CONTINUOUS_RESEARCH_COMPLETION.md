# Jev与Qwen工业异常检测：连续研究完成报告

本文件汇总服务器阶段01—13的完整冻结结果，补充原FINAL_FOLLOWUP_REPORT.md；历史合同、预测、数据划分与旧报告保持原样。
最终确认评价完成时间：2026-11-11 03:14:36（北京时间）。后续推理、付费API评测、验证和统计均在指定AutoDL服务器完成。

## 1. 完成情况与主要结论

预声明主系统为Jev 1.14.0＋4张同类正常参考图（seed43）＋448输入＋固定Logistic概率映射，阈值p≥0.5；完整240张确认图AUROC为 **90.18%**，Recall **85.00%**，FPR **20.83%**，Brier **0.126974**。
阶段13只评估这一固定Jev系统及同次raw概率诊断，没有在confirmation上测试Qwen或其他候选，因此不作确认集跨模型优越性结论。

- 共同448开发输入下，监督Qwen LoRA相对量化底座的pooled排序改善明确；逐类别宏平均区间跨零。
- 两组固定支持bank中的few-shot正常参考改善开发排序；它是推理上下文，不更新Jev模型权重。
- ROI复核尚未证明定位净收益；路由仅为离线回放，未证明在线Agent或省费收益。
- 独立校准与heldout开发评价、最终一次性confirmation有不同角色，不能把各阶段分数混为同一测试。

## 2. 阶段01—13工作总表

| 阶段 | 工作 | 范围 | 完成结果与边界 | 主要证据 |
|---|---|---|---|---|
| 01 | 来源与隔离审计 | 旧开发120、原生训练768 | 核验source/SHA；字节隔离不证明物理零件独立 | stages/01_source_audit.json |
| 02 | Teacher对齐 | 旧query120、支持48 | query训练暴露96、验证9、测试15；未见24不是新盲测 | stages/02_teacher_alignment.json |
| 03 | 固定候选提取 | 旧开发120 | 60/120为整图框，候选几何质量是瓶颈 | stages/03_candidates.json |
| 04 | ROI预测冻结 | ROI120＋duplicate120 | 预测完整持久化后评价；不据结果重挑框 | stages/04_prediction_freeze.json |
| 05 | 候选区域评价 | 全60异常＋60正常 | 尚未证明ROI定位净收益，保留负结果 | stages/05_evaluation.json |
| 06 | 顺序路由回放 | 已采集240响应 | 逻辑调用预算回放，不等于在线实际省费 | stages/06_routing_results.json |
| 07 | 开发OOF校准/拒答 | 旧120、5折、25个拟合 | 开发交叉拟合诊断，不替代独立校准或确认 | stages/07_calibration_results.json |
| 08 | 支持种子稳健性 | 同120×5条件＝600判断 | 共享0shot一次；43/44各1/4shot；独立运维救援补齐3条 | stage08_rescue/metrics.json |
| 09 | 共同448输入Qwen | base/DD各120 | 同PNG、两路native tensor一致；480候选forward | stage09_qwen_matched/final_audit.json |
| 10 | 320/448分辨率 | 新320120＋复用448120 | 只复用shared0，排除480支持响应；调用时间未交错 | stage10_resolution/metrics.json |
| 11 | 完整研究汇总 | 阶段01—10冻结统计 | 保留负结果、区间及审计，原报告不覆盖 | stages/11_summary.json |
| 12 | 独立校准与heldout | cal120＋heldout120 | 只用cal120拟合；系数保存后评价heldout，不再择主系统 | stage12/run/metrics.json |
| 13 | 一次性最终确认 | 原冻结test240 | 最终bundle后一次开封；calibrated主/raw次；全240阈值评价 | stage13_confirmation/metrics.json |

## 3. 数据角色和唯一来源

| 角色 | 来源数 | 用途 | 评价定位 |
|---|---:|---|---|
| 历史原生监督训练 | 768 | Qwen LoRA训练；支持从训练正常来源固定抽取 | 训练 |
| 旧开发query | 120 | 基线、ROI、OOF、支持和分辨率 | 已多次查看的开发数据 |
| 独立calibration | 120 | 固定一维Logistic映射拟合 | 校准数据 |
| 来源互斥heldout | 120 | 映射固定后检验可靠性和阈值 | 开发评价，不是confirmation |
| 原冻结confirmation | 240 | 最终系统后一次开启，全240主评价 | 本轮最终确认 |

输入独立审计已核：旧开发120＋阶段12的240＋阶段13的240，唯一query来源与原图SHA并集均为600，三组两两source/byte交集为0。训练768不计入这600个query。
阶段12来源虽然与旧120和训练来源隔离，但native验证剩余池可能曾参与历史验证，不能称全新盲测。阶段13历史构建使用类别/标签元数据分层，本轮预测与评价标签在最终系统前封存；“盲”仅指本轮操作顺序，不证明所有历史模型完全没见过图像。
Teacher划分不同于原生Qwen：旧query96在Teacher训练划分，未见24只有22异常/2正常；Teacher未见子集不是新盲测。公开VisA预训练污染、物理零件与近重复独立性仍未知。

## 4. zero-shot和few-shot正常参考

固定模型、问题/criteria、query和448渲染。0-shot每请求只给query；1-shot增加1张同类正常图；4-shot增加4张。1-shot嵌套4-shot首张，seed与顺序预先固定，不按query结果挑图，不训练Jev参数。

| 条件 | AUROC | AP | Recall（p≥0.5） | FPR |
|---|---:|---:|---:|---:|
| shared0 | 83.46% | 86.84% | 78.33% | 35.00% |
| s43_1 | 89.21% | 90.97% | 86.67% | 30.00% |
| s43_4 | 91.33% | 92.84% | 85.00% | 23.33% |
| s44_1 | 89.14% | 90.48% | 88.33% | 25.00% |
| s44_4 | 91.12% | 92.78% | 85.00% | 23.33% |

- 1-shot两bank均值89.17%，范围[89.14%, 89.21%]；是两固定bank点估计，非支持总体方差区间。
- 4-shot两bank均值91.23%，范围[91.12%, 91.33%]；是两固定bank点估计，非支持总体方差区间。
- s43_4-minus-shared0 AUROC：+7.8750个百分点，95%区间[+4.6247, +11.7365]。
- s44_4-minus-shared0 AUROC：+7.6667个百分点，95%区间[+3.9583, +11.6250]。

384张训练正常候选每类32张经原图SHA核验，与query来源/字节无交。4-shot仍有23.33% FPR，ECE不保证单调改善。旧seed42历史0/1/4shot AUROC82.54%/87.19%/88.85%；renderer不同，只保留历史对照，不纯归因seed。

## 5. 共同448输入的监督Qwen LoRA对照

| 路线 | AUROC | AP | Recall（logodds>0） | FPR | TP/FP/TN/FN |
|---|---:|---:|---:|---:|---|
| base | 82.44% | 85.97% | 23.33% | 0.00% | 14/0/60/46 |
| decision_decision | 93.49% | 94.33% | 71.67% | 5.00% | 43/3/57/17 |

- DD−base pooled AUROC：+11.0417个百分点，95%区间[+7.4028, +14.9729]。
- 逐类别宏平均AUROC：+1.0000个百分点，95%区间[-4.3375, +6.5000]，区间跨零，pcb3/capsules等仍有弱项。
- base/DD均252语言NF4 double-quant层、BF16 compute、视觉BF16、SDPA；DD有144个q/k/v/o注意力LoRA模块，r16/alpha32/dropout0.05。本轮只推理，无新训练。
- 同query两路PNG与native像素tensor一致，grid=[1,28,28]、196个merged视觉token；每图true/false各1个候选forward，240判断＝480forward，无额外预热。
- 历史原图82.39%→94.56%（+12.17个百分点）与新共同448的82.44%→93.49%属于不同输入预算，不能混写。
- Jev/Qwen只匹配PNG/图数，问题、输出协议、监督训练预算与内部算力不同；不能把Qwen监督LoRA和Jev推理参考数当等训练成本。

## 6. 分辨率和ROI负结果

| 0shot输入 | AUROC | AP | Recall | FPR |
|---|---:|---:|---:|---:|
| size320_0shot | 79.04% | 83.63% | 66.67% | 21.67% |
| shared0_448 | 83.46% | 86.84% | 78.33% | 35.00% |
320−448 AUROC：-4.4167个百分点，95%区间[-9.1826, +0.0833]；区间跨零。下采样同时改变细节/预算，调用时间不交错，不独立归因算力或分辨率。

| 候选门控 | 全60异常meanIoU | IoU≥0.5 | 保留召回 | 全60正常出框率 |
|---|---:|---:|---:|---:|
| unconditional_teacher | 0.089529 | 6.67% | 100.00% | 100.00% |
| roi | 0.069719 | 5.00% | 71.67% | 21.67% |
| duplicate | 0.068542 | 5.00% | 65.00% | 18.33% |
ROI−duplicate meanIoU：+0.0012，95%区间[-0.0007, +0.0032]；保留召回：+6.6667个百分点，95%区间[+0.0000, +15.0000]。
60/120候选为整图框；门控只能保留/删除，不能修复几何。ROI定位净收益未被证明，负结果保留。
不确定路由离线162逻辑请求/42升级，Recall70.00%、正常出框率23.33%；相对duplicate召回+5.0000个百分点，95%区间[-1.6667, +11.6667]，正常出框率+5.0000个百分点，95%区间[+1.6667, +10.0000]。已实际采集的响应不因回放而省费，也不是在线Agent部署。

## 7. 阶段12：独立校准与heldout开发评价

固定映射：p_cal = sigmoid(1.56720531322167 × logit(clip(p_raw,1e-6,1−1e-6)) -0.15632219069120)。只用calibration120拟合，固定C1/L2/lbfgs/random_state42；系数保存后才评价heldout120。
系数为正使映射严格保序，AUROC/AP不变是数学性质，不是重新训练Jev；概率值与固定0.5阈值的决策可以改变。

heldout120固定结果

| 版本 | AUROC | AP | MacroF1 | Recall | FPR | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 92.82% | 94.24% | 84.98% | 88.33% | 18.33% | 0.124228 | 0.158333 |
| calibrated | 92.82% | 94.24% | 89.17% | 88.33% | 10.00% | 0.107742 | 0.110424 |

| 版本 | TP/FP/TN/FN | Recall计数与Wilson95% | FPR计数与Wilson95% | Binary log loss |
|---|---|---|---|---:|
| raw | 53/11/49/7 | 53/60；[77.82%, 94.23%] | 11/60；[10.56%, 29.92%] | 0.404792 |
| calibrated | 53/6/54/7 | 53/60；[77.82%, 94.23%] | 6/60；[4.66%, 20.15%] | 0.348591 |

- calibrated−raw Brier：-0.0165，95%区间[-0.0237, -0.0087]；负值代表平方误差改善。
- calibrated−raw ECE10：-0.0479，95%区间[-0.0708, +0.0084]，区间跨零。
- raw FPR11/60→calibrated6/60；不能由10%点估计声称生产FPR≤10%。
- 主calibrated/rawsecondary在12结果前固定，未依heldout结果切换主系统。旧120上的OOF25拟合只是历史诊断，不等同本次独立cal120映射。

固定映射−raw binary log loss：-0.0562，95%区间[-0.0761, -0.0338]；这与Brier一起描述概率质量，ECE仍受分箱与小样本影响。

## 8. 阶段13：固定系统的一次性确认240

| 版本 | AUROC | AP | MacroF1 | Recall | FPR | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary_calibrated | 90.18% | 91.70% | 82.07% | 85.00% | 20.83% | 0.126974 | 0.072835 |
| secondary_raw | 90.18% | 91.70% | 81.58% | 88.33% | 25.00% | 0.140102 | 0.123000 |

| 版本 | TP/FP/TN/FN | Recall计数与Wilson95% | FPR计数与Wilson95% | Binary log loss |
|---|---|---|---|---:|
| primary_calibrated | 102/25/95/18 | 102/120；[77.53%, 90.30%] | 25/120；[14.53%, 28.95%] | 0.399576 |
| secondary_raw | 106/30/90/14 | 106/120；[81.37%, 92.92%] | 30/120；[18.11%, 33.44%] | 0.440692 |
主映射−raw binary log loss：-0.0411，95%区间[-0.0583, -0.0228]。Wilson区间对应独立二项率描述，bootstrap区间固定本次来源/校准器，二者估计对象不同。

| 主系统指标 | 点估计 | 95%固定映射分层bootstrap区间 |
|---|---:|---|
| auroc | 90.18% | [86.68%, 93.39%] |
| ap | 91.70% | [88.93%, 94.49%] |
| macro_f1 | 82.07% | [77.90%, 86.25%] |
| recall | 85.00% | [79.17%, 90.83%] |
| fpr | 20.83% | [14.17%, 26.69%] |
| brier | 0.126974 | [0.107590, 0.146460] |
| ece_10_equal_width | 0.072835 | [0.059239, 0.117387] |

- 主映射−raw Brier：-0.0131，95%区间[-0.0184, -0.0076]。
- 主映射−raw ECE10：-0.0502，95%区间[-0.0721, -0.0071]。
- 主映射−raw Recall：-3.3333个百分点，95%区间[-6.6667, -0.8333]；FPR：-4.1667个百分点，95%区间[-7.5000, -0.8333]。
- 本次映射减少5个误报，同时少检出4个异常；Recall损失的配对区间不跨零，不能写成校准全面提高分类表现。MacroF1变化+0.4833个百分点，95%区间[-1.6533, +2.9791]，区间跨零。
- 不论方向好坏均不再选择模型、问题、seed、阈值或映射；240主评价全部保留，不拒答。以下仅预声明回顾覆盖率风险诊断，不等于部署覆盖保证。

| 版本 | 覆盖率 | 接收/240 | 接收异常/正常 | 接收错误率 |
|---|---:|---:|---:|---:|
| primary_calibrated | 50.00% | 120/240 | 70/50 | 5.83% |
| primary_calibrated | 75.00% | 180/240 | 95/85 | 10.56% |
| primary_calibrated | 100.00% | 240/240 | 120/120 | 17.92% |
| secondary_raw | 50.00% | 120/240 | 76/44 | 6.67% |
| secondary_raw | 75.00% | 180/240 | 99/81 | 11.11% |
| secondary_raw | 100.00% | 240/240 | 120/120 | 18.33% |
最终确认仅有固定Jev及其raw secondary，不能声称确认集优于未跑的Qwen、AnomalyGPT或其他系统。teacher历史暴露需另列，不宣称对所有既往模型完全未知。

## 9. 正式API执行量与恢复过程

| 正式实验 | 接受logical targets | Started | 显式失败 | 未决 |
|---|---:|---:|---:|---:|
| 固定ROI/duplicate | 240 | 278 | 38 | 0（完整冻结审计） |
| 支持0/1/4-shot | 600 | 699 | 99 | 0 |
| 新320分辨率 | 120 | 120 | 0 | 0 |
| calibration120＋heldout120 | 240 | 251 | 11 | 0 |
| confirmation240 | 240 | 240 | 0 | 0 |
| 正式合计 | 1440 | 1588 | 148 | 0 |

1440＝ROI240＋支持600＋新320120＋stage12 240＋stage13 240，是正式API目标数，不是1440张唯一图。独立source/byte审计核实query并集为600；同development120的多臂比较不能当新增独立query。
独立单目标预检另列在其预检目录，不混入正式合计；旧360条历史基线也不混入。原生Qwen另列240图像判断＝480候选forward。
stage08原597接受后3个目标均三次明确502耗尽cap3，原科学合同/seed/PNG/问题/请求UUID/body/日志保持不变；显式单并发运维修订补3条。旧status仍保留未完整，独立completion证书诚实标识合并完整。
最终确认单目标最多3次、全局started≤720；累计失败30停止新提交，已reserved inflight最多drain到32。timeout/transport/HTTP200 typed解析异常永久unknown，不得当明确502重发。
费用只依据实际返回usage，缺失/失败usage未知、不补0；没有可靠货币账单，不从token编造费用。服务model回显也不是上游权重不可变的证明。

## 10. 审计与解释边界

阶段12独立审核1550通过/0失败；阶段13独立审核548通过/0失败。最终bundle另42项核验及7组合成ledger边界通过；审计项可能重复，不相加当独立证据数。
表中率的Wilson区间为二项比例区间；其余所列bootstrap区间采用2000次、seed42、category×label分层，固定当前query/支持/拟合校准器/服务响应，不含训练seed、calibration拟合、provider重复与支持总体方差，未做多重比较校正。
类均衡50%异常先验不等于生产先验；误报/漏检分母和区间不可省略。来源/byte隔离不等于近重复或物理零件独立，公开基准预训练污染未知。阶段01—11“未打开confirmation”是历史阶段状态；阶段13现已按冻结流程完成最终评价。
本报告只读取完整冻结聚合，不打开legacy确认成员、保护labels、RGB或GT mask，不新增模型/API/拟合。

## 11. 后续工作

本轮确认应作为不可回调终点。任何新问题、支持、模型、阈值、候选器或校准器修改都建立新开发和独立测试协议，不再把已打开240称新盲测。

1. 以训练数据冻结更好的候选器，验证meanIoU、IoU≥0.5、整图框比例与Teacher监督暴露。
2. 预声明更多支持bank和时间交错provider重复，测支持/服务方差、实际usage与账单。
3. 在新来源与生产异常先验上验证固定系统的概率可靠性、FPR上界和业务效用。
4. 预测收益已建立后，再做在线路由/Agent端到端时延、真实费用与误报/漏检效用。

## 12. 冻结证据索引

### 12.1 结果图索引

| 图 | PNG | 来源/审计 |
|---|---|---|
| 最终固定系统确认240 | figures/confirmation_fixed_system_analysis.png | figures/confirmation_fixed_system_analysis.provenance.json |
| 独立calibration/heldout | figures/calibration_heldout_analysis.png | figures/calibration_heldout_analysis.provenance.json |
| 支持bank与共同448Qwen开发对照 | figures/support_qwen_matched_comparison.png | figures/support_qwen_matched_comparison.provenance.json |
| 320/448零参考分辨率 | figures/resolution_tradeoff.png | figures/resolution_tradeoff.provenance.json |

确认图与heldout图对应不同来源，不横向拼成一份测试；Qwen图仅是旧开发120的共同输入对照。各图PNG/SVG/provenance均按下列SHA索引留存，独立visual QA结果由图形审计单独记录。

### 12.2 文件SHA256

- stages/11_summary.json：f89bb7adfb19dcd0708460729e9e959cc2c2596ecb655495bbed22ff627b9280
- stages/11_summary_status.json：5080c721368bc652aae9bfa2093131de8ef49c6df2678dcad1ad42ac8c3f615f
- stage12/run/metrics.json：075f823dc6fc0617844e3d5578ffd0c04b3b0d077e07052f5cff9628408afea4
- stages/12_postexecution_audit.json：595c853d71a3dee79bfa11d40497cf9c0017ab5a07b54f5625bc6a6f3e8a31bb
- stage13_confirmation/metrics.json：2838095b23bba31a4c233be7a385f3e010c4af7a978673de40bccf10f6e63976
- stage13_confirmation/completion_certificate.json：415f40f52644c0d716c3f563add5d4d03c540c37cbd249c197b6b8a80d899241
- stages/13_postexecution_audit.json：c6a590ef9470c644cec4848a5205d4d68b60a312f58667adbb3b1a5d6a0ae668
- stage12/run/STATISTICAL_SUPPLEMENT.json：25d0cf2b74f2b9c9441cb9bdaeae7eb11a22b96b6f6c89e6811cb6acb0e38912
- stage13_confirmation/STATISTICAL_SUPPLEMENT.json：ff5193c86beb5d85ddc0f70324b906efef023e2e08f689e214d321114332af36
- stages/13_crossstage_source_audit.json：1ab7183e0f376d35ec5fb20a083d5146524df5988596c8d4ac69ad5149c60159
- FINAL_FOLLOWUP_REPORT.md：8dced54c1da46b6ae5dfd7f5051537a265ccd3352a1d0e36cd4a4e772b8781f7
- stage12/run/calibrator.json：0bac063df108bd44938c25a8bb2c07261d79bfe2350b6d1bdb8b8cf1c1bcd9ec
- stage12/run/prediction_freeze.json：07c742b039dac77f345682fe7c5c3f82482e867aa1dfbdb58f9e7adc98f5c30a
- stage13_confirmation/final_system_bundle.json：72ab9b8706cdf77c102f7fb53c372e3a183ecdf69e67599d9d58732838fd392e
- stage13_confirmation/prediction_freeze.json：785e76ea8593c1dcba52111581d92fd090b456ba22e95421ee07a339c152a20f
- stages/13_runner_preopening_audit.json：6944bf46fafd772acf826e3cdb82dd44dbfed96d52509e05b347c8a386ade5d3
- stages/13_bundle_preopening_audit.json：a770c8053ea40ef82486871684c6c7cb75e3ccdabc41aed8d5901130254afdd3
- code/compile_continuous_completion.py：adc0612bc898d34834e5e8578fa52bea49e9a9caaf43112f6b1a90f741171cd2
- figures/confirmation_fixed_system_analysis.png：4884529a769664dbc5416dba547a2af588d846632c5186da77543bd0da29d79f
- figures/confirmation_fixed_system_analysis.svg：02eb299a09d9881f4591036547eddf9b7dac84729e407e26c2c5b3af66cf1386
- figures/confirmation_fixed_system_analysis.provenance.json：4ada6c1dfacd902ba3c5b66775648af1abb69ed84c57a4f031a404af7c453bc0
- figures/calibration_heldout_analysis.png：cb9054cbf4c5ea86ab021b690d916e11720c5c7b5a68b096b7e8d7aac9059423
- figures/calibration_heldout_analysis.svg：0dc6526fa15b048a9daa5cfe092055f46ccfae1ad0436c290b611436dd76703c
- figures/calibration_heldout_analysis.provenance.json：39e0fcb5b4e784e5889cada017ac23292b4771d26584c212ee424c2ad59433bf
- figures/support_qwen_matched_comparison.png：a244becf400709a260da075c6cb71cb6515412466193a02f0d3cd98f434a82d3
- figures/support_qwen_matched_comparison.svg：fee6a213c4614d678a57cefb77fa0f6db624e9a91a6d7cec0667e0543fde45e2
- figures/support_qwen_matched_comparison.provenance.json：8a17816e3beef3bbc64e758c9f1bc8c313ac051bfe7fe269ec75429d10f7d12d
- figures/resolution_tradeoff.png：baa6e29e554fab1cbb22bc20e7de2e637ffe74c3451f98c77f70f2d2cff9e268
- figures/resolution_tradeoff.svg：ca09902404813dc181714c22263cfe92d91251cc834bc49805c79148442f5c70
- figures/resolution_tradeoff.provenance.json：bd80a2e272ad79694c0b85011ec8ff81d886ba87bcbd56d3a096d7b3bdd3659d

各阶段原始结果保留，新增报告与stages/14_completion_report_manifest.json同步本地；不拷legacy确认元数据或保护标签。

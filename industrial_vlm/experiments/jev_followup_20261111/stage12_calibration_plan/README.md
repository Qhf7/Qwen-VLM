# Stage12：新来源校准与留出评价预协议

当前状态：两个互斥来源角色已按metadata冻结；最终全图系统待stage08–10完成后固定。未启动API，未打开RGB、GT mask、native test全清单或confirmation。

## 固定的数据角色

从native val 1,624来源扣除当前已查看开发120，候选1,504来源。对每个category×label，以SHA256("jev-new-calibration-v1|42|" + source_id)升序排序，source_id破同分。每层前5条进入calibration、随后5条进入heldout_eval。12类×2标签×5=每角色120来源，其中60正常/60异常。每类剩余异常恰为10，全部分配到两个互斥角色；正常只使用每类前10个，保留其余正常来源。

两个角色的source与原始字节SHA交集0；与Qwen固定训练768、当前120交集均0。原始SHA来自DATA_PROTOCOL定义的source_id编码，本metadata阶段未重新打开图像字节；运行时必须先校验所选240原图SHA再发请求。不得用文件名或类别路径当额外模型输入。

这是native validation remainder研究，不称blind。完整val可能过去用于loss/模型选择；本次只证明对当前120及固定Qwen训练来源隔离，不能声称所有历史模型从未见过。Teacher不参与本阶段，不读取Teacher map，不借助其监督划分筛选样本。

## 最终系统与执行门

最终系统仅从stage08–10的完整开发结果选择一个固定全图0-shot或4-shot正常参考系统。模型固定jev-1.14.0。prompt、问题语义、448等最终输入尺寸、参考图身份/seed、几何和解析规则由最终system contract明确；这些选择必须发生在240个新响应之前，不依据新校准或留出结果反选。

新请求为calibration120+heldout_eval120，共240个accepted logical targets；0-shot和4-shot都只有一个固定arm，不各跑一遍。started HTTP attempts可能大于240，执行器须在最终contract固定上限、累计错误/费用停机门与重试策略。当前计划0新API。

## 固定校准方法

校准只使用calibration120：p clip到[1e-6,1-1e-6]，变换logit后拟合LogisticRegression(C=1,L2,lbfgs,fit_intercept=True,class_weight=None,max_iter=1000)。不搜索C、不用heldout120调映射、不按结果换方法。目标先验是category×label平衡设计，异常比例0.5；不能把概率解释为真实产线缺陷率。

raw和calibrated均固定p>=0.5为异常，明确记录exact ties。按heldout120共享source比较AUROC/AP、macroF1、Recall/FPR、混淆矩阵、Brier/logloss/ECE10；配对category×label bootstrap2000、seed42，FPR/recall给Wilson95。校准不保证收益，负结果原样报告。

## 冻结顺序

1. 固定stage08–10选出的唯一全图system及预算/失败策略。
2. 逐个验证已选原始SHA，渲染资产，冻结payload/contract和依赖版本。
3. 执行全部240目标，保留失败/拒答于账本，冻结原始预测与完整错误账本SHA。
4. 仅用calibration120标签拟合固定映射，冻结系数。
5. heldout评价端在预测与系数冻结后加载heldout标签，全120分母评价，不反调任何系统或校准参数。
6. 保存结果JSON、指标区间、尝试数/费用/延时、负结果、局限和下一阶段动作。

当前metadata选样已经读取分层标签；“预测后才开标签”是后续推理/评价入口的角色隔离，不能宣称研究者从未知道这些标签。API推理入口只读*_inputs.jsonl，不读取*_labels.jsonl或selected_sources.jsonl；评价入口严格检查预测/系数freeze才访问heldout_labels。

## 文件与核验

- selected_sources.jsonl：240条选中metadata、固定角色、标签、原始SHA、哈希排序值。
- calibration_inputs.jsonl / heldout_eval_inputs.jsonl：不含label的120条推理入口，各自保留身份用于对齐，不向模型发送source_id/image路径。
- calibration_labels.jsonl / heldout_eval_labels.jsonl：单独标签入口，后者仅评价gate后加载。
- metadata_audit.json：角色分层计数、交集、输入与输出SHA、未证明事项。
- preprotocol.json：机器可读冻结方法、依赖门、metrics与失败政策。

数据角色、预协议和元数据审计SHA在交付时核验；最终system contract另立不可覆盖文件。现有confirmation继续封存，其启动须符合ISOLATED_VALIDATION_PROTOCOL_DRAFT.md。后续该路线的heldout结果不是最终confirmation。

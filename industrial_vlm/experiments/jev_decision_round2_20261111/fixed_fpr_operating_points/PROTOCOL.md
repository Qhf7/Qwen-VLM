# 固定经验 FPR 运行点诊断协议

本协议于任何本诊断的阈值计算与 held-out 标签读取之前冻结。唯一执行位置为指定 AutoDL 实例 `/root/autodl-tmp/industrial_vlm`；只做已有分数统计，不执行 API、推理或训练。

1. 仅允许 stage12 的冻结 `contract.json`、`predictions.jsonl`、`calibrator.json`、`code/isolated_calibration.py`、`calibration_labels.jsonl`、`heldout_eval_labels.jsonl`。这些输入逐文件 SHA256 校验。禁止读取 stage13、图像、掩码或其他结果。
2. 使用已有 calibration120（60 normal / 60 anomaly）与既有 heldout120（60 normal / 60 anomaly），按 source_id 对齐，核验不重叠。不重拟合原校准器。
3. 分数为 raw Jev probability，以及冻结正系数 logistic 校准映射：sigmoid(a×logit(clip(p,1e−6,1−1e−6))+b)。a、b 仅来自冻结 calibrator。
4. 同时事前报告 α∈{0.05,0.10}，不以 held-out 结果反选 α。对每种分数，将 calibration 的 60 个 normal 降序排序，取第 floor(α×60)+1 高分为阈值，采用严格 score > threshold 判异常。分别为第 4 和第 7 高；与阈值并列均判正常。
5. 先完成并持久化 `thresholds.json`，随后才允许读取 heldout 标签；记录读取顺序、文件哈希与时间。已有 heldout 预测可用于映射核验，但不用于阈值选择。
6. 报告两种分数在两个 α 下的 calibration 实际 FPR、heldout FPR / Recall 和 TP/FN/FP/TN。用 z=1.959963984540054 的二项 Wilson95% 描述性区间，明确正常样本数60的精度风险。
7. 逐对核验 raw/calibrated 排名与 ties，并核验两种分数决策一致性；披露 clip、阈值并列、有限精度差异。
8. 这是既有开发 heldout 上的运行点诊断，不是新盲测、不是 stage13 确认结果，也不构成真实总体固定 FPR 保证。不能因校准分数尺度变化宣称排名或固定 FPR Recall 提升。
9. 主 runner 一次冻结执行；输出协议、阈值、结果、中文报告、逐样本 heldout 决策与 SHA manifest，随后另由独立脚本在服务器重算。

校准标签已参与旧 logistic 拟合，这里沿用冻结映射；这不将 calibration 的拟合/经验尾部表现变成外推保证。Wilson 区间对 calibration 自适应阈值不包含阈值选择不确定性；heldout 区间亦仅为边际、未校正、基于样本的描述。

# Qwen3 分割 baseline 实施记录

## 当前检查（2026-09-12）

- 服务器连接成功，RTX 4090 24GB 可用，检查时没有 GPU 任务。
- 基座：Qwen3-VL-2B-Instruct，模型配置为 Qwen3VLForConditionalGeneration。
- 原生模型输出文本；当前仓库的像素 mask 来自额外的 PatchDecoder，不能将随机初始化分割头的结果视为预训练 Qwen3 的分割能力。
- 旧 e1/evaluate_fold1.py 仍含五视角聚合和旧分类阶段门，不能直接用作本次单图分割评测器。
- 零样本框转 mask baseline 已完成 3,600 张 Dev：有效输出率 0.851389、平均 Dice 0.080246、非空 Dice 0.026978、平均 IoU 0.074283、非空像素召回 0.292041、正常图预测面积均值 0.344239；耗时 2,078.4 秒。数值来自服务器日志，原始产物位于 artifacts/segmentation_zeroshot_qwen3。
- 数据审计通过：训练 18,775 张（异常 15,782、正常 2,993），Dev 3,600 张（异常 1,775、正常 1,825），各 18 类，审计错误数为 0。
- GPU 单批前向实际输出 mask 为 (1,1,448,448)，base_grad=False、decoder_grad=True。此检查没有验证 prompt_learner 无梯度；日志中的 deepstack_ok 是固定输出，不能作为独立的 DeepStack 参数传递断言。当前不能据此宣称“仅 decoder 有梯度”的全部检查通过。
- 分割训练尚未启动。当前训练入口仍包含联合损失和 prompt learner 更新，需完成分割专用入口与验证后再训练。

## 与分类实验保持一致的数据约束

- 沿用 artifacts/splits/fold1 的训练与 Dev 清单，启动前核对 SHA256。
- 历史冻结协议记录训练 18,775 张、Dev 3,600 张、18 类；需要重新核验逐图 mask 可读性与标签一致性。
- 保留真实正常、真实异常；以 view_label 判断单张图片，不使用零件级标签替代。
- 训练使用 Train448_verified_20260905，Dev 使用 Dataset448，原始 GT 来自 Dataset。
- 图片保持原有 448×448 预处理；mask 使用最近邻缩放，记录原始尺寸及坐标映射。
- 缺失或不可读 GT 必须单独报错，不得当成空 mask。
- 不使用伪异常；不混入 Dev 或 holdout 训练。

## 已确定的实验顺序

1. 原生 Qwen3 零样本定位：通过提示词输出异常框并栅格化为 mask。应命名为框定位转 mask 基线，框覆盖区域不代表精细分割。记录格式错误、空输出及预测框，不额外引入其他模型。
2. 推荐的监督分割基线：冻结原生 Qwen3，训练轻量分割头，使用真实 GT mask。该实验是训练后的分割基线，与原生分类零样本 baseline 的定义不同；需确定训练日程和 loss 后执行。

## 后续评测要求

- 按单图、逐类别汇总 Dice、IoU、像素 Precision/Recall；空 mask 与非空 mask 分开报告。
- 报告正常图片预测异常面积、缺陷面积分层结果和失败样本。
- 连续像素分数可用时报告像素 AUROC/AUPR；二值框 mask 不伪装成连续异常分数图。
- 若使用分类器先筛图，同时报告分割单模块和分类后分割的端到端结果；端到端漏检计入最终 mask 指标。
- 正常图双空 mask 的高 Dice 不作为分割收敛的主要证据。
- 保存逐图预测、评测配置、提示词（如使用）、数据指纹和汇总指标。

模型参考：https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

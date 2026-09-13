# Qwen3 异常单图分割训练

任务：在已知包含异常区域的单张图片上预测像素 mask。此实验评价分割模块本身；两阶段系统整体性能还需要独立计入分类阶段的漏检与误报。

2026-09-13：新服务器 RTX 4090 D 已通过完整数据审计、梯度验证和18类评测导出检查，并开始训练。检查时已完成 step 90，latest checkpoint 已保存。可训练参数共 3,285,505，全部来自 decoder；base 和 prompt 的 requires_grad 与梯度检查均为 false。进度以服务器日志为准。

## 数据

- 训练：`artifacts/segmentation_real_anomaly/train.parquet`，15,782 张。
- Dev：`artifacts/segmentation_real_anomaly/dev.parquet`，1,775 张。
- 两个集合均为 18 类，`view_label=1`，对应 mask 必须在原尺寸及 448×448 最近邻缩放后非空。
- 图片分别来自 `Train448_verified_20260905` 和 `Dataset448`；GT 来自 `Dataset`。
- 启动时逐文件检查图片解码、图像尺寸、GT 对齐、非空 mask、train/Dev 图像及样本组交集。

## 固定协议

- 冻结 Qwen3-VL-2B-Instruct；取视觉 block 5/11/17 的完整 28×28 patch 特征。
- 使用 288 通道的 decoder；更新其投影、融合和 mask head。
- Qwen3、prompt learner 与 region head 均冻结。分割前向直接使用视觉特征，不调用语言模型。
- BF16 autocast，FP32 可训练权重和损失，micro batch=1，梯度累积=8。
- AdamW，学习率 1e-4，weight decay=0.01，warmup 50 step，cosine schedule，梯度裁剪=1。
- 本轮沿用现有 Focal+Dice 实现：Focal gamma=2、positive alpha=0.25；Dice 计算非空 mask。loss 的后续消融需独立记录。
- 类别循环打散，类内均匀抽样零件组，再抽一张异常视角；种子 20260901。
- 1000 个 optimizer step，共 8000 次样本抽取，有放回抽样，不表示已经遍历全部 15,782 张。
- 保存本轮新生成的 `sampling_schedule.parquet` 与数据/代码 SHA256。

## 验证与运行

```bash
cd /root/autodl-tmp/vlm
.venv/bin/python -m unittest e1.test_segmentation_only
# 可单独运行验证，使用独立输出目录
.venv/bin/python -m e1.train_segmentation_only --validate-only \
  --out artifacts/segmentation_qwen3_validation
# 正式入口也会自动完成全部数据/梯度检查及18类评测导出检查
screen -dmS seg_decoder_anomaly bash e1/start_segmentation_only.sh
```

正式目录：`artifacts/segmentation_qwen3_decoder_only`。启动脚本使用绝对 Python 路径及进程锁，已有训练历史时拒绝覆盖。

每个 optimizer step 检查仅 decoder 有梯度且所有梯度有限。单批验证记录视觉特征、DeepStack 输出、mask 尺寸、基座和 prompt 的冻结及梯度状态。DeepStack 输出存在不等同于通过语言模型；本分割模块直接读视觉特征。

## 评测与产物

在 step 250/500/750/1000 对全部 1,775 张异常 Dev 图评测，阈值固定 0.5。记录逐图及逐类别 Dice、IoU、像素 Precision/Recall、预测面积、缺陷面积分层结果；保存二值 mask。以类别宏 Dice 选择 best checkpoint。

- `data_validation.json`、`forward_validation.json`、`frozen_training_protocol.json`
- `preflight/`：每类一张的评测导出检查，不作为正式 baseline。
- `training_history.csv`：每步写入并刷新 loss/梯度/学习率/面积/耗时。
- `tensorboard/`：实时训练与 Dev 曲线。
- `latest.pt`：每50步保存；`step0250.pt` 等四个阶段 checkpoint；`best.pt`。
- `step*_metrics.json`、`step*_views.csv`、`step*_per_class.csv`、`step*_per_area.csv`。
- `step*_masks/`、`training_summary.json`、`run.log`、`exit_code.txt`。

文件存在或 screen 创建成功只表示任务已提交；应从 `[train] 1/1000` 及持续增长的历史确认开始训练，从最终 summary 和四个阶段评测文件确认完成。

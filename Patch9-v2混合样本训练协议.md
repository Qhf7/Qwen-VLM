# Patch9-v2 混合样本训练协议

## 数据和采样

训练读取 `artifacts/splits/fold1/train.parquet`：18,775 张，正常视角 2,993、异常视角 15,782；Dev 读取 `artifacts/splits/fold1/dev.parquet`：3,600 张，正常 1,825、异常 1,775。正常/异常均依据 `view_label`。正常视角可以来自异常零件，不能用零件级标签代替视角标签。

每个 optimizer step 恰好八个输入：

| 来源 | 数量 | 目标 |
|---|---:|---|
| 异常整图 | 2 | 真实非空 mask |
| 异常图的含缺陷 patch | 2 | 同坐标裁剪的真实 mask |
| 异常图的无缺陷 patch | 1 | 空 mask |
| 正常图的局部 patch | 1 | 空 mask |
| 正常整图 | 2 | 空 mask |

每种来源按类别循环，类别内均匀选样本组、组内均匀选视角。日程预生成，可复现。原图 patch 非空但缩放后消失时，不将该 patch 伪装为空 patch，而是记录并排除出含缺陷候选。所有实际输入再次验证目标是否符合来源。

训练覆盖所有推理输入类型，但正/空 patch 比例是人为平衡而不是自然分布。相较全异常全图 baseline，非空监督为4/8，必须报告曝光量差异，不能将未来全部性能变化单独归因于放大效果。

## 模型与损失

Qwen3-VL-2B-Instruct 全部冻结，仅训练 decoder（288通道，视觉层5/11/17）。输入/输出448×448，原图切patch，fraction=0.4、patch相邻重叠约25%。

非空目标：现有 Focal(gamma=2, positive alpha=0.25) + Soft Dice。
空目标：同一个 Focal，Dice贡献精确为0。通过显式 `dice_nonempty_only=True` 与 `allow_empty=True` 解耦控制。既有调用未指定新参数时保留原行为。

AdamW lr=1e-4，weight decay=0.01，warmup=50，cosine，梯度裁剪1，micro batch=1、累积8、BF16/FP32损失，seed=20260901，目标12000步。

## 评测与保存

250/500/750/1000/1500/2000/2500/3000及4000至12000每1000步真实执行完整Dev评测。每图运行全图和全部九个patch，禁止用GT选择推理patch。

输出 `full` 与 `mean_w0.5` 两套逐图、逐类别、面积分层和汇总指标。异常图Macro Dice/IoU、小缺陷Dice/Recall与正常图非空预测率/面积分开汇总，避免正常空mask提高总Dice造成误读。像素precision/recall分别标明逐图均值和全像素池化口径。

最优checkpoint依据 `mean_w0.5` 的异常类别Macro Dice。每50步保存latest，每个评测节点保存独立checkpoint；评测汇总JSON、训练历史和checkpoint采用临时文件替换。原实验目录保留，新实验应选择新的目录。

## 验证与运行

CPU回归测试：

```bash
python -m unittest e1.test_patch9_training e1.test_patch9_v2_geometry -v
```

服务器完整门禁（全数据检查、八输入真实优化步、四张Dev十视图推理）：

```bash
bash e1/start_patch9_geometry.sh validate artifacts/patch9_geometry_mixed_preflight_20260920
```

门禁执行一次优化步骤仅用于验证，不保存为正式训练模型；正式训练重新初始化。正式命令（本轮修复不会自动运行）：

```bash
bash e1/start_patch9_geometry.sh train artifacts/segmentation_qwen3_patch9_mixed_12000
```

每节点完整评测包括3600×10次前向，耗时显著多于全图评测。代码验证通过不代表定位精度已提高，效果须由后续正式实验确认。

## 2026-09-20 实测验证

- 本地8项CPU回归测试通过；Python编译、Shell语法、diff空白检查通过。
- 服务器全量数据门禁通过：Train18,775、Dev3,600；两类样本池、mask标签与尺寸、类别一致性、样本组隔离均检查通过。
- 预生成12000步×8=96,000输入的日程；每种来源均覆盖18类，各类别次数最多相差1。
- GPU完整8输入优化步通过，decoder权重实际改变，梯度范数4.07833；基座无梯度，可训练参数3,285,505。
- 所有空目标的Dice项为0；输出形状为[1,1,448,448]。
- 2正常+2异常Dev冒烟评测通过，共40次前向；两种融合的逐图/逐类别/面积分层/汇总文件均已生成。此为初始化模型的功能验证，不作为精度结果。
- 服务器验证产物已回传本地 `artifacts/patch9_geometry_mixed_preflight_20260920/`。本轮完成修复与门禁验证，未启动正式12000步训练。

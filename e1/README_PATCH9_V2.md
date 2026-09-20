# Patch9-v2 几何与数据接口（未训练）

这是独立的 Patch9 几何、混合样本采样和训练评测实现。Qwen3基座冻结，decoder结构和Focal参数保持现有设置；空mask的Dice贡献为0。

## 为什么重写

旧实验的失败并不是九宫格本身无效：stitching 后 tiny Recall 从约 0.245 提升到约 0.673，说明局部放大确实能找回部分小缺陷。但整体 Dice 下降，主要因为训练只使用含缺陷 patch，推理却使用全部 9 个 patch；无缺陷 patch 没有空 mask 监督，旧的 max 融合又会放大其中任意假阳性。此外 patch 训练每步异常曝光减半，造成整图性能退化。

## 新实现的边界

- `patch9_v2_geometry.py`：原始分辨率上生成 3×3 重叠框、同步裁剪 image/mask、保留空 patch 记录、概率回投和 weighted-mean stitching。
- `test_patch9_v2_geometry.py`：只做 CPU 几何测试。
- 默认 patch fraction `0.4`、overlap `0.25`，输出 `448×448`。
- 不使用旧 `max` 融合作为默认；weighted mean 的 patch 权重默认 `0.5`，防止单个 patch 噪声直接覆盖整图证据。
- 训练和验证通过独立入口运行；本轮已完成门禁，正式训练尚未启动。

混合训练入口为 `train_segmentation_patch9_v2_geometry.py`，采样实现见 `patch9_training_data.py`，完整Dev评测实现见 `patch9_validation.py`。当前训练协议及验证结果见项目根目录的 `Patch9-v2混合样本训练协议.md`。通过 `start_patch9_geometry.sh validate` 执行数据、梯度、优化与评测门禁，正式训练使用 `train` 模式和全新的输出目录。

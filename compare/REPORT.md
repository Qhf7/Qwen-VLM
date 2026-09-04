# 448 缩放与 mask 对比报告

## 选择和处理规则

- 数据范围：`artifacts/splits/fold1/dev.parquet`，不读取 holdout-zs。
- 每个类别选择一个 OK 组和一个 NG 组的 C1（`view_idx=0`）视角，共 18 类、36 张图。
- 原图保持原始 PNG 尺寸；模型图使用当前生产预处理：`cv2.INTER_AREA` 直接缩放到 448×448。
- 原始 mask 使用当前训练/评测协议中的 `cv2.INTER_NEAREST` 缩放到 448×448。
- 每个面板包含原图、模型输入图、原始 mask、448 mask 以及两种 overlay；每个组件也单独保存。

## 观察结果

在这 18 张异常示例上，原始 mask 面积比例与 448 mask 面积比例非常接近：

- 相对面积变化最小：`-0.54%`
- 相对面积变化最大：`+0.74%`
- 18 张异常图的相对变化均值：约 `+0.09%`
- 正常样本没有 mask，原始和 448 mask 均为全零

这说明对当前抽到的、面积相对明显的异常区域，`INTER_AREA` 图像缩放加
`INTER_NEAREST` mask 对齐没有造成大块区域直接消失。可是它不能排除微小缺陷
受影响：原图边长在本批样本中为 787--3623 像素，缩到 448 后每个模型像素对应
约 1.8--8.1 个原图像素；宽度只有几个原图像素的细纹、针孔或划痕仍可能被平均、
变暗或完全落入背景。当前一类一正常/一异常的抽样不是对最小缺陷的压力测试。

## 文件说明

- `class_name/ok_c1_resize_compare.png`、`class_name/ng_c1_resize_compare.png`：六格对比面板。
- `class_name/*_original.png`、`*_448.png`：原图和模型输入图。
- `class_name/*_mask_original.png`、`*_mask_448.png`：原始及 448 mask。
- `selected_samples.csv`：每个类别实际抽到的文件路径。
- `resize_mask_summary.csv`：尺寸、mask 面积和缩放变化的逐样本统计。

本目录仅用于诊断，不改变任何 baseline、E1 训练结果或当前冻结协议。后续“两阶段”
方案（先异常分类，再对异常图做像素分割）目前只作为问题拆分假设，尚未启动实现或训练。

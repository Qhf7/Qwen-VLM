# Resize / mask comparison

每个 fold1 Dev 类别选取一个 OK 和一个 NG 的 C1（view_idx=0）视角。

- 原图：原始 PNG，不改变尺寸。
- 模型图：`cv2.INTER_AREA` 直接缩放到 448×448，与当前 B0/E1 输入协议一致。
- GT mask：原始 mask 与 `cv2.INTER_NEAREST` 缩放到 448×448，与训练/评测对齐代码一致。
- 红色 overlay：帮助观察异常区域在缩放后是否仍可见。
- `resize_mask_summary.csv`：记录面积、中心点、边界框、连通域、反向映射 Dice/IoU 和边界距离；正常样本没有 GT mask 时按全零 mask 处理。

本目录仅用于缩放诊断，不改变任何训练、评测或冻结配置。

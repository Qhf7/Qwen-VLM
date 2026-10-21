# 跨零件异常分割（Cross-Part Anomaly Segmentation）

用 **DINOv2 的密集视觉特征** + **VLM 的语言语义**，验证「语义能弥补纯视觉模型在
未见零件类别上的泛化缺陷」。

> **先读这两份文档，再动代码：**
> - [`docs/项目说明.md`](docs/项目说明.md) —— 任务、数据、协议、每个实验的设计与读法（唯一依据）
> - [`docs/实验进度与版本记录.md`](docs/实验进度与版本记录.md) —— 版本迭代、实验台账、已知问题
>
> 框架图：[`docs/框架图.png`](docs/框架图.png)

## 目录

```
seg_cross_part/           实验代码（分割）
├── common.py             路径 / 配置 / split 解析 / 台账
├── metrics.py            AUROC、AUPR、F1max、阈值
├── metrics_pixel.py      AUPRO@0.3、池化像素指标
├── build_mask_manifest.py
├── train_seg.py          A0 训练器（A1/A2/A3 待接）
└── protocols/

artifacts/
├── manifest.parquet      冻结 manifest
├── splits/               Protocol A/B/C（冻结）
└── seg_cross_part/       manifest_pixel、results.csv、各 exp 产物

docs/                     文档与图
Dataset/                  24 类工业零件图像 + mask
models/                   dinov2-base、Qwen3-VL-2B
third_party/              WinCLIP / AdaCLIP 参考实现
论文参考/ 文献阅读/        文献
```

## 快速开始

```bash
# 开跑前复核（强制；train_seg 也会自动跑）
python -m seg_cross_part.preflight --protocol protocol_c_unseen_category --fold 0

python -m seg_cross_part.build_mask_manifest --verify 500   # 数据地基
python -m seg_cross_part.train_seg --smoke                   # 离线冒烟（无需权重）
python -m seg_cross_part.train_seg --protocol protocol_b_seen_category \
  --backbone models/dinov2-base --arm a0 --exp-id b_a0_s20261020

# 离线复评（分层 + 统计量）与 bootstrap CI
python -m seg_cross_part.evaluate_pixel --run artifacts/seg_cross_part/b_a0_fix_s20261021 --split test
python -m seg_cross_part.bootstrap_pixel --stats artifacts/seg_cross_part/b_a0_fix_s20261021/stats_test.npz
```

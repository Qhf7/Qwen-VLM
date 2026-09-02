# AdaCLIP 复现说明

本目录用官方源码提交 `b762ac40c3f33c77e7e513e48cb436f059d456da` 作为实现主体，
源码位于 `third_party/AdaCLIP`，许可证为 MIT。适配代码不改变网络结构和论文超参，
只解决本项目的路径、批量推理、结果落盘和补充评测问题。

唯一的官方模型文件兼容补丁位于 `method/adaclip.py`：进入现代 scikit-learn KMeans 前，
把 MPS 产生的 float16 patch 特征转为 CPU float32。旧版 CUDA 环境隐式走的也是适合
KMeans 的 CPU 数值路径；不改网络参数、K、聚类算法或模型输出 dtype。

## 已准备的文件

- OpenAI CLIP 骨干：`models/adaclip/ViT-L-14-336px.pt`
  - SHA256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- 官方 prompt 权重：`third_party/AdaCLIP/weights/`
  - `pretrained_mvtec_colondb.pth`：测试 VisA 及其他未见数据集
  - `pretrained_visa_clinicdb.pth`：测试 MVTec AD、ColonDB
  - `pretrained_all.pth`：仅用于演示；包含所有目标数据，不能算零样本
- BPE 词表：`third_party/AdaCLIP/method/bpe_simple_vocab_16e6.txt.gz`

这些大文件均被项目 `.gitignore` 排除，不应提交到 Git。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| ViT-L-14-336px.pt | 934088680 | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| pretrained_all.pth | 42673907 | `33e8d3db1cb4aab030866b8b70a46e10aa27ebf2c23b5463cb07f2574addd98c` |
| pretrained_mvtec_colondb.pth | 42673907 | `be51a42c052bd4cf060e54f503a1f5d0b2a3b899bc8dc2e243042f18b215427e` |
| pretrained_visa_clinicdb.pth | 42673907 | `3deabbbaf1e412cfdfcb42923a500b986f4b9ee96ccbc7a735d89dbc87df44c8` |
| bpe_simple_vocab_16e6.txt.gz | 1356917 | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

## 环境

论文原始环境是 Python 3.9.5、PyTorch 1.10.1+cu111、CUDA，ViT-L/14@336，
训练使用 48 GB A6000。官方训练超参为：输入 518、batch size 1、5 epochs、
AdamW、学习率 0.01、prompt depth 4、prompt length 5、静态+动态 prompt、
视觉+文本双分支、HSF 开启、K=20。

当前机器已验证 Python 3.12 + PyTorch 2.10 的 CPU/FP32 推理，也验证了 MPS/FP16
前向。MPS 用于本机完整测试提速，但它不是论文 CUDA 环境，数值差异必须与版本、设备一起
记录，不能把 MPS 结果直接称为原论文数字。

## 本地图像推理

```bash
python3 -m adaclip_repro.infer \
  --device cpu \
  --checkpoint third_party/AdaCLIP/weights/pretrained_all.pth \
  --item 'Dataset/3_adapter/NG/FiberorBurrs/S0001/3_adapter__0001_NG_FiberorBurrs_C1_20240530131319341.png::adapter' \
  --item 'Dataset/3_adapter/OK/S0001/3_adapter_0001_OK_C1_20240530102209163.png::adapter' \
  --output artifacts/adaclip/inference
```

输出含连续 `anomaly_score`、原始 `.npy` 异常图、灰度异常图、叠加图、CSV 和运行配置。
AdaCLIP 是排序模型；不要默认使用 0.5 作为部署阈值。阈值只能从独立 dev 集确定。
HSF 的官方 KMeans 未设置 `random_state`，所以 runner 固定 seed=111 且保留输入顺序；
比较两次结果时必须使用相同的样本顺序。

## 论文数据集评测

先按官方 README 下载处理后的数据，目录必须包含 `meta.json`。例如：

```text
/path/to/adaclip_datasets/
├── VisA_20220922/meta.json
├── mvtec_anomaly_detection/meta.json
└── ...
```

VisA 严格零样本评测（训练权重来自 MVTec AD + ColonDB）：

```bash
python3 -m adaclip_repro.evaluate_dataset \
  --dataset visa \
  --data-root /path/to/adaclip_datasets \
  --checkpoint third_party/AdaCLIP/weights/pretrained_mvtec_colondb.pth \
  --device cuda --precision fp32 \
  --output artifacts/adaclip/paper_eval
```

MVTec AD 严格零样本评测（训练权重来自 VisA + ClinicDB）：

```bash
python3 -m adaclip_repro.evaluate_dataset \
  --dataset mvtec \
  --data-root /path/to/adaclip_datasets \
  --checkpoint third_party/AdaCLIP/weights/pretrained_visa_clinicdb.pth \
  --device cuda --precision fp32 \
  --output artifacts/adaclip/paper_eval
```

当前已准备好的 KSDD 全量严格零样本评测（训练权重来自 MVTec AD + ColonDB）：

```bash
python3 -m adaclip_repro.evaluate_dataset \
  --dataset sdd \
  --data-root Dataset/adaclip_paper \
  --checkpoint third_party/AdaCLIP/weights/pretrained_mvtec_colondb.pth \
  --device cuda --precision fp32 \
  --output artifacts/adaclip/ksdd_full_cuda
```

`--limit N` 只用于代码冒烟，带 limit 的输出绝不是完整论文结果。

## 训练与 checkpoint 选择

官方命令可以重训，但 `train.py` 每个 epoch 都在 `--testing_data` 上计算像素 F1，
并用它保存最佳模型。如果同一数据又被当成最终测试集，这会泄漏测试标签。
严格协议应拆成：辅助训练集、辅助验证集、完全未见的目标测试集。只用辅助验证集选
checkpoint；目标测试集固定配置后只运行一次。

## 自检

```bash
python3 -m adaclip_repro.metrics
python3 -m adaclip_repro.paper_results \
  --output artifacts/adaclip/paper_reported_results.csv
```

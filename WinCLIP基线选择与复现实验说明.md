# 异常检测基线选择与 WinCLIP 0-shot 复现实验

## 1. 当前实验推荐基线

| 模型 | 轨道 | 连续图像分数 | 连续异常图 | 当前实验可评测内容 |
|---|---|---:|---:|---|
| Qwen2.5-VL-3B-Instruct | 零样本 VLM | ✓ | × | 视角/五视角图像级、对话格式、粗位置、延迟 |
| Qwen3-VL-2B-Instruct | 零样本小模型对照 | ✓ | × | 同上，主要作速度和负对照 |
| WinCLIP 0-shot | 无目标图像 CLIP | ✓ | ✓ | 图像级、像素级、区域级、五视角、延迟 |
| AdaCLIP | 外部辅助数据训练后目标类零样本迁移 | ✓ | ✓ | 图像级、像素级、区域级、五视角、延迟 |
| AnomalyGPT | 专用对话式异常 VLM | ✓ | ✓ | 需额外适配；另报 Yes/No、3×3 位置和类型 |
| WinCLIP+ | 1/2/4 normal-shot | ✓ | ✓ | 必须单列少样本 support/query 轨道 |

`Transformer.pdf` 是《Attention Is All You Need》，是机器翻译架构论文，不能作为异常检测模型基线。

## 2. 当前项目的统一口径

- 视角标签：mask 非空才是异常；NG 零件中没有 mask 的视角按该视角正常。
- 五视角样本标签：`sample_label = max(view_label_C1...view_label_C5)`。
- 五视角主分数：`sample_score = max(view_score_C1...view_score_C5)`。
- 视角/样本图像级主指标：I-AUPR、I-AUROC、F1-max；另报 dev 固定阈值下的 precision、recall、FPR、FNR、specificity。
- 像素级主指标：P-AUPR、P-F1、AUPRO@0.3；P-AUROC 只作参考。
- 所有方法都保存连续 score 和连续 anomaly map；二值判定/二值 mask 由 dev 冻结阈值产生。
- 结果先按类别计算，再做 class macro；五视角 group 作为 bootstrap 单位。

严格 holdout-zs 只比较 Qwen-VL、WinCLIP、AdaCLIP 和具备外部 checkpoint 的 AnomalyGPT。WinCLIP+、PatchCore k-shot 等需要另外建立 support/query 轨道。

## 3. WinCLIP 0-shot 复现对象

本次实现复现 WinCLIP 的算法而不是直接抄论文表格：

1. normal/anomalous 两组状态词与视觉检查模板的 compositional prompt ensemble；
2. CLIP 图像特征的 image-scale、2×2 patch-window、3×3 patch-window；
3. 每个窗口的异常文本相似度；
4. 重叠窗口的 harmonic averaging；
5. 多尺度异常图平均并上采样到原图评测网格。

论文默认是 LAION-400M 的 ViT-B/16+ 240。由于该 794 MB 权重在当前网络环境下载过慢，本机复现使用已经存在的 `models/adaclip/ViT-L-14-336px.pt`，因此正式标注为：

```text
WinCLIP-0shot (algorithm reproduction, local OpenAI CLIP ViT-L/14@336 backbone)
```

这可以作为当前数据集的可运行 CLIP 基线，但不能称为论文默认 backbone 的精确复现。若要严格对齐论文，只需替换为 `vit_b_16_plus_240-laion400m_e31-8fb26589.pt` 并保持其窗口网格与预处理。

## 4. 运行

安装依赖（已有 `torch`、`PIL`、`numpy`、`scikit-learn` 即可）：

```bash
python3 -m winclip_repro.run \
  --device cpu \
  --clip-checkpoint models/adaclip/ViT-L-14-336px.pt \
  --item 'Dataset/3_adapter/NG/FiberorBurrs/S0001/3_adapter__0001_NG_FiberorBurrs_C1_20240530131319341.png::3 adapter' \
  --item 'Dataset/3_adapter/OK/S0001/3_adapter_0001_OK_C1_20240530102209163.png::3 adapter' \
  --output artifacts/winclip/smoke
```

`--item` 的格式是 `图片路径::英文类别名`。输出包括 JSON/CSV 分数、`.npy` 连续异常图、灰度图和运行配置。

## 5. 已完成的本地冒烟结果

使用一张有 mask 的 NG 视角和一张无 mask 的 OK 视角，本地 ViT-L/14@336 验证已完成：

| 指标 | 结果 |
|---|---:|
| NG 视角图像分数 | 0.228050 |
| OK 视角图像分数 | 0.219041 |
| 两图 I-AUROC | 1.000 |
| 两图 I-AUPR | 1.000 |
| 两图 map-grid P-AUROC | 0.183888 |
| 两图 map-grid P-AUPR | 0.005506 |

这个结果只是链路冒烟，不是当前数据集的正式成绩：样本数仅 1 正 + 1 负，异常图仅为 24×24 网格，微小 mask 会在下采样时丢失。原始产物位于 `artifacts/winclip/smoke_pair/`。

WinCLIP 正式评估应在 CUDA 主机上遍历 S0 的 dev/holdout manifest，每个视角调用一次 `predict()`，将 24×24 anomaly map 对齐到冻结的 448 评测网格，再按 `group_path` 做五视角 max 聚合。

在 CUDA 主机上运行整个 S0 split：

```bash
python3 -m winclip_repro.evaluate_manifest \
  --fold 1 --split dev --device cuda \
  --checkpoint models/adaclip/ViT-L-14-336px.pt \
  --output artifacts/winclip/fold1_dev_vitl
```

只做链路冒烟时可加 `--limit-groups 2`。该脚本输出五视角 max 样本级指标、视角级 native-grid 像素指标、`view_scores.csv` 和 `view_maps_native.npy`。正式报告时需将 native 图对齐到 S0 的 448 网格，并补充 AUPRO、F1-max 和类别宏平均。

## 6. 结果解释纪律

- 以上本地 backbone 结果只用于验证算法链路和当前数据集对比，不得与论文 MVTec/VisA 数字直接并列为同一实验。
- WinCLIP 0-shot 不使用当前目标类图像；类别英文名和 prompt 集合必须在推理前冻结。
- 若使用具体缺陷名称 prompt，必须另标为 `defect-taxonomy-aware`，不能与 generic zero-shot 混合。
- 连续分数用于 AUROC/AUPR 和五视角聚合；生成的硬标签不是唯一结果。
- 若使用当前 S0 编排器，需把模型原生 336 输入得到的 anomaly map 对齐到 S0 的 448 评测网格；不能把 336 的 mask 尺寸直接当成 GT 尺寸。

## 7. 正式对比表建议

| Model | Setting | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | AUPRO@0.3 | 5-view p50/p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | ZS | ✓ | ✓ | ✓ | N/A | N/A | N/A | N/A | ✓ |
| WinCLIP-0shot-local | ZS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AdaCLIP | ZS-transfer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AnomalyGPT | ZS-transfer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ours | 对应轨道 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

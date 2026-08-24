# DINOv2 + VLM Finetune v4

独立于 `v5_multiscale_v3` 的训练型 VLM 版本，不覆盖旧代码、checkpoint 或提交包。

## 主要模块

- 冻结 DINOv2 ViT-B/14 多层特征；
- 从 v3 初始化多尺度 DINO pixel decoder；
- 冻结通用 OpenAI CLIP ViT-B/16 主干；
- CoOp 式 Static Prompt；
- 图像条件 Dynamic Prompt；
- CLIP patch Visual Adapter；
- 1×1/3×3/5×5 局部图文对齐；
- background/normal surface/edge/reflection + 8类缺陷的 12 状态像素头；
- 8个 Anomaly Query；
- DINO/VLM/Query 动态 Gate；
- 五视角 MIL；
- 有界流式类别/视角和 global DINO memory bank；
- 五折 leave-category-out Zero-Shot proxy。

完整设计见根目录：`VLM_FINETUNE_V4_METHOD.md`。

## 合规边界

- 训练只使用官方 `Train` 正常图；
- 伪异常来自程序化变换和官方图自裁剪；
- 只使用通用 DINOv2、CLIP 基础权重；
- 不使用外部异常数据、异常专项 checkpoint、在线 API 或 Test 标签。

## 文件

| 文件 | 用途 |
|---|---|
| `states.py` | 12状态与生成器类型映射 |
| `pseudo_states.py` | 类型化伪异常和 hard-normal 状态标签 |
| `clip_dense.py` | CLIP dense token、Prompt、Adapter |
| `model_v4.py` | Query、Pixel Head、Gate、五视角 MIL |
| `losses_v4.py` | 联合损失 |
| `memory_v4.py` | 有界正常样本 memory bank |
| `splits.py` | 类别级五折划分 |
| `train_v4.py` | 训练 |
| `proxy_v4.py` | held-out category proxy |
| `infer_v4.py` | Test推理和ZIP生成 |
| `validate_v4_submission.py` | 提交格式完整校验 |

## 推荐执行顺序

如果服务器还没有通用 OpenAI CLIP ViT-B/16 权重，先在本机运行：

```bash
bash setup_v4_server_dependencies.sh
bash prepare_v4_clip_weight.sh
```

脚本只下载 OpenAI 官方通用基础权重并校验固定 SHA256，不会下载异常专项权重。

先做一折 Zero-Shot 验证：

```bash
export DATA_ROOT=/root/autodl-tmp/Dataset
export WEIGHT_DIR=/root/autodl-tmp/weights
export CLIP_WEIGHT=/root/autodl-tmp/weights/ViT-B-16.pt
export V3_CHECKPOINT=/root/autodl-tmp/qwen_dinov2_multiscale_v3_run_20260821_190619/ckpt/decoder.pt
bash run_v4.sh
```

只有 held-out Pixel AP/F1max 相对冻结 VLM 基线确有改善，才运行全部类别最终训练：

```bash
bash run_v4_final.sh
```

## 注意

`run_v4.sh` 默认只运行 fold 0，目的是先确认 VLM 微调有效，不直接消耗五折完整训练时间。
正式结构选择需要轮换 `--heldout-fold 0..4`。最终训练使用 `--heldout-fold -1`，此时全部
50类进入训练。

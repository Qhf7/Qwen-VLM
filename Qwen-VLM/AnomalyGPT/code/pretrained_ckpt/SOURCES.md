# 权重来源与校验记录

> 下载时间：2026-10-22 19:39–19:52（Asia/Shanghai）
> 校验方式：文件大小 + PyTorch 实际加载 + 键名核对

## 1. 已下载

| 文件 | 大小 | 来源 | 状态 |
|---|---|---|---|
| `imagebind_ckpt/imagebind_huge.pth` | 4,803,584,173 B（4.5G） | https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth | ✅ 可 `torch.load`，键名匹配 `imagebind_huge()` |
| `vicuna_ckpt/7b_v0/` | 13,476,966,481 B（≈12.6G） | HF `Ejafa/vicuna_7B_vanilla`（社区镜像） | ⚠️ 见第 3 节 |
| `pandagpt_ckpt/7b/pytorch_model.pt` | 75,557,949 B（72M） | HF `openllmplayground/pandagpt_7b_max_len_1024` | ✅ |
| `ckpt/train_mvtec/pytorch_model.pt` | 215 MB | HF `FantasticGNU/AnomalyGPT` | ✅ |
| `ckpt/train_visa/pytorch_model.pt` | 215 MB | 同上 | ✅ |
| `ckpt/train_supervised/pytorch_model.pt` | 215 MB | 同上 | ✅ |

合计约 **18.7 GB**。

## 2. 校验结果

**ImageBind**
```
顶层键示例：modality_preprocessors.vision.cls_token / modality_trunks... / modality_heads...
```
与 `openllama.py:171-172` 的 `load_state_dict(..., strict=True)` 对齐（官方 checkpoint 就是该模块的 state_dict）。

**AnomalyGPT 权重**（21 个键，全部对上）
```
image_decoder.fc.{0..3}.{weight,bias}       → 4 × (1024, 1280)
prompt_learner.base_prompts                 → (9, 4096)
prompt_learner.meta_net.{0,3,6,9,12,15}     → 6 个卷积层
  └ 最后一层 (4096, 1024, 5, 5)             → 对应 9 个 prompt token 的 3×3 输出
```
与 `model/AnomalyGPT_models.py` 的 `LinearLayer(1280,1024,4)` + `PromptLearner(1,4096)` 完全一致。

**Vicuna**
- 两个 shard 都是合法的 torch zip 存档（`pytorch_model-0000x/data.pkl`）
- 索引 323 个权重键，`total_size` 13,476,855,808
- 实际两个文件合计 13,476,966,481 B，**比索引多 110,673 B**——经核对 HF 上的真实文件大小（`x-linked-size`）与本地一致，是索引里的 `total_size` 元数据略旧，**文件本身没问题**

## 3. ⚠️ Vicuna 的来源需要你知道

论文要求 **Vicuna-7B v0**。官方通道是：

```
LLaMA-7B（需向 Meta 申请）+ lmsys/vicuna-7b-delta-v0（HF 公开）
   └ 用 fastchat 的 apply_delta 合并 → 才是官方 v0
```

本次下载的是 **社区已合并的镜像** `Ejafa/vicuna_7B_vanilla`，原因：
1. `lmsys/vicuna-7b-v0`（官方合并版）已变为 **401 不可访问**；
2. 官方 delta 需要 LLaMA 底座，而合并过程要额外 26GB+ 磁盘，本机当时只剩 32GB。

**风险**：无法 100% 确认该镜像与官方 v0 逐位一致（它是 `LlamaForCausalLM`、fp16、13.47GB，结构正确）。
**如果你要严格复现**，请换成"官方 delta + LLaMA"合并出来的版本，替换 `vicuna_ckpt/7b_v0/` 即可，代码无需改动。

## 4. 还没有的东西

| 缺什么 | 用途 | 怎么补 |
|---|---|---|
| MVTec-AD | 训练/评测 | mvtec.com 填表下载，放 `data/mvtec_anomaly_detection/` |
| PandaGPT 图文数据 | 交替训练 | HF `openllmplayground/pandagpt_visual_instruction_dataset`，约 20GB |
| VisA（可选） | 跨域评测 | amazon-science/spot-diff |

# B0 本地模型冒烟测试

固定子集来自 `fold1/dev.parquet`，按分辨率覆盖选择 10 类，每类 1 个 OK + 1 个
NG，共 20 个完整样本、100 个视角。两个模型必须复用同一份子集和 prompt。

## 实测结果

设备为 MacBook Pro M3 Pro（18 GB 统一内存），后端 MPS + FP16。图片先用
`INTER_AREA` 直接缩放到 448×448；生成的第一个 token 读取 Yes/No logits，样本分数
为 5 视角 `P(Yes)` 的 max。warm 延迟排除了含 MPS 图编译的首个完整样本。

| 模型 | 参数 | 视觉网格 | 契约解析 | Yes 输出 | 5-view AUROC / AUPR | 宏平均 AUROC | 单视角 p50/p95 | 5 视角 p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B-Instruct | 2.13B | 1×28×28 | 100% | 100% | 0.5400 / 0.5785 | 0.5000 | 2.89 / 3.42 s | 14.48 / 16.14 s |
| Qwen2.5-VL-3B-Instruct | 3.75B | 1×32×32 | 100% | 0% | 0.7250 / 0.7879 | 0.7000 | 4.06 / 5.07 s | 21.30 / 23.62 s |

两者都能在本机完整跑通 448 预处理、next-token logits、5 视角聚合、JSON
与位置词解析，无需服务器完成 B0。但零样本的 0.5 阈值均已退化：2B 全部
输出 Yes，3B 全部输出 No，样本级 accuracy 都为 50%。3B 的连续分数排序更有信息，
因此作为 B0+ 的准确性主线；2B 保留为较快的延迟对照。两者都远未达到
1 s/完整样本的目标。完整表见 `artifacts/b0/comparison.md`，详细结果解读与
后续决策见 `../B0-baseline分析与下一步.md`。

## 复现

```bash
python3 b0/make_smoke_subset.py

python3 b0/run_smoke.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --slug qwen3_vl_2b --resume

python3 b0/run_smoke.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --slug qwen25_vl_3b --resume

python3 b0/compare_models.py
```

每个视角都会立即写入 CSV，因此中断后加 `--resume` 可继续。输出首 token 的
Yes/No 归一化概率、生成文本、JSON/位置词解析、预处理/生成耗时；最终按 5 视角
max 聚合并报告图像级指标。

本次 3B 因 Hugging Face 匿名链路过慢，实际从 ModelScope 官方镜像下载，
再将 `--model-id` 指向本地 snapshot；权重名称与测试协议不变。

## B0+ smoke20 结果

`B0plus-preregistered-v1` 在推理前冻结 P0/P1/P2、prompt 选择规则和 5-crop 定义。
B0f 使用 `use_cache=False`、`logits_to_keep=1`：它与 generate 首 token 的最大
P(Yes) 误差为 `4.87e-06`，batch5 与单图评分的最大误差为 `0`。

| 实验 | AUROC | AUPR | F1max | 宏 AUROC |
| --- | ---: | ---: | ---: | ---: |
| P0 + max（预注册胜者） | 0.7250 | 0.7879 | 0.7500 | 0.7000 |
| P1 + max | 0.5550 | 0.5299 | 0.7407 | 0.6000 |
| P2 + max | 0.6900 | 0.8101 | 0.7500 | 0.6000 |
| P0 + top-2 mean（离线） | 0.7500 | 0.8001 | 0.7826 | 0.7000 |
| P0 + mean（离线） | 0.7700 | 0.8261 | 0.7619 | 0.7000 |
| P0 + 5-crop | 0.5700 | 0.6695 | 0.7143 | 0.7000 |

单视角 B0f 总延迟 p50/p95 为 1.875/3.398 s；真实 5 图 batch prefill 为
28.757/44.400 s，在本机 18GB 统一内存上反而比 5 次串行中位估计慢 3.07 倍。
P0 5-crop 的 5 视角延迟为 39.324/50.294 s，且准确性显著下降，因此按预注册
阶段门停止 B0w，不打开原尺度滑窗。

复现完整 B0+（支持逐组续跑）：

```bash
python3 b0/run_b0plus.py
python3 b0/report_b0plus.py
```

详细分析见 `../B0-baseline分析与下一步.md` 第 11 节；机器可读表和独立报告位于
`artifacts/b0plus/`。

## B0+ Dev-mini 720 最终基线

冻结的 `Qwen2.5-VL-3B + P0 + direct448 + B0f` 已在 RTX 4090D 上扩展至
18 类、720 个完整样本、3600 个视角。主口径 max 的 pooled
AUROC/AUPR/F1max 为 `0.6674/0.6697/0.6781`，18 类宏平均为
`0.7208/0.7495/0.7380`。top-2 mean 和 mean 宏 AUPR 为 `0.7648/0.7734`，
作同一次推理的离线对照保留，不事后替换冻结主口径。

CUDA 上 forward/generate 首 token 差 `3.81e-06`，但 FP16 batch5/single 最大
差 `0.00522` 超过预注册容差，因此正式分数全部使用逐图 B0f。
4090D 上一个 5 视角样本端到端 p50/p95 为 `0.608/0.865 s`，
独立 batch5 prefill 测量为 `0.584 s`。

复现：

```bash
python -m b0.run_dev720 \
  --model-id models/Qwen2.5-VL-3B-Instruct \
  --subset artifacts/splits/fold1/dev.parquet \
  --root Dataset448 \
  --preprocessed-cache-manifest artifacts/b0plus/dev720_cache_manifest.csv

python -m b0.analyze_dev720
```

完整报告和原始产物位于 `artifacts/b0plus_dev720/`。

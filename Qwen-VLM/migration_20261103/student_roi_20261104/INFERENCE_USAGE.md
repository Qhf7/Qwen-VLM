# Student ROI 单图推理

脚本：`/root/autodl-tmp/AnomalyGPT/migration_infer_student_roi.py`。输入任意可读取的 RGB 图片；不需要 label、category、GT mask、teacher cache 或历史 control 结果。流程为 Qwen pre448 frozen visual＋已选 SpatialHead → float32 map224 → 原图与预测 ROI 两图 → 完整 native Qwen BF16 评分与回答。两图各 448×448，合计 392 个 visual tokens、470 个 prompt tokens；ROI 相对面积为 1/9。

## 服务器命令

先将 `--image` 换成待测图片绝对路径，`--outjson` 使用全新路径。以下用 GT head；GPU 锁与实验队列共用，锁被占用时 `flock -n` 直接退出，请待队列结束后重试。

```bash
flock -n /root/autodl-tmp/AnomalyGPT/runs/migration_20261103/gpu.lock \
  /root/autodl-tmp/conda-envs/qwen3vl2/bin/python \
  /root/autodl-tmp/AnomalyGPT/migration_infer_student_roi.py \
  --model /root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct \
  --head /root/autodl-tmp/AnomalyGPT/runs/migration_20261103/dense_pre_gt_s42/best.pt \
  --image /path/to/any_image.JPG \
  --outjson /root/autodl-tmp/AnomalyGPT/runs/migration_20261103/student_demo/gt_new_image.json \
  --device cuda:0 --max-new-tokens 32 --heatmap --run
```

使用 KD head 时，只替换 `--head` 为 `/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/dense_pre_kd_s42/best.pt`，并换一个新的输出路径。

- `--run`：实际执行 CUDA 推理，需要支持 BF16 的 GPU。
- 默认不带 `--run`：只做 CPU 来源、权重与 head 兼容性审计，写 plan/status，不加载视觉塔或完整语言模型，不生成预测 JSON 或 map。
- `--heatmap`：额外输出固定 [0,1] 概率标尺 PNG，不做逐图归一化。
- 所有输出防覆盖。CPU plan 与实际 run 应使用不同的新 `--outjson` stem；例如 `gt_plan.json` 与 `gt_run.json`。

## 输出

若 `--outjson /path/demo.json`，成功时输出：

| 文件 | 内容 |
|---|---|
| `demo.json` | `score_yes_minus_no`、Yes/No logits、`generated_text`/token IDs、两种决策、ROI、dtype/source/head/input 身份、SHA、计时与 GPU 峰值 |
| `demo.map.npy` | float32、224×224、有限且范围 [0,1] 的原始概率图 |
| `demo.map.heatmap.png` | 可选固定标尺热图 |
| `demo.plan.json` | CPU 审计计划 |
| `demo.status.json` | planned/running/complete/failed 状态；失败记录错误及部分产物 |

`score_decision` 使用 Yes−No score > 0 判 Yes；`generated_decision` 独立解析自由回答开头的 Yes/No，无法解析时为 null，并标记 `generated_invalid`。只有全部推理成功才发布最终 `demo.json`。

## 使用边界

GT head 训练使用过异常 GT mask；KD head 还使用过 weight=0.5 的训练 teacher 概率监督。此入口推理时不读取这些监督，不代表无监督训练或原生 Qwen 零样本定位。当前 head 由同一开发 val 选 best，再在 val380 报告结果；不构成独立 holdout 证据。

单图运行依次加载视觉塔＋head、释放其 GPU 引用，再加载完整 Qwen；每次调用均有两次模型加载。日志分列加载与推理耗时，这一验证入口不应作为低延迟部署吞吐的测量结果。dense 使用确定性 cuDNN 设置，结束后恢复原 cuDNN 三项状态再进入 native 阶段。

## 固定样例实测

2026-11-04 已在远程 GPU 完成 GT head 两例单图推理，并与已有批量结果精确对照。[demo/verification.json](demo/verification.json) 为 `status=complete`、`passed=true`、`n=2`：map array SHA、map 文件 SHA、完整 ROI、Yes/No logits、score、生成文本/token IDs 及两图 392/470 token 预算全部一致，两例 score 最大绝对误差均为 0。该 verification 本身只读保存产物，所以其中 `gpu_used=false` 指离线核验过程。

固定输入位于 `/root/autodl-tmp/industrial_vlm/datasets/VisA/candle/Data/Images/`：

| 输入 | Yes−No score | 自由生成 | 含审计与两次加载耗时 | 结果 |
|---|---:|---|---:|---|
| `Normal/0784.JPG` | −3.75 | No, the target product appears normal and consistent with the close-up view. | 20.382602 s | 正常判定正确；与批量精确一致 |
| `Anomaly/010.JPG` | −1.25 | No, the target product appears normal with no visible anomalies. | 20.216672 s | 异常漏检；与批量精确一致 |

两例 allocated 显存峰值均为 17,694,053,376 bytes（16.478871 GiB），dense 阶段释放后 allocated 均为 8,519,680 bytes。原始 JSON、map、heatmap、plan/status 保存在 `demo/normal0784.*` 与 `demo/anomaly010.*`；本地已复核 JSON、数组、NPY/PNG 文件 SHA。

实测日志出现 cuDNN execution-plan fallback 提示及 `pynvml` 弃用提示；两次运行均 exit 0、结果 complete，上述精确对照通过。

这两例用于入口与批量路径一致性验证，不是新的统计性能评测。异常图保留真实漏检，不因验证通过而视为检测正确。

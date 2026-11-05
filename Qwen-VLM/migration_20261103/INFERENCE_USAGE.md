# 路线1单图推理

2026-11-04已在远程RTX4090D实测。脚本：/root/autodl-tmp/AnomalyGPT/migration_infer_language.py。本地代码快照：code/migration_infer_language.py。该入口属于“ImageBind＋BigDec＋Qwen语言骨干”路线；不是Qwen原生视觉map分支或独立GT/KD head入口。

## 已验证命令

在服务器上运行；选择全新outjson路径即可换图。类别须为已支持的VisA类别。脚本没有--run开关，正常调用即执行推理。flock避免与本实验队列同时占GPU。

```bash
flock -n /root/autodl-tmp/AnomalyGPT/runs/migration_20261103/gpu.lock \
  /root/autodl-tmp/conda-envs/qwen3vl2/bin/python \
  /root/autodl-tmp/AnomalyGPT/migration_infer_language.py \
  --backend qwen \
  --ckpt /root/autodl-tmp/AnomalyGPT/runs/migration_20261103/lm_qwen_s42/best.pt \
  --model /root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct \
  --decoder /root/autodl-tmp/AnomalyGPT/runs/dec_big_fixed_s42/best.pt \
  --imagebind /root/autodl-tmp/AnomalyGPT/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
  --image /root/autodl-tmp/industrial_vlm/datasets/VisA/candle/Data/Images/Normal/0784.JPG \
  --category candle \
  --outjson /root/autodl-tmp/AnomalyGPT/runs/migration_20261103/route1_demo/qwen_s42_new_image.json \
  --device cuda:0 --max-new-tokens 32 --heatmap
```

代码严格核验decoder SHA和训练checkpoint的backend/pipeline/桥接state，且重新读取训练后的PromptLearner与cls_proj。正常/异常输入均无需标签、GT mask、split或teacher cache；类别文本是输入条件。输出保存JSON、float32原概率图.map.npy及可选固定[0,1]标尺热图，不做每张图独立归一化。

## 实测产物

| 输入 | 选择方式 | Yes−No score | 自由生成 | 含加载耗时 | Allocated峰值 |
|---|---|---:|---|---:|---:|
| candle Normal/0784.JPG | 前置接口检查已使用的正常图 | −1.750 | No, there is no anomaly in the image. | 43.48s | 18.375 GiB |
| candle Anomaly/010.JPG | 已有完整val顺序中的首张异常图，不按预测结果筛选 | 2.625 | Yes, there is an anomaly in the image. | 44.11s | 18.375 GiB |

两例score决策与生成一致、invalid=false、strict加载state哈希通过、地图shape224×224/dtypefloat32/有限/范围[0,1]和文件SHA核验通过。产物在route1_demo/。这些是入口验证样例，不是新的统计性能评测；两例属于已使用数据。

## 使用边界

- score是原始logit差，不是校准后的异常概率；默认score>0判Yes。自由生成单独解析，不把文本自动等同于候选分数决策。
- 本入口不加载Qwen视觉tower。先重算FP16 ImageBind CLS与四层特征、FP32 BigDec概率图，释放teacher GPU后才载BF16 Qwen语言模型和FP32桥接。
- singleton teacher与batch8缓存的数值可有差异；单图结果不能要求与旧批量缓存逐位相等。
- native_map_s42/best.pt和dense_*_*/best.pt的结构不匹配该CLI，不应当作--ckpt传入。独立head现已通过另一入口接入ROI，见[Student ROI单图说明](student_roi_20261104/INFERENCE_USAGE.md)。该CLI参数仍不可混用。
- 两次实测出现cuDNN某候选执行plan不支持的warning后自动选择可用路径，进程exit0并保存全部产物；并未出现训练/推理失败。原有pynvml弃用warning不影响本轮输出。
- 任一目标JSON/map/heatmap已存在时脚本拒绝覆盖。保留已验证样例，更换输出文件名。

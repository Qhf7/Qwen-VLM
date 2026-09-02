# S0 数据与评测地基

S0 已完成。这里的代码固定数据口径、四折划分、指标、resize 和延迟测量；后续
B0/E1/E2/E3 只能替换 `Predictor`，不能临时修改评测定义。

依赖可用 `python3 -m pip install -r requirements-s0.txt` 安装。

## 冻结口径

- 一行 manifest = 一个视角；`view_label=1` 仅当 mask 非空。
- 图像级样本 = `group_path`（一次 5 视角拍摄），分数以 5 视角 max 聚合。
- `source_part_key=<类>/<Sxxxx>` 仅供追溯，不是跨批物理 ID。
- train/dev 防近重复泄漏代理 = `visual_group_id`：C1 64×64 灰度图在不同采集批
  间互为最近邻、MSE≤25，并保证一组内同一采集批最多一个成员。
- holdout-zs 按整类冻结；`splits.py` 默认复用已有 `holdout_classes.txt`。
- 主输入/指标网格 = direct 448；GT 用 nearest-neighbor 对齐。
- P-AUROC 只输出，不决策；决策看 P-AUPR、P-F1max、面积占比和小缺陷分档。

## 复现顺序

```bash
python3 s0/manifest.py --root Dataset --out artifacts/manifest.parquet
python3 s0/part_clusters.py --root Dataset --workers 8
python3 s0/splits.py
python3 s0/checks.py
python3 s0/metrics.py
python3 s0/eval.py --fold 0 --split dev --predictor constant \
  --policy r448 --workers 8 --out artifacts/run_dev_constant_r448.txt
python3 s0/latency.py --fold 0 --split holdout --predictor constant \
  --policy r448 --out artifacts/latency_constant_r448.txt
python3 s0/resize_audit.py --workers 8
```

## 已验证的不变量

- 24 类、5971 个样本目录、29855 个视角。
- mask 文件 23378 张：非空 23369、空 9；另有 482 个 NG 视角无 mask 文件。
- 四折各 6 个整类 holdout；每折 18 个 train 类的 dev 均为 20 OK + 20 NG。
- 每折 train/dev 的 `visual_group_id` 与 `group_path` 均不相交。
- 恒定基线全量 dev：I-AUROC=0.5、I-AUPR=0.5、P-AUROC=0.5。
- 随机 predictor 在 workers=1/8 下报告逐位一致。
- resize 分层样本 731 张：direct 448 后高覆盖核心消失率 0%；但原尺度滑窗
  每个 5 视角样本窗口数 p50=42、p95=174。

## 主要产物

```text
artifacts/manifest.parquet
artifacts/manifest_with_groups.parquet
artifacts/part_clusters.parquet
artifacts/part_clusters_report.txt
artifacts/splits/fold{0..3}/{train,dev,holdout}.parquet
artifacts/splits/split_report.txt
artifacts/run_dev_constant_r448.txt
artifacts/latency_constant_r448.txt
artifacts/latency_constant_rnative.txt
artifacts/resize_audit.txt
artifacts/resize_audit.parquet
```

`visual_group_id` 只是从图像与采集批结构推断出的近重复代理。没有真实实体 ID，
因此不能声称绝对消除了物理工件泄漏；这个限制必须随实验结论一起披露。

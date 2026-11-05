# 数据来源与说明

> 下载时间：2026-10-22 20:00–20:xx（Asia/Shanghai）

## 1. 已下载

| 数据 | 大小 | 来源 | 用途 |
|---|---|---|---|
| `mvtec_anomaly_detection/` | **5.9 GB**（14,601 个文件） | HF 镜像 `foersben/mvtec-ad`（原始为 MVTec 官方，CC BY-NC-SA 4.0） | 训练 + 评测 |
| `pandagpt4_visual_instruction_data.json` | 238 MB | HF `openllmplayground/pandagpt_visual_instruction_dataset` | 交替训练（防遗忘） |
| `images/` | 13 GB，84,918 张 | 同上（images.zip 解压） | 上者配套图片 |

## 2. 目录结构（代码期望的格式）

```
data/
├── pandagpt4_visual_instruction_data.json      ← 161,151 条图文指令
├── images/                                     ← 84,918 张 jpg
└── mvtec_anomaly_detection/
    ├── bottle/
    │   ├── train/good/000.png ... (209 张)
    │   ├── test/good/...
    │   ├── test/broken_large/...
    │   └── ground_truth/broken_large/000_mask.png
    ├── cable/ ... 共 15 类
```

`code/datasets/mvtec.py` 读取的正是这个结构；训练时异常图由 `self_sup_tasks.patch_ex` **在线合成**，不需要预先生成。

## 3. 校验（与官方数字逐项对齐）

| 项 | 本地 | 官方 |
|---|---|---|
| 类别数 | **15** | 15 |
| train/good | **3,629** | 3,629 |
| test 图 | **1,725** | 1,725 |
| ground_truth mask | **1,258** | 1,258 |

> 注：镜像里还带了一个 `results/`（1.8 GB，别人跑的实验曲线），与本数据集无关，已删除。

- PandaGPT JSON：161,151 条，字段 `image_name` + `conversation`；抽查 `image_name` 对应的文件在 `images/` 里存在 ✅
- MVTec：类目录齐全（15 类），每类含 `train/good`、`test/*`、`ground_truth/*` ✅

## 4. 许可提醒

| 数据 | 许可 | 限制 |
|---|---|---|
| MVTec-AD | **CC BY-NC-SA 4.0** | **仅限非商业研究/教学**；商用需向 MVTec 申请 |
| PandaGPT 数据 | 见原仓库 | 用于研究 |

如果这个项目要商用，MVTec 的许可需要单独处理。

## 5. 还缺（可选）

| 数据 | 用途 | 来源 |
|---|---|---|
| VisA | 跨域评测（论文报了训 VisA / 训 MVTec 两个方向） | github.com/amazon-science/spot-diff |

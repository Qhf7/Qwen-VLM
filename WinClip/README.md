# WinClip 材料目录

WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation (Jeong et al., CVPR 2023, arXiv 2303.14814)

## 目录里有什么

| 路径 | 是什么 |
|---|---|
| `paper.pdf` | 论文原文（arXiv v1，含 21 页：正文 10 页 + 附录 11 页） |
| `source/` | **arXiv 的 LaTeX 源码**（`main.tex` + `appendix.tex` + `tables/*.tex` + `figures/*.png` + `cvpr.sty` + `egbib.bib`），可以直接编译 |
| `paper.tex` | 上面源码拍平成一个文件（`\input` 已展开、注释和参考文献已去掉），带行号，读公式和表格以它为准 |
| `paper.txt` | PDF 逐页抽出的纯文本（按页分隔） |
| `outline.md` | 章节 / 公式在 `paper.tex` 里的行号，按需跳读 |
| `meta.json` | 标题、作者、日期、摘要、代码链接等元数据 |
| `figures/` | 每张图、每张表裁成的 PNG + `index.md`（caption 与正文引用位置）+ `_contact.png`（全部缩略拼图） |
| `code/` | **源代码**：社区复现 [caoyunkang/WinClip](https://github.com/caoyunkang/WinClip)（*unofficial*，见下） |
| `精读笔记.md` | 按六站式精读整理的笔记（① 动机 → ② 架构 → ③ 训练 → ④ 数据 → ⑤ 信息流 → ⑥ 例子 → ⑦ 实验） |

## 关于源代码：这篇论文没有官方实现

论文正文只给了两个链接——[OpenCLIP](https://github.com/mlfoundations/open_clip)（CLIP 的代码和权重）和
[amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)（baseline）。
作者没有放出 WinCLIP 自身的代码。社区有两份常用复现：

| 仓库 | 说明 |
|---|---|
| [caoyunkang/WinClip](https://github.com/caoyunkang/WinClip) | 本目录 `code/`。MVTec-AD + VisA 都能跑，`run_winclip.py` 一键跑全类，README 里有逐类结果对照表 |
| [mala-lab/WinCLIP](https://github.com/mala-lab/WinCLIP) | 你工作区 `third_party/WinCLIP` 里已有的那份，基于 InCTRL 的代码框架，另加了几个数据集 |

**注意**：`code/` 的复现指标明显低于论文（README 自报 MVTec 0-shot i-AUROC 70.17 / p-AUROC 80.67，
论文是 91.81 / 85.12）。已知的差异至少有：发布的脚本默认只用 2×2 和 3×3 两个尺度、缺掉论文的
image-scale；跨尺度是算术平均而非调和平均；非正方形图直接拉伸而没有做论文里的 tiling。
拿它当"论文的参考实现"读结构可以，拿它的数字当基线要谨慎。详见 `精读笔记.md` 的「代码 ↔ 论文」表。

## 怎么复现（只记录，未在本机执行）

```bash
cd code
sh install.sh                       # python 3.7 + torch 1.10
python run_winclip.py               # 改脚本里的 datasets / classes
# 或单类：
python eval_WinCLIP.py --dataset mvtec --class-name screw --k-shot 1
```

需要先按 MVTec-AD / VisA 官方协议下载数据并组织成 `DATA_PATH/<class>/train/good`、`test/<defect>/` 的结构。
预训练权重是 OpenCLIP 的 `ViT-B-16-plus-240` / `laion400m_e32`（首次运行自动下载）。

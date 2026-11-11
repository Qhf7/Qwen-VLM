# Qwen-VLM 技术栈与逐步实践教程

> 本文面向希望读懂并复现 `Qhf7/Qwen-VLM` 的学习者。内容根据仓库当前源码、配置、实验协议和项目报告整理，目标是让你能回答三类问题：
>
> 1. 这个项目到底用了哪些技术，它们各自解决什么问题？
> 2. 如果从零开始，应该按什么顺序准备环境、数据、模型和实验？
> 3. 每一步如何验证自己做对了，失败时应该先查哪里？
>
> 仓库地址：[https://github.com/Qhf7/Qwen-VLM](https://github.com/Qhf7/Qwen-VLM)
>
> 重要边界：代码仓库不携带完整数据集、底座权重、LoRA/decoder 权重和 tensor cache。教程中会明确区分“仓库里已有的代码”和“你必须另外准备的资产”。

## 1. 先建立正确的项目地图

### 1.1 项目要解决什么问题

项目研究的是工业异常检测的视觉语言模型适配。输入是一张工业零件图片和一条指令，输出至少包含“是否异常”，某些任务还要求输出一个覆盖缺陷区域的归一化框。项目同时研究：

- 把已有的 AnomalyGPT 视觉前端迁移到 Qwen3-VL 语言骨干；
- 使用 LoRA/QLoRA 做参数高效后训练；
- 让输出从自由文本变成可解析 JSON；
- 用连续分数、AUROC/AP、F1、FPR、框 IoU 和像素指标分别评估检测与定位；
- 通过哈希、协议、参数审计和保存重载检查，让实验能够复核。

这不是一个只有一个入口的“应用脚本”，而是一组有关联但协议不同的实验路线。阅读时先把路线分开，才能避免把一个实验的数字、依赖和训练口径套到另一个实验上。

### 1.2 三条主要路线

| 路线 | 视觉输入路径 | 语言部分 | 训练方法 | 主要代码入口 | 适合学习的主题 |
|---|---|---|---|---|---|
| A：迁移主线 | ImageBind ViT-H → BigDec 异常图 → CLS/PromptLearner 桥接 | Vicuna-7B 或 Qwen3-VL-8B 的语言部分 | 桥接训练、语言 q/v LoRA、联合适配、ranking、结构化 SFT | `migration_20261103/`、`repair_20261103/`、`AnomalyGPT/` | 视觉语言接口、软提示、受控消融 |
| B：原生 Qwen3-VL | Qwen3-VL 自带视觉塔、merger、DeepStack | Qwen3-VL 语言层 | 语言层 NF4 + LoRA 的 QLoRA | `industrial_vlm/` | 原生 VLM 输入、QLoRA、评分/推理工程 |
| C：早期跨零件分割 | DINOv2 patch 特征，CLIP/WinCLIP 作为语义对照 | 可选 Qwen3-VL SFT 语言分支 | 分割解码器、LoRA SFT、WinCLIP 基线 | `seg_cross_part/` | 密集预测、跨类别泛化、基线设计 |

### 1.3 先记住四个容易混淆的事实

1. **路线 A 不调用 Qwen 原生视觉塔。** 它把 ImageBind/BigDec 的结果转为语言侧的连续向量，再喂给冻结或部分微调的语言骨干。
2. **路线 B 保留 Qwen 原生视觉路径。** 视觉模块冻结，语言线性层做 NF4 量化并挂 LoRA。
3. **主线训练使用真实异常标注。** 它是项目自定义的有监督开发协议，不应直接标作原论文的 normal-only 无监督结果。
4. **不同阶段有不同的 Python/Transformers 版本。** 原生 QLoRA、后期迁移和旧版 AnomalyGPT 不应强行安装到同一个环境。

### 1.4 仓库目录与职责

```text
Qwen-VLM/
├── README.md                         项目总览、结果口径、快速开始
├── ASSETS.md                         需要外部准备的资产
├── THIRD_PARTY.md                    上游组件、许可证和来源边界
├── industrial_vlm/                   原生 Qwen3-VL QLoRA 工程
│   ├── src/industrial_vlm/
│   │   ├── data.py                   VisA/MVTec 清洗、切分、指令构造
│   │   ├── modeling.py               NF4 加载、LoRA、输入构造、参数审计
│   │   ├── train.py                  手写 SFT 训练循环
│   │   ├── evaluate.py               生成式评测、断点续跑、指标
│   │   ├── score.py                  双候选似然评分、校准、AUROC/AP
│   │   ├── metrics.py                JSON 解析、框指标、F1/FPR
│   │   ├── compare.py                配对比较和 bootstrap
│   │   ├── report.py                 报告生成
│   │   └── infer.py                  单图推理 CLI
│   ├── configs/                      实验配置
│   ├── ops/                          预检、运行编排、绘图、验证
│   └── reports/                      实验协议与结果
├── migration_20261103/               Qwen/Vicuna 语言迁移与后训练
│   ├── code/                         桥接、特征缓存、语言训练、指标
│   ├── lora_20261104/                B/C LoRA 配对实验
│   ├── advanced_20261104/            D/Cpair/R/S、结构化、工具、外评
│   └── student_roi_20261104/         学生 ROI 选择和推理
├── repair_20261103/                  mask/几何/特征布局修复
├── seg_cross_part/                   Real-IAD 跨零件分割及 Qwen SFT
├── AnomalyGPT/                       必要的 AnomalyGPT/ImageBind 运行时与脚本
├── docs/                             项目精读、协议、报告、图表
└── artifacts/、data/                 轻量数据、统计与证据文件
```

## 2. 全部技术栈清单

下面的“全部”指仓库实现、运行配置和实验协议中实际出现的技术；同一技术如果只属于旧路线，会在“所属路线”列中标明。

### 2.1 硬件、操作系统和运行方式

| 技术 | 项目中的具体形态 | 用途 | 证据入口 |
|---|---|---|---|
| NVIDIA CUDA GPU | RTX 4090D 24GB；迁移协议还记录 CUDA 12.1，原生运行时记录 CUDA 12.4 | 训练、Qwen 推理、ImageBind/DINOv2 前向 | `industrial_vlm/README.md`、`migration_20261103/README.md` |
| BF16 | `torch.bfloat16` | 原生 Qwen 视觉塔、迁移语言骨干和混合精度 | `industrial_vlm/src/industrial_vlm/modeling.py` |
| Linux 服务器 | 历史路径以 AutoDL 容器为主 | 运行 GPU 实验、保存 checkpoint 和日志 | 各路线协议 |
| macOS/本地开发机 | 只做阅读、CPU 逻辑自测、文档和图表 | 不承担 8B 训练 | 项目交付范围 |
| SSH/rsync | `industrial_vlm/ops/remote.py` | 本地与服务器同步代码/结果 | `industrial_vlm/ops/remote.py` |
| `flock` | GPU 文件锁 | 保证多实验串行，避免争抢同一张卡 | `migration_20261103/` 的运行脚本 |
| Git/GitHub | `main` 分支、提交哈希、远端仓库 | 版本管理、代码身份追踪 | `.git`、README |

### 2.2 Python、打包和环境

#### 工程语言与配置格式

- **Python**：模型、数据、训练、评测和审计主语言；
- **Bash/Shell**：服务器启动、串行实验、环境变量和 GPU 锁；
- **JSON/JSONL**：配置、逐样本指令、预测、指标和身份清单；
- **YAML**：`seg_cross_part/protocols/` 中的实验协议；
- **Markdown**：README、实验协议、报告和学习材料；
- **SVG/HTML/PNG**：架构图、实验路线图和可视化交付；
- **Chat template/Jinja**：Qwen processor 使用的模板由底座模型目录提供，仓库代码通过 `apply_chat_template()` 调用，不手写完整模板字符串。

#### 原生 QLoRA 路线

`industrial_vlm/pyproject.toml` 声明的下限是：

| 包 | 声明 |
|---|---|
| Python | `>=3.10` |
| PyTorch | `>=2.5.1` |
| torchvision | `>=0.20.1` |
| Transformers | `>=4.57.0` |
| PEFT | `>=0.17.0` |
| Accelerate | `>=1.0` |
| bitsandbytes | `==0.48.2` |
| Pillow | `>=10` |
| NumPy | `>=1.26` |
| safetensors | `>=0.4` |

`industrial_vlm/requirements-runtime.txt` 是服务器实测冻结版本：

```text
torch==2.5.1
torchvision==0.20.1
transformers==5.14.1
peft==0.21.2
accelerate==1.14.0
bitsandbytes==0.48.2
numpy==2.5.2
Pillow==10.4.0
safetensors==0.8.0
huggingface-hub==1.29.0
```

这里要理解“下限”和“冻结版本”的区别：下限用于创建一个兼容环境，冻结文件用于复现某次服务器运行。不要把“安装成功”解释成“数值一定逐位一致”。

#### 迁移路线环境

`migration_20261103/README.md` 记录的后期服务器环境是 Python 3.10、PyTorch 2.3.0+cu121、Transformers 4.57.1、PEFT 0.7.1。它还依赖 AnomalyGPT/ImageBind 运行时、图像处理和实验脚本。迁移路线应该单独建环境，并按每个 run 的 runtime 记录回放。

#### 旧版 AnomalyGPT 环境

`AnomalyGPT/code/requirements.txt` 固定过一套更老的依赖，例如：

```text
torch==1.13.1+cu117
torchvision==0.14.1
transformers==4.29.1
peft==0.3.0
deepspeed==0.9.2
pytorchvideo==0.1.5
timm==0.6.7
kornia==0.7.0
PyYAML==6.0.1
scikit_learn==1.3.0
opencv_python==4.8.0.74
Pillow==10.0.0
```

文件中的完整列表还包含 `easydict`、`einops`、`ftfy`、`gradio`、`h5py`、`iopath`、`ipdb`、`matplotlib`、`mdtex2html`、`open3d_python`、`regex`、`termcolor`、`tqdm`、`sentencepiece` 等。它服务于旧 AnomalyGPT 训练/演示，不能直接替换原生 QLoRA 环境。

#### 仓库根层和 ImageBind 辅助依赖

根目录的 `requirements-s0.txt` 是轻量数据/统计环境：

```text
numpy
pandas
pyarrow
Pillow
opencv-python
scipy
scikit-learn
```

ImageBind 自己还有一份更窄的依赖声明（`AnomalyGPT/code/code/model/ImageBind/requirements.txt`），包含 `torchvision==0.14.0`、`torchaudio==0.13.0`、固定 commit 的 `pytorchvideo`、`timm==0.6.7`、`ftfy`、`regex`、`einops`、`fvcore` 和 `decord==0.6.0`，并指向 PyTorch cu113 wheel 源。它和原生 QLoRA 运行时不兼容，应该只在迁移/旧路线环境中使用。

### 2.3 深度学习框架与工具库

| 组件 | 项目中的角色 |
|---|---|
| PyTorch | 张量、自动微分、模块、优化器、AMP、CUDA、梯度检查点 |
| torchvision | 图像变换、加载器和传统视觉工具 |
| Transformers | Qwen3-VL、DINOv2、CLIP、tokenizer、processor、生成接口 |
| PEFT | LoRA 配置、注入、adapter 保存/加载和导出 |
| Accelerate | `init_empty_weights`、设备映射和低 CPU 内存加载 |
| bitsandbytes | 4-bit NF4、double quant、BF16 compute |
| Pillow | 图像/掩码读取、EXIF 处理、尺寸变换、PNG/JPG I/O |
| NumPy | 数值处理、哈希前的字节处理、统计和绘图数据 |
| safetensors | 安全权重序列化依赖 |
| huggingface-hub | 本地/远端模型目录规范和模型文件管理 |
| scikit-learn | 早期基线和部分指标/统计工具 |
| pandas | `seg_cross_part` 的 manifest、表格数据和统计 |
| PyYAML | 早期分割协议 YAML |
| matplotlib | 训练曲线、方法对比、结果图和示例图 |
| tqdm | 部分数据/训练脚本的进度显示 |
| kornia | 旧 AnomalyGPT 图像增强 |
| pytorchvideo | ImageBind/旧 AnomalyGPT 依赖 |
| timm | 视觉模型/旧基线依赖 |
| sentencepiece | Vicuna 等 tokenizer 依赖 |
| OpenCV | 早期视觉数据处理依赖 |
| Gradio | 旧 AnomalyGPT web demo |
| python-docx | `md2docx.py` 将 Markdown 转 Word |
| Mermaid / SVG / HTML | 架构图、实验路线图和可编辑图表 |

### 2.4 模型和视觉语言组件

| 模型/组件 | 维度或配置 | 在项目中做什么 |
|---|---|---|
| Qwen3-VL-8B-Instruct | 8B 级多模态生成模型 | 原生路线完整 VLM；迁移路线只使用语言部分 |
| Vicuna-7B-v1.1 | LLaMA 系语言骨干 | 与 Qwen 语言骨干做受控迁移对照 |
| ImageBind ViT-H | 视觉 token 序列 257（CLS + 256 patch），宽度 1280，中间层 4 路 | 冻结视觉特征、全局语义和局部 patch 表示 |
| BigDec | 4 路 1280→512 投影 + ConvTranspose 上采样 | ImageBind patch 特征 → 224×224 异常图 |
| PromptLearner | 9 个基础 token + 9 个自适应 token | 异常图 → 18 个 soft token |
| CLS 投影 | `Linear(1024→4096)` | ImageBind 全局 CLS → Qwen/Vicuna 语言维度 |
| DINOv2-B/14 | patch 维度 768；448 输入对应 32×32 patch | 早期跨零件密集特征 |
| OpenAI CLIP ViT-L/14@336 | 文本/视觉投影空间 | WinCLIP 风格文本原型和窗口 CLS |
| SpatialHead | 轻量 MLP/head | 学生 ROI 选择 |
| Qwen3-VL processor | image grid、chat template、图像预处理 | 原生路线的多模态输入构造 |
| MRoPE | t/h/w 多模态位置编码 | 由 Qwen 原生实现处理 |
| DeepStack + merger | 多层视觉特征注入和维度整理 | 原生路线冻结，不作为迁移路线输入 |

### 2.5 训练方法和优化技术

| 技术 | 具体实现 | 目的 |
|---|---|---|
| SFT | 自回归监督微调，assistant-only CE | 让模型学习工业检查回答 |
| Teacher forcing | 用真实回答前缀预测下一个 token | 训练语言模型 |
| Assistant-only loss | prompt/image 前缀标签设为 `-100` | 不让输入部分污染损失 |
| LoRA | 低秩 `W + α/r·BA` | 只训练少量适配参数 |
| QLoRA | NF4 量化冻结底座 + LoRA | 让 8B 模型适配 24GB 显存 |
| NF4 | 4-bit NormalFloat | 对近似正态权重使用分位数量化 |
| Double quant | 量化量化常数 | 进一步压缩显存 |
| BF16 compute | 量化层计算时反量化到 BF16 | 速度、稳定性和显存折中 |
| Gradient checkpointing | non-reentrant checkpoint | 用重算换激活显存 |
| Gradient accumulation | microbatch 1 + accum 4/8 等 | 模拟更大有效 batch |
| Token-count normalization | 先按 token 求和，再按总监督 token 除 | 让不同回答长度公平加权 |
| AdamW | 解耦权重衰减 | LoRA/桥接优化器 |
| Warmup + 衰减/OneCycle | 训练初期升 LR，再衰减 | 稳定训练 |
| Gradient clipping | 通常 `clip_norm=1.0` | 防止梯度爆炸 |
| Multi-seed | 42/43/44 | 区分方法收益和随机波动 |
| Matched control | 相同数据顺序、曝光、评测集 | 让消融更接近单因素比较 |
| Ranking loss | `softplus(-(s_pos-s_neg))` | 直接优化异常/正常排序 |
| 结构化监督 | JSON schema、框、位置字段 | 提高可解析性 |

### 2.6 数据、评测和统计

| 技术 | 作用 |
|---|---|
| VisA | 主线工业异常数据；自定义有监督 70/15/15 划分 |
| MVTec AD | 冻结外部评测和跨域诊断 |
| Real-IAD | 早期跨零件/跨类别分割 |
| SHA256 图像去重 | 防止同字节图片跨 split 泄漏 |
| mask class-index 语义 | VisA 中 0=背景，非 0=异常前景 |
| union bounding box | 将所有缺陷像素包成一个 `[x1,y1,x2,y2]` 框 |
| 0–1000 坐标 | 与原图无关的整数归一化坐标 |
| JSONL | 逐样本指令、预测和日志，方便流式处理/断点续跑 |
| ChatML/chat template | 把 user/image/assistant 组织为模型输入 |
| Greedy generation | `do_sample=False`, `num_beams=1` |
| 双候选 log-likelihood | `logP(Yes)-logP(No)` 生成连续分数 |
| AUROC/AP | 连续排序能力 |
| Macro F1/Recall/FPR | 离散生成结果的分类质量 |
| IoU/box recall | 框定位质量 |
| Pixel AUROC/AP/AUPRO | 像素级分割质量 |
| Bootstrap | 估计样本重采样不确定性 |
| Paired bootstrap | 在同一图像上比较两个模型差值 |
| Threshold calibration | 用验证集正常样本确定目标 FPR 阈值 |

### 2.7 工程、审计和交付

- `manifest.jsonl`、`run_config.json`、`parameter_audit.json`：记录输入身份、超参数和可训练参数。
- `source_sha256`、`cache_sha256`、`decoder_sha256`、`base_manifest_sha256`：绑定代码、缓存、模型和权重身份。
- 原子写：临时文件写完后使用 `os.replace`，避免崩溃留下半截 JSON。
- 失败关闭：模型类型不对、视觉层被量化、mask 缺失、token 前缀不一致、输出 schema 非法时直接报错。
- 断点续跑：评测 JSONL 逐行写入，重启时读取已经完成的样本；配置/数据身份变化则拒绝续跑。
- 重载检查：保存后重新加载 adapter 或 decoder，比较 logits/参数/指标，确保保存路径没有改变模型行为。
- `pyproject.toml` 的 `src` 布局：使用 `pip install -e industrial_vlm` 或 `PYTHONPATH=industrial_vlm/src`。
- `matplotlib`、SVG、HTML、Mermaid：生成论文/答辩图和可编辑架构图。

## 3. 必须掌握的前置知识

### 3.1 Python 环境与 CUDA

一个虚拟环境是“Python 包和解释器的隔离边界”。本项目至少建议分成：

```text
.env-native   原生 Qwen3-VL + bitsandbytes + PEFT
.env-migrate  ImageBind/BigDec + Qwen/Vicuna 迁移
.env-legacy   旧 AnomalyGPT（必要时才建）
```

CUDA 不是 Python 包本身，而是驱动、运行时和 PyTorch wheel 之间的兼容关系。遇到 `libcudart.so`、`no kernel image` 或 `bitsandbytes` 报错时，先比较：

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

项目的 `modeling.load_model()` 会主动拒绝没有 CUDA/BF16 的环境；这是为了避免在错误设备上悄悄跑出不可比较的结果。

### 3.2 Tensor、shape、dtype 和 device

你至少要能读懂下面四个概念：

```python
x.shape       # 张量形状，例如 [batch, tokens, hidden]
x.dtype       # torch.float32 / torch.bfloat16 等
x.device      # cpu 或 cuda:0
x.requires_grad
```

本项目典型形状：

- ImageBind 原始中间层：`[257, batch, 1280]`；
- 去掉 CLS 并转置后：`[batch, 256, 1280]`；
- BigDec 输出：`[batch, 1, 224, 224]`；
- Qwen 语言 hidden size：4096；
- soft prompt bridge：1 个 CLS token + 18 个 PromptLearner token，共 19 个连续向量。

形状错误往往比数值错误更早出现。看到 `mat1 and mat2 shapes cannot be multiplied`，先打印每个模块的输入输出 shape，再检查 batch/token/hidden 的顺序。

### 3.3 ViT、patch、CLS 和密集预测

Vision Transformer 把图片切成 patch。例如 224×224 输入、patch size 14：

```text
224 / 14 = 16
16 × 16 = 256 个 patch
加 1 个 CLS token = 257 个视觉 token
```

CLS 汇总整张图的全局语义，patch token 保留局部空间信息。路线 A 用 CLS 做全局语言 token，用 patch 做异常图；路线 C 用 DINOv2 patch 特征做密集分割。

### 3.4 VLM、processor 和 chat template

多模态模型不只是“把图片转成一个向量”。processor 需要同时处理：

1. 图片缩放、归一化和 patch 网格；
2. 对话模板中的 image 占位符；
3. 文本 token、attention mask、`image_grid_thw`；
4. 生成时的 assistant 起始位置。

训练时要把“用户输入部分”设为 `-100`，只让 assistant 答案参与交叉熵。项目通过分别编码 prompt 和 prompt+answer，检查 prompt token 是否是完整序列的严格前缀；如果模板发生漂移，直接报错。

### 3.5 LoRA 和 QLoRA

LoRA 在冻结矩阵 `W` 旁边加入低秩更新：

```text
W' = W + (alpha / r) * B * A
A: [r, d_in]
B: [d_out, r]
```

`r` 越小，参数越少；`alpha` 控制更新尺度。项目中常见两类配置：

- 迁移 C 臂：`r=8, alpha=16, q_proj/v_proj`；
- 原生 QLoRA：`r=16, alpha=32, q_proj/k_proj/v_proj/o_proj`。

QLoRA 还会把冻结的语言线性层压到 NF4。`bnb_4bit_compute_dtype=torch.bfloat16` 表示计算时使用 BF16，不代表 LoRA 参数也是 NF4。项目会把可训练 adapter 参数提升到 FP32，并审计所有视觉层是否保持 BF16。

### 3.6 交叉熵、logits 和连续评分

语言模型输出 logits。训练时对正确答案 token 计算交叉熵：

```text
loss(token) = -log p(correct_token | prefix)
```

检测时如果只靠生成的 “Yes/No”，分数是离散的；项目还实现了二选一条件评分：分别计算 `logP(Yes)` 与 `logP(No)`，用它们的差作为连续异常分数。这样可以排序、画 ROC/PR 曲线并校准阈值。

注意：连续评分与自由生成是两条不同的测量路径。报告时要分开写，不能用一个替代另一个。

### 3.7 AUROC、AP、F1、FPR、IoU

二分类混淆矩阵：

```text
                 预测异常   预测正常
真实异常            TP        FN
真实正常            FP        TN
```

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
FPR       = FP / (FP + TN)
```

- AUROC：随机抽一张异常图和一张正常图，异常图分数更高的概率；
- AP：更关注异常样本在排序前部的命中质量；
- Macro F1：正常类 F1 与异常类 F1 的平均；
- IoU：预测框和真值框交集面积除以并集面积；
- Pixel AUROC/AP/AUPRO：对每个像素异常概率图进行评估。

项目对并列分数使用 tie-aware 处理；对异常框采用 union box，而不是把每个缺陷实例当成独立目标。读指标时必须同时看协议、数据分布和是否使用了开发集选点。

### 3.8 可复现性和实验身份

随机种子不是完整的实验身份。至少还要记录：

```text
代码 SHA256
数据 manifest SHA256
模型文件/配置 SHA256
缓存 SHA256
tokenizer / processor 版本
seed、采样顺序、曝光数、更新数
GPU、PyTorch、CUDA、Transformers 版本
```

这也是为什么项目会在输出目录保存 `run_config.json`、`parameter_audit.json`、`metrics.jsonl`、`source_sha256` 等文件。

## 4. 从零开始的操作教程

下面按“先能检查，再准备数据，再跑原生路线，最后学习迁移路线”的顺序安排。每一步都写出目标、前置知识、命令、验收标准和常见错误。

### Step 0：定位仓库并建立工作区

**目标**：确认你操作的是 `Qwen-VLM` 仓库，不把生成结果写回旧实验目录。

**前置知识**：Git 的工作区、分支、远端和提交哈希。`git status` 只说明当前工作区，不代表远端没有变化。

```bash
git clone https://github.com/Qhf7/Qwen-VLM.git
cd Qwen-VLM
git remote -v
git rev-parse HEAD
git status --short --branch
```

建议为本次学习建立独立输出根目录：

```bash
export QWEN_VLM_ROOT="$PWD"
export RUN_ROOT="$PWD/_runs/tutorial"
mkdir -p "$RUN_ROOT"/{prepared,runs,evaluations,reports,logs}
```

不要把新的模型、缓存或预测直接写入仓库已有结果目录；实验变更应使用新的输出路径。

**验收标准**：

- `git remote -v` 指向 `https://github.com/Qhf7/Qwen-VLM.git`；
- `git rev-parse --show-toplevel` 与 `pwd` 对应仓库根目录；
- `$RUN_ROOT` 存在且为空白的新目录。

**常见错误**：在父项目目录执行 `PYTHONPATH=src`，导致导入到另一份源码；解决办法是先 `cd Qwen-VLM`，再使用仓库内相对路径。

### Step 1：创建原生 QLoRA 环境

**目标**：获得一套可以运行 `industrial_vlm` 的 CUDA 环境。

**前置知识**：虚拟环境只隔离 Python 包，不会自动修复驱动/CUDA；必须先确认 GPU 架构支持 BF16。

```bash
python3 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 按你的 CUDA 选择 PyTorch wheel；项目历史服务器使用 cu124
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -e ./industrial_vlm
```

如果你需要严格回放项目记录的服务器环境：

```bash
python -m pip install -r industrial_vlm/requirements-runtime.txt
```

但这不保证所有公共镜像都能提供未来日期记录的版本；安装失败时应保留 `pip freeze`，并在报告中写清实际版本。

验证：

```bash
python - <<'PY'
import torch, transformers, peft, bitsandbytes
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('bitsandbytes', bitsandbytes.__version__)
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

**验收标准**：`cuda=True` 且 `bf16=True`。如果只有 CPU，继续做 Step 2–3 的逻辑检查，不要启动 8B 训练。

### Step 2：准备模型和数据资产

**目标**：在开始训练前，列出每条路线真正依赖的文件。

**前置知识**：模型权重、tokenizer、processor、adapter 和数据集是不同资产；仅有代码不能完成推理。

#### 原生路线必需

- Qwen3-VL-8B-Instruct 底座；
- VisA 图片、mask、`split_csv/1cls.csv`；
- CUDA/BF16 环境；
- `bitsandbytes`、PEFT、Transformers。

#### 迁移路线额外需要

- ImageBind 权重；
- AnomalyGPT/ImageBind 运行时目录；
- Vicuna-7B-v1.1（若跑 Vicuna 对照）；
- BigDec/PromptLearner/CLS 投影的阶段性权重或重新训练产物；
- teacher cache（CLS、异常图、items、哈希）；
- 迁移协议所需的 VisA 训练/验证清单。

#### 早期分割路线额外需要

- Real-IAD 数据和 manifest；
- DINOv2 权重；
- CLIP ViT-L/14@336 权重（WinCLIP 基线）；
- 早期 `seg_cross_part` 配置 YAML。

资产来源和许可证入口见：

- [ASSETS.md](../ASSETS.md)
- [THIRD_PARTY.md](../THIRD_PARTY.md)
- [industrial_vlm/DATA_PROTOCOL.md](../industrial_vlm/DATA_PROTOCOL.md)

建议把外部资产放在仓库外：

```bash
export MODEL_ROOT=/data/models
export DATA_ROOT=/data/datasets
export OUTPUT_ROOT="$PWD/_runs/tutorial"
mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT"
```

**验收标准**：模型目录能被 `AutoConfig.from_pretrained(..., local_files_only=True)` 读取；VisA 根目录下能找到 `split_csv/1cls.csv`；异常图片都有对应 mask。

### Step 3：先跑不加载大模型的 CPU 检查

**目标**：在消耗 GPU 前验证评分、解析、配置和 Python 语法。

**前置知识**：纯函数测试可以验证算法接口，不需要下载 8B 权重；这是工程上很重要的“分层验证”。

```bash
python -m compileall -q industrial_vlm seg_cross_part migration_20261103 repair_20261103

(cd industrial_vlm && PYTHONPATH=src python ops/test_score.py)

PYTHONPATH=industrial_vlm/src python -m industrial_vlm.data --help
PYTHONPATH=industrial_vlm/src python -m industrial_vlm.train --help
PYTHONPATH=industrial_vlm/src python -m industrial_vlm.evaluate --help
PYTHONPATH=industrial_vlm/src python -m industrial_vlm.infer --help
PYTHONPATH=industrial_vlm/src python -m industrial_vlm.score --help
```

`ops/test_score.py` 的重点不是训练，而是验证：

- 双候选 log-softmax 数值稳定；
- 布尔答案 token 边界可识别；
- 并列分数的 AUROC 处理正确；
- 不完整的 JSONL 结果不会参与指标；
- 校准阈值和异常分数的字段一致。

**验收标准**：`compileall` 退出码为 0，评分测试无失败，所有 CLI 能打印帮助。

**常见错误**：

- `ModuleNotFoundError: industrial_vlm`：使用 `PYTHONPATH=industrial_vlm/src` 或 `pip install -e industrial_vlm`；
- `bitsandbytes` 在 CPU 环境导入失败：暂时跳过模型加载，只运行不依赖它的测试；
- 直接执行 `python industrial_vlm/src/industrial_vlm/data.py`：相对导入/包入口可能不一致，优先使用 `python -m industrial_vlm.data`。

### Step 4：准备和审计 VisA 数据

**目标**：生成可训练的 manifest、训练指令和验证/测试指令。

**前置知识**：数据切分的第一原则是来源级隔离；同一张图片即使文件名不同，也不能跨 train/val/test。

原生路线的数据入口是 `industrial_vlm/src/industrial_vlm/data.py`。它会：

1. 读取 `split_csv/1cls.csv`；
2. 校验图片、mask、尺寸和 EXIF；
3. 识别 VisA mask 的 class-index 语义；
4. 按原始图片 SHA256 去重；
5. 按 `(category, label)` 分层切分；
6. 生成 union box 和 0–1000 坐标；
7. 生成区域任务和判断任务 JSONL；
8. 写出 `rejects.jsonl`、`duplicates.jsonl` 和 `audit_report.json`。

运行：

```bash
export VISA_ROOT=/data/datasets/VisA
export PREPARED="$OUTPUT_ROOT/prepared/visa"

PYTHONPATH=industrial_vlm/src python -m industrial_vlm.data \
  --data-root "$VISA_ROOT" \
  --output "$PREPARED" \
  --train-size 768 \
  --eval-size 120 \
  --seed 42
```

你会看到类似文件：

```text
manifest.jsonl
split.json
train_quality_region.jsonl
train_random_region.jsonl
train_quality_decision.jsonl
val.jsonl / val_decision.jsonl
val_balanced_120.jsonl / val_balanced_120_decision.jsonl
test.jsonl / test_decision.jsonl
rejects.jsonl
duplicates.jsonl
audit_report.json
```

#### 为什么 VisA mask 不能除以 255

VisA 原始 mask 可能是调色板索引：0 表示背景，非零值表示缺陷类别。正确逻辑是：

```python
binary = mask_values > 0
```

把索引先除以 255 再阈值化，会把 1–8 全部压到很小的值，导致异常区域被错误地当作背景。仓库的 `repair_20261103/REPAIR_REPORT.md` 记录了这个问题以及修复后的定位复评。

#### 为什么框是 union box

数据脚本不推断多个实例框，而是把所有前景像素合成一个外接矩形，并把它按原图宽高映射到 `[0,1000]`：

```text
x1 = floor(left  * 1000 / width)
y1 = floor(top   * 1000 / height)
x2 = ceil (right * 1000 / width)
y2 = ceil (bottom* 1000 / height)
```

因此模型回答的框是整图坐标系中的一个 union box，不是实例检测框集合。

**验收标准**：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("$PREPARED")
r = json.loads((p / "audit_report.json").read_text())
assert r["valid_unique_images"] > 0
assert r["checks"]["source_id_disjoint"] is True
print('prepared data audit passed')
PY
```

如果你的 `audit_report.json` 字段名与上面版本不同，以实际输出为准，但必须确认：唯一图片数量大于 0、train/val/test 无 source_id 交集、异常 mask 非空、正常 mask 无前景。

### Step 5：理解并运行原生 Qwen3-VL QLoRA

**目标**：先完成一个原生视觉路径的训练臂，再做生成式评测和连续评分。

**前置知识**：原生 Qwen3-VL 的 processor 会产生 `input_ids`、`attention_mask`、图像像素和 `image_grid_thw`。不能把它简化成普通文本 LLM。

#### 5.1 模型加载到底做了什么

`industrial_vlm/src/industrial_vlm/modeling.py::load_model()` 的关键顺序：

1. 用 `AutoConfig` 读取本地 Qwen3-VL 配置；
2. 在 meta device 上创建空壳，找出语言 attention 的 q/k/v/o 层；
3. 只把语言线性层加入 NF4 量化，其余视觉/merger/DeepStack/embedding/lm_head 跳过；
4. 用 `device_map={"": cuda_device}` 把模型放到单卡；
5. 语言层注入 PEFT LoRA；
6. 训练时启用 non-reentrant gradient checkpointing；
7. 审计所有可训练参数必须是语言 attention LoRA；
8. processor 设置 `max_pixels` 并保持本地加载。

如果你看到视觉层是 4-bit，或者 `audit_trainable_parameters` 报 `Forbidden trainable parameter`，应该停止训练并修复配置。

#### 5.2 选择一个训练配置

配置文件位于 `industrial_vlm/configs/`，常见实验臂包括：

- `quality_region.json`：按异常面积分层的区域任务；
- `random_region.json`：同样本量的随机区域任务；
- `quality_decision.json`：判断任务；
- `quality_region.json`、`quality_decision.json` 等共同定义数据、步数、学习率和输出路径。

先打开配置确认模型路径、数据文件和输出目录：

```bash
python -m json.tool industrial_vlm/configs/quality_decision.json > /tmp/quality_decision.pretty.json
```

#### 5.3 运行训练

```bash
cd "$QWEN_VLM_ROOT/industrial_vlm"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

python -m industrial_vlm.train \
  --config configs/quality_decision.json
```

训练脚本是手写循环，而不是直接使用 `transformers.Trainer`。这样做的原因是把 microbatch、监督 token 数、梯度累积、验证频率和保存逻辑显式固定下来。

**前置知识：为什么要按监督 token 归一化？** 如果一个回答有 4 个 token，另一个回答有 40 个 token，直接对两个“平均 loss”求平均，会让短回答和长回答在 token 级别权重不一致。项目会把 microstep loss 还原成 token sum，再除以本次更新的监督 token 总数。

**验收标准**：输出目录中至少出现训练配置、loss/metrics、参数审计、checkpoint/adapter 和 summary；参数审计显示视觉模块没有可训练参数，LoRA 参数数量与配置一致。

### Step 6：原生路线的生成式评测

**目标**：在固定 JSONL 上生成回答，解析 JSON，报告 F1、Recall、FPR、框指标和结构有效率。

**前置知识**：生成式评测关注“模型实际写了什么”，连续评分关注“模型对候选答案的相对偏好”，二者不能混为一个指标。

```bash
cd "$QWEN_VLM_ROOT/industrial_vlm"
export PYTHONPATH="$PWD/src"

python -m industrial_vlm.evaluate \
  --model /absolute/path/to/Qwen3-VL-8B-Instruct \
  --adapter /absolute/path/to/adapter \
  --data /absolute/path/to/prepared/visa/val_balanced_120.jsonl \
  --image-root /data/datasets \
  --output /absolute/path/to/new_evaluation \
  --max-pixels 524288 \
  --max-length 2048 \
  --max-new-tokens 128 \
  --bootstrap-samples 2000
```

脚本会：

- 使用 greedy decoding：`do_sample=False, num_beams=1`；
- 逐行写 prediction JSONL；
- 在每个样本上记录输入 token、生成 token、耗时和峰值显存；
- 解析 `is_anomalous`、`boxes` 和错误原因；
- 输出 metrics JSON 和运行 metadata；
- 如果输出目录已有部分结果，只能用完全相同配置的 `--resume` 续跑。

**验收标准**：`meta.json` 状态为 `complete`，预测条数等于输入 JSONL 条数；如果有解析失败，必须保留失败计数，不能偷偷从分母中删除。

### Step 7：连续分数、校准和报告

**目标**：用 `logP(true)-logP(false)` 计算排序指标，并在验证正常集上校准阈值。

**前置知识**：AUROC 只需要排序，不要求阈值；FPR/Recall 需要阈值。把一个阈值从验证集搬到外部数据，不代表外部 FPR 一定相同。

先查看 CLI：

```bash
cd "$QWEN_VLM_ROOT/industrial_vlm"
export PYTHONPATH="$PWD/src"
python -m industrial_vlm.score --help
```

根据 `score.py` 的实际参数运行评分（参数名称以你当前 checkout 的 `--help` 为准）：

```bash
python -m industrial_vlm.score score \
  --model /absolute/path/to/Qwen3-VL-8B-Instruct \
  --adapter /absolute/path/to/adapter \
  --data /absolute/path/to/prepared/visa/val_balanced_120_decision.jsonl \
  --image-root /data/datasets \
  --output /absolute/path/to/new_scores \
  --data-role validation
```

评分器会做 token 边界审计：如果 `true/false` 被 tokenizer 拆成不符合预期的多个 token，或者 prompt/answer 的 `image_grid_thw` 不一致，就拒绝该样本，而不是给出一个看似合理的分数。

生成式评测和连续评分完成后生成报告：

```bash
python -m industrial_vlm.report \
  --runs /absolute/path/to/evaluations \
  --output /absolute/path/to/reports

python -m industrial_vlm.compare \
  --evaluations /absolute/path/to/evaluations \
  --output /absolute/path/to/reports
```

报告至少要写清楚：数据 split、标签比例、连续分数定义、生成解析规则、阈值来源、bootstrap 设置、是否为开发集。

### Step 8：单图推理

**目标**：对一张没有标签的图片运行原生 Qwen3-VL，得到原始文本和解析结果。

```bash
cd "$QWEN_VLM_ROOT/industrial_vlm"
export PYTHONPATH="$PWD/src"

python -m industrial_vlm.infer \
  --model /absolute/path/to/Qwen3-VL-8B-Instruct \
  --adapter /absolute/path/to/adapter \
  --image /absolute/path/to/image.png \
  --task region \
  --max-pixels 524288 \
  --max-length 2048 \
  --max-new-tokens 128 \
  --output /absolute/path/to/new_prediction.json
```

如果只判断是否异常，把 `--task region` 改为 `--task decision`。输出包含：

- `raw_text`：模型原始回答；
- `parsed`：通过 schema 解析后的字段；
- `parse_error`：解析失败原因；
- `coordinate_frame`：框坐标是整图、0–1000 的 `xyxy`；
- `input_tokens`、`generated_tokens`；
- 模型和 adapter 路径。

脚本会拒绝覆盖已有输出文件，因此每次实验都使用新文件名。

### Step 9：学习路线 A——修复 ImageBind/BigDec 视觉前端

**目标**：理解迁移主线如何把视觉特征转成语言模型可消费的连续 token。

**前置知识**：需要理解冻结 backbone、特征缓存、异常图、空间对齐和 soft prompt。

#### 9.1 ImageBind 特征布局

`repair_20261103/code/dec_big.py::patch_features()` 明确检查 ImageBind 输出：

```text
raw:       [257, batch, 1280]
去 CLS:    [256, batch, 1280]
转置后:    [batch, 256, 1280]
```

如果把第一维误当 batch，单张图会被错误解释为 257 个样本，BigDec 输出和标签都失去空间对应关系。

#### 9.2 BigDec

BigDec 的结构是：

```text
4 路 ImageBind [B,256,1280]
    ↓ 每路 Linear(1280→512)
4 个 [B,512,16,16] map
    ↓ 跨层平均
ConvTranspose2d × 4 + Conv2d
    ↓ bilinear interpolate
[B,1,224,224] anomaly map
```

训练损失是 focal 风格的 BCE 加 dice。视觉编码器冻结，只训练解码器。

#### 9.3 数据和 mask 对齐

修复记录强调三件事：

- 图像和 mask 使用相同的直接缩放，不对图像中心裁剪、对 mask 直接拉伸；
- 图像用双三次/双线性，mask 用最近邻；
- 异常 mask 缺失必须报错，不能静默把样本当作正常。

运行前先做数据审计或对齐自检，再训练 BigDec：

```bash
cd "$QWEN_VLM_ROOT/repair_20261103/code"
python dec_big.py --help
```

具体训练参数以当前脚本的 `--help` 和 `migration_20261103/PHASE1_PROTOCOL.md` 为准；历史服务器路径需要通过环境变量/参数替换。

**验收标准**：

- `patch_features` 通过 shape 检查；
- 异常 mask 前景面积非零；
- 图像/mask 变换后的合成对齐测试通过；
- BigDec 输出固定为 `[B,1,224,224]`；
- 像素指标保存了类别宏平均和整体值。

### Step 10：学习路线 A——19 个 soft token 和语言桥接

**目标**：理解路线 A 中“视觉 → 语言”的接口，而不是把 ImageBind 特征直接塞给 Qwen。

**前置知识**：语言模型输入通常是 token id；soft token 则直接提供 embedding 向量，不对应词表中的某个词。

`migration_20261103/code/migration_language.py` 的桥接逻辑可抽象为：

```text
ImageBind CLS [B,1024]
    ↓ cls_proj
image_token [B,1,4096]

BigDec map [B,224,224]
    ↓ PromptLearner
prompt_tokens [B,18,4096]

concat
    ↓
features [B,19,4096]
    ↓ 插入 prompt 前后文本 embedding
Qwen/Vicuna language model
```

这里的 19 个向量不是文本 token id。代码通过 `inputs_embeds` 路径把它们拼到语言 embedding 序列中，再构造 attention mask 和 labels。

迁移协议强调：

- 视觉前端、decoder、语言骨干的冻结状态要固定；
- 重新初始化桥接时，Vicuna/Qwen 要共享可比较的初始化；
- 记录 cache、decoder、base model 和采样顺序的 SHA256；
- 只使用 val 选 best，历史 test 不能参与选点。

### Step 11：学习路线 A——语言骨干迁移和 q/v LoRA

**目标**：理解 A、B、C、D、R、S 等实验臂的关系。

| 实验臂 | 训练内容 | 学习问题 |
|---|---|---|
| Vicuna 对照 | 桥接训练，语言骨干冻结 | 语言骨干更换前的基线 |
| A / Qwen bridge | 重新训练 PromptLearner + CLS 投影 | Qwen 语言骨干能否提高开发性能 |
| B | 从 A best 继续训练桥接 | 继续更新桥接是否仍有收益 |
| C | 冻结桥接，只训练 q/v LoRA | 少量语言参数是否足够 |
| D | 桥接与 LoRA 联合 | 联合适配的收益/参数代价 |
| Cpair | 匹配 CE 控制 | 排序实验的对照 |
| R | CE + ranking loss | 直接提升异常排序 |
| S | 结构化回答继续训练 | 提升 JSON/schema 遵循 |

C 臂的典型命令：

```bash
cd "$QWEN_VLM_ROOT/migration_20261103/lora_20261104"
python migration_lora_train.py \
  --arm C \
  --seed 42 \
  --cache /absolute/path/to/teacher_cache.pt \
  --bridge /absolute/path/to/qwen_bridge_best.pt \
  --base-manifest /absolute/path/to/base_manifest.json \
  --out /absolute/path/to/new_c42_run
```

脚本会检查：

- target modules 是语言模型的 q/v；
- 训练参数是 FP32；
- adapter 注入前后未训练 logits 一致；
- 冻结参数版本未变化；
- cache、bridge、base manifest 和 sample schedule 身份一致。

**常见坑**：把 `q_proj` 后缀全局匹配到视觉塔；把 LoRA 参数保留在 BF16；B/C 使用不同训练样本顺序；把 `best` 检查点选择看成“看过 test”。

### Step 12：学习路线 C——DINOv2/CLIP 跨零件分割

**目标**：理解仓库早期为何保留一条密集预测和基线路线。

**前置知识**：分割和 VLM 生成是两种输出空间。分割模型直接预测每个像素的 logit，VLM 则生成文本/JSON；两者的指标不能混用。

`seg_cross_part/train_seg.py` 的核心结构：

```text
RGB
 ↓ DINOv2（冻结，取第 3/6/9/12 层 patch）
4 路 [B,768,H,W]
 ↓ 1×1 projection + concat + GroupNorm/GELU
SegDecoder
 ↓ interpolate 到 448×448
pixel logits
```

`seg_cross_part/winclip_b2.py` 额外提供 CLIP 文本原型和 DINOv2 参考图最近邻融合：

- CLIP 语言侧使用 normal/anomalous 状态词和模板集成；
- DINOv2 视觉侧不具备文本对齐能力，只做 patch 特征最近邻；
- 这是 WinCLIP 思路的工程对照，不是把 DINOv2 当作 CLIP 文本塔。

先跑几何对齐自检：

```bash
python -m seg_cross_part.train_seg --help
python -m seg_cross_part.eval_sft --help
python -m seg_cross_part.winclip_b2 --help
```

再按 `seg_cross_part/protocols/` 的 YAML 运行真实数据实验。训练前检查：

- train/val/test 的类别是否按协议隔离；
- mask 与图像裁剪/翻转是否同步；
- 留出类别是否完全没有出现在训练源；
- Pixel AUROC、AP、AUPRO 的计算口径是否一致。

### Step 13：结果、审计和可复现交付

**目标**：把一次实验变成别人能检查的证据，而不是只有一个最终数字。

每次新 run 至少保存：

```text
run_config.json             参数、版本、路径、seed
parameter_audit.json        训练参数名、数量、dtype
metrics.jsonl               step/loss/lr/grad/memory
predictions.jsonl           每张图的原始回答和解析结果
source_sha256.json          代码、数据、权重身份
summary.json                完成状态和最终指标
```

推荐的检查顺序：

```bash
# 1. 代码语法
python -m compileall -q industrial_vlm seg_cross_part migration_20261103 repair_20261103

# 2. 纯逻辑测试
(cd industrial_vlm && PYTHONPATH=src python ops/test_score.py)

# 3. 数据审计
PYTHONPATH=industrial_vlm/src python -m industrial_vlm.data \
  --data-root "$VISA_ROOT" --output "$PREPARED" --train-size 16 --eval-size 8 --seed 42

# 4. 训练前置检查（该脚本读取 industrial_vlm/prepared/visa 的固定协议目录）
python industrial_vlm/ops/validate_prepared.py

# 5. 训练/评测后检查
python industrial_vlm/ops/verify_completion.py --help
```

如果要比较两个方法，必须同时报告：

- 相同/不同的训练数据、曝光数、更新数和回答模板；
- 可训练参数量和显存；
- seed 列表及均值/标准差；
- 是否使用同一张图的配对比较；
- val/test/external 的角色；
- 生成 F1 与连续 AUROC 的来源；
- 失败解析、正常 FPR 和外部域偏移。

## 5. 代码精读路线：按什么顺序打开文件

### 5.1 原生 QLoRA

1. `industrial_vlm/src/industrial_vlm/data.py`：先懂输入 JSONL 和 mask/box 规则；
2. `modeling.py`：读 NF4 skip list、LoRA target、输入构造和标签 mask；
3. `train.py`：读 microbatch、token normalization、checkpoint；
4. `evaluate.py`：读生成式结果和断点续跑；
5. `score.py`：读候选 token、log-likelihood 和阈值校准；
6. `metrics.py`：读 JSON schema、框 IoU、F1/FPR；
7. `compare.py`：读配对 bootstrap 和方法比较；
8. `infer.py`：读最小单图推理接口。

### 5.2 迁移主线

1. `repair_20261103/code/visa_pipeline.py`：数据/几何/mask 的修复；
2. `repair_20261103/code/dec_big.py`：ImageBind layout 和 BigDec；
3. `migration_20261103/code/migration_cache.py`：缓存 CLS、map 和 manifest；
4. `migration_20261103/code/migration_language.py`：19 soft token bridge；
5. `migration_20261103/code/migration_train.py`：桥接训练和评分；
6. `migration_20261103/lora_20261104/migration_lora_train.py`：B/C LoRA；
7. `migration_20261103/advanced_20261104/`：ranking、结构化、工具和外评。

### 5.3 早期密集路线

1. `seg_cross_part/common.py`：数据根和协议；
2. `train_seg.py`：DINOv2 特征、分割解码器、几何变换；
3. `metrics_pixel.py`：像素指标；
4. `winclip_b2.py`：CLIP 文本原型 + DINOv2 参考关联；
5. `sft_qwen3vl.py`：另一套原生 Qwen LoRA SFT 接口。

## 6. 排错手册

### 6.1 加载阶段

| 现象 | 先检查 | 修复方向 |
|---|---|---|
| `bf16 support` 报错 | GPU 架构、PyTorch/CUDA | 换支持 BF16 的 GPU 或仅做 CPU 逻辑测试 |
| `bitsandbytes` 动态库错误 | `torch.version.cuda`、驱动、bnb 版本 | 重建与 CUDA 对齐的环境 |
| CPU 内存/显存爆炸 | 是否误加载视觉层量化/重复模型 | 使用项目 loader，确认单卡、关闭重复进程 |
| `model_type` 不匹配 | 底座是否真是 Qwen3-VL | 检查 `config.json` 和模型路径 |
| `local_files_only` 找不到文件 | 模型目录缺 shard/tokenizer/processor | 完整下载底座并核对文件清单 |

### 6.2 数据阶段

| 现象 | 常见根因 | 处理 |
|---|---|---|
| 异常 mask 为空 | 把 class index 当 0/255 或路径后缀错误 | 非零即前景；严格解析 `.png` mask |
| mask 与图像尺寸不一致 | 原始数据不完整或错误 resize | 先报错，补齐/修正数据，不静默拉伸 |
| train/val 有重叠 | 只按文件名去重 | 使用原始文件 SHA256 做 source-level 去重 |
| 箱坐标越界 | 把像素坐标直接当 0–1000 | 按图像宽高归一化并检查 0≤x1<x2≤1000 |
| 训练臂样本不公平 | 分层配额或展示顺序不同 | 保存 sampling schedule，做 matched audit |

### 6.3 训练阶段

| 现象 | 常见根因 | 处理 |
|---|---|---|
| loss 为 NaN | dtype/学习率/梯度爆炸 | 检查 BF16/FP32 参数、clip、grad norm |
| LoRA 没梯度 | target module 错、checkpoint 输入无梯度 | 看 parameter audit、启用 `enable_input_require_grads()` |
| 训练参数太多 | 全局匹配 `q_proj` 误碰视觉塔 | 用语言层完整名称正则并运行审计 |
| 显存逐步增长 | 记录了带图计算图的 tensor、缓存未清理 | `detach()`、`inference_mode()`、检查列表和 cache |
| 保存后结果变了 | adapter/base/cache 身份不一致 | 运行重载检查并比较 logits/hash |
| 续跑被拒绝 | 数据/参数/模型身份变化 | 使用原配置续跑，或新建 run 重新开始 |

### 6.4 评测阶段

| 现象 | 常见根因 | 处理 |
|---|---|---|
| AUROC 很高但 FPR 也很高 | 排序好但阈值不适合目标域 | 分开报告 AUROC 与校准 FPR |
| 生成 Yes/No 与评分不一致 | 连续评分和生成是不同路径 | 两者都保留，不把一个覆盖另一个 |
| JSON 解析失败 | 额外解释、代码围栏、坐标非法 | 先看 `raw_text`，检查 schema 和 prompt |
| 外部集性能下降 | 域偏移、提示/类别变化 | 固定权重后做 external audit，不修改阈值掩盖问题 |
| 框 IoU 很低但结构有效率高 | JSON 格式学会了，空间定位没学会 | 分开分析 schema、框 recall、IoU |

## 7. 学习检查题

完成教程后，尝试不看代码回答：

1. 为什么原生路线只量化语言层而保留视觉塔 BF16？
2. ImageBind 的 `[257,batch,1280]` 为什么必须转为 `[batch,256,1280]`？
3. 为什么 VisA 的 mask 不能简单除以 255？
4. `assistant-only loss` 如何通过 labels 的 `-100` 实现？
5. 为什么生成 F1 不能替代候选 log-likelihood AUROC？
6. C 臂只训练哪些 LoRA 层，如何防止误训练视觉层？
7. QLoRA 的 NF4、double quant 和 BF16 compute 分别是什么意思？
8. 为什么需要记录 image/data/model/cache 的 SHA256？
9. 一个外部数据集正常 FPR 很高时，为什么不能只报告 AUROC？
10. DINOv2 为什么不能直接当作 CLIP 文本对齐模型？
11. `best` 检查点为什么必须只由 val 选择？
12. 为什么“单 seed 提升”不足以说明方法稳定有效？

如果你能用项目中的文件名和函数名回答这些问题，说明已经从“知道名词”进入“能读代码”的阶段。

## 8. 最小可复现实验清单

### 只想先跑通代码

```bash
python -m compileall -q industrial_vlm seg_cross_part migration_20261103 repair_20261103
(cd industrial_vlm && PYTHONPATH=src python ops/test_score.py)
```

### 有一张支持 BF16 的 GPU，先跑原生路线

1. 准备 Qwen3-VL-8B-Instruct；
2. 准备 VisA；
3. 用 `industrial_vlm.data` 生成 JSONL；
4. 运行一个较小 steps 的训练配置；
5. 用 `industrial_vlm.evaluate` 生成回答；
6. 用 `industrial_vlm.score` 计算连续分数；
7. 用 `report.py`/`compare.py` 汇总；
8. 复核 `parameter_audit`、`summary`、`source_sha256`。

### 想学习迁移主线

1. 先完成 repair 数据/mask/布局审计；
2. 准备 ImageBind 与语言底座；
3. 生成 teacher cache；
4. 训练并保存 BigDec/bridge；
5. 先复现 Qwen bridge，再做 B/C LoRA；
6. 只用 val 选点，最后冻结权重做 external；
7. 读最终报告中的正结果和负结果。

## 9. 如何正确解读项目结果

学习技术栈时，不要只背最高分，还要把数字和实验协议绑定起来。仓库 README 中给出的几个代表性结果是：

| 结果 | 数字 | 读法 |
|---|---:|---|
| 共同 ImageBind/BigDec 前端下的 Vicuna 开发 AUROC | 0.889684 | 语言迁移的对照起点 |
| 共同前端下的 Qwen bridge 开发 AUROC | 0.925779 | 重新训练桥接后的开发结果 |
| C 臂 q/v LoRA 开发 AUROC | 0.936983 | 只训练约 383 万个语言 LoRA 参数的结果 |
| 原生 QLoRA DD 臂 AUROC | 0.9456 左右 | 独立协议、单 seed、开发验证 |
| 外部 MVTec 正常 FPR | 约 0.61–0.72 | 说明跨域误报仍然很高，不能直接支持部署 |

这些结果不能互相直接相减，因为它们的视觉路径、数据划分、训练预算、评分接口和评测集不同。正确的复现实验报告至少要同时写：

1. 训练数据是否包含真实异常标注；
2. checkpoint 是否由 val 选择，test 是否被查询；
3. 生成结果和候选似然是否分开计算；
4. 单 seed 还是多 seed，是否有配对差值；
5. 外部数据是否完整、是否只是一部分 pilot；
6. 正常 FPR、解析失败率和框定位质量是否随 AUROC 一起报告。

如果一个方法的 AUROC 上升但正常 FPR 也大幅上升，应该把它解释为“排序能力和阈值/域适配存在取舍”，而不是简单宣布模型更适合生产。

## 10. 参考入口

### 仓库内

- [根 README](../README.md)
- [资产与复现说明](../ASSETS.md)
- [第三方来源与许可](../THIRD_PARTY.md)
- [原生 Qwen 项目说明](../industrial_vlm/README.md)
- [原生数据协议](../industrial_vlm/DATA_PROTOCOL.md)
- [迁移路线说明](../migration_20261103/README.md)
- [第一阶段迁移协议](../migration_20261103/PHASE1_PROTOCOL.md)
- [B/C LoRA 协议](../migration_20261103/lora_20261104/PROTOCOL.md)
- [定位修复报告](../repair_20261103/REPAIR_REPORT.md)
- [跨零件分割协议](../seg_cross_part/protocols/seg_default.yaml)
- [AnomalyGPT 运行时说明](../AnomalyGPT/code/README.md)

### 外部

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)
- [ImageBind](https://github.com/facebookresearch/ImageBind)
- [FastChat/Vicuna](https://github.com/lm-sys/FastChat)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [VisA / spot-diff](https://github.com/amazon-science/spot-diff)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Real-IAD](https://github.com/M-3LAB/Real-IAD)
- [PEFT](https://huggingface.co/docs/peft)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

本教程把项目的技术、前置知识、操作步骤、验证标准和排错路径串在一起。真正开始训练前，先读对应路线协议，再为本次实验创建新的输出目录并保留完整身份信息。

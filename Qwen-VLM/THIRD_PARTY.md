# 代码、模型与数据来源

> 整理日期：2026-11-05。本文件说明 `Qwen-VLM/` 交付包中项目实现与上游组件的边界。阅读入口见 [README](README.md)，未随包提供的资产见 [ASSETS.md](ASSETS.md)，实际纳入的文件以 [DELIVERY_MANIFEST.tsv](DELIVERY_MANIFEST.tsv) 为准。

## 项目实现与第三方组件的边界

`industrial_vlm/`、`seg_cross_part/`、`repair_20261103/`、`migration_20261103/` 和 `AnomalyGPT/followup_20261026/` 中的实验脚本记录本项目的数据处理、接口适配、训练、推理、对照与审计。它们使用了Qwen、ImageBind、DINOv2、LoRA/QLoRA等已有模型与方法；本项目贡献落在具体适配、修复、训练目标组合和实验控制，不把这些上游算法作为个人原创。

独立论文库、论文原文与LaTeX材料、CLIP/WinCLIP/AdaCLIP等完整阅读克隆未随本交付包复制。与本项目实现和结果解释有关的文档保留在 `docs/` 等目录；这些笔记不改变原论文、模型、数据和代码的权利归属。

## AnomalyGPT与ImageBind必要运行时

`AnomalyGPT/followup_20261026/server_snapshot/` 是实验时保存的AnomalyGPT派生运行时，包含PromptLearner、旧语言接口及本项目严格Qwen迁移修改；它不是上游官方发行版，也不是另一个完整独立阅读仓库。

| 组件 | 来源 | 包内来源和许可 |
|---|---|---|
| AnomalyGPT派生运行时 | [CASIA-IVA-Lab/AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT) | 保留 [CC BY-NC-SA 4.0全文](AnomalyGPT/followup_20261026/server_snapshot/LICENSE)；修改仍需遵守适用上游条款 |
| ImageBind子树 | [facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind) | 保留 [随附许可证](AnomalyGPT/followup_20261026/server_snapshot/code/model/ImageBind/LICENSE)，本地随附文本为CC BY-NC-SA 4.0 |

快照的 [SOURCE_COMMIT](AnomalyGPT/followup_20261026/server_snapshot/SOURCE_COMMIT) 原文记录其来自2026-10-25的本地AnomalyGPT归档，但明确远端归档没有可用Git提交号。因此不能把该文件称为已验证的上游commit SHA。各次实验仍可用运行时源码清单及字节SHA追溯到实际保存的代码。

代码、模型权重和数据可能各有独立条款。保留源代码许可证不表示取得了任意模型或数据的商业使用权；应分别按相应上游授权获取和使用。这里未给整个混合目录添加统一的新许可证。

## 底座模型与开源依赖

以下入口供获取原始实现、模型卡和对应版本的许可；包内没有附上这些底座权重或Python环境。

| 依赖/模型 | 作用 | 获取与条款入口 |
|---|---|---|
| Qwen3-VL-8B-Instruct | 原生视觉语言底座；迁移主线只保留其语言部分 | [Qwen3-VL源码](https://github.com/QwenLM/Qwen3-VL)、[官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Vicuna-7B-v1.1 | 语言骨干对照 | [FastChat](https://github.com/lm-sys/FastChat)、[模型卡](https://huggingface.co/lmsys/vicuna-7b-v1.1)；同时核对对应底层模型条款 |
| ImageBind权重 | 冻结全局与局部视觉特征 | [ImageBind官方项目](https://github.com/facebookresearch/ImageBind)及其权重说明 |
| DINOv2 | 前期冻结密集视觉特征 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| LoRA / PEFT | 语言层低秩参数适配、导出与加载 | [LoRA原始实现](https://github.com/microsoft/LoRA)、[Hugging Face PEFT](https://github.com/huggingface/peft) |
| Transformers | Qwen模型、tokenizer、processor及生成接口 | [huggingface/transformers](https://github.com/huggingface/transformers) |
| bitsandbytes | 独立原生QLoRA路线的NF4量化 | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| PyTorch / torchvision | 张量、自动微分、优化和图像变换 | [PyTorch](https://github.com/pytorch/pytorch)、[torchvision](https://github.com/pytorch/vision) |
| NumPy / scikit-learn / Pillow等 | 数值处理、指标复算及图像I/O | 依赖各自上游许可；实际版本见相应运行记录 |

原生QLoRA与后期BF16迁移运行环境不同。原生路线的精确版本记录见 [requirements-runtime.txt](industrial_vlm/requirements-runtime.txt)，后期迁移版本见 [migration README](migration_20261103/README.md)与各run的runtime记录；不要用一套当前最新版环境替代所有历史版本后宣称精确复现。

## 数据集来源

| 数据集 | 本项目用途 | 官方入口 |
|---|---|---|
| VisA | 自定义有监督训练/开发；异常mask、区域目标和主线缓存来源 | [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)中的VisA说明与数据入口 |
| MVTec AD | 历史pilot及排除已查询来源后的冻结跨数据集评测 | [MVTec AD官方页面](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| Real-IAD | 前期跨零件密集分割及多视角/近重复分组研究 | [Real-IAD项目](https://github.com/M-3LAB/Real-IAD) |

交付包含选定来源ID、划分、内容hash、指令目标和评测记录，不包含原始数据图片、完整标注图或下载归档。数据集必须从适用来源另行获取并遵守其条款。自定义训练使用真实异常监督，报告数字不应标作官方normal-only基准。

## 资产与发布范围

本包保留必要运行时源码、许可证、配置、实验协议、轻量划分、预测与审计结果。底座/LoRA/decoder权重、tensor缓存、训练图像、服务器认证、原始会话和无关论文库均不在此交付范围。历史manifest可能记录未随包资产的路径与SHA，这些记录用于身份追溯，不表示文件已附带；具体路径映射与准备要求见 [ASSETS.md](ASSETS.md)。

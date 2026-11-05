# 资产与复现说明

本目录是代码、配置、实验文档和轻量证据的交付包。原工作目录的文件保留；整理过程没有启动新训练或推理。实际交付路径、大小和 SHA256 见 [DELIVERY_MANIFEST.tsv](DELIVERY_MANIFEST.tsv)。

## 已包含

- 项目 Python/Shell 实现、环境声明、协议、配置和冻结运行时源码。
- 小型数据划分、指令 JSONL、样本来源 manifest、轻量逐图预测、指标及审计结果。
- 项目图表、全程精读、代码索引、成果总结、简历和面试问答。
- AnomalyGPT/ImageBind 必要运行时代码、tokenizer/BPE 及原有许可；LoRA 配置和保存时的 tokenizer 元数据。adapter 的 tensor 权重不包含。

`industrial_vlm/prepared/` 的训练文本与来源身份用于解释历史协议，不包含对应完整图片数据集。数据获取及使用须遵守原数据集条款。历史报告与旧 run 可能引用未随包交付的服务器资产，这是资产身份记录。

## 需单独准备的资产

| 资产 | 公开来源/记录 | 用途 |
|---|---|---|
| Qwen3-VL-8B-Instruct | [Qwen 官方模型](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 语言迁移与原生视觉路线；加载路径不同 |
| AnomalyGPT 与 ImageBind 权重 | [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)、[ImageBind](https://github.com/facebookresearch/ImageBind)、[checkpoint 来源](AnomalyGPT/code/pretrained_ckpt/SOURCES.md) | 历史视觉前端、PromptLearner 与语言基准 |
| Vicuna 及其底座/转换资产 | [FastChat](https://github.com/lm-sys/FastChat)、[Phase1 协议](migration_20261103/PHASE1_PROTOCOL.md) | 严格配对的 Vicuna 对照，遵守对应许可 |
| VisA | [官方仓库](https://github.com/amazon-science/spot-diff)、[数据协议](industrial_vlm/DATA_PROTOCOL.md) | 主线与原生分支的自定义有监督划分 |
| MVTec AD | [官方数据页](https://www.mvtec.com/company/research/datasets/mvtec-ad)、[冻结外部协议](migration_20261103/advanced_20261104/external_protocol.json) | 从官方 test 扣除项目历史 pilot 后的外部评估 |
| Real-IAD | [官方仓库](https://github.com/Real-IAD/Real-IAD)、[项目说明](docs/项目说明.md) | 早期跨零件实验 |
| DINOv2 / CLIP | [DINOv2](https://github.com/facebookresearch/dinov2)、[CLIP](https://github.com/openai/CLIP) | 分割与语义对照；按对应代码准备依赖 |
| 本项目训练 checkpoint / adapter | 各阶段 run、checkpoint SHA、保存重载报告 | 语言桥接、BigDec、LoRA 与学生 head；此包未提供 tensor |
| 特征、teacher map、tensor cache | 各阶段 cache manifest、来源及配置 | 在准备数据和权重后按原代码重新构建 |

公开来源仅说明获取入口，不保证资产版本、许可或数值与历史实验自动相同。须核对对应 manifest、版本和哈希；私有训练产物不因仓库有配置就变成可直接下载的权重。

## 运行路径与历史证据

项目代码、文档和结果直接位于 `Qwen-VLM` 仓库根目录；克隆后进入仓库即可按原有项目相对路径运行，无需再进入同名子目录。历史脚本可能假设服务器工程位于 `/root/autodl-tmp/...`，或要求源码和上游包处于相应 PYTHONPATH。应先阅读对应协议和 CLI，准备资产、复制到新的执行目录或明确修改新运行的路径配置，并保留新 run 身份。

原 run、split、模型哈希和 source manifest 保持原字节。旧绝对路径是当时环境的记录，不是克隆后的可用路径；包内报告对代码和资产的引用应结合来源说明理解。重新运行得到的缓存、权重或预测应写入新目录，不能覆盖旧证据。

## 本次省略范围

原始数据集及样例来源大目录、完整模型/adapter 权重、tensor 缓存、环境、系统缓存、原始 `.log`、重复归档和原始对话/认证文件不发布。独立论文精读、论文翻译、论文原文与无运行依赖的上游克隆不发布。必要 tokenizer 的 `.model`/`.gz` 是运行资源，仍保留。

此次清理仅改变 GitHub 当前版本的跟踪内容，不重写提交历史，也不删除原工作目录资料。[THIRD_PARTY.md](THIRD_PARTY.md) 列出保留运行时的来源与许可；[核验记录](DELIVERY_AUDIT.json) 区分源码复制检查和本次文档适配。

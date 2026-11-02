# 代码与资料来源

`industrial_vlm/` 与 `seg_cross_part/` 保存本项目的实验实现；`AnomalyGPT/followup_20261026/*.py` 为迁移与审计实验脚本。论文笔记记录阅读和代码核对，不将上游方法作为本项目原创算法。

## AnomalyGPT 迁移运行时

`AnomalyGPT/followup_20261026/server_snapshot/` 保存实验时使用的 AnomalyGPT 运行时及本项目迁移修改，包括 `openllama_qwen3vl_strict.py` 和缓存训练/评估脚本。它来源于 [CASIA-IVA-Lab/AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)，保留 [CC BY-NC-SA 4.0 许可证](AnomalyGPT/followup_20261026/server_snapshot/LICENSE)。上游提交身份见该目录的 `SOURCE_COMMIT`；项目修改并非上游官方发布。

其中 `code/model/ImageBind/` 来源于 [facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)，保留其 [许可证](AnomalyGPT/followup_20261026/server_snapshot/code/model/ImageBind/LICENSE)。模型权重及数据未随代码同步。

## 阅读与对照的上游项目

独立克隆的原始上游仓库、论文原文与 LaTeX 源码保留在本地。仓库提供阅读笔记和来源信息；复现实验应从对应上游获取指定代码、依赖与模型，并保留其许可。

| 项目 | 上游来源 |
|---|---|
| CLIP | https://github.com/OpenAI/CLIP |
| WinClip | https://github.com/caoyunkang/WinClip |
| AnomalyCLIP | https://github.com/zqhang/AnomalyCLIP |
| AdaCLIP | https://github.com/caoyunkang/AdaCLIP |
| AACLIP | https://github.com/Mwxinnn/AA-CLIP |
| DualAnchor | 见对应精读笔记与 meta.json |
| MoECLIP | https://github.com/CoCoRessa/MoECLIP |
| FB-CLIP | https://github.com/Xi-Mu-Yu/FB-CLIP |
| AnomalyVFM | https://github.com/MaticFuc/AnomalyVFM |
| AnomalyGPT | https://github.com/CASIA-IVA-Lab/AnomalyGPT |

本次同步未对整个仓库另行添加统一开源许可证；各第三方部分的原有许可继续适用。

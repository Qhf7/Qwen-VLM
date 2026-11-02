# AnomalyGPT 迁移实验运行时快照

本目录包含实验时的 AnomalyGPT 模型、数据处理和训练/评估代码，以及 Qwen 语言迁移修改。上游为 [CASIA-IVA-Lab/AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT)，其完整安装与模型下载说明见 [上游 README](https://github.com/CASIA-IVA-Lab/AnomalyGPT#readme)。

上游提交身份保存在 `SOURCE_COMMIT`，代码遵循随附 [LICENSE](LICENSE)。`code/model/ImageBind/` 保留自身 [许可证](code/model/ImageBind/LICENSE)。原上游依赖清单见 [requirements.txt](requirements.txt)；迁移实验还需对应 Qwen/PEFT 运行环境，实际版本见最终结果中的运行记录。

迁移脚本入口和最终协议见 [上一层 README](../README.md)；本目录不包含模型权重、特征缓存和训练图片。

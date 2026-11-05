# 源码版本与证据对应

`source_manifest.json` 是队列启动时生成的依赖清单，保留原始字节。每个实验 `run.json/source_sha256` 记录实际执行的训练脚本；`implementation_source_sha256.json` 记录交付时脚本，而不是声称所有历史运行使用同一版本。

| 版本 | migration_lora_train.py SHA256 | 对应提交 | 使用范围 |
|---|---|---|---|
| 初始版 | `dd1c93e90e2b5d39a50584375514096a2608fe869f43d7d798ab8aafb22e1b46` | `a79b747` | B/C seed42工程预检、正式B42 |
| 生成复用版 | `8819300a9abfac48229bc99fd7ae3d6fdf001a19b50e16ea82f246c2eba46f6d` | `f08bab7` | 正式C42、B43、C43、B44、C44 |

后一个版本省去同一checkpoint的重复自由生成。复用前检查完整380条分数与8条原始生成完全一致；final始终重新生成380条，best为中间步时也重新生成380条。其余训练、采样、连续分数、模型选择和参数配置保持一致。具体规则见 [PROTOCOL.md](PROTOCOL.md)。两个版本均可从Git提交恢复。

独立进程核验脚本在训练期间补充了源bridge/cache/manifest身份检查，并选取两个类别中正常、异常代表图所在的原评估批次。保留原batch大小、顺序与padding形状，避免将BF16中改变批次形状带来的数值差异误判为权重重载失败。检查结果记录实际 `verifier_sha256`、图像路径和标签；这些图像均已在本轮val评估中使用，不新增test查询。

最终的 `source_versions.json` 列出每个run实际trainer SHA与交付脚本SHA。权重和特征缓存保存在服务器，GitHub只同步代码、文档及小型结果证据。

`asset_integrity_audit.json` 记录结束后底座13个文件的重新哈希核验，以及15份正式检查点/标准adapter文件的大小和SHA256。它提供资产身份清单，不表示仓库含有权重本体。

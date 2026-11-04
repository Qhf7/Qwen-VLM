# 外部 teacher 固定框后验控制

该控制只用于解释已完成外部评估的定位结果。它不新增模型、不训练、不调整阈值、不参与 checkpoint 或方法选择，也不能支持因果结论。

执行前，`external_teacher_control.py` 要求 `external_summary.json` 为 `complete`，七个方法 × 三个 seed 的 21 份结果均已重算；同时检查 Cstructured/S × seed42/43/44 六份结构化结果的 complete、inference_complete、冻结配置、原始预测与既存 posthoc_targets 的 SHA。六份目标必须逐项完全一致。任何条件未满足即失败，不产出部分均值。

完成门控后只读取 CPU `external_teacher_cache.pt`、已有预测及已有 posthoc_targets；不重新打开 RGB、mask、模型、checkpoint 或 decoder 权重。cache、manifest、decoder 的身份必须与外部选择冻结记录一致。

几何规则完全复用已有 S 的 VisA validation 控制：每张 teacher 概率图取图内 95 分位，使用 `>=`，保留所有并列值，取二值区域 union bbox，向外取整归一化至 0..1000，并以框中心确定 grid3。使用 NumPy 既有默认 linear 分位规则；没有拟合参数。比较条件为：

1. **Unconditional teacher box**：每张图均输出 teacher 框；共享一份结果，不伪装成三个独立 seed。
2. **Cstructured score > 0 gated teacher box**：逐 seed 使用已存 Cstructured 分数，严格大于零才出 teacher 框；其余输出 No/null。
3. 已有 **Cstructured generated** 与 **S generated**：原文保留，采用冻结严格 parser；No、格式无效和缺框在全异常定位指标中按零计。

结果覆盖全部 1183 张异常、392 张正常及 15 个类别，输出全异常 IoU、IoU≥0.5、框中心落入 GT union bbox、grid3、异常出框覆盖率、条件于有效正框的定位指标，以及正常图出框比例。正常图出框比例保留了 unconditional 控制不能承担检测任务的代价。中心指标不是像素 mask 命中率。JSON 同时给出各 seed、各类别和三 seed 描述性均值/标准差。

先运行 CPU 合成检查：

```bash
python3 external_teacher_control.py --self-test
```

21 份完整外部汇总和六份结构化结果都完成后，在已有结果根目录运行：

```bash
python3 external_teacher_control.py --root /path/to/advanced_20261104
```

默认生成 `external_teacher_control.json`、`external_teacher_control.md`、`external_teacher_control_boxes.jsonl`，不修改冻结评估源或已有结果。可通过 `--manifest`、`--protocol`、`--cache` 指向同 SHA 的实际文件；不提供跳过门控或改分位/分数阈值选项。

这是一项**外部后验固定控制**：teacher 几何来自既有 ImageBind/BigDec maps，C 门控来自模型分数，S 框来自文本生成，三者是不同系统。规则化控制的格式合法性和 grid 一致性由代码赋予；其差异不能归因于 LoRA、语言推理或 mask 监督中的单一因素。

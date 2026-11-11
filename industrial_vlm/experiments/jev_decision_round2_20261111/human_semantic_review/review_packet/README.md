# 24图人工视觉事实复核材料

当前状态：等待两位复核者独立填写，尚无人工金标。

1. 复核者A、B分别复制自己的annotator_A/B.jsonl，独立填写；先只打开01_whole_images.md，完成whole_image部分。
2. 再打开02_candidates.md，填写candidate部分；不要事先查看模型分数、原标签、mask或另一位的表单。
3. visible_anomaly填 yes / no / uncertain。yes须写可见证据；no不等于保证没有不可见缺陷；uncertain须写原因，不能强行二选一。
4. evidence记可见形态和位置；defect_type_open_text允许自由描述或unknown，不猜原因，不据类别名称推断答案。
5. relationship_to_whole_image填 consistent / inconsistent / uncertain，比较候选区域的证据是否支持已记下的整图描述。
6. 完成单项后填reviewer_id、completed_utc并将status从pending改为completed。未经人工填写的null必须保留，不得当成no。
7. 保留两个原始表单。分歧由第三人或双方逐项裁决，保存独立adjudication表及理由；未裁决的条目保持不确定。

图像是冻结的448输入，不提供更高清原图，判断仅对应本输入条件。阶段2提供的候选来自已有方法；候选图不构成答案。
本包隐藏了预测、标签和source ID，来源映射另存于包外。所有样本都来自已看开发数据，不能称为新的盲测。
包内manifest提供图像SHA用于核验，不包含缺陷标签。这里的盲标指复核者看不到旧结果，不保证未曾见过图。

后续验收：24项×两人，所有状态/必填证据/时间完整、分歧逐项保留；以“图像来源”为统计单位，不能将两位复核者当48个独立来源。
这些人工记录完成后才可成为视觉事实参考。缺陷类型/解释支持评测还需固定待验证命题和独立确认来源；本包不自行证明模型准确率。

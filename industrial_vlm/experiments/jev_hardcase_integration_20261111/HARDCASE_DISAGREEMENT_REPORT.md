# 困难样本分歧分析（摘要）

本分析只读取已有 Qwen DD、Jev gate、duplicate/ROI 结果和 24 图 pilot 元数据，不发 API、不做 GPU forward。复核排序只使用模型分歧、分数边界和 gate 不确定性，不使用标签或 GT IoU 选样。

## 聚合结果

- Qwen 与 duplicate 预测分歧：**5/24**
- Qwen 与 ROI 预测分歧：**5/13** 个有 ROI 结果的来源
- Qwen viewed-diagnostic errors：**4/24**
- duplicate viewed-diagnostic errors：**7/24**

## 边界

- 24 个来源的逐样本排序、Qwen log-odds、duplicate/ROI/gate 概率保留在服务器端分析文件，仓库不发布逐样本模型输出。
- 该排序只用于安排人工复核，不构成新的测试集、泛化结论或部署依据。
- 人工填写仍使用现有匿名 review packet，并保留 `uncertain`。

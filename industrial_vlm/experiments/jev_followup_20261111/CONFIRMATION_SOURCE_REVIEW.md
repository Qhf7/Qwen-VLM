# 最终确认流程独立源码审阅

审阅仅覆盖新runner和非保护构建源码；未读取confirmation schema、文件成员、元数据、图像或标签。所有合成验证在服务器临时目录完成，0API、0模型。

- 主系统在heldout结果前固定为calibrated Jev4参考／seed43／448／p≥0.5，全240主分母；raw只作secondary。
- 开封门要求stage12完整240预测冻结、calibration120系数及heldout评价完成，并将准确系数、源码、依赖与来源SHA绑定到最终bundle。
- 一次性原子marker先于旧确认元数据读取；broker分离输入和布尔标签，API只读取输入及参考图；全部240预测与固定映射冻结后再评价标签。
- 已修复timeout、transport和HTTP200无效typed响应被误当可重试失败的问题；这些情况永久unknown，停止提交并drain，不得重新发送。
- 已接受结果与failed终态冲突、随后重发、滚动并发超过3、单目标超过3与持久预算越界均拒绝。
- 累计失败30是停止新提交阈值；已预留inflight最多使终态失败到32。跨invocation保留累计started/failed，只有显式新invocation重置连续失败计数。
- 独立7组合成ledger边界全部通过；producer自测11组另行通过。
- 报告保留历史元数据标签抽样、公开基准污染未知、近重复独立性未知及teacher暴露需单列审计的限制。

## 来源

- `code/final_confirmation.py` SHA256：`ba312df2a904122e1a9dab72c62fe01d5beed40dbe507e1091bc8c21e8366bc4`。
- `stages/13_preopening_source_review.json`记录全部审阅条件。

本审阅不代替stage12实际完整性和数值审计，不开启确认数据、不跳过最终bundle或一次性访问门。

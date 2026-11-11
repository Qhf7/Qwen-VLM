# Qwen DD gate 离线回顾协议 v1

状态：已在候选几何审计冻结后执行；全程不调用 API、不做 GPU forward。

## 目的

判断 Qwen DD 自身的异常分数门控是否已经不差于 Jev crop-needed gate，从而决定是否值得为 gate 单独支付 Jev 调用成本。

## 固定输入

- 样本：既有在线 pilot 的 24 张开发图，12 正常、12 异常；不扩样、不换标签。
- 候选：在线 pilot 的同一 eligible 候选集合。
- 预算：固定 `K=8`。
- Qwen 分数：读取 stage09 `scores.jsonl` 中已有 DD `anomaly_log_odds`，固定既有规则 `>0`，不在 24 张标签上重新选阈值。
- 对照：Qwen DD gate、Jev gate、new duplicate、同 K 随机。

## 指标与限制

报告 AUROC、AP、Recall、FPR、保留候选数、共同 K 和 Jev 调用数。所有分类指标均标记为 `image-label proxy`，因为任务本身是候选区域判定。

Qwen DD 的异常分数与 Jev 的 crop-needed 概率语义不同；本回顾只回答“在这 24 张开发图和固定 K 下，已有 Qwen 分数是否表现出额外门控价值”，不能当成新的部署结果，也不能据此调阈值。

## 决策规则

- 若 Qwen DD gate 不差于 Jev gate，则停止单独的 Jev gate 付费调用；
- 若三种 gate 接近，则停止路由线，优先修复候选几何或整图判断；
- 只有出现预先定义的稳定正信号，才新建本地决策头的 source/byte-disjoint calibration 与 holdout 协议。

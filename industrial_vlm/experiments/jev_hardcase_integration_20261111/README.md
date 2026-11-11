# Jev 困难样本二次判定器：下一轮实验目录

本目录承载“Qwen DD 主模型 → 困难样本路由 → Jev 二次判定”的新实验，不覆盖既有 Jev 第一阶段、stage03/stage05 候选结果、stage09 Qwen 结果或 stage13 confirmation。

> 代码与结果中的 `/root/autodl-tmp/...` 是历史服务器路径；换环境运行需重新映射输入、权重与输出目录。

## 执行顺序

1. 无 API 候选几何审计：固定 `square224 → 原图 → 448` 坐标链，比较三种预注册候选规则。
2. 使用既有 stage09 Qwen DD 分数做同 24 图、同 eligible、K=8 的离线 gate 回顾。
3. 若前两步出现预先定义的正信号，再建立 source/byte-disjoint 的本地决策头协议。
4. 最后才进行新的来源互斥确认实验。

所有服务器实验必须使用指定 AutoDL 实例和已核验环境；本机只保存协议、代码和查看结果。confirmation240 已查看，不能用于本轮调参或重新称为盲测。

## 文件

- `protocol_candidate_geometry_v1.md`：候选几何审计协议。
- `protocol_qwen_dd_gate_retro_v1.md`：Qwen DD gate 离线回顾协议。
- `STATUS.md`：连接和执行状态。
- `REPORT.md`：候选几何实际结果。
- `QWEN_DD_GATE_RETRO_REPORT.md`：Qwen DD gate 实际结果。

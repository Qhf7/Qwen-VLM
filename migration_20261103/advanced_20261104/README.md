# Qwen→AnomalyGPT 进阶实验（2026-11-04）

用户授权按顺序执行：外部协议冻结 → D 联合桥接/LoRA → 同采样 CE/排序损失 → mask监督结构化定位 → 模型自主选择 ROI 工具。

- [冻结协议](PROTOCOL.md)：可比条件、损失权重训练集校准、外部边界。
- [CPU及数据审计](AUDIT_NOTES.md)：真实代码契约回归，内容哈希排重。
- [结构化定位](STRUCTURED_PROTOCOL.md)：继续原 C best，输出 Yes/No + bbox/位置，train mask只做输出监督。
- [ROI工具](TOOL_PROTOCOL.md)：原生Qwen独立路线，真实JSON请求及裁剪返回，五臂控制与成本。
- advanced_runner.py：共享GPU锁下串行D与Cpair/R三seed；verify_advanced_queue.py随后新进程回载。
- continue_stages.py：前序训练/回载完成后串行S预检/三seed/回载，再工具预检/全val及后验审计。
- summarize_advanced.py：仅纳入已完成正式运行，逐份原始预测重算，best/final与配对差并列。

历史 B/C 源码与结果未改写；新训练器为 migration_advanced_train.py。运行中目录/日志在服务器，轻量结果同步本目录。外部1575图manifest已固定；在全部checkpoint和prompt冻结前不打开其模型结果。

当前这里只报告实际完成的审计与运行，不将已实现或排队的阶段视为训练完成。汇总以 RESULTS.md / summary.json 的状态和独立回载报告为准。

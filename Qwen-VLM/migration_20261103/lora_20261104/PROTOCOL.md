# 桥接基线上的语言 LoRA：B/C 配对延续实验

2026-11-04 执行协议。以下为预先固定的起始配置，不代表最优超参。

1. 先完成当前代码/文档 GitHub 同步，再进行工程预检，门槛全部通过后进入正式训练。
2. A 是各 seed 的既有 Qwen best bridge，不新增训练。B 从同一 bridge 出发继续训练桥接；C 冻结 bridge，仅训练语言 q_proj/v_proj 的 LoRA。分别使用 seed42/43/44，不拿单 seed 与旧三 seed 均值比较。
3. B/C 各新增500updates，microbatch1、accum4、共2000曝光。每 seed 使用 `seed+100000` 的共同正常/异常均衡 train 采样序列，两臂序列相同。沿用原答案模板、teacher/cache、评分与生成代码；无新 test 查询。
4. 冻结 BF16 基座，桥接/LoRA/AdamW 为 FP32；无量化、无梯度检查点。C：r8、alpha16、dropout0、bias none、q/v；候选学习率 C=1e-4，B=5e-4；共同 OneCycleLR、weight_decay0.01、clip_norm1。每条图像独立计算答案token均值CE再累积，此口径两臂一致。B/C不同参数组学习率属于各自起始设置，不作等学习率算法隔离的表述。
5. step0完整val380并生成；每100步完整val380评分；按Image AUROC选best，step0也有资格，严格提高才替换。报告best与final的AUROC/AP、生成Macro-F1、召回、FPR/invalid、类别等权指标、训练参数量、峰值显存、吞吐。step0被选中时明确表示新增训练未改善选择指标。
6. seed42工程预检各40步，只用固定8个train样本拟合、8个val样本检查接口；工程预检数值不作效果结论。验证零adapter全logits精确一致、冻结参数版本不变、梯度有限且进入optimizer、LoRA A第二步获得梯度、桥接冻结、固定样本loss改善、主动扰动后checkpoint重载评分精确恢复及PEFT标准导出一致。
7. B存best/final桥接权重，C存best/final adapter与标准PEFT best_adapter。绑定source bridge、基座文件、cache、采样与源码SHA。未保存optimizer状态，失败后使用新目录重跑，不能称断点精确恢复。
8. D（联合训练）在B/C三seed结果分析后决定。多轮使用的val380仍是开发验证；固定模板不能证明复杂解释/推理能力提高。

运行优化：首个B42完整重复起点/best生成。后续起点在全部380连续分数与同seed历史结果精确一致、8例自由生成逐字符串一致后复用已保存的380条原始答案；best为step0或final时也在完整分数与8例生成核验后复用对应答案。final始终重新生成全部380图，best为中间步时重新生成全部380图。这只省去同一checkpoint的重复生成，不改变评分、生成或选择规则；每run记录真实执行源码SHA。

运行入口为 `migration_lora_runner.py`。它持有既有GPU文件锁，串行执行B42/C42工程预检，再B42/C42/B43/C43/B44/C44，任何检查失败即停止。

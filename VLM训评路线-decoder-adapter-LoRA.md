# 从「整图 Qwen 出有无 + mask」往下做：decoder / adapter / LoRA

> Baseline = 整图进 Qwen-VL，让它直接说有没有缺陷，并给出 mask。  
> 原则：**一次只开一种可训练模块**；mask 不再由 VLM 生成。

---

## 0. 先改 Baseline 的输出契约（不训，先做）

当前设定有两个问题叠在一起：

1. 通用 Qwen **没见过工业缺陷指令**，整图语义会盖住小缺陷。
2. 让它 **直接出 mask**（框/多边形/格子再涂满）——这是 v5 同类失败，Pixel AP 会被面积打死。

在开 decoder 之前，把 Baseline 收成可评的 **B0**：

```text
整图 → Qwen（零样本，冻权重）
     → 只允许：是/否 + 3×3 位置词
     → 图级分：P(Yes) 或 anomaly token 概率
     → 不交 VLM mask；像素级标「暂缺」或只用 3×3 上采样当弱定位（表上分开写）
```

可选 **B0w**：同一 prompt，只加 3×3/5 窗聚合。仍不训。  
B0 / B0w 必须在 **冻结 Dev-mini** 上留下一行数，后面所有实验都和它对。

---



## 1. 三块东西各管什么（不要一起拧）


| 模块          | 在 Qwen 里对应谁                                               | 你在优化什么                     | 先不做什么                         |
| ----------- | --------------------------------------------------------- | -------------------------- | ----------------------------- |
| **Decoder** | 外挂：视觉塔中后层 patch → 热图 M                                    | 让模型**看见局部**                | 热图不直接当提交 mask                 |
| **Adapter** | ① Prompt Learner：M→prompt token；② 视觉连接器/浅层 Visual Adapter | 让 LLM **读到**局部信号 / 轻度对齐工业图 | 不解开视觉塔全参                      |
| **LLM 微调**  | Qwen 语言塔 LoRA（q/v）                                        | 让它按你的 JSON/Yes-No **说稳**   | 第一轮不开；数据 <200 且 decoder 未通时不开 |


开训顺序固定：**数据 → Decoder+Prompt（Adapter①）→ 必要时 Visual Adapter → 最后 LoRA。**  
AnomalyGPT 主路径也是冻 encoder/LLM，只训 decoder + prompt learner。

---



## 2. 冻 / 训 总表（按轮）


| 轮次      | 名称                       | 冻                       | 训                        | 相对谁比                    |
| ------- | ------------------------ | ----------------------- | ------------------------ | ----------------------- |
| **B0**  | 零样本整图 Qwen               | 全部                      | 无                        | —                       |
| **B0w** | 零样本 + 切窗                 | 全部                      | 无                        | B0（只归因窗口）               |
| **E1**  | Decoder + Prompt Adapter | 视觉塔 + LLM               | Decoder + Prompt Learner | B0（主消融）                 |
| **E2**  | + Visual Adapter         | 视觉塔主干 + LLM             | E1 的模块 + 浅层 adapter/连接器  | E1（只归因 adapter）         |
| **E3**  | + LLM LoRA               | 视觉塔                     | E2 已收敛模块 + LoRA          | E2（只归因 LoRA）            |
| **N1**  | 负对照                      | —                       | Decoder 热图阈值当 mask       | 证明「头当 mask」不行           |
| **H**   | 赛线                       | DINO + 现网 Qwen residual | 不在这张表里训                  | 和 E* 的 mask **不要比分数混谈** |


E2、E3 只有上一轮 **识别过线** 才开。三轮损失、prompt、子集必须同一套。

---



## 3. 数据（三轮共用，先造齐再开训）

> S0 已于 2026-08-31 冻结：主输入 direct 448；GT nearest-neighbor 对齐；
> 图像级样本按 `group_path`（5 视角）聚合；train/dev 按跨批互为最近邻得到的
> `visual_group_id` 原子化拆分；四折 holdout 类表默认不可重建。

和 schema 对齐，建议第一批就能训 E1：

- ≥ 伪异常（Cut-Paste/NSA + 近似 mask）+ 正常 + **hard-normal**（边、反光、纹理）  
- 对话：`Is there any anomaly?` → `Yes, at bottom left` / `No`  
- 解析稳定后可加短 JSON（`is_anomaly`, `region`），**不要让模型输出多边形 mask**  
- `train` / `dev-mini` / `Holdout-zs` 划死；Holdout 类不进训练、不进调参  
- 可选混少量通用图文，防 E3 之后只会答质检

Holdout 不用于决定是否开 E2/E3；阶段门只看 Dev-mini 与冻结折稳定性。
B0+、E1、E2、E3 配置全部冻结后，再一次性跑 holdout，并且不根据结果返回调参。

没有 hard-normal 不要开 E1：decoder 会学成物体性。

---



## 4. 每一轮怎么做



### E1 · Decoder + Prompt Adapter（第一刀，主消融）

```text
冻 Qwen 视觉塔 + LLM
中间层 patch → Decoder → 热图 M
M → Prompt Learner → 若干 prompt embedding
与图像 token、用户问题一起进 LLM → 只生成是/否 + 3×3
```

- 实现冻结为视觉塔第 7/15/23/31 层 patch，256-channel decoder，4 个条件虚拟 prompt token。
- 损失：`BCE(Yes/No logits) + 0.2×(focal+dice) + 0.1×CE(3×3+none)`。
- Checkpoint 只按 **Dev-mini 18 类宏平均 I-AUPR** 选，不按 Pixel AUROC。
- 过线：宏 I-AUPR 相对最终 B0+ `0.7495` 绝对提升≥0.03，宏 F1max 不低于
  `0.7380`，至少 3/4 个冻结折为正增益，小缺陷召回不降且预测面积不爆炸。
- 不过线：查 hard-normal、层数、λ，**不要立刻上 LoRA**。

2026-09-01 fold1 已完成 1000 步：四次完整 Dev-mini 中，step 500 的
宏 AUPR/F1max 最优为 `0.7489/0.7679`，小缺陷召回 `0.8214`。
宏 AUPR 低于 `0.7795` 阶段门，且 step 750/1000 出现空掩码塌缩。
结论是 **E1 fold1 未过线，不开 E2 Visual Adapter 或 E3 LoRA**；
下一步只做 E1 内部的损失、hard-normal、decoder 取层或 `lambda_seg`
单变量诊断。完整记录见 `E1-fold1实验结果分析.md`。



### E2 · Visual Adapter（E1 过线才开）

在 E1 上只加一块：视觉连接器或 1–2 层 Visual Adapter（patch 投影）。  
仍冻 ViT 主体和 LLM。  
过线：图级或 crop F1 比 E1 再好一截；差不多就 **停在 E1**，写「adapter 增量不显著」。

### E3 · LLM LoRA（最后）

- 只挂 `q_proj, v_proj`，rank 8 或 16，lr 大于 decoder 一档。  
- 继续冻视觉塔；decoder/adapter 可冻住只训 LoRA，或用更小 lr 联训。  
- 阶段门只看 Dev-mini 是否掉点。配置冻结后在最终一次性 Holdout 报告中
  检查是否过拟合指令格式 / 品类词；若掉点，作为负结果记录，不用 Holdout 调回去。
- 消融里保留「只 LoRA、无 decoder」一行（E3−）：论文里这条往往不如 decoder+prompt。用来证明 **不是调了 LLM 就行**。



### N1 · 负对照（E1 训完顺手评）

Decoder 热图阈值 → mask，报 Pixel AP / Hit@1%。  
预期差。这行是面试和实验卡上的「为什么不让头出 mask」。

---



## 5. 指标（三轮同一张表）


| 指标                       | B0      | E1–E3       | N1     |
| ------------------------ | ------- | ----------- | ------ |
| 图级 Acc / F1 / AUROC / AP | 必报      | 必报          | 可不报    |
| 3×3 格子对不对                | 选报      | 选报          | 否      |
| Pixel AP / F1 / Hit@1%   | 无（或弱格子） | 只诊断，不选 ckpt | **主报** |
| 像素 AUROC                 | 不决策     | 不决策         | 不决策    |
| p50 延迟                   | 抽 20 张  | 抽 20 张      | —      |


赛线 H 的 Pixel AP 不要写进「E1 比 B0 涨了」同一句里。

---



## 6. 两周排期（交 A 榜之后的主推日历）


| 天     | 做                              | 不做              |
| ----- | ------------------------------ | --------------- |
| D1    | 钉 B0：改输出、补 P(Yes)、Dev-mini 一行数 | 不改模型            |
| D2    | B0w 窗口对照（作业直推1）                | 不叠 prompt 模板    |
| D3–D4 | 指令 + 伪异常 + hard-normal         | 不开 LoRA         |
| D5–D7 | 训 E1，评 vs B0，跑 N1              | 不上 adapter/LoRA |
| 第 2 周 | E1 过线再 E2；E2 过线再 E3 和 E3−      | 三模块一起搜超参        |


A 榜期间：提交仍走 DINO hybrid；这条线只占训评叙事。

---



## 7. 面试怎么讲这条线

> Baseline 是整图 Qwen 零样本，既判有无又出 mask。mask 这路和 CLIP 直接分割一样，会把零件涂亮，所以我把输出改成是/否加格子，图级用 token 概率。  
> 然后冻视觉塔和 LLM，只训 decoder 和 prompt adapter，让局部热图以 prompt 形式进 Qwen，消融是有无 decoder。  
> Adapter 和 LoRA 是后两刀，各只加一个模块；只有 decoder 的识别过线才往下开。提交定位仍以 DINO 为主，decoder 热图不当交榜 mask。

---



## 8. 实验卡（课上填，不许事后编）


| 项          | B0    | E1    | E2    | E3  |
| ---------- | ----- | ----- | ----- | --- |
| 冻 / 训      | 全冻    |       |       |     |
| 图级 F1 / AP |       |       |       |     |
| 相对上一行归因    | —     |       |       |     |
| N1 Hit@1%  | —     |       | —     | —   |
| 是否开下一刀     | 先做 E1 | □是 □否 | □是 □否 | 停   |

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

和 schema 对齐，建议第一批就能训 E1：

- ≥ 伪异常（Cut-Paste/NSA + 近似 mask）+ 正常 + **hard-normal**（边、反光、纹理）  
- 对话：`Is there any anomaly?` → `Yes, at bottom left` / `No`  
- 解析稳定后可加短 JSON（`is_anomaly`, `region`），**不要让模型输出多边形 mask**  
- `train` / `dev-mini` / `Holdout-zs` 划死；Holdout 类不进训练、不进调参  
- 可选混少量通用图文，防 E3 之后只会答质检

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

- 损失：`λ_seg (focal+dice)` + `CE(回答 token)`，`λ_seg` 先小。  
- Checkpoint 按 **Dev-mini 图级 AP / F1** 选，不按 Pixel AUROC。  
- 过线：Holdout 上图级 AP 或 F1 **明显高于 B0**，且同方向可复现。  
- 不过线：查 hard-normal、层数、λ，**不要立刻上 LoRA**。



### E2 · Visual Adapter（E1 过线才开）

在 E1 上只加一块：视觉连接器或 1–2 层 Visual Adapter（patch 投影）。  
仍冻 ViT 主体和 LLM。  
过线：图级或 crop F1 比 E1 再好一截；差不多就 **停在 E1**，写「adapter 增量不显著」。

### E3 · LLM LoRA（最后）

- 只挂 `q_proj, v_proj`，rank 8 或 16，lr 大于 decoder 一档。  
- 继续冻视觉塔；decoder/adapter 可冻住只训 LoRA，或用更小 lr 联训。  
- 必须看：Holdout 是否掉点（过拟合指令格式 / 品类词）。掉了就回 E1/E2。  
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



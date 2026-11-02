# AnomalyGPT、基础注意力与本项目实现核对

阅读日期：2026-10-23。依据本地论文、报告、源码的静态阅读；没有启动训练、下载权重或重新评测模型。

## 已读材料

- `AnomalyGPT/paper.tex` 正文及附录、`paper.txt` 对照、`outline.md`、`meta.json`；核看 `figures/fig-02.png` 架构图与 `tab-04.png` 消融表。
- `AnomalyGPT/精读笔记.md`、`复现指南.md`、`code/README.md`、`code/pretrained_ckpt/SOURCES.md`。
- 核心实现：`code/code/model/openllama.py`、`AnomalyGPT_models.py`、`agent.py`；`train_mvtec.py`、`test_mvtec.py`、`test_visa.py`、`web_demo.py`；`datasets/mvtec.py`、`self_sup_tasks.py`、`utils/loss.py`、DeepSpeed JSON。中文监督训练入口作路径与协议补充检查，没有逐条阅读大型问答模板。
- 总报告 DOCX 的全部正文与表格（618 段）；四篇对照、七篇总表、复现避坑报告。`docs/paper_notes/` 五份文件与根目录对应报告逐字节相同。
- `README.md`、`docs/项目说明.md`、`docs/Anomaly-VL_v2框架设计.md`，实验进度的运行台账与 I-17～I-30 重点节点；`train_seg.py` 视觉/解码器路径、`align_dino_llm.py`、`sft_stage2.py`、`build_sim_anomaly.py`、`eval_pair_big.py`。
- `文献阅读/Transformer.pdf` 主文 pp.1–9。没有在本地找到对应原始 tensor2tensor 仓库。

## AnomalyGPT 的方法与信息流

中心命题是：用专门的密集定位模块补充通用 LVLM 缺乏的局部异常证据，把异常图编码为软提示，保留语言交互，同时支持正常参考样本迁移。其像素图由视觉分支产生，语言分支负责根据视觉与定位证据生成判断和粗位置。

1. 冻结 ImageBind-Huge，取第 8/16/24/32 层。图像为 224×224，patch 网格 16×16，每个 patch 1280 维。四个线性层把 patch 投影到 1024 维，与 ImageBind 文本塔的正常/异常原型对齐。源码位置：`model/openllama.py:176`、`:259`，`model/AnomalyGPT_models.py:17`。
2. 当前实现先算 `100 * cosine(patch,text)`，把两通道 logits 双线性插值到 224×224，再 softmax；四层异常概率取平均。顺序是 **插值 logits → softmax → 层平均**，不能任意换为插值概率。`openllama.py:443–469`、`:602–615`。
3. 正常参考路径使用原始 1280 维 patch 的最近邻余弦距离；每层先找最大相似度，再跨层平均相似度、插值，得到 `1-sim`。提供参考图时，这个结果**替换**文本匹配图；源码中的两路融合是注释。`openllama.py:619–636`。
4. Prompt Learner：5 次 3×3 Conv/ReLU/Pool 将 224×224 压到 7×7；最后 5×5 Conv 得到 3×3×4096，reshape 为 9 个动态 token，再拼 9 个 base token。整图全局向量 1024→4096 另占一个 token。LLM 接收 1 个图像 token、18 个软提示 token 和文字。`AnomalyGPT_models.py:32–73`、`openllama.py:372–415`。
5. 九个动态 token 来自空间网格，但不能等同于九块互不重叠的九宫格。按卷积核/池化步幅推算，单个输出位置的理论感受野约 222×222，几乎覆盖整张 224×224 输入；模型没有显式把每个 token 监督成一个方位词。
6. 可训练模块必须看训练代理，不能只看模型构造。构造时有 PandaGPT LoRA 与 `llama_proj`，但 `agent.py:13–20` 冻结全部参数后仅解冻 decoder、prompt learner。冻结 LLM 参数仍须允许损失对软提示的输入梯度回传。

## 训练、数据与成本

普通 MVTec 入口只收集 `train/good`（`datasets/mvtec.py:78–83`）；在线 NSA 风格粘贴/Poisson 融合生成异常。mask 为基于强度变化的 logistic-intensity 标签，进模型后以 0.3 二值化；并非简单将整个粘贴矩形都当真实异常。`self_sup_tasks.py:93–108`、`openllama.py:455–458`。

每个原始实例在 collate 中展开成正常与合成异常两条（`mvtec.py:198–224`）。损失为答案 token CE 加四层 focal、异常通道 Dice 的和；通用 PandaGPT batch 仅用 CE。`openllama.py:461–528`、`:530–581`。论文说三种损失（CE/Focal/Dice）与代码把它们组织为 `loss+loss_pixel` 一致，**不是损失数量冲突**。原笔记所谓额外独立“图像描述 CE”没有原文依据。

`train_mvtec.py:86–100` 交替执行工业 batch 与通用 batch，按调用次数为 1:1；工业 collate 又加倍了样本数，不能直接称“每种图像各占一半”。符合路径长度条件时以 50% 概率改用参考距离图给 Prompt Learner，**但像素分支的分割损失此前仍然计算**。这更准确地说是在切换语言端收到的地图来源。训练中的正常参考还做 4 个直角旋转。

按卷积和线性层形状逐项计算：decoder 5,246,976 参数；Prompt Learner 107,416,504；总计 112,663,480。最后一层卷积为 104,861,696；所以“最后一层 107M”不精确，107M 是整个 Prompt Learner。这里是由源码形状计算的参数量，没有用模型实测显存作佐证。

论文 §4 implementation（`paper.tex:363`）写 50 epoch、两张 3090、lr=1e-3、batch=16，以及 one-cycle cosine；当前 DeepSpeed JSON 实际指定 `WarmupDecayLR`，不能把配置自动解读为论文所说 cosine。配置 microbatch=1、累积=8、world=2 得到名义 batch=16，工业样本展开与交替更新应另行记账。复现指南的 6–10 小时、20–22GB、单卡 ZeRO-3 改法均属于估计/建议，没有本次运行证据。

## 评测协议与报告校正

| 设置 | 训练 | 测试 | Image AUROC | Pixel AUROC | 语言 Accuracy |
|---|---|---|---:|---:|---:|
| 同域无监督 | MVTec 正常图+合成 | MVTec | 97.4 | 93.1 | 93.3 |
| 跨域 1-normal-shot | VisA 正常图+合成 | MVTec+正常参考 | 94.1 | 95.3 | 86.1 |
| 跨域 1-normal-shot | MVTec 正常图+合成 | VisA+正常参考 | 87.4 | 96.2 | 77.4 |

来源：原文 introduction `paper.tex:45`、few-shot 表、unified 表；表中数字为作者报告。根目录七篇总表把 VisA image AUROC 写成 77.4，是把 Accuracy 放进 AUROC 列。单篇笔记实验表这一处正确。

`test_mvtec.py:150–183` 的 Image AUROC 来自地图最大值，Pixel AUROC 来自地图，Accuracy 来自回答中的 `Yes`/`No`；三者并非同一输出头。论文消融表保留 decoder 而去掉 prompt learner，地图仍有很高 AUROC，但语言 Accuracy 没有改善，正好说明“有像素图”与“语言实际利用像素图”是两个问题。

“无需手调阈值”只表示由 LLM 输出离散语言判决；不保证经过校准、也不保证得到二值 mask 无需阈值。softmax 的两个分量相加为 1 不证明跨类别概率校准；margin、余弦距离同样可在验证集选择阈值计算 F1。不能用输出形式给出“能不能做决策”的绝对区分。

### 已由静态源码直接确认的复现细节

| 位置 | 观察 | 含义 |
|---|---|---|
| `openllama.py:21` | 实际只启用 2 个文本模板，22 模板列表已注释 | 原文附录列出 22；报告须区分配置 |
| `openllama.py:469` | 四层 mean，原文式(1)写 sum | 对 AUROC 的固定正缩放不改排序，但软提示网络对幅度敏感 |
| `openllama.py:621` | 判断为 `'mvtec' in 'normal_img_paths'`，比较固定字符串 | 此条件恒假，推理中的旋转增强分支不会启用；训练分支会增强 |
| `test_mvtec.py:12` | `argparse type=bool` | 传字符串 `False` 仍会得到 True，不能照指南当可靠模式开关 |
| `test_mvtec.py:115–117` | `round` 控制参考图索引 `4*round` | 指南解释成“重复次数并平均”错误；单次脚本不完成五次实验 |
| `test_mvtec.py:44` | 默认加载 `train_visa` 权重 | 默认是跨域 MVTec few-shot 评测，不是同域 |
| `web_demo.py:17`、README | demo 默认 `train_supervised` | 不能把在线演示当论文仅正常样本训练的证据 |
| `openllama.py:714–723` | `do_sample=True`，按固定后缀截文本 | Accuracy 有解码随机性，应固定种子、记录解析失败 |
| `ImageBind/data.py:79` / `:106` | 普通入口短边Resize+CenterCrop，web入口方形Resize | 非正方图两种入口几何不同，mask要严格对齐 |

另一个开销细节：few-shot 推理先走普通图像/decoder 路径，再重新编码 query 和 reference（`openllama.py:595–624`），因此“每图一遍 ImageBind”只适用于没有正常参考的路径。文本锚点在当前调用中也会反复编码，“可缓存”是优化机会，不是已经实现的缓存。

这些是本地版本的静态结论，尚未量化各自对指标的影响。原报告“取消 50% 切换就会崩”“LoRA 参数量一定更大”“LLM 视觉完全来自 decoder”等说法比现有证据更强，需要收窄。

## Transformer 对后续方法的连接

原文 pp.2–6：encoder/decoder 各 6 层，d_model=512、8 heads、d_ff=2048；`softmax(QK^T/sqrt(d_k))V`；原版采用 Post-LN，不能误套现在 LLM 的 Pre-Norm。decoder causal mask 约束当前位置只能看到前缀。跨模态软 token 只要维度匹配即可进入序列，但“可以输入”不等于模型已经学会理解这些特征；还需训练对齐与检验证据利用。

pp.7–9：WMT14 En-De/En-Fr；Adam、4000-step warmup、inverse-square-root decay、label smoothing 0.1；big 报告 28.4/41.0 BLEU。该文证明注意力架构的机器翻译能力，不直接证明工业异常定位、自然语言解释可靠性或当前 Qwen 位置编码的处理方式。

## 与当前项目文档和代码的关系

`docs/项目说明.md:16–39` 的原始中心命题是：seen 类 mask 监督下，在完全留出类别上提高像素 AUPRO，且归因于语言。AnomalyGPT 的语言判决与该像素主指标不同；把它接入系统后，应分别评价像素定位、语言判断、图文位置一致性，不能仅以可生成解释替代 AUPRO 增益证据。

v2 是融合设计草案，不能把其中的 DINOv3 分组分支、定位 Prompt Learner 与工业:通用=4:1 写成已经实现。已有 `align_dino_llm.py` 是 DINOv2 最后一层→64视觉token→冻结 Qwen text model，`sft_stage2.py` 再加 LoRA；当前读到的这两条路径没有定位图→9 token 分支、没有 Dual Anchors 分组模块、没有交替通用数据。AdaCLIP 借鉴点应写为 hybrid prompting+投影的 PEFT，而非称其原方法采用本项目的 LoRA adapter。

### 会影响既有实验解释的代码证据

1. `align_dino_llm.py:138–142` 遍历 batch 内每条问答时始终拼接 `vis[:1]`。batch>1 时，第二张及后续图的文字也被配给第一张视觉特征。这是明确的图文错配，会影响对“对齐训练仍退化”的归因。本次只记录，不改训练脚本、不重跑结果。
2. `build_sim_anomaly.py:143` 合并 train+val 正常图作为源图；`:150–152` 从全局 `MANIFEST_PIXEL` 采样真实缺陷面积，而没有按本折 train 过滤。如果产出的 sim 图进训练，验证实例会回流，holdout mask 的面积统计也会影响生成分布，违背项目自己的严格 holdout 规则。是否已影响某个 run 仍要查其实际使用的数据；不能直接宣判所有旧 run 作废。
3. `eval_pair_big.py:117` 的 `ans_first_rate` 统计 `anom_first` 真值，实际上是“异常位于第一张的比例”，不是模型选第一张比例。因此 I-30 用该字段“排除位置偏置”证据不成立。当前两份 `docs/results/pair_*.json` 仅有汇总，没有 `records`，无法据现有本地文件重算模型位置偏置；这也与实验记录声称“含逐组记录”不同。
4. `eval_pair_big.py:91–92` 有放回抽样图像，bootstrap 在 pair 布尔结果上重采样（`:109`）；并未按零件组重采样，重复视角的相关性没有覆盖在当前 CI 中。79.25% 为已保存汇总数字，不能在本次阅读中当作重新独立验证。
5. I-30 自身已写明 SFT 79.25% 对零样本 75% 的增益 p=0.124、样本不配对；所以它支持这批样例的相对判别有信号，不能据此证明微调有稳定增益，也不能从 448 与 1024 的不同样本推出“分辨率无关”。

后续更合理的验证是：先确保图文逐样本配对与 split 权限，再比较真实地图、打乱地图、无地图三种语言输出，并以固定 heldout、同一批样本测量语言端是否确实消费定位证据。这是阅读导出的可检验建议，不是本次已实施实验。

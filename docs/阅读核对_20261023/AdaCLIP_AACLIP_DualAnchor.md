# AdaCLIP、AA-CLIP、Dual Anchors：原文—精读报告—代码对照证据笔记

阅读日期：2026-10-23。工作区：`/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm`。

## 阅读范围与证据层级

已完整读三篇主文的摘要、引言、相关工作、方法、实验和讨论/结论；采用展开版 paper.tex 对照 source 分节源码定位。另完整读三篇 `精读笔记.md`、`outline.md`、`meta.json`。主文没有引用附录不等于源码包没有附录，因此额外检查了补充源码。

- AdaCLIP：`source/sources/{0-abstract,1-intro,2-related-work,3-method,4-experiment,5-conclusion}.tex`；主表 `source/tables/Table-{industrial,medical,prompts,HSF,backbones}.tex`；`paper.txt` 中补充 §1–5、§7.1，尤其 HSF、数据和 full-shot 对照。未逐图审读全部 26 张补充定性图，不以此声称穷尽失败情况。
- AA-CLIP：`source/sec/{0_Abstract,1_Introduction,2_RelatedWork,3_Methods,4_Experiments,5_Discussion,maintab,X_suppl}.tex`；`source/main.tex`。`X_suppl.tex` 在源码包中存在，但 main.tex:88 的 include 被注释，11 页 paper.pdf 不包含它。
- Dual Anchors：`source/sec/{0_abstract,1_intro,2_related,3_method,4_experiment,5_conclusion,X_suppl}.tex`；全部四张主文表源码；`source/main.tex`，并目视检查 `figures/fig-02.png` 总体架构。`X_suppl.tex` 未被 main.tex 引入，部分补充图表依赖在本地缺失；不能把该文件视作当前 10 页 PDF 已发表补充。
- AdaCLIP 核心代码：`README.md`、`train.py`、`test.py`、`train.sh`、`test.sh`、`method/adaclip.py`、`method/trainer.py`、`dataset/base_dataset.py`、`dataset/__init__.py`、`dataset/mvtec.py`、`tools/metrics.py`、`loss.py` focal/dice 实现、ViT-L-14-336 配置。比对 `third_party/AdaCLIP` 和 `AdaCLIP/code` 全目录差异及核心单文件 diff。
- AA-CLIP 核心代码：`README.md`、`train.py`、`test.py`、`scripts.sh`、`model/adapter.py`、`model/adapter_modules.py`、`forward_utils.py`、`dataset/__init__.py`、`dataset/constants.py`；查看 `model/transformer.py` 的 V-V attention / DAPM 接口。
- DualAnchor `code/` 只有 README，无本地训练、推理、模型实现。
- 为迁移建议阅读根 README；未对本项目实验结果做重新评测。

证据分三级：**原文声称**（主文/源码包补充文字，注明版本边界）、**静态实现可确认**（当前文件内容）、**运行验证**（本次没有执行训练、模型推理或基准评测，所以本笔记不声称复现成功）。本次未联网核验发布状态或下载材料；meta 的“官方代码”字段不代替远端核验。

## 1. 三篇实际研究的问题

共同任务是跨类别/数据集异常定位：在辅助类别上用正常/异常图及像素 mask 训练，然后对未参与该优化的目标类别输出分数和 mask。它们的 zero-shot 是“目标类别不用于模型权重训练”，并不是整个系统零监督、零数据或没有人工/图像信息。

| 方法 | 适配对象 | 文本是否依赖当前图 | 辅助监督 | 真正改变的机制 |
|---|---|---|---|---|
| AdaCLIP | CLIP 图像和文本提示、视觉投影、动态生成器 | 是，默认动态文本提示逐图生成 | 图像标签 + 像素 mask | 用静态通用偏置和图像条件偏置共同改动冻结 Transformer 的信息流 |
| AA-CLIP | 先文本 residual adapter/projector，后视觉 residual adapter/projector | 否，训练后每类可缓存 | 图像标签 + 像素 mask；类别描述另有正常图→GPT-4 构造信息 | 先建立较分离的语义靶点，再用稳定靶点适配视觉 |
| Dual Anchors | DINOv3 视觉特征上的层级分组、GGTR、图像条件文本 prompt；图示部分文本 block 可训练 | 是 | 图像标签 + 像素 mask，并复用同一 mask 做 group 语义约束 | 从图像中构造 normal/anomaly 状态锚点，改变仅以文本定义判别方向的结构 |

“共享异常模式”是这组论文的核心假设：辅助集上的缺陷模式能迁移到新物体。它不保证逻辑异常、所有成像模态或任意分布偏移都被覆盖。

## 2. AdaCLIP

### 2.1 原文方法与实际张量

原文 `source/sources/3-method.tex:11–14` 明确辅助数据包含正常、异常图和标注 mask，测试类别不与辅助类别重叠。方法不是靠对话生成 mask，而是 CLIP 特征与 normal/anomaly 文本原型的密集相似度。

在默认 ViT-L/14@336、518×518 输入下，视觉宽度 1024，37×37=1369 patch；文本和最终 CLIP embedding 768。取视觉第 6/12/18/24 层，每层独立 `Linear(1024,768)`。文本用 7 normal states、5 abnormal states，各乘 4 模板，共 48 句，逐句归一化后组内平均并归一化，得到 `[B,768,2]`。

- `method/adaclip.py:11–33`：ProjectLayer；patch 分支先去掉 CLS。
- `:146–232`：48 条文本模板、tokenization、normal/anomaly 分组。
- `:334–405`：模型组装；`:377–388` 除 patch projector 外还有 **CLS 768→768** projector。
- `:407–416`：无 prompt 的冻结视觉塔先完整前向获取 `[B,768]` image embedding，再由两组线性层生成 `[B,5,1024]` 和 `[B,5,768]` 动态提示。
- `:419–477`：带提示视觉塔第二次前向；不是只运行一次图像编码。

静态提示是逐层学习的 `[M,C]` 参数，动态提示由同一图生成后在多个浅层复用，二者逐元素相加；并非把 static 和 dynamic 拼接成长度 2M。视觉提示在第 0 层末尾追加，随后到 J 之前替换尾部 M 个提示，其余层继续传播；`:98–137`。文本实现与论文抽象式不完全同形：`:81–90` 在 index 0 **不插入**提示，index 1..J-1 替换 SOT 后 M 个 token，占原句位置，未增加 77 token 长度；`:493–509` 最终依原 token 序列 EOT 位置取特征。报告笼统说“两塔前四层都追加 5 token”掩盖了这个实现差异。

### 2.2 像素图、HSF 与训练

原文公式是 cosine→二类 softmax→reshape/upsample；实现明确用 `100*cos`（`:516`）。训练每层独立 upsample/softmax、各自监督；推理是 **先上采样各层 logits，再跨层平均，最后 softmax**（`:532–548`）。最终 `(p_a+1-p_n)/2` 在严格二类 softmax 下等于 `p_a`，不是新增独立集成。评估另在高分辨率图上施加 Gaussian σ=4（`trainer.py:193–196`）。

HSF 必须区分论文版和发布代码版：

- 原文 `3-method.tex:66–75`、paper.txt 补充 §2.2：各层 patch 聚成 K 簇；按簇平均异常值选择最高簇，再聚合这些簇的质心。
- `adaclip.py:235` 作者主动声明实现已变更。`:244–288`：先将四层 logits 平均/softmax，选 top-(5K)=100 patch；取这 100 个位置四层向量，沿通道拼成 `[100,3072]`，CPU KMeans K=20；依该分簇，把各层特征共同平均得到每簇 768 维质心，再**等权平均所有 20 簇心**。这相对于所有选中 patch 直接平均会重加权小簇。
- `:519–524`：再做 `normalize(0.2*clustered_feature+0.8*projected_CLS)`，不是直接仅用 HSF 输出替代 CLS。`:528–530` 与文本算图像二类 softmax。
- KMeans 标签从 detach 后的 NumPy 得来，分簇决策不传梯度；选中向量本身求均值仍属于 Torch 图。不能把整个 HSF 简化为“端到端可微聚类”。

实际训练 `trainer.py:103–139`：图像 focal 加四个层级各自 focal、anomaly dice、normal dice。四层分割损失求和，不在聚合后只算一次。优化器 `:68–87` 仅收集六类新增模块参数，AdamW lr=0.01、betas=(0.5,0.999)。没有统一 `requires_grad_(False)` 冻结所有 backbone 参数的显式代码；因此可以精确说“不被 optimizer 更新”，不能据此保证不存在 backbone 梯度存储。`loss.py:34–82` 内部直接求 `pt.log()`，默认 apply_nonlin=None，输入已经是 softmax 概率正是这个实现的接口；报告说“该 focal 实现期望 logits”不成立。

主文 5 epochs、J=4、M=5、单 A6000；`train.py:155–179` 默认配置吻合，且入口硬性禁止 batch≠1，理由是 unresolved bugs。不能把 batch=1 上升成动态提示数学上只能 batch=1。

### 2.3 数据、版本、评估权限

原文主文默认写 MVTec+ClinicDB、评这两集时用 VisA+ColonDB；仓库 `README.md:20–22` 更正为 **MVTec+ColonDB**、评 MVTec/ColonDB 用 **VisA+ClinicDB**，`:104–113` 命令、`train.py:138` 和 `test.sh` 遵循更正版。README 权重表 `:97–98` 的训练集标签仍有旧命名，应该按实际 checkpoint 命名和脚本记录来源，不可只抄表头。

`dataset/base_dataset.py:45` 始终用标准数据集的 test split，无论 auxiliary training 或 evaluation；这是因为无监督 AD 的标准 train 常无异常。这里“使用 test split”是辅助数据集内的 split，不自动等于目标类别泄漏，关键看类别/数据集归属。

更正版默认样本池是 MVTec 1725 + ColonDB 380 = **2105**，并非报告把 ClinicDB 612 加进去得到 2337。正常/异常为 467/1638；2337 对应论文旧默认。实际运行仍须以所用 manifest 为准。

此外存在更细的 zero-shot 条件：`train.py:84–127` 以验证集 pixel max-F1 选择 best checkpoint，README 建议多次训练按验证集选最佳；默认 train MVTec+ColonDB、validation VisA，而 test.sh 又评 VisA。这意味着按所给 workflow，VisA 没进入梯度优化，但用于 epoch/run 选择，不能把它描述为从未使用任何目标类别信息的严格 model-selection-disjoint 评估。并不据此断定论文数字无效；需要单独登记验证集权限，避免本项目照搬后破坏冻结 split。

指标 `tools/metrics.py:14–37` 是每类收集预测后 AUROC、AP、扫 PR 得 max-F1；`:40–49` 类间宏平均。单类别标签指标返回 0；论文医学 image average 仅 3 个检测集、pixel average 仅 4 个分割集，不是每项都在 7 个医学集上平均。

### 2.4 论文结果支持的边界

主表工业 I-AUROC/P-AUROC 90.2/94.2，医学 94.6/85.3；HSF 消融工业 image 86.0→90.2。投影 frozen→learnable 工业 82.9/90.1→90.2/94.2。以上只是该实验设计下的作者结果，未在本次运行。

补充 full-shot 对照 Table 8：PatchCore/CDO/AdaCLIP 工业平均 image AUROC 95.2/96.2/90.2、pixel 97.7/98.2/94.2。报告“在 7 工业集和 full-shot 打平”夸大了原始数据，应该是少数集/部分指标能超过，平均仍有差距。原文承认逻辑异常、显著偏离辅助分布的失败；逐图有 GT 的 prompt 优化只是 oracle 分析（`4-experiment.tex:111`），不是 zero-shot 推理程序。

### 2.5 两份 AdaCLIP 的区别（已静态核验）

`AdaCLIP/code` Git remote 为官方 `https://github.com/caoyunkang/AdaCLIP`，HEAD `b762ac40c3f33c77e7e513e48cb436f059d456da`，2025-07-08。`third_party/AdaCLIP` 无独立 .git，代码、README、train/test、metrics 基本一致；全目录 diff 唯一实质源代码差异在 `method/adaclip.py`：third_party 将 KMeans 输入从 `.detach().cpu().numpy()` 改为 `.detach().float().cpu().numpy()`（third_party:270），注释说明适配 Apple MPS 的 float16 CPU KMeans 慢路径。它是已存在的本地适配，不是本次修改，也不是论文新版本。

third_party 额外有三个 pretrained .pth、log.txt、__pycache__，而 `AdaCLIP/code` 没有本地权重。文件存在不证明质量或对哪一个目标协议适用，尤其 pretrained_all 为 all-datasets 演示权重，若包含目标类不能作为严格跨类 zero-shot 结果。

### 2.6 精读报告需修订项

1. `精读笔记.md:89` 说文本可缓存，`:307` 又正确说每图重算。默认 SD+VL 应用后者：动态文本提示依赖图，缓存同类原型会改变算法。
2. 其信息流把最终 CLS 写为 1024→768 新投影，实际 CLIP 已投影至 768，新增 cls_token_layer 是 768→768（adaclip.py:383–388、474–475、488）。
3. `:205`“focal 期望 logits”应改为上述概率接口。
4. `:49/:401` 从低/高 max-F1 推断绝对分数不可用/更可用没有证据：max-F1 对严格单调变换不变，不证明概率校准、固定阈值迁移或报警成本。
5. `:175` dice 算例有算术错误：518²=268324，100 个缺陷内 p=.8、其余 p=.05 时 Σp=80+268224*.05=13491.2，并非 575；loss≈1−161/13592.2≈0.9882。另把两个 focal loss 值之比说成梯度之比也不严谨。
6. “主干不能微调，一调就毁泛化”“dice 天然对不平衡免疫”“4 层求和所以像素权重正好 4 倍”等是过度绝对化解释；实现/消融只支持该设定下有效，损失实际量级和梯度贡献需测。
7. 默认数据统计与 full-shot 对照见前述更正。

## 3. AA-CLIP

### 3.1 方法的关键并非只有正交惩罚

原文 `3_Methods.tex:80–132` 的完整设计是：第一阶段以固定视觉特征稳定地学文本 normal/anomaly anchors，第二阶段以固定文本 anchors 稳定地学视觉。浅层 residual adapters 防止过大漂移；不是依靠把两个向量正交就自动得到正确异常语义。

`model/adapter_modules.py:6–22` 是不带 bias 的 Linear+LeakyReLU（projector 的 relu 可开关）。`adapter.py:90–111/:125–140`：每个 backbone block **之后**接 adapter；先把 adapter 输出的 L2 norm 匹配到原特征，再 `x'=0.1*adapt_out+0.9*x`。文本 K_T=3、图像 K_I=6。文本最后使用新 `SimpleProj(768,768, relu=True)`，并非原 `text_projection` 本身被训练；图像每层 patch 用各自 1024→768 projector。

混合前模长匹配不代表混合后模长不变。若两分量等长 r、夹角 θ，则 `||x'||²=r²[(1−λ)²+λ²+2λ(1−λ)cosθ]`。除同向或退化情况会缩短。报告的“只改方向，不改大小”必须改为“控制支路尺度，限制每层扰动”；其“每层 4 度、6 层 25 度”只是人为例子，不能成为实际旋转角度或遗忘机制的结论。

### 3.2 训练代码比原文多出/少掉的内容

第一阶段 `train.py:73–85` 使用另一个经 `DAPM_replace(DPAM_layer=20)` 的 CLIP Surgery/V-V 图像编码器，多层 patch 经 ln_post、原 CLIP projection 和归一化后，**加上普通 CLIP 的归一化 CLS**作视觉参照。不是普通冻结 CLIP 原 patch 随意直接对齐。

关键差异：

- 原文第一阶段写 image BCE+seg+γ dis；当前 `train.py:87–100` 没有分类 loss，README:55–56 明确解释当前方法改用“给 visual embeddings 加 class embeddings + segmentation”，BCE 对工业和医学影响不同。因此这是官方实现可见差异，不宜擅自“修复”为主文。
- 第一阶段循环里 `loss = calculate_seg_loss(...)`，未累加；循环结束后才 `backward()`。故四层虽依次计算，**仅最后一个层级的 seg loss 和一次 orthogonal 项参与本批优化**。笔记没有识别此点，不应声称 stage 1 是四层 loss 求和。
- orthogonal 项 `train.py:91–95` 是 `mean_b(dot(T_N,T_A))**2`，不是 `mean_b(dot²)`，混合类别内积不同符号可能抵消；文本已归一化所以这里 dot 是 cosine。
- 第二阶段 `:145–159`：`CE(det_feature @ text, label)`，无温度；四层分别算 `100*cos→interpolate→softmax` 的 focal+双通道 dice 并**累加**，然后 optimizer.step，MultiStepLR 逐 iteration.step，里程碑 16000/32000。
- 原文方法写四层 projection 后先求和再做 cosine；代码训练逐层监督，推理逐层算 map 后相加，且每层独立归一化。只有在严格限定线性运算、不对聚合向量额外归一化等条件下，sum-feature 点积与 sum-logit 可等价；不能统称完全一模一样。
- 优化器分别只收集 text_adapter 或 image_adapter（`:263–274`）；底座参数并未全局设置 requires_grad=False。因此“不会更新 CLIP 参数”成立，“完全不产生 backbone 梯度”“只反传 6 层”不成立；浅层新增参数的梯度仍需经过后续冻结层。
- `:198–215`：text batch16/5 epoch/lr1e-5，image batch2/20 epoch/lr5e-4，Adam betas(.5,.999)，λ/γ=.1。模型长期 model.eval，autograd 不因 eval 而自动禁用。第 2 阶段 text embeddings 在 `no_grad` 下预计算固定（`:338–344`）。

### 3.3 推理与评测的实际定义

文本每类 6 normal+10 abnormal=16 句，均值归一化 `[768,2]`，`forward_utils.py:138–162` 和 `dataset/constants.py:135–148`；与当前图无关，可缓存。

图像第 6/12/18/24 层返回去掉 CLS 的 patch，经 ln_post、seg_proj、L2 normalize；det feature 来自最后层所有 patch `normalize(det_proj(...)).mean(1)`（adapter.py:100–111）。det feature 平均后未重新 L2 normalize，因此 `det @ normalized_text` 更准确说“归一化 patch 相似度的平均”，不完全是均值图像向量与文本的 cosine。

测试 map `forward_utils.py:196–216`：每层 `100*patch @ T`，在 37×37 上取 `(s_a+1−s_n)/2`，工业 Gaussian σ1/kernel7、医学 σ1.5/kernel9，然后 bilinear align_corners=True 到 518；`test.py:87–94` 四层相加。没有 softmax，map 为 margin，不是概率。单层理论原始 margin 可在约 [-99.5,100.5] 内（两 cosine 差∈[-2,2]），并非笔记写的大约 [-50,50]；真实范围还受锚点夹角等约束。

**最终 image 指标不是仅 det 分支。** `test.py:83–85` 得到 `(det@T_A+1)/2` 后，`forward_utils.py:241–254` 针对当前整类测试预测分别 min-max map 和 det：工业 `0.5*max(map)+0.5*det`，医学直接 `max(map)`。这涉及整类未标注测试分数的分布归一化；单一 score 的 AUROC/AP 对正单调缩放不变，但两分支各自缩放再融合会改变排序，单图部署不能机械照搬测试集归一化。代码无常数分数的零分母保护，是可定位静态边界。

`forward_utils.py:256–278` 既算 AUROC 也算 AP，故“只报 AUROC”只适用于主文主表，不是当前实现全部输出。主文像素 average 覆盖11集、图像 average 只有4工业+3医学=7集，息肉全异常数据集无有效图像级 ROC。

`test.py:172–179` 遍历每个 image_adapter_epoch checkpoint 逐个评估；若要从目标测试集选最优 epoch，必须作为额外 model-selection 权限披露，不能默认为严格零接触。脚本本身没有自动 best selection。

### 3.4 数据与来源

辅助默认 VisA，评 VisA 时 MVTec；2/16/64-shot 是辅助类别的样本数，不是目标类 few-shot。正文说随机选且 normal/anomaly 1:1，2-shot 等于每辅助类一正常一异常。当前 metadata 只提供 full-shot 文件，`dataset/__init__.py:189–202` 仅加载外部提供的 N-shot.jsonl，不实现随机平衡采样；用户须另生成，不能声称默认 train.py 会替你随机选好。

`dataset/constants.py:1` 硬编码作者数据目录，README 说可修改；仅把数据放 README 所说 `./data` 并不会自动覆盖常量。数据 augmentation 实现 `dataset/__init__.py:30–52/:89–94`：图像和 mask 共同几何变换；image 阶段另加 p=.7 色彩变化，与补充文字“一律 p=.5”的摘要不全相同。

最关键的额外信息权限：补充源码 `X_suppl.tex:57/:86` 明确描述 **class description 由正常样本输入 GPT-4 自动生成**。constants 的 descriptions 是静态保存的生成结果，不能因为存在字典就说是“作者手写整句”。目前材料没有把生成用图像的具体 split、数量和目标类信息隔离写明；不能断言训练泄漏，但与“只知道类名、不看任何目标图”的更严格设定不能不加说明就比较。可以区分“无目标标签优化模型”与“描述构造看过类正常样本”。

AACLIP/code Git remote 官方 `https://github.com/Mwxinnn/AA-CLIP`，HEAD `53db195f230442aa118c246876c94ba1c76139cc`，2025-05-26。代码 README 自称 official implementation；没有本地已训练 adapter 权重，README 只指导下载 CLIP backbone。本次未访问远端，故只确认本地 provenance，不判断当前全网权重发布状况。

### 3.5 结果与报告需修订项

主文表 1 pixel full 93.4，2-shot 92.0；MVTec full pixel91.9，2-shot91.0；表2 image best average83.1在64-shot，而 full82.5。MVTec image 在64-shot已 **92.0**（超过 WinCLIP91.8），full90.5；因此 DualAnchor 报告“首次超过 WinCLIP”在这组本地材料里已经不成立。

AA-CLIP 报告需改：

1. “无附录”只能说 PDF 无附录；本地 source/sec/X_suppl.tex 存在完整文本。
2. “手写类别描述”改为补充原文 GPT-4 正常样本生成；记录构造信息权限。
3. “只改方向不改大小”与“margin 不能扫描阈值算 F1”均数学错误。任何可排序实数分数都能扫阈值算 max-F1；没有 softmax不妨碍，只是概率未校准。
4. 文本第1阶段层级损失覆盖、batch orthogonal 平方顺序、图像分数后处理皆应补入。
5. 主文全程 frozen 是优化参数权限，不等于整个计算图都 no_grad。
6. 报告“2-shot只要30张MVTec图，MVTec pixel91.0”混淆了辅助集与测试集：MVTec 指标是默认以 VisA 类训练得到，VisA 12类2-shot应24张；30张是反向以MVTec15类训练评VisA。具体数应对应主表脚注与实际 metadata。
7. 本文只提供主文/代码的受控适配证据，不足以把灾难性遗忘描述成所有其他方法必然发生的机制。

## 4. Dual Anchors

### 4.1 核心理解及源码包补充

本文不是“两个文本向量”的 dual：是图像状态锚点和文本状态锚点互补。主文 `3_method.tex:11–12` 选 DINOv3 视觉，解释为优先密集视觉表示而非 CLIP 先验图文对齐；`:14–106` 四阶段 hierarchical grouping 用不同深度 patch progressively 建立 group token，并最终取前/后两个为 normal/anomaly anchors。

HGM：cross-attention 初始化→Gumbel assignment→ToMe式 bipartite top-r 合并→最后一块将 group 重新 assign 至 patch。group count 的主表 `ablation_group_table.tex:12` 已给 **[16,8,4,2]**，不仅在未展示补充中。补充源码 `X_suppl.tex:39–47` 进一步给四层6/12/18/24、初始16、逐级减半、**10 epochs、batch8、Adam lr1e-4、3090 24GB、约1小时**。因此当前目录报告/README 的“epoch/lr/batch/K全缺、完全不可复现”均应撤回为“缺官方可执行实现，仍有关键细节不充分”。

注意：补充所谓“四个block每次减半，16最终2”在文字计数上需按表中四阶段[16,8,4,2]解释，不能自行按四次减半实现为1；主表是可核对的具体序列。

GGTR (`3_method.tex:111–129`)：`w=[cos(CLS,G_N),cos(CLS,G_A)]`，`CLS_refined=CLS+sigmoid(Linear(Norm(w)))`。2维权重通过 Linear扩展至所需输出维；公式更像基于两分数生成逐通道残差，不直接等价于“只对 G_N/G_A加权求和”。

Dynamic State Prompt (`:132–143`)：学习 context/state token 与图像 G_N/G_A 一起进入 CLIP 文本编码器，得到逐图 T_N/T_A。正文写 [class]，图2显示 [Object]；无代码无法最终裁定是否真正 object-agnostic 或用实际类名。图中 DINO雪花、group/GGTR火焰、文本前块雪花后块火焰，支持“部分文本塔适配”，但层数、优化参数集合无实现可核验。

loss (`:171–199`) 为 image BCE+pixel Dice/Focal+group orthogonality+mask-supervised group cosine。GT mask用于分割监督并将 assigned tokens划分normal/anomaly算Z_N/Z_A；这是**同一套标注的复用**，不等于标注成本翻倍或“两份mask”。group loss明确固定锚点语义，但“没有它谁是谁完全随机”仍太强：分类/分割和带状态的prompt也能提供非对称监督，要靠去掉group loss的专门消融判断必要性，而当前主表只有去掉三个模块。

### 4.2 不能静默补全的维度/公式空白

1. 图像 feature 1024，CLIP文本 output768，见 `4_experiment.tex:13–14`；patch与文本点积、G token插入文本、CLS与文本点积都需要明确的共同空间映射。当前公式没有交代完整投影/归一化拓扑，不应直接断言“必然是一个线性层”并当成已验证实现。
2. `3_method.tex:42–50` 按所写 `F'[M,C]`、`G[K,C]`，Pi为[M,K]，`Pi*(G W_V)`是[M,C]，不能直接加到[K,C]的G。图2看起来是 group-query对patch-key/value的聚合，可能与主文符号/转置不一致。应标为需澄清的公式维度问题，不能自行选择改式后声称精确复现。
3. `:154` 的 `[(mean layers),F_assign]` 是何维拼接、如何保持空间网格、为何后面有L+1张图，定义不充分。L层图如何逐个产生也未完整写明；2类softmax下 `(A_a+1−A_n)/2=A_a`。
4. `:163–169` 文字提τ，式子却没写温度或BCE前的sigmoid/softmax；图2另显示β将refined-global分支和max(map)融合，主文最终score式并没有说明β。报告把这当完全确定的推理算法过早。
5. Gumbel原式写log alpha却又称alpha为logit，温度/straight-through/训练推理离散化未提供细节；“每patch严格0/1且反向可微”是某种ST实现选择，不是仅见Gumbel-Softmax公式即可确认的代码行为。
6. DINOv3/16配518输入不整除。用标准无padding卷积会floor至32×32是条件推导；缺实现前不能断言实际resize/crop/pad/grid协议。没有代码不能把报告张量流程当运行证据。
7. 对正常图不存在anomalous patches时Z_A如何定义、空group如何处理、L_ortho前是否L2归一化/各项权重均未足够说明。原式是dot²；若未归一化就不能直接当cos²用0.9→0.81举例。

### 4.3 实验与材料边界

主文14数据集=8工业+6医学；辅助默认MVTec，评MVTec用VisA，不同于AdaCLIP工业+医学辅助混合，也不同于AA默认VisA。医学全为分割集，主表image没有其ROC。

`source/table/main_table.tex:31–49` image只有8个工业数据集，Average **92.9** 是8集平均；`main_table_pixel.tex:32–62` pixel含14集，Average92.9是14集平均。报告称“14集图像92.9/像素92.9”分母错误。MVTec image92.7/pixel92.4，组件消融表去HGM pixel82.8、去GGTR80.6、去DSP84.6，都是作者在该协议的报告值。

该image表caption“both industrial and medical”却实际只列工业；两个表都用同一个 `label{tab1}`；主文“consistently outperforms all baselines”也强于数字，RSDD/KSDD2图像、MPDD/RSDD/DAGM/ISIC/ClinicDB/Endo像素等仍有基线更高。可以说均值改善，不能说每数据集全胜。

本地source/sec/X_suppl.tex包含训练设置及backbone/效率讨论，然而未被main.tex:70–89引入；其 `suppl_table/`、`suppl_figure/` 依赖未随本地目录存在。可以据原文字段说补充源码声称CLIP/DINOv2替换也有效，但缺表体不能核验具体比较数字，也不能说“替换主干已经排除了所有容量竞争解释”。

DualAnchor/meta.json official_code=null、code目录仅README，是**本地没有官方实现**的证据；README写2026-10-22曾检索无仓库，属于先前记录，本次未独立联网验证。没有代码不等于方法不成立，也不等于永远不可复现。当前合理边界是：有主文和部分额外源码文字，可设计独立实现，但精确实现细节、完整补充和checkpoint仍需作者澄清或补齐。

### 4.4 报告中的其他关键差异

- `精读笔记.md:97` 说“AdaCLIP dynamic prompt不经过文本塔”错误；AdaCLIP的dynamic text prompt正是经text transformer传播（其adaclip.py:503）。不能把这一点当Dual独有创新。
- `:340`“六篇第一个超过WinCLIP91.8”与其自己表中的AdaCLIP92.0及AA-CLIP64-shot92.0矛盾。
- “首次换掉CLIP主干”“2026-10公开数字最好”“必然成为/不成为标准baseline”需要更广检索或未来证据；本地这三篇材料不支持全领域首创/排名断言。
- pixel92.4与其他论文91.9/88.7的单列排序混入了数据集/训练辅助/重跑设置、backbone和score处理差异；应当用同一协议的对照行，不能拼成公平排行榜。

## 5. 对当前项目的可迁移内容（只基于现有证据）

根README的目标是DINOv2密集特征+VLM语言验证未见零件泛化；v2计划结合AnomalyGPT骨架/AdaCLIP PEFT/DINOv3。由这三篇可支持以下具体设计判断，而不能直接保证性能：

1. **固定语义靶点再适配视觉**是AA提供的可执行训练顺序。可比较“冻结文本、只训视觉”“先文本后视觉”“联合训练”，以同辅助数据、同参数量/预算检验稳定靶点是否真的必要。迁移前须选择跟随论文式还是发布代码式stage1，记录偏差，特别是loss覆盖不能悄悄带入。
2. **视觉表征和文本对齐是两种能力**。换DINO密集特征之后，必须明确1024/768或项目实际维度的projection、normalize、loss位置；Dual论文恰恰暴露直接拼接模块仍有维度空白。不能把CLIP视觉prompt原样套在DINO后就当完整多模态适配。
3. **图像条件提示不等于目标监督/测试时优化**。AdaCLIP的DPG只用当前输入做前向生成；oracle的GT prompt优化不属于合法推理流程。若project动态prompt进语言塔，应按每图文本编码预算实现，不能缓存与图相关原型。
4. **结构化视觉聚合可以作为独立比较项**：max map、top-k均值、Ada代码HSF、group anchor之间应相同特征/相同辅助监督对照。HSF含CPU KMeans和图像塔双前向，部署成本须测；third_party float32 KMeans适配已存在，保留版本注记。
5. **信息权限先统一**：auxiliary类别、epoch选择validation类别、测试类别、类名/描述、GPT看过的正常样本、正常参考图分别登记。AA的描述生成和Ada的validation selection说明“zero-shot”标签本身不足以证明公平。根项目已有冻结protocol时，应严格沿用自己的split与阈值选择，不移植论文的目标集挑epoch和整类test min-max。
6. **输出score与校准分开**：max-F1/AUROC/AP可评实数margin；softmax也不自动等于校准概率。单图部署阈值应从允许的验证集固定，另报告固定阈值F1/误报/漏报。对image指标明确是否含map最大值与det融合。
7. **同一mask用于多个loss不是增加标注量**，可以在相同辅助mask上试锚点正交和group约束；但正常图空异常集、G_N/G_A定位、归一化方式必须写明。正交仅去相关，不自动教会哪类是异常。
8. **这些论文仍以相似度/密集分割为核心**。不能据此声称语言模型已经在推理中改正mask、进行了多步因果诊断或能解释任意逻辑异常；若v2要证明VLM语言对mask有贡献，需要独立信息流和消融证据。

本次产出是阅读与静态核对；未改原报告或代码，未下载，未运行训练或模型评测。

补充到项目 v2 的术语边界：AdaCLIP 原方法的 PEFT 是**插入静态/动态 prompt token + 独立投影层/生成器**，不含在权重矩阵旁增加低秩 A/B 分解的 LoRA。若 v2 用 LoRA 适配 DINO 或 VLM，只能说借鉴“保留预训练参数、用少量新增参数适配”的原则，不能称作已复刻 AdaCLIP，亦不能照搬其10.7M、提示层数或消融解释到 LoRA 上。AA-CLIP 的 adapter 是 full-rank d×d 线性层和残差混合，也不是 LoRA。

## 6. 另读 `WinClip/与AdaCLIP对照.md` 的核对

这份短对照表同样需要按版本修订：AdaCLIP 的动态提示先运行冻结视觉塔，再运行带提示的视觉塔，并逐图运行文本塔，不能按“一次视觉前向”估计速度；当前默认辅助集为 ColonDB，非表中所写 ClinicDB。当前 HSF 是高异常 patch 聚类后等权平均簇心，再与 CLS 融合，并非只取“最异常的簇”。各项细节的代码依据见上文 AdaCLIP 小节。

对照中由不同骨干/分辨率推断“本身就能涨几个点”、把 VisA 的差距直接归因为 PCB 文字不可分，均需要同条件消融才能成为因果结论。WinCLIP 的分数可追溯到预设正常/异常文本相似度，但没有原生的缺陷原因解释输出；不能仅以手写 prompt 就断言其解释忠实性优于可学习 prompt。两者的 0/非0 任务训练参数这一基本区别仍成立。

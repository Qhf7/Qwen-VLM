# CLIP、WinCLIP、AnomalyCLIP：原文—精读报告—代码证据笔记

审阅日期：2026-10-23。工作目录：`/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm`。以下相对路径均相对此目录。只做本地阅读和静态追踪；没有运行训练、推理、权重加载或重新计算指标。仓库附带日志属于既有材料，不是本轮复现实验。

## 已读范围与来源

- CLIP：`meta.json`、`outline.md`、全部 `精读笔记.md`；`paper.tex` 完整主文 L1–708（方法、实验、人类对比、重叠分析、局限、社会影响及结论），另读附录线性探针协议 L714–797、预测说明 L1039–1043、超参 L1261–1312；用 `paper.txt` 交叉定位对称目标等内容。代码读 `README.md`、`model-card.md`、`clip/model.py` 核心架构和前向、`clip/clip.py` 预处理/加载/tokenize、`simple_tokenizer.py` 词表/BPE、ImageNet prompt notebook 的原型聚合片段。
- WinCLIP：`meta.json`、`outline.md`、全部 `精读笔记.md`、材料 `README.md`；`paper.tex` 完整主文 L1–496，附录实现 L517–605、额外消融 L606–678、失败案例 L679–716；用 `paper.txt` 核对 harmonic 和失败案例。代码读 `code/README.md`、`run_winclip.py`、`eval_WinCLIP.py`、`WinCLIP/model.py` 全文、`ad_prompts.py` 全文、`CLIPAD/transformer.py` 窗口生成与前向 L561–760、对应模型 JSON、`datasets/dataset.py`、`datasets/mvtec.py`、`utils/metrics.py` 的分类与像素指标、PRO 主逻辑。
- WinCLIP 第二份实现：`third_party/WinCLIP/README.md`、`main.py` 全文、`open_clip/model.py` WinCLIP 类 L359–570 和 window_masking、`open_clip/transformer.py` 窗口/全图前向 L645–715、`datasets/mvtec_dataset.py` 数据/参考采样/tiling L1–226。
- AnomalyCLIP：`meta.json`、`outline.md`、全部 `精读笔记.md`；`paper.tex` 完整主文 L1–401，附录实现/基线 L405–419、数据与 DPAM 分析 L420–475；`paper.txt` 交叉确认 top-only 说明。代码读 `README.md` 全文、`train.py` 全文、`test.py` 全文、`train.sh`、`test.sh`、`prompt_ensemble.py` PromptLearner 全文、`AnomalyCLIP_lib/AnomalyCLIP.py` 注意力/双路/提示词/视觉前向/编码接口、`model_load.py` 骨干来源/相似度、`build_model.py` 加载、`utils.py`、`dataset.py`、`loss.py`、`metrics.py`；读两个已有零样本结果日志。未反序列化 checkpoint。
- 三篇并非同一种“zero-shot”：CLIP 无下游训练但有 400M 图文预训练；WinCLIP 无异常任务梯度训练、可使用目标物体名/缺陷描述，WinCLIP+ 还允许 K 张目标正常参考；AnomalyCLIP 先用辅助数据集正常/异常图像标签及像素 mask 监督适配，再对未用于适配的目标数据集零样本。

## 1. CLIP：真正提供了什么接口

### 核心机制与维度

论文方法 L98–100：对 batch 内 N 对图文，分别得到表示并线性投影到联合空间，L2 归一化后构造 `Z_ij = exp(t) * <I_i,T_j>`，优化 `0.5*(CE(Z, arange(N)) + CE(Z.T, arange(N)))`。这既是图找文也是文找图的匹配，不是生成 caption，也不是异常检测的像素监督。N=32768，整个矩阵有约 10.7 亿项，但实现训练时分片计算以节省显存（L121）。

视觉模型有修改 ResNet 与 ViT 两类。ResNet 用三层 stem、anti-aliasing、attention pooling；代码 `clip/model.py:58` 和 `:94` 可核对。ViT 在 `:223` 由 patch、CLS、位置编码组成序列，transformer 后只对 CLS 做 `ln_post` 和投影（`:235`–`:240`）。文本在 `:343`–`:354` 用因果 transformer，取 EOT 位置，再做 text projection。`forward :358`–`:372` 做双侧 L2、可学习 exp(logit_scale)、点积并返回转置 logits；**此函数不算 CE，也不含训练循环**。

必须区分视觉内部宽度、文本内部宽度与联合空间 D。ViT-B/32 与 B/16 的视觉宽768、文本宽512、联合512；L/14 的视觉宽1024、文本宽768、联合768（论文附录 L1303–1306）。L/14 原生224分辨率是16×16 patches；@336 是24×24。不能把所有 CLIP 输出统一写512。

论文主文 L115 的49152与附录 L1270 的49408不一致；公开 tokenizer 的 `simple_tokenizer.py:67`–`:74` 实际组成49408词表。tokenize 的 context_length=77，含起止符，超长默认报错，可显式 truncate（`clip/clip.py:205`–`:245`）。模型列表 `clip/clip.py:30`–`:40` 当前有9个名称（含 L/14@336），不是笔记笼统说的8个。

### 训练、推理和实验边界

论文 L121：32 epoch，batch32768，Adam/decoupled decay/cosine，温度初值0.07，限制放大倍数不超过100，模型从头训练；数据是WIT 400M（L87），未公开。公开仓库主要是模型与预训练权重使用代码，不能靠本地 `train.py` 重做作者预训练——因为本目录没有该文件，也没有WIT。

推理是把 K 个候选文字变成 K 个分类器权重，图像编码一次后评分（L138）。文本可以缓存。Prompt ensemble 在**嵌入空间**进行，而不是平均 softmax 概率（L182）。本地真实实现位于 `notebooks/Prompt_Engineering_for_ImageNet.ipynb:966`–`:978`：单条 embedding 先归一化，再按类平均，再归一化。`clip/clip.py` 并没有80个模板或 prompt ensemble API。

论文 ImageNet zero-shot76.2/top5=95；27数据集胜过ResNet50线性探针16个（L159、L197）。这不是全部任务SOTA，也不是泛化保证；专业病理、计数、距离等任务弱（L199、L416–418）。自然分布偏移的 effective robustness gap 可减少75%，但不能推成任何分布都稳健。ImageNet适配的实验是**冻结特征上的监督 logistic regression**，增加9.2点ID准确率却略降平均OOD准确率（L321），不能把此证据直接说成全部端到端微调都如此（作者 L325 明确未知）。

另一个评测细节：论文线性探针对ViT用**联合空间投影前的视觉表示**（L794–795）；README简单示例用 `encode_image` 的投影后特征，不能把示例当完全复刻论文协议。模型开发曾反复查询验证集（L424）；zero-shot 是下游模型参数不训练，不等于从未接触评测反馈。重叠检测中位2.2%、均值3.2%、最大检出精度增益0.6，但作者承认不能保证400M全集召回、Overlap/Clean存在难度混杂（L404–406）。

### 精读笔记需纠正

1. `精读笔记.md:20` 提到 `train.py`，`:23`、`:138` 把80条ensemble放在 `clip/clip.py`，均不是当前公开代码布局。
2. `:114` “只做行方向文本塔永远只当负样本”错误。每行图像对其匹配文本仍有正样本，梯度同时进入图像与文本；双向目标的意义是同时优化两种检索归一化方向。
3. `:9` “任何分类任务都能……”应理解为可定义候选文本接口，不是所有任务可靠。论文列出近随机任务与预训练分布外失败。
4. `:305` “大家用的其实都是OpenCLIP在LAION上复现”错误。当前 CLIP 仓库提供OpenAI WIT权重；本项目 AnomalyCLIP 的下载地址也是OpenAI权重。WinCLIP才明确用LAION400M。
5. `:156` 以400M/500K=800推出“每query只拿到800对”不严谨。一个样本可命中多个query，原文并未给独立分桶/去重统计。
6. 原文0.07是温度**初始化**，公开前向读checkpoint中的学习值，不能把所有预训练模型推理统一当乘14.3。公开 `forward` 没有clamp，而是训练论文报告clamp过；README手工示例乘100也应区分。
7. 词表以代码和附录49408为准；基础63M文本塔不是L/14文本塔统一规格。MNIST弱的原文依据主要是训练分布稀少，不能简单说“自然语言对抽象符号帮不上忙”。

## 2. WinCLIP：窗口CLS绕过局部图文未对齐

### 原文方法

论文 L125–129：正常/异常二分类；7正常state×22模板=154条、4异常state×22=88条，共242，平均成2个文本原型。目标类名[o]可用object替代；主实验使用各目标子集物体名（附录L521）。任务专属缺陷词可额外加入，但VisA多加这些词提高0.8点是额外消融（L492），不能混入默认91.8/78.1。

论文ViT-B/16+是LAION400M OpenCLIP，输入240、视觉宽896、文本及联合宽640、12层、15×15=225 patches（L309、L601；代码 JSON）。窗口2×2有196、3×3有169，再加整图1个；窗口保留自己的patch与原位置编码，丢掉其余token，然后取经过全图文本对齐监督的**CLS**。不是把窗外RGB填零、不是每个小crop resize240、也不是直接用最后patch token做零样本分割。窗口内序列长度5或10，按尺度批处理，不能称366次串行Python前向。

L164–166：原始patch没有被直接语言监督，且混有全局上下文，原文此处是经验诊断，不是“数学上绝不在同一空间”。patch-token baseline22.4 pAUROC，window85.1说明直接局部文字匹配很差。论文聚合：每个窗口先计算**异常概率**p_a，然后对覆盖每个位置的窗口异常分做调和均值，再对尺度做调和（L170–177）。

WinCLIP+允许K∈{1,2,4}目标正常图，各自提取small/mid/window与penultimate patch三套参考。`d(q)=min_r 0.5*(1-cos(q,r))`（L183–188）。三种视觉图算术均值，再与语言图融合；原文未给此最后像素融合完整公式。图像级则明确 `0.5*(global_text_score + max(visual_map))`（L197–200）。**视觉patch参考库并不需要patch与文本对齐**。

原文主结果MVTec0-shot91.8/85.1、1-shot93.1/95.2，VisA0-shot78.1/79.6、1-shot83.8/96.4。K参考采样5 seeds报告均值标准差（L245、L389）。F1-max为测试分数扫最优阈值的上界，不能当已确定部署阈值；pAUROC95+同时像素F1-max<60，论文明确任务未解决（L597）。逻辑异常、极小缺陷、可接受变化、边界误差是原文列举失败类型（L707）。

### `WinClip/code` 的实际实现与论文差异

README L3明确 unofficial；meta.json:47 的字段虽然叫official_code，但 :48 说明“--code手动指定”。不是作者官方实现。README自报MVTec复现70.17/80.67，相比论文91.81/85.12；这只是该README声称，不是本轮跑得。

- `eval_WinCLIP.py:149` 默认scales=(2,3)，无image scale；CLIPAD默认的(2,3,15)不会覆盖传入值。`run_winclip.py` 默认仅VisA、zero-shot。
- `WinCLIP/model.py:32` 无视传入precision，强制fp16。`:44`–`:54` 先拉伸成方形再center crop。`datasets/dataset.py:45` OpenCV以BGR读图；参考图在 `eval_WinCLIP.py:31` 显式BGR→RGB，但测试图`:46`没有换色。**静态代码显示测试/参考颜色通道处理不一致**，不能凭其指标归因于某一种聚合设计。
- `model.py:115`–`:141` 默认V1是先平均未逐条L2的文本嵌入，再归一化类原型；第三方另一份先逐条归一化。
- **最重要的聚合订正**：`:163`两类softmax；`:172`给覆盖token填`1/p_normal`；`:179`和`:190`求覆盖窗口的倒数均值；`:194`跨尺度平均这些倒数均值；`:195`统一输出`1-1/x`。令a_w=p_anomaly，则代码输出是 `1 - 1 / mean_scales(mean_windows(1/(1-a_w)))`，即**1−正常概率的调和均值**。这既不是论文的“异常分的调和均值”，也不能概括为“窗口调和、跨尺度算术平均异常分”。笔记以各尺度最终异常图作算术均值的数例不对应此代码。
- 两个窗口异常概率0.9与0.1：论文H(0.9,0.1)=0.18；代码1−H(0.1,0.9)=0.82，差异方向相反。不能用“单调所以AUROC影响不大”带过，聚合多项后可改变跨样本排序。
- `model.py:143`–`:154` reference gallery只收 `encode_image` 返回的窗口CLS；`CLIPAD/transformer.py:751`–`:760`虽然形成tokens_list但默认只返回pooled_list，因此**没有论文WinCLIP+的225个penultimate patch参考分支**。即使手动加整图scale，也只是一个整图CLS，不会变出patch库。笔记声称此实现K×590个640维库不成立；默认是K×365个窗口640维向量。论文不同分支本来就不必同维度。
- `calculate_visual_anomaly_score :210` 使用当前gallery后，`:222`才在scale边界切换gallery；静态顺序显示下一尺度第一个窗口会先与上一尺度库比较，是需运行单测确认的边界实现问题。
- `forward :260`输出 `1/(1/M_text+1/M_vis)`，少普通两数调和均值的因子2；zero-shot时两图相同，因此输出恰好M_text/2，改变数值范围但此常数本身不改变排序。`:262`插值到400；`:268`高斯注释掉。
- `utils/metrics.py:20`直接map.max当图像score，未实现原文global语言score与视觉map.max的均值。
- `datasets/mvtec.py:44`、`datasets/dataset.py:15`只接受K∈{0,1,5,10}，3个seed index；与原文0/1/2/4、5seeds不同。不能直接用默认脚本复刻原文few-shot。

### `third_party/WinCLIP` 的版本差别

README同样明确unofficial，是mala-lab/InCTRL体系。它不等价于上面caoyunkang版，不能按同名文件夹互换。

- `main.py:79`本地指定LAION `e31`权重，caoyunkang版使用laion400m_e32；main没有调用model.eval，虽然前向在no_grad/autocast上下文（:181），需关注模型状态。
- `main.py:98`–`:108`每张图重新编码文本；逐条L2→平均→L2，正确但未缓存。
- 0-shot `main.py:115`–`:135`只以全图CLS语言概率做**分类**，无语言窗口分割路径；few-shot `:143`–`:144`才合并视觉map.max与global语言score。主函数最终只输出分类AUROC/AUPR/F1；数据loader的异常mask也为全零（`mvtec_dataset.py:200`–`:206`），不能把这份默认程序当零样本分割评测。
- 视觉参考分支 `open_clip/model.py:424`–`:502`确有2×2、3×3窗口640维和全图patch896维，三图均值，接近原文参考框架；窗口覆盖采用调和距离。视觉`transformer.py:645`–`:715`同时返回窗口CLS、原始patch、全图CLS。
- 严重的入口不一致：`main.py:174`默认shot=2，但 `open_clip/model.py:506`–`:510`固定解包query+4张参考图，shot1/2会索引不足；这是静态可判定的配置与函数契约冲突，未运行。
- `datasets/mvtec_dataset.py:185`每次用RandomState(10).choice选择参考，默认可放回，不是5seed协议。
- `:134`–`:168`为非方图做tiling，横向长图逻辑可见，但PIL `.size`是(width,height)，代码变量名反了且固定沿x方向crop；高纵图需额外验证。caoyunkang版直接拉伸。

### 现有WinCLIP笔记需纠正/降级的推断

- “窗口内调和、跨尺度算术”及跨尺度13倍数例与实现不符，以上已给确切公式。
- 尾表写“异常分数=1−p(anomalous)”有符号错误；异常分是p_anomalous=1−p_normal。
- “1/p_normal单调所以AUROC影响不大”不适用于多窗口非线性聚合。
- K×590统一640维参考库是把论文算法和本地社区实现混在一起。
- “任何形容词都无法描述装反”“没有任何域适应能力”“跨尺度算术是PRO偏低来源”超出证据。原文是视觉参考对逻辑异常有帮助；没有做因果消融不能把复现落差归因到某个实现差异。

## 3. AnomalyCLIP：用辅助像素监督学通用异常文字原型

### 原文与实际配方

对象无关template为 `[V_1...V_E] object` 与 `[W_1...W_E] damaged object`，E=12。two-class相似度可用于全图与patch；global CE，local Focal+normal Dice+anomaly Dice，`L=L_global+4ΣL_local`（原文L99–124）。**是使用正常和异常标签、像素mask的有监督辅助适配**，不是只靠正常图学出来。

文本额外8组4×768的deep prompt，替换文本第2–9层的前4个内容token，保留SOT（`AnomalyCLIP.py:238`–`:249`）。有效学习向量数 `2*12*768+8*4*768=43008`。

DPAM是内容自相似attention `softmax(VVᵀ/sqrt(d_head))V`，保留原始QK分支用于global。代码 `Attention :76`–`:94`复用原来的qkv与out投影权重；`DAPM_replace :344`–`:353` range(1,20)替换最上19层，也就是第6–24层。它不是全24层都双路。

更细的实现：`ResidualAttentionBlock :139`–`:149`以**原始分支x_ori**产生当层QKV，原始分支做attention残差+MLP，local分支只累加对应VV增量并跳过FFN。不是两条分别独立喂自身token、各自完整运行transformer的网络。局部投影有ln_post（`:383`），global返回却是 `x_ori[0] @ proj`（`:387`），没有同样ln_post；不能直接以原版CLIP的global encode公式推断完全相同。

骨干实际来源 `model_load.py:26`–`:27`是OpenAI `ViT-L-14-336px.pt`地址，尽管论文L178给OpenCLIP仓库脚注。518输入经位置编码插值，37×37=1369 patches；视觉1024→联合768，文本768。图像score是global异常概率；像素score是各选层异常概率上采样后相加、Gaussianσ4。双类联合softmax使 `(1-S_n)/2+S_a/2 = S_a`，笔记这一点订正正确（`test.py:102`–`:104`）。图像score**不使用像素图最大值**。

### 版本与配置的关键区别

当前原文v12（meta.updated2026-01-01）L85脚注明确实现只用top feature，L178、L407亦重复。README Updates L8记录2025-04-13修复“不同中间层local输出相同”的bug、更新图解/文本适配，作者称不影响性能。本地实际 `AnomalyCLIP_forward :293`已经 `.clone()` 保存层特征。

`train.py:128`及`test.py:184` parser默认[6,12,18,24]，但**发布train.sh:16/:35、test.sh:17/:36明确 --features_list 24**。因此读取默认帮助不能取代实验启动命令；文件夹名字_multiscale也不是配置证据。当前推荐配方top-only与v12论文一致。

训练实际 `Dataset(... mode='test')` 默认（dataset.py:41、:48）；`train.py:35`未改mode，故用辅助集test split。train.sh第一组VisA训练→文件夹无_visa→test.sh测MVTec；第二组MVTec训练→文件夹带_visa→test.sh测VisA。名称易误导，应按命令记录训练域。

训练图像视觉encode在no_grad（train.py:68–74，视觉forward本身亦no_grad）；optimizer只有prompt_learner（:44）；但代码仅model.eval而没显式requires_grad_(False)。**eval不冻结梯度**：CLIP文本参数仍可能计算/累积梯度，只是未被optimizer更新。说“权重不更新”准确，说“梯度只回到43K”过强。有效学习参数是43K，但 `prompt_ensemble.py:165`–`:166`另外注册8个768→896 Linear且forward不使用（:264只返回compound_prompts_text）；注册参数额外 `8*(768*896+896)=5,512,192`，总prompt模块注册约5,555,200参数。统计时需区分实际参与损失的参数、注册requires_grad参数、优化器参数与checkpoint体积。

当前局部使用顶层一次损失，故笔记按四层×λ推出局部占96%的示例既不是实测梯度比例，也不适用于脚本配方；各项数值占比还不能等价梯度贡献。

### 实验与评测

17集=7工业+10医学；MVTec训练测其它集，MVTec自身由VisA辅助训练。原文表MVTec91.5/91.1，VisA82.1/95.5；与WinCLIP相比MVTec图像级略低0.3但像素+6，VisA图像+4、像素+15.9。不能把这些增益全归于prompt模块：骨干、分辨率、训练监督与推理方式均不同；模块消融才是内部证据。医学分割数据集只有异常图、无正常图像（L437–443），故不能对它们计算正常/异常图像AUROC；像素AUROC仍可因为有正常背景像素而计算。

医学辅助ColonDB+Endo后，结肠等任务提升而ISIC、COVID退化（L251–253），说明“通用异常”受辅助数据分布影响，不能保证任意新域。固定object不是一种数学上保证去除所有对象信息的操作，只是避免显式目标类别文本的偏置。

`metrics.py:32`–`:42` image AP用average_precision_score；pixel AUROC先在每子集内flatten，`test.py:168`再对类别均值，不能与全类别混池AUROC等同。PRO `metrics.py:5`–`:29`扫200 thresholds，FPR<0.3后按采样到的min/max重新归一化再积分；要严格比较时需一致评测脚本。AUROC/API并不自动提供可部署阈值。

仓库已有日志（不是本轮运行）：MVTec`results/9_12_4_multiscale/zero_shot/log.txt:19`为pAUROC91.1/PRO81.4/iAUROC91.6/AP96.4；VisA`..._visa/.../log.txt:16`为95.5/86.7/82.0/85.3，与论文91.5/96.2和82.1/87.0略不同。差异必须如实记录，不应说完全复现原文。

### AnomalyCLIP精读笔记需纠正

- “自带2个checkpoint”实际是2个训练目录，各epoch1–15共30个文件；可说两套训练域checkpoint系列。
- “给图+类别名”作为必需输入不准确；类别名仅归档/metrics，prompt固定object（test.py:82、prompt_ensemble.py:88）。
- 多层[6,12,18,24]是Python默认能力，不是当前sh实际配方。正文多处四层流程图、显存次数及“真正让图好的是多层聚合”应更新为top-only默认及可选多层。
- “论文没写DPAM替换范围”错误，当前附录L407明确第6层开始。
- “视觉塔唯一改动只QK→VV”省略local跳过FFN、双路残差以及global返回与原版的差别。
- “43K可训练参数”须注明有效向量参数；忽略unused投影会使代码numel核对不上。
- “CLIP eval因此冻结”“梯度只回43K”应改为权重未进入optimizer，视觉no_grad，文本仍为prompt传梯度。
- “ISIC89.7不如VAND”直接与表矛盾：原文VAND89.4/77.2，AnomalyCLIP89.7/78.4，是后者略高。数据太少/不确定性未报告也不应扩大解读。
- “WinCLIP换数据集就得重来242prompt”夸张；WinCLIP模板state固定，换类别主要替换名词，专属缺陷词是可选。
- “object固定遇到逻辑异常必然失效”是待测假设，当前阅读没有证明此必然结论。

## 对当前项目可直接迁移的、有证据支持的点

1. 首先固定允许信息：完全无目标图、可用K张目标正常图、还是允许辅助域像素标签。WinCLIP与AnomalyCLIP不能在不说明这些信息权限的情况下当同等训练预算方法排名。
2. 文本原型可缓存，降低重复文本编码开销；两份WinCLIP实现已展示缓存/未缓存的明确差别。规范单条L2→组均值→L2，并记录温度/权重来源。
3. 小缺陷定位必须保留足够空间分辨率。窗口CLS与DPAM两者都是针对原始CLIP patch局部语义不足的不同处理：前者免任务训练、窗口开销大；后者单次高分辨率视觉前向但需要辅助像素监督。不能仅按“单次前向”推断跨骨干速度，需同硬件实测。
4. 正常参考最近邻可以补充相对/逻辑正常性，且视觉patch无需与文字对齐；实现时应严格分别存窗口640维与patch896维（WinCLIP骨干），不能强凑联合空间。
5. 本项目若沿用WinClip/code当baseline，至少应先审查RGB/BGR、默认丢全图尺度、正常概率反向调和、缺patch库、AC取max、few-shot采样、非方图处理等已证差异，再说明要复刻论文还是评估此代码版本。不能把README低分解释成论文方法本身效果。
6. AnomalyCLIP可作为辅助域有mask时的轻量prompt适配基线，但必须按训练脚本记录辅助域、top-only、检查点；token与DPAM变更必须和模型实现配套，不能只把prompt接到原版CLIP。
7. 评测至少按类别宏平均并同时看像素AUROC/PRO与阈值指标；对全异常医学集只报可定义的指标。已有日志和静态代码阅读仅支撑工程可读性，不是成功复现或可部署证据。

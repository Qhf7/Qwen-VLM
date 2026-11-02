# MoECLIP、FB-CLIP、AnomalyVFM：原文—精读报告—代码交叉阅读证据

阅读日期：2026-10-23。工作区：`/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm`。本笔记中的相对路径均以此为根。没有改原论文、原笔记或代码，没有联网下载，没有运行训练或模型推理。下文“实现”只代表本地代码的静态控制流；论文指标是作者报告值，不能当成本次复现结果。参考 cvpr-paper-playbook 的引用与复现原则，明确区分主文、补充材料、当前代码版本和数学分析。

## 已读材料与版本

- 三篇的 `meta.json`、`outline.md`、`精读笔记.md` 已完整读取；以拍平后的 `paper.tex` 读主文，公式和表格按其行号核对。
- MoECLIP：`paper.tex:6–396` 完整主文；补充算法 `429–479`、实现/文本/损失 `583–673`、计算开销与理论和专家分析 `938–1097`、局限 `1125–1127`。代码完整核心 `model/moe_adapter.py`、`model/adapter_modules.py`、`train.py`、`forward_utils.py` 的计算与评测部分，`test.py` 的模型装载/预测/评测控制流、`model/config.py` 和 `dataset/__init__.py` 相关分支、`scripts.sh`、README。
- FB-CLIP：`paper.tex:8–519` 完整主文；补充软掩码定义 `544–615`、局限/遮挡 `1133–1200`。代码 `FBCLIP_lib/FBCLIP.py` 的完整有效模型路径（视觉塔、FB 参数初始化、文本池化、SCR、软掩码、SEM/SPA、FB attention、BS、最终打分），`cross_domain_test.py` 训练/冻结/损失、`test_with_trained_model_pic.py` 装载与推理/指标、`dataset.py`、`prompt_ensemble.py` 核心 prompt 定义、运行 shell、README。
- AnomalyVFM：`paper.tex:6–373` 完整主文；补充 `400–578` 的局限、数据、消融和基线重训。代码 `train.py` 训练主循环、`decoder.py`、`models/model.py`/`models/radio.py`、`peft_local/peft_func.py`/`peft_wrapper.py`/`lora.py`、`generate_dataset.py`、`aux_dataset.py`、`image_gen_models/img_gen_model.py`/`flux.py`、`hf_model.py`、`test.py` 推理/聚合/指标、README。
- 来源：三份 git remote 均与元数据/论文或 README 的官方 repo 相符，但这只是本地来源关联，本次未联网核验托管状态。MoECLIP origin `https://github.com/CoCoRessa/MoECLIP`，本地 HEAD `f9b2574f3527f7b0e6d470cf31df32276f7302c8`（2026-10-14）；FB-CLIP origin `https://github.com/Xi-Mu-Yu/FB-CLIP`，HEAD `c097e7c39479e27651c0d94257053072f18d3549`（2026-07-03）；AnomalyVFM origin `https://github.com/MaticFuc/AnomalyVFM`，HEAD `4da96b493a0b12e4e380fa5bf6573c126342855c`（2026-04-21）。
- 论文版本：MoECLIP arXiv 2603.03101 v2；FB-CLIP 2603.19608 v1；AnomalyVFM 2601.20524 v2。MoECLIP/AnomalyVFM 元数据写 CVPR 2026；FB-CLIP 元数据 comment 为空，精读笔记写 CVPR 2026，不能仅靠这份元数据确认其录用状态。

## 1. MoECLIP

### 核心理解与信息权限

真正的设计是“在冻结 CLIP 的若干层后接入内容条件化的残差变换，并把改变后的 token 继续送入后续层”。它不只是把四个冻结层输出拿出来各自做独立后处理。`model/moe_adapter.py:414–430` 中每层先执行原始 transformer block，选中层后直接覆盖 `x`；后续 block 消费适配后的 `x`。这保留主干权重，却改变了后续内部激活。

原文 `paper.tex:79` 明确辅助已见类别有监督训练，目标未见类别测试；`281` 是 VisA 辅助训练，评 VisA 改用 MVTec。训练实际取图像、mask、图像级 label、class_name（`train.py:84–87`）。所以 zero-shot 是目标类别/目标数据集没有训练图，并不是全系统无训练、没有异常标签。文本需要测试类别名，经 `forward_utils.py:135–158` 的 REAL_NAMES 映射、正常/异常模板编码，属于可用元信息权限。

### 公式、维度与路由

- 输入 518×518、patch14，因此有 37×37=1369 个 patch；视觉 hidden 是 1024，文本/对齐空间是 768。原文超参 `paper.tex:287,585`；代码硬编码 `moe_adapter.py:283,309–324`。
- 四层位于 6/12/18/24，K=4、top-k=2、rank8、alpha16、dropout0.05；router softmax→top-k→重归一化见 `192–196`。
- 专家只学输出矩阵 B，FOFS 的 A 分割 1024 维为四个各256维坐标块，各取一个 8×256 的正交行投影，别的坐标置零。QR 构造在 `130–163`；A 冻结在 `93–95` 和训练开关 `train.py:264–281`。这种互不相交是“通道坐标子空间”的互不相交，不是已验证存在四个天然语义簇。
- 专家输出 `B(A(dropout(x))) * alpha/r`（`102–107`）；先匹配原输入 L2 范数，再 `0.1*moe_normalized + 0.9*x`（`426–430`）。因此保留方向变化的幅度约束。
- ETF 对每个 token 的所有 K 专家输出做 L2 norm，构造 K×K Gram，target 对角1、非对角−1/(K−1)，MSE（`9–36`）；K4 的理想值−1/3对应四面体角109.47°。这是优化目标，不代表实际专家输出已经达到理想角度。
- 原文式 `paper.tex:138` 和 `139` 把 Frobenius squared norm 与除 K² 的元素均值写成相等，严格数学上差一个 K² 因子；当前代码为元素平均 MSE，对应后者。固定 K 时可吸收进权重，但跨 K 消融要注意尺度。
- balance 用所有 token 的 router probability 求和，再 CV²（`199–208`），并非强制每个图/每类 top-1 选择平均；std 默认估计约定与论文“总体方差”也不是完全同一写法。K4时即使 loss 低，特定目标数据集某专家仍可很少用。论文 `1057` 正式讨论 Expert4 的数据集依赖利用率。

### 训练与推理的实际链条

1. `train.py:256–281` 先冻结全部模型，再解冻 `text_adapter`、image_adapter 中除 FOFS A 外的参数。`moe_adapter.py:467–486` 文本编码最后没有直接用 CLIP 原 `text_projection`，而用可训练 `SimpleProj(768,768)` 处理 EOT。这一步被精读笔记漏掉，也未在主文简单“预训练 Text Encoder 输出平均”的叙述中交代清楚。
2. 训练态专家对所有 token 计算全部 K 个输出，用 top-k gather 再加权（`210–219`），以便 ETF 看到所有专家。推理态才按 expert mask 仅计算被选 token（`168–179,221–234`）。论文附录伪代码 `paper.tex:460–465` 本身也明确训练计算所有 K 个专家，所以它不是“代码悄悄违背论文”的问题。
3. PAA 对每个选中层按 `[1,3,5]` 固定顺序生成三个邻域均值特征（`342–374`，unfold 使用零 padding），四层共12组。`_create_gaussian_kernels` 虽构造 Gaussian buffer，但 PAA 实际在 `361` 用普通 mean，没有用该 Gaussian kernel。
4. `440–457`：LN后去 CLS，各层三个尺度共用一组 `SimpleProj(1024,768)`，得到 `[B,1369,768]`。
5. 图像头 `459–460` 用 `tokens[-3]`。12组按每层 s1/s3/s5 排序，`-3` 是最后层 **s=1**，并不是“第三尺度”；经 LN+depthwiseConv1d3+GELU+pointwise1024→768（`adapter_modules.py:80–103`），逐 token L2 normalize 后 patch mean。原文 `paper.tex:182–186` 就是最后层 patch+depthwise+GAP，未声称用 CLS。
6. 训练分类用 `F.cross_entropy`（`train.py:102–104`），并非字面 BCE；分割为 *100 相似度→上采样→softmax（`forward_utils.py:193–213`），每组 focal+正常dice+异常dice（`220–224`）。总和加0.01 ETF和0.01balance（`train.py:106–113`）。
7. 当前默认 `train.py:176` 是 lr=5e-5，而论文 `paper.tex:287,585` 写5e-4；`scripts.sh` 没覆盖lr。因此直接运行 README 默认命令不会使用论文lr。batch2/20epochs/Adam betas(.5,.999)、iteration调度16000/32000见 `train.py:174–175,283–289`。
8. 推理并非“12张异常概率图平均”：`forward_utils.py:196–207` 把 *100 的正常/异常余弦 logits 组合成 `(a + 1 - n)/2`，按工业/医学使用不同 Gaussian kernel/sigma，再插值；`test.py:94–100` 对12组求和。没有 test softmax。
9. 图像最终评测也非直接图像头：`forward_utils.py:239–253` 必要时按当前类别预测范围做minmax；工业为 0.5*mapmax + 0.5*image_head；医学直接用 mapmax。`test.py:89–91` 的head原始分是异常相似度平移到[0,1]。复现实验必须复制这个域依赖聚合，否则即使权重相同也可能不同。

### 结果与证据边界

主表 `paper.tex:231–272`：MVTec I/P-AUROC 93.9/92.5，VisA 83.6/95.6。89.6的图像平均来自**9个有图像评测的数据集**，94.3的像素平均来自**13个有像素评测的数据集**，不能写成“14集每个都报两个指标”。总的确涉及14个独立数据集。VisA I-AUROC低于Bayes-PFL86.8；MVTec P-AP45.7低于Bayes-PFL48.4，体现AUROC不等于高精度分割。

四集消融 `333–352`：去PAA像素平均92.2→90.7，去FOFS和ETF92.2→91.2、图像95.3→93.1。本文实际证据支持“在这些配置下有增益”，不能仅凭两三个数据集结果归因为“类别多的集合一定更适合MoE”。`938–975` 作者报告MoECLIP158.9ms，AA-CLIP119.6ms；FLOPs接近但 wall time明显增加，动态路由开销不能忽略。没有本次运行验证这些时间。

### 精读报告应改的点

- `精读笔记.md:63,180,300` 的CLS图像头及“论文用CLS、实现绕过”错误；正文与实现都用patch，`-3`为s1。
- `120,176` 的1024→1024投影和文本[2,1024]错误，应1024→768、文本[2,768]。
- `10,122` 漏掉可训练text_adapter；CLIP文本塔原权重冻结不等于“文本无适配”。
- `137` 只抄论文lr，须注明当前默认代码5e-5的版本差异。
- `190` “每patch只算两个专家”只适用于推理；报告末表提到dense训练是对的，但应补论文伪代码也如此。
- `12/主要结果` 的“14集平均”需要说明两个分母；`局限` 还应保留原文 `1127` 的未探索高级文本适配、缺少文字解释，而不是仅自己推测超参成本。

### 可迁移内容（条件化建议）

优先能独立迁移的是可训练投影前的PAA、多层输出监督、残差范数匹配及明确的图像/map聚合，而不是默认堆完整MoE。若比赛允许辅助异常mask训练、且主干暴露patch hidden，可对照单adapter与MoE，再用同样数据/总参数检验FOFS和ETF是否真有用。专家语义需要用目标域routing/low-level统计验证，不能命名成固定“裂纹专家/背景专家”。该方法原生输出分数和热图；不会生成异常解释文字。

## 2. FB-CLIP

### 核心理解与权限

它同时修改文本表示和视觉token读出：文本以generic object状态prompt学习；视觉把原token按ID/SEM/SPA三条变换，各过共享FB Attention，再从所有层/视图构造一个背景原型，进行相减与余弦抑制。这里“multi-view”主要是特征变换视图，不等于输入多个相机图像。真正多相机遮挡是原文 `paper.tex:1183–1191` 的未来方向。

`paper.tex:413–415` 默认以MVTec test split作为有标注辅助训练，评MVTec时用VisA test；这与报告泛称“VisA训练”不同。`dataset.py:55,62` 默认读取meta的`test`分支；`cross_domain_test.py:247` 训练构造没有改mode，确实使用该辅助test split。使用已见数据的test子集训练在本cross-dataset协议中是明确设定，不能把它误说成对当前目标测试标签训练；也不能带入比赛未知测试集照做。

### 前向与维度

- 518²输入、ViT-L/14@336px位置插值，得到1370 tokens×1024；`FBCLIP.py:286–320` 保留CLS和选中层输出。shell使用层 `[1,6,12,18,24]`，不是只有四层（`train_on_mvtec.sh`/`train_on_visa.sh`）；每层三视图，共15组。
- `FBCLIP.py:1001–1008`：先 `enhanced_text_encoding`→`encode_text_learn`，再 `encode_all_image`。两个类别prompt文本hidden768，输出768。
- MSTFF有效计算 `488–504`：EOT@预训练projection、所有77 token mean@projection、token-selector attention pool→两层MLP；最后固定 `global + .5*attn + .5*EOT`。原文 `paper.tex:111` 正是1/.5/.5。
- `FB_params:436` 虽有`fusion_weights` Parameter，`get_trainable_parameters:583` 虽把它加入optimizer，但最终forward `503–504` 完全没有读取。**声明参数、加入optimizer不等于参与计算/收到梯度。** 同理 `gate_network`（441–446）、`fusion_weight`（449）、`layer_selector`（456）在这条有效forward里未使用。
- 有效learnable FG gate 是 `FB_Attention.gate`（45–49，1024→512→1sigmoid），不等于上面unused768×3 gate_network。`76–102`：按gate拆FG/BG，分别独立QKV attention，**fg_out+bg_out** 后LN；`bg_weight=.1`仅保存成员，forward未引用。

### 软掩码与三视图

`FBCLIP.py:758–822` 的软掩码由四个手工指标算出，不是learned gate sigmoid阈值化：

1. `826–850`：token序列上的1D avgpool3局部差；这不是严格2D四/八邻域。
2. `766–778`：每patch与**特征均值中心**的L2距离，不是与图像坐标中心距离。
3. `786–787`：1−cos(patch,CLS)。
4. `780–784,916–926`：当前选出层与前一个选出层特征的L2差；不是跨时间帧/增强/相机视角。通常是第6层对第1层、第12层对第6层，并不一定相邻block。

四指标逐样本minmax，权重 .3/.3/.3/.1，1D平滑5，再阈值.5取{.5,1}。原文 `paper.tex:133–141` 已明确给阈值，附录 `553–600` 已明确feature-center/层变化。由于torch.where阈值是离散选择，所谓“soft”只是保留非零的0.5权重，不是端到端可微的连续前景概率。

SEM有效代码 `711–753` 构造fg/bg mask外积，与CLS不一致性/一致性相乘后softmax，做全patch汇聚，0.6增强+0.4原特征。SPA `665–708` 在2D网格unfold5×5，对mask加权的fg/bg邻域，按稳定度/丰富度softmax汇聚，再除mask权重和。三路后均过共享FB_Attention（`916–938`）。前景高不等于异常高，这里只是启发式中间表征，仍需辅助异常标签监督。

### BS、SCR、最终分数

- 独立BS在 `encode_all_image:949–994`，不是dual attention branch：**各特征组取序列前一半token**（955，包含CLS和靠前空间位置，而不是按mask排序出的背景），拼bank；`597–618` 做0.5mean+0.5channelwise max得到背景原型。每token减原型，乘(1−cos)，最终0.5原特征+0.5残差，再通过冻结visual.proj1024→768。这是需要重点检查的结构先验；“前一半一定背景”不是由模型保证的事实。
- SCR `517–551` 对最后一组（最后层SPA后BS，含CLS）token mean和text norm，除temperature.07，entropy+0.5ReLU(1−|sA−sN|)，乘.15。原文 `paper.tex:337,346` 明确margin1、we1、wm.5、lambda.15；不是“论文没给”。SCR是训练正则，不在推理时修改scoremap；代码测试仍计算它但丢弃返回loss。
- 熵与绝对margin都对类别交换对称，不保证类别方向正确，也不能单独阻止全预测normal的类塌缩；类别方向由有标签的分类/分割loss决定。报告“margin能修复错的自信”无该数学保证。
- `FB_encode:1021–1045` 每组做cosine/0.07，按每层softmax(scale_weights)[i//3]求和，最终softmax，CLS异常概率做图像分数，patch异常概率reshape37²。scale_weights长度6（454），当前shell五层只用前五项，但六项共同参与softmax分母，需锁定checkpoint配置。
- `1030–1038` 将同一重建误差重复到两类并加到两个logit：`scores + .5*[e,e]`。按softmax的平移不变性，**这项在精确数学上对两类概率完全抵消**，可能只剩浮点误差。它不等于整个BS无效，BS在token域的相减确实影响score；只是不应声称额外error logit提高检测。此结论是静态代数分析，未运行数值试验。
- 测试 `test_with_trained_model_pic.py:323–362` 每batch重新编码固定prompt（可以缓存但当前没有），最终maps插值到518²并可Gaussian sigma4，图像分数直接取CLS概率（340–342），类别宏平均统计（382–386）。

### 训练与结果边界

`cross_domain_test.py:243–244` 冻结**整个**原始CLIP，不是仅冻结视觉塔/部分训练原文本层；随后新建prompt、FB模块、selector/MLP及scale参数（252–269）。有效可训练的是prompt ctx、FB_Attention、文本MLP/selector和scale。deep_compound_prompts参数虽通过接口，但此`encode_text_learn`并未使用它改变transformer；不能仅凭parameter存在断言深层prompt有效。

训练有效总loss `324–331` 是图像focal + maskfocal + 0.2dice + SCR，不是报告写的单独分类BCE；focal函数 `61–75` 内部是BCE乘(1−pt)^4。shell设3epochs、bs4、lr5e-5，parser默认lr.001被shell覆盖；`train_steps_per_epoch`截断代码在 `284–285` 已注释，不能把shell的20理解成每epoch仅20步。

原文表 `paper.tex:368–385`：MVTec I-AUROC92.4/P91.9，VisA89.5/96.3；Real-IAD I80.6优于AF-CLIP79.2 **1.4**，优于FAPrompt77.3 **3.3**。报告称“+3.3”时需要交代比较对象，不是同表最强对手增益。Real-IAD P-AUPRO88.2 vs82.1为+6.1，不能与MoECLIP的P-AP直接比。工业7集+医学9集共16集，有的医学集仅一种任务。

消融 `449–479` 表明MVFBE是大头：单独加MVFBE已92.2/91.7、VisA88.4/96.0。BS单独使MVTec图像67.1→44.8；MSTFF+MVFBE+BS的MVTec图像92.8高于完整92.4；不能讲成每个组件总是单调改善。原文局限 `1169–1180` 有PCB颜色/材质过响应、细针脚漏检、密集相似物体异常淹没、边界发散；作者测215ms/3.3GB vsAF-CLIP112ms/2.5GB，无实时部署验证。

### 精读报告应改的点

- `10,72,114,238`：文本融合固定权重，原文本塔全部冻结；报告把声明的unused参数当成有效模块。
- `85,87,180–186`：center是feature center、temporal是跨层差；据此构造的“离图像中心远所以压低”的玩具示例不对应真实计算。
- `94,239–240`：软mask不是gate_network生成；bg_weight.1没参与forward；双attention是加法、BS另有原型相减。
- `123`：具体训练loss应含focal分类与0.2dice；`134` 的lr大就崩只是未验证推测。
- `161`：SCR不是推理图变换；`169` “每图3次注意力”漏掉层数与FG/BG双分支，当前五层×三视图×两attention。
- `241`：margin/weights/threshold论文均已给。
- `基本情况`“开源权重未确认”可以更新为README提供Google Drive预训练权重链接，但本次未下载、不确认权重实际完整可用。

### 可迁移内容

可独立借鉴固定通用object prompt、多种token池化与视觉前景背景差异建模；建议先做少量可归因消融。完整实现含15组双attention和启发式前半序列background bank，成本与先验都需实测。若比赛存在多相机输入，本法ID/SEM/SPA并不是多相机融合的现成解法。训练正则可按数据协议使用，不能当作无需监督保证正确性的校准器。

## 3. AnomalyVFM

### 核心命题、权限与数据链

它证明在作者协议内，“更多样的合成异常监督 + 深入backbone的参数高效适配”可以让**没有推理文本分支**的VFM达到强零样本AD。不是声称整个开发流程不用语言：对象/背景/异常词表由GPT-4o产生，FLUX也是文本条件生成器（原文 `paper.tex:80,91`）。目标图没有正常参考库，不用类别名prompt做检测，但模型训练仍是有图像和像素标签的监督学习，标签来自合成。

三阶段：正常对象/背景图（100 objects×50 backgrounds词表）→IS-Net前景内采样矩形、RePaint受控inpainting→正常/异常图的DINOv2余弦差，阈值生成mask并拒绝无明显差异样本（`paper.tex:72–96`）。原文默认1024²生成、矩形宽高[50,350]、T=.3（246）；10000图、204异常词、4596对象背景组合、平均异常面积2.52%（465–472）；过滤拒绝率约30%（518）。作者 `453` 声称人工排除与评价类重合的object tags，本次没有独立证明生成模型预训练无目标数据，不能扩大成严格无任何预训练泄漏。

原文 `243` 明确 **AnomalyVFM solely generated data**。其中“评MVTec用VisA、评别集用MVTec”是**其他竞争方法和控制实验的训练**，不是主AnomalyVFM添加真实辅助集。现有报告把两者混了。

### 论文架构/损失与可核算维度

原文 `113`：每个transformer block attention的Q/V/out插rank64 LoRA；最后patch reshape进两层conv+GroupNorm+ReLU+bilinear upsample decoder；CLS线性分类。RADIOv2.5 ViT-L patch16、768²（248）→48²patch=2304，hidden1024；不是518/16→32²。RADIO本地封装 `models/radio.py:21–22` 正是dim1024/patch16。RADIO有3个summary CLS，当前训练 `train.py:152–153` 用3072维分类输入。

论文优化器/训练不是未给：`paper.tex:248` 明确AdamW，lr1e-4、500iterations、batch32；`346` 报训练约2h A100、总345.8M/可训练35.4M；`363` 合成约1天A100。

论文loss：image focal；pixel `b=L1+5*focal`，`C=1+exp(c)>1`，`Lpixel=C*b−.1*log(C)`（115–125）。对固定b>0，导数`∂L/∂C=b−.1/C`，内点最优`C=.1/b`（还受C≥1约束）。因此`−.1 logC`本身是**奖励较高confidence/防止confidence塌缩**，而正的`C*b`惩罚在高误差处增大C；不是报告所写“负log惩罚过度自信”。因为C至少1，所谓downweight是相对更容易像素的增权而言，不是把困难像素权重降到基础loss以下。

### 当前代码是已公开声明的后续配置，不能等同论文

README `103–104` 直说当前超参和论文不同，追求新数据/扩展的稳定配置；`137` 改FLUX.1-krea与DINOv3。

1. `train.py:51,71` 默认 **DoRA** rank64、**200**steps，不是论文LoRA500。冻结原权重后再注入PEFT（139–142），Q/V位置LoRA实现可查 `peft_local/lora.py:6–46`，一般out/proj `49–75`；`peft_func.py:6–29,83–89` 按模块名qkv/proj递归替换，CLIP另走q_proj/v_proj/out_proj（32–45）。说明只看LoRA文件不能推断默认训练在跑LoRA。
2. `train.py:148–150` 调用 `SimpleDecoder(feat_dim,1,1)`，因此当前只有**一个**上采样block；类默认2不代表调用用了2。`decoder.py:30–35` `nn.Upsample(scale_factor=2)`默认nearest，且先upsample后两层Conv/GN/ReLU；名字“DecoderBlockBilinear”不等于有效mode=bilinear。这与原文two blocks/bilinear不一致。
3. 当前training decoder输出96² mask与confidence；`train.py:185–188` maskGT也resize到96²。训练default有效batch32，用4次microbatch8累积（32–33,193–211），AdamW1e-4/weightdecay.01（35–43）。训练数据来自synthetic目录（62,191），无真实VisA分支。
4. 当前L1前做`sigmoid().clip(.1,.9)`（225–229），作者注释稳定训练变化会在followup解释；confidence公式仍同paper（231–238）。这是训练版本差异，不是证明原论文不完整到无法运行。
5. 对当前**binary**GT（`aux_dataset.py:29–37`显式阈值化），上述clip使每像素L1至少.1、focal非负，所以`b≥.1`。结合alpha=.1，固定mask时`b−.1/C≥0`（C>1），confidence分支的像素局部目标总倾向于C→1。共享decoder/有限步训练会影响实际状态，但静态数学说明这套clip与置信度原设计有非平凡交互；不能只说“完全一致”。此处未运行验证confidence最终分布。
6. `generate_dataset.py:44–50` 默认DINOv3、T=.25（注释又说.3）；矩形 `84–96` 用randint50到300（上界不含），不是paper350；`image_gen_models/img_gen_model.py:27` 选flux-dev-krea。`generate_mask:113–141` 将两个feature map先上采样到图大小再算cosine difference，并**乘采样box**，有任意阳性pixel才接受；这比“全图max差”多了位置约束。
7. 当前测试 `test.py:197–214` summary→predictor.sigmoid作image score，decoder mask.sigmoid、插值、可AvgPool5 smoothing；confidence测试时被丢弃，不拿confidence乘最终预测。`hf_model.py:50–65` 输出不含评测的AvgPool，只返回原decoder概率。要比指标，应区分Hub接口输出与test评测后处理。
8. `test.py:391–392` 同样一层decoder，`314–350` 每类评测后类别平均、再数据集平均；F1-max `253–259` 使用test labels扫阈值选最大，只是评测上界，不是部署阈值；AUPRO-0.3由adeval accumulator（37,232–245）。

### 实验怎样支持命题

- 主表9工业集平均I-AUROC94.1/P96.9，Bayes-PFL90.8/96.0（`paper.tex:188,201`）。这不是所有18工业/医学数据集一起的平均。医疗另报3个图像/6个像素任务，平均94.0/90.0（224,234）。
- 同backbone四条件 `143–158`：RADIO简单头+真实辅助89.1/84.9；只换合成92.1/95.9；只换适配93.0/95.2；两者94.1/96.9。这个因子实验支持数据与适配共同重要。对DINOv2像素增益，合成数据作用尤其大。
- 去过滤I−3.8/P−14.6，去前景I−1.4/P−5.8，去confidenceI−.6/P−2.0（318–321）。这些作者消融是优先关注数据质量的直接证据，但不能推断任意合成图越多越好。
- 合成数据重训CLIP类方法 `426–439` 有升有降，AdaCLIP还下降，说明不能从“AnomalyVFM受益”推广为“给所有模型喂这个数据都会受益”。
- MVTec94.9/92.7、VisA93.6/96.2来自原文主表。精读笔记用MoECLIP/FB-CLIP的独立论文数值作“此前最强”对照，跨训练/数据/预处理协议，不构成同表公平对比，且AnomalyVFM主表没有这两行。
- KSDD像素AUROC99.0而F1-max10.1（196），平均P-F1也不是SOTA（201），明确说明良好ranking不足以保证细轮廓和precision。医学不是全赢；图像平均94.0低于FAPrompt95.0。
- 原文速度只比较Bayes-PFL208.5ms/AdaCLIP82.4ms/AnomalyVFM20.5ms（355，A100统一测），不是“比所有CLIP方法都轻”。CLIP文本通常也能缓存，不能从无文本分支推出总速度排序。

### 精读报告应改的点

- `38` 4亿与几千的比值不是百万分之一，避免夸张数量级；`213` 合成10000对真实约20000并非规模高一量级，关键是类别/组合多样性，不是样本数十倍。
- `62–63` 对K“只影响归一化”的解释不成立：K直接参与QK点积注意力分布；原文 `562` 只说基于开源实践并做位置消融，不能虚构机制证明。
- `68,159` 改768→48²；`110` 改明确AdamW1e-4/500steps/bs32；`111` 去掉主模型额外VisA错误。
- `98,101` 改负log项符号解释；`233` 应写“可选LoRA实现位置对应论文，当前默认DoRA且decoder/生成配置已变”。
- `171` 改为论文对所列两个竞争者的测量结论，当前代码存在AvgPool后处理。
- `219`“尚无第三方复现”本地材料无法证明，应该写“本次未查到/未核验第三方复现”。
- “完全抛弃文本”的范围限于异常检测训练/推理网络的文本分支；数据生成明显使用语言和文本条件模型。

### 可迁移内容

对本项目最有证据支撑的是：合成异常后必须验证生成成功并产生mask；前景内采样比任意位置合成可靠；把数据质量控制和适配机制分开消融。若已有真实比赛数据与规则，不必默认从零跑整条10000图生成管线，且对象/缺陷名权限要按比赛约束核对。可把AnomalyVFM当视觉评分/热图模块；与AnomalyGPT的回答层对接需要额外接口与监督，本文不自带文字解释能力。

## 三篇一起看

| 维度 | MoECLIP | FB-CLIP | AnomalyVFM |
|---|---|---|---|
| 主要改变 | patch条件化LoRA专家与多尺度 | 文本池化+三视图FG/BG+原型减背景 | 合成数据+内部PEFT+卷积decoder |
| 训练标签 | 真实辅助图/异常mask/label | 真实辅助图/异常mask/label | 合成图/自动mask/label |
| 目标类别名 | 文本模板用class名 | generic object，不需class特定prompt | 推理无text prompt |
| 检测输出 | 分数、热图 | 分数、热图 | 分数、热图 |
| 原生自然语言解释 | 无 | 无 | 无 |
| 当前最大复核风险 | lr/文本adapter/域依赖score聚合 | unused参数误读/多模块调用链/原型先验 | 当前代码明确使用后续配置 |

不能按三篇纸面平均AUROC直接排一个总榜：MoECLIP14数据集且9/13任务平均，FB-CLIP16集且AP/PRO，AnomalyVFM9工业+9医学且F1-max；BrainMRI等还可能来自不同版本/划分（MoECLIP BMAD与另外两篇的分类BrainMRI不可只凭名字等同）。本次完成了静态阅读和交叉定位，尚没有依赖、权重、真实数据、统一协议运行证据。

# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ---------- 基础样式 ----------
st = doc.styles['Normal']
st.font.name = 'Calibri'
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.15

for i, sz in [(1, 16), (2, 13), (3, 11.5)]:
    h = doc.styles[f'Heading {i}']
    h.font.name = 'Calibri'
    h.font.size = Pt(sz)
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)

sec = doc.sections[0]
sec.left_margin = Cm(2.2); sec.right_margin = Cm(2.2)
sec.top_margin = Cm(2.0); sec.bottom_margin = Cm(2.0)

def para(text, bold=False, size=10.5, align=None, space_after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    if align: p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p

def bullet(text, style='List Bullet'):
    p = doc.add_paragraph(text, style=style)
    p.paragraph_format.space_after = Pt(2)
    for r in p.runs:
        r.font.size = Pt(10.5)
    return p

def kv(label, text):
    p = doc.add_paragraph()
    r1 = p.add_run(label + '：'); r1.bold = True; r1.font.size = Pt(10.5)
    r2 = p.add_run(text); r2.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(2)
    return p

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def table(headers, rows, widths=None, font=8.5, header_fill='1F3A5F', first_col_bold=True):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        r = p.add_run(h); r.bold = True; r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(hdr[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v)); r.font.size = Pt(font)
            if i == 0 and first_col_bold:
                r.bold = True
    if widths:
        total = sum(widths)
        target = 16.4
        scale = target / total if total > target else 1.0
        widths = [w * scale for w in widths]
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

# ---------- 封面 ----------
for _ in range(4):
    doc.add_paragraph()
para('零样本异常检测十篇精读报告', bold=True, size=26, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
para('从 CLIP 到 2026 年的四条技术路线', size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
para('CLIP · WinCLIP · AdaCLIP · AnomalyCLIP · AA-CLIP · AnomalyGPT · Dual Anchors · MoECLIP · FB-CLIP · AnomalyVFM',
     size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=60)
para('2026-10-22', size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para('材料：各论文 PDF、LaTeX 源码、图表与精读笔记，随报告一同保留在工作区', size=9,
     align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ---------- 执行摘要 ----------
doc.add_heading('执行摘要', level=1)
para('本报告精读十篇与零样本异常检测（ZSAD）相关的论文，覆盖从 CLIP 地基到 2026 年最新工作的完整脉络。')
para('一、任务定义。工业质检要求「冷启动」：新产品上线时没有正常图、更没有缺陷标注，却要判出缺陷并定位。'
     '传统 one-class 方法需要目标类的成百上千张正常图，无法应对；零样本异常检测用辅助数据训练、在未见类别上直接推理。')
para('二、四条技术路线。', bold=True)
bullet('文本锚点路线（WinCLIP → AdaCLIP → AnomalyCLIP → AA-CLIP）：保持 CLIP 双塔，改的是「怎么用文本描述异常」。')
bullet('图像锚点路线（Dual Anchors）：用图像自身分组产生锚点，反过来注入文本 prompt。')
bullet('纯视觉基础模型路线（AnomalyVFM）：彻底不用 CLIP 文本，改用 RADIO / DINOv2 这类 VFM 加合成数据。')
bullet('LVLM 路线（AnomalyGPT）：把检测器接进大模型，输出「判断 + 位置 + 对话」，绕开阈值问题。')
para('三、精度演进。MVTec 像素级 AUROC 从 WinCLIP 的 85.1 涨到 2026 年的 92.7；图像级从 91.8 涨到 94.9。'
     '但同一方法在不同论文里的数字差异可达 2~3 个点（如 AdaCLIP 在自家表中 89.2、在 Dual Anchors 表中 92.0），'
     '说明评测协议仍未统一，跨表比较必须谨慎。')
para('四、一个反直觉的结论。', bold=True)
para('精度最高的几篇，恰好是复现最差的几篇：Dual Anchors（92.7/92.4，无代码/无权重/无超参）、'
     'AA-CLIP（91.9，有代码无权重）、MoECLIP / FB-CLIP / AnomalyVFM（有代码但结果尚未被第三方复现）。'
     '而长期作为标准 baseline 的 AnomalyCLIP（91.1），恰恰是唯一开箱即跑的那一篇。')
para('五、选型结论。没有训练样本选 WinCLIP；有 1~4 张正常图选 WinCLIP+；有辅助标注且要开箱即用选 AnomalyCLIP；'
     '辅助数据极少选 AA-CLIP；不想设阈值、要对话解释选 AnomalyGPT；追求最前沿数字且有工程能力，参考 2026 三篇。')
doc.add_page_break()

# ---------- 一、全景 ----------
doc.add_heading('一、全景：四条路线与时间线', level=1)
para('ZSAD 的全部方法都可以看成对同一个问题的不同回答：如何把「异常」这个概念注入一个预训练模型？')
table(
    ['路线', '代表工作', '注入位置', '输出'],
    [
        ['文本锚点', 'WinCLIP → AdaCLIP → AnomalyCLIP → AA-CLIP', '文本 prompt / 文本塔内部', '分数 / 概率 / margin'],
        ['图像锚点', 'Dual Anchors (2026)', '图像分组 token → 反向注入 prompt', '概率图'],
        ['纯视觉', 'AnomalyVFM (2026)', 'VFM 特征 + 合成数据适配', '分数'],
        ['LVLM', 'AnomalyGPT (2024)', '检测器结果 → prompt token → LLM', '判断 + 对话'],
    ], widths=[2.2, 6.6, 4.6, 3.2])
para('时间线上有一条清晰的「改动量」变化曲线：')
paste = [
    ['2021 CLIP', '对比预训练，什么都不改', '—'],
    ['2023 WinCLIP', '只改输入 prompt', '0 参数'],
    ['2024 AdaCLIP', '改 prompt + 投影', '10.7M'],
    ['2024 AnomalyCLIP', '只改文本塔 + 换注意力', '43K'],
    ['2025 AA-CLIP', '两阶段改两个塔的浅层', '12.6M'],
    ['2026 Dual Anchors', '换主干 + 图像锚点', '未给'],
    ['2026 MoECLIP', 'patch 级路由到多个 LoRA 专家', '未给'],
    ['2026 FB-CLIP', '前景背景解耦 + 文本多表示', '未给'],
    ['2026 AnomalyVFM', '换掉 CLIP，纯 VFM + 合成数据', '未给'],
]
table(['版本', '核心改动', '可训练参数'], paste, widths=[3.4, 9.4, 3.4])
doc.add_page_break()

# ---------- 二、十篇总表 ----------
doc.add_heading('二、十篇总表', level=1)
rows = [
    ['CLIP', "ICML'21", '地基', '4 亿图文对对比预训练', '4 亿对', '—', '—', '✅'],
    ['WinCLIP', "CVPR'23", '文本锚点', '手写状态词 prompt + 滑窗 CLS', '0', '91.8', '85.1', '❌'],
    ['AdaCLIP', "ECCV'24", '文本锚点', 'static+dynamic prompt + 投影 + HSF', '2337 张', '89.2', '88.7', '✅'],
    ['AnomalyCLIP', "ICLR'24", '文本锚点', 'object-agnostic prompt + V-V 注意力', '1725 张', '91.5', '91.1', '✅'],
    ['AA-CLIP', "CVPR'25", '文本锚点', '两阶段：文本解耦 → patch 对齐', '30 张起', '90.5', '91.9', '✅'],
    ['AnomalyGPT', "AAAI'24", 'LVLM', 'ImageBind + Vicuna，检测器出定位图', '正常+模拟', '94.1*', '95.3*', '✅'],
    ['Dual Anchors', "CVPR'26", '图像锚点', 'DINOv3 分组锚点 → 注入 prompt', '未给', '92.7', '92.4', '❌'],
    ['MoECLIP', "CVPR'26", '文本锚点', 'patch 级 MoE + 正交专家分离', '未给', '93.9', '92.5', '✅'],
    ['FB-CLIP', "CVPR'26", '文本锚点', '前景背景解耦 + 多策略文本', '未给', '92.4', '91.9', '✅'],
    ['AnomalyVFM', "CVPR'26", '纯视觉', 'VFM + 三阶段合成数据 + LoRA', '未给', '94.9', '92.7', '✅'],
]
table(['方法', '年份', '路线', '核心思想', '训练数据', 'MVTec 图像', 'MVTec 像素', '代码'],
      rows, widths=[2.4, 1.7, 1.9, 5.4, 2.0, 1.9, 1.9, 1.1], font=8)
para('* AnomalyGPT 为跨域 1-shot 结果（训 VisA / 测 MVTec）；其论文首页的 97.4/93.1 为同域结果，不可与其余并列。', size=8.5)
para('注：不同论文对同一方法的复跑结果差异可达 2~3 个点（如 AdaCLIP 在自家表中 MVTec 图像 89.2，'
     '在 Dual Anchors 表中为 92.0），表中数字取自各方法原始论文。', size=8.5)
doc.add_page_break()

# ---------- 三、逐篇精读 ----------
doc.add_heading('三、逐篇精读', level=1)

def paper(name, meta, oneline, core, nums, limits, code, extra=None):
    doc.add_heading(name, level=2)
    kv('出处', meta)
    kv('一句话', oneline)
    para('核心做法', bold=True, space_after=2)
    for b in core:
        bullet(b)
    para('关键数字', bold=True, space_after=2)
    for b in nums:
        bullet(b)
    para('局限', bold=True, space_after=2)
    for b in limits:
        bullet(b)
    kv('代码 / 权重', code)
    if extra:
        para(extra, size=9.5)

paper('1. CLIP（ICML 2021, OpenAI）', 'arXiv 2103.00020 · 对比语言图像预训练',
      '用 4 亿对（图，文）做对比预训练，让任何分类任务都能用一句文本当分类器。',
      ['双塔结构：图像塔（ResNet 或 ViT）与文本塔（63M Transformer）各自出一个向量，L2 归一化后点积。',
       '目标函数：一个 batch 内 N×N 的相似度矩阵，用对称交叉熵（行 + 列）抬高对角线。',
       'batch = 32768，训练 32 epoch，温度 τ 可学习（初始化 0.07，限制 logits 放大不超过 100 倍）。',
       '零样本推理：把 K 个类名套进模板「a photo of a {}」编码成 K 个向量，即为分类器权重。'],
      ['ImageNet 零样本 76.2%，追平原版 ResNet-50 的全监督结果（76.1%）。',
       '27 个数据集上零样本赢过 ResNet-50 线性探针 16 个；Stanford Cars / Food101 高出 20%。',
       '零样本 ≈ 别人的 16-shot；在分布偏移下把鲁棒性差距缩小最多 75%。',
       'prompt engineering 带来 +1.3%，80 个模板集成再 +3.5%（合计约 +5%）。'],
      ['MNIST 只有 88%（抽象符号描述不了）；低分辨率数据集略输 EfficientNet；计数与空间关系弱。',
       'WIT 数据集未公开，因此「CLIP」实际指一类模型，业界多用 OpenCLIP 在 LAION 上的复现。'],
      '代码 ✅ / 权重 ✅（数据 ❌）',
      '对后续九篇最重要的两条限制：① 只有 CLS 与 EOS 两个位置被语言监督过，patch token 从未对齐；'
      '② 对齐是在自然图文上学的，不是工业缺陷上学的。这两条直接导致了所有下游方法的动机。')

paper('2. WinCLIP（CVPR 2023, AWS AI Labs）', 'arXiv 2303.14814 · 零样本异常分类与分割的开山之作',
      '冻结 CLIP，靠「状态词 × 模板」的组合式 prompt 与滑窗取 CLS，零训练完成分类与分割。',
      ['组合式 prompt 集成（CPE）：7 个正常状态词 + 4 个异常状态词 × 22 个模板 = 242 条，平均成 2 个类原型。',
       '滑窗特征：对每个 2×2 / 3×3 / 整图窗口，丢弃窗口外 patch，只把窗口内 token 送入 ViT，取 CLS 向量。',
       '多尺度调和平均：重叠窗口与多尺度都用调和平均聚合，偏向「正常」一侧以收紧缺陷边界。',
       'WinCLIP+：用 1~4 张正常图建参考特征库，与语言分支按调和平均融合，图像级取两路平均。'],
      ['MVTec 图像 91.8 / 像素 85.1；VisA 78.1 / 79.6。',
       '0-shot 已超过 PatchCore 1-shot（83.4）；4-shot 95.2/96.2 与 full-shot 的 CutPaste 打平。',
       '消融显示状态词层的贡献最大（74.0 → 89.8），模板层仅 +1.0。'],
      ['像素 max-F1 仅 31.6（VisA 14.8），分数不可直接卡阈值。',
       '240² 输入会吃掉小缺陷；三个尺度写死；prompt 是领域工程，换数据集要重调。'],
      '无官方代码；社区复现 caoyunkang/WinClip 自报 70.17/80.67，与论文差约 20 点')

paper('3. AdaCLIP（ECCV 2024, 华科 + 米兰理工）', 'arXiv 2407.15795 · 混合可学习 prompt',
      '用辅助数据训练 static + dynamic prompt 与投影层，把「异常知识」注入冻结的 CLIP。',
      ['prompt 插入图像塔与文本塔的前 4 层（每层 5 个 token），主干全冻。',
       '动态 prompt 由当前图的 CLS 经线性层生成，逐图不同；与静态 prompt 相加后注入。',
       '投影层把 4 层 patch 特征（1024 维）投到文本空间（768 维），这是该方法贡献最大的一块。',
       'HSF 模块：对 patch 嵌入做 KMeans，聚成图像级语义嵌入，替代「取最大分数」。'],
      ['7 个工业数据集平均：图像 90.2 / 像素 94.2；医学 7 集平均 94.6 / 85.3。',
       'MVTec 89.2 / 88.7；VisA 85.8 / 95.5；1-shot 比 PatchCore 高 9.7 个点。',
       '消融：可学习投影比冻结投影高 7.3（图像）/ 4.1（像素），贡献大于 prompting 本身。'],
      ['推理要跑两遍图像塔（动态 prompt 依赖当前图）；FP16 训练不稳定，需多次重跑。',
       'HSF 代码实现与论文不一致（top-k 后聚类并对所有簇心取平均）；默认辅助集在 README 中被更正。'],
      '代码 ✅ / 权重 ✅（HuggingFace Space）')

paper('4. AnomalyCLIP（ICLR 2024, 浙大 + SMU + 哈佛）', 'arXiv 2310.18961 · 物体无关的 prompt 学习',
      '把物体名从 prompt 里去掉，只学「正常/异常」本身的语义，用 43K 参数完成适配。',
      ['两条物体无关 prompt：[V1…V12][object] 与 [W1…W12][damaged][object]，共 24 个可学习词向量。',
       '文本塔第 2~9 层各注入 4 个可学习 prefix token，逐层重新提醒「注意异常」。',
       'DPAM：把视觉塔的 Q-K 注意力换成 V-V，使注意力图更「对角」，抑制全局语义对局部细节的污染。',
       'glocal 损失：全局交叉熵（分类）+ 局部 focal/dice（分割），λ = 4 让像素信号占九成。'],
      ['MVTec 图像 91.5 / 像素 91.1；VisA 82.1 / 95.5；覆盖 17 个数据集。',
       '医学域全面领先：HeadCT 93.4、Br35H 94.6。',
       '消融：物体无关 prompt 贡献最大（像素 +21.1、图像 +24.5）；DPAM 让像素从 46.8 到 68.4。'],
      ['需要辅助集的像素标注；没有 few-shot 模式；DPAM 改动模型结构，无法「原版 CLIP + 换 prompt」。',
       '推理图公式 (I−S_n)/2 + S_a/2 在二类 softmax 下恒等于 S_a，属写法对称。'],
      '代码 ✅ / 权重 ✅（仓库自带两个 checkpoint，开箱即跑）')

paper('5. AA-CLIP（CVPR 2025, 中科大 + 中科院 + 斯坦福）', 'arXiv 2503.06661 · 两阶段异常感知适配',
      '先让文本塔把正常/异常两个锚点推开，再让视觉塔的 patch 对齐过去，全程只训残差 adapter。',
      ['阶段一：文本塔前 3 层加 residual adapter（λ=0.1），用 Disentangle Loss 把两个锚点推正交。',
       '阶段二：视觉塔前 6 层加 adapter，4 个粒度的 patch 各过一个投影层，去贴住已冻结的锚点。',
       'adapter 在混合前先做模长对齐，保证只改方向、不改大小（论文未写，代码中有）。',
       '每类需要一句手写的自然语言描述（如 pcb1 → dual ultrasonic distance sensor pcb module）。'],
      ['MVTec 图像 90.5 / 像素 91.9；VisA 84.6 / 95.5；11 个数据集平均 93.4 像素。',
       '2-shot（每类 1 正 1 负，共 30 张）即可达 MVTec 85.9 / 91.0。',
       '消融：普通 adapter 会把性能打崩（88.9 → 48.9），残差是必要设计。'],
      ['输出是 margin 而非概率，只能算 AUROC，无法直接卡阈值；full-shot 反而过拟合（64-shot 图像 92.0 > full 90.5）。',
       '两阶段超参（lr 1e-5 / 5e-4、batch 16 / 2）敏感；未释放训练权重。'],
      '代码 ✅ / 权重 ❌')

paper('6. AnomalyGPT（AAAI 2024, 中科院自动化所）', 'arXiv 2308.15366 · 首个把 LVLM 用于工业异常检测的工作',
      '冻结 ImageBind 与 Vicuna，中间接一个检测器与一个 prompt learner，让大模型直接说出「有没有异常、在哪」。',
      ['ImageBind-Huge 出 4 层 patch，经线性层与 2 个文本锚点比对得到定位图（本身即 WinCLIP 式做法）。',
       '定位图经 5 层 CNN 压成 9 个 token，与 9 个可学习 base token 拼接后送入 Vicuna-7B。',
       '训练数据由 cut-paste + Poisson 融合合成，mask 自动获得，零人工标注。',
       '只训 decoder（5.25M）与 prompt learner（107M），并与 PandaGPT 通用数据交替训练以防遗忘。'],
      ['跨域 1-shot：训 VisA / 测 MVTec 得 94.1 / 95.3 / 86.1%（图像 AUC / 像素 AUC / 准确率）。',
       '同域（训 MVTec 测 MVTec）得 97.4 / 93.1——注意这不是零样本。',
       '关键对比：PatchCore 图像 AUC 99.3%，但用统一阈值时准确率仅 79.76%；AnomalyGPT 直接给出 86.1%。'],
      ['LLM 只拿到 1 个图像 token，视觉能力实际来自检测器；位置只到 3×3 宫格（像素图另由 decoder 给）。',
       '模拟异常与真实缺陷存在分布差；7B 模型训练推理都重；对话存在幻觉风险。'],
      '代码 ✅ / 权重 ✅')

paper('7. Dual Anchors（CVPR 2026 Findings）', 'arXiv 2608.11933 · 图像锚点 + 层级分组',
      '用图像自己分组产生「正常/异常」两个锚点，既门控 CLS，又当成词写进 prompt。',
      ['主干换成 DINOv3 ViT-L/16（冻结），文本侧仍是 CLIP ViT-L/14。',
       '4 个 Merge Block：交叉注意力初始化 → Gumbel-Softmax 硬分配 → 余弦匹配合并 → 最终更新，聚出 G_N / G_A。',
       'GGTR：用 CLS 与两个锚点的余弦做门控，只 2 个标量就修正整图表示。',
       'Dynamic State Prompt：[V…][W][G_N][class]——图像锚点直接作为 token 进文本塔。',
       'L_group：用 GT mask 切分出正常/异常 token，约束哪个锚点是哪一个（否则两个锚点可能整体反色）。'],
      ['MVTec 图像 92.7 / 像素 92.4，首次超过 WinCLIP 的 91.8。',
       'VisA 88.3 / 96.6；14 个数据集平均 92.9 / 92.9。',
       '消融：去掉层级分组，像素从 92.4 掉到 82.8（−9.6）。'],
      ['无代码、无权重、无超参（全在未发布的补充材料中），基本不可复现。',
       '文本塔每图都要重跑（锚点进了 prompt）；1024 维 patch 与 768 维文本之间的投影未交代。'],
      '代码 ❌ / 权重 ❌')

paper('8. MoECLIP（CVPR 2026）', 'arXiv 2603.03101 · patch 级专家混合',
      '给每个 patch 动态路由到专门的 LoRA 专家，实现 patch 级的差异化适配。',
      ['MoE 结构：K=4 个 LoRA 专家，Top-2 路由，按 patch 自身特征动态选择。',
       'FOFS（Frozen Orthogonal Feature Separation）：把输入特征空间正交切分，逼专家关注不同信息。',
       'simplex ETF 损失：约束专家输出形成最大等角表示，进一步抑制功能冗余。',
       '仍以 CLIP ViT-L/14@336 为主干（冻结），MoE 模块接在第 6/12/18/24 层输出处。'],
      ['MVTec 图像 93.9 / 像素 92.5；14 个数据集平均 89.6（图像）/ 94.3（像素）。',
       '在医学子集上提升明显：Brain MRI 图像 88.5（对比 AA-CLIP 79.5）。'],
      ['VisA 图像 83.6，反而低于 Bayes-PFL 的 86.8——跨域一致性仍有波动。',
       'MoE 带来额外推理开销；专家数量与路由策略需要调。'],
      '代码 ✅（GitHub: CoCoRessa/MoECLIP）')

paper('9. FB-CLIP（CVPR 2026）', 'arXiv 2603.19608 · 前景背景解耦',
      '把前景缺陷与背景干扰在文本、视觉两侧同时解耦，提升细粒度定位。',
      ['文本侧多策略表示：EOT 特征 + 全局池化 + 注意力加权 token，三者融合提供更丰富的语义线索。',
       '视觉侧多视角软分离：沿 identity / semantic / spatial 三个维度分离前景与背景，并做背景抑制。',
       '语义一致性正则（SCR）：对齐图像特征与正常/异常文本原型，抑制不确定匹配。'],
      ['MVTec 图像 92.4 / 像素 91.9；VisA 89.5 / 96.3；Real-IAD 图像 80.6。',
       '在 18 项指标中 14 项进入前二，超过 FAPrompt（10 项）。'],
      ['指标口径与部分同期工作不同（用 AP / PRO 混排），跨表比较需注意。',
       '解耦模块较多，超参与训练稳定性未见详细报告。'],
      '代码 ✅（GitHub: Xi-Mu-Yu/FB-CLIP）')

paper('10. AnomalyVFM（CVPR 2026）', 'arXiv 2601.20524 · 把视觉基础模型改造成异常检测器',
      '彻底不用 CLIP 文本，改用纯视觉基础模型（RADIO / DINOv2）加合成数据适配。',
      ['三阶段合成数据生成：从真实 AD 数据出发生成更多样、更贴近真实缺陷的辅助数据。',
       '参数高效适配：低秩特征 adapter 插进 VFM 的中间层。',
       '置信度加权像素损失：对不确定区域降权，缓解合成数据标签噪声。',
       '无需文本编码器，整个流程是纯视觉的。'],
      ['以 RADIO 为骨干，9 个数据集图像级平均 AUROC 94.1%，比此前 SOTA 高 3.3 个点。',
       'MVTec 图像 94.9 / 像素 92.7；VisA 93.6 / 96.2——两项均为目前最高。'],
      ['合成数据生成管线复杂，复现成本高；纯视觉方案放弃了文本的可解释性。',
       '结果较新，尚缺第三方独立复现。'],
      '代码 ✅（GitHub: MaticFuc/AnomalyVFM）')
doc.add_page_break()

# ---------- 四、横向对比 ----------
doc.add_heading('四、横向对比', level=1)

doc.add_heading('4.1 数据需求', level=2)
para('这一栏比精度更能决定能不能落地——很多场景根本没有异常标注。')
table(['方法', '训练图', '真实异常 mask', '物体名', '换新类要准备'],
      [['WinCLIP', '0', '不需要', '需要', '类名'],
       ['AdaCLIP', '约 2337', '需要', '需要', '类名'],
       ['AnomalyCLIP', '约 1725', '需要', '不需要', '什么都不要'],
       ['AA-CLIP', '30 起', '需要', '需要（整句）', '一句描述'],
       ['AnomalyGPT', '正常图 + 模拟异常', '不需要（模拟生成）', '需要（整句）', '一句描述'],
       ['Dual Anchors', '未给', '需要（用两次）', '需要', '类名'],
       ['MoECLIP', '未给', '需要', '未给', '未给'],
       ['FB-CLIP', '未给', '需要', '未给', '未给'],
       ['AnomalyVFM', '合成数据', '需要（真实 AD 数据出发）', '不需要', '无']],
      widths=[2.8, 4.2, 4.2, 3.2, 2.4], font=8.5)

doc.add_heading('4.2 输出形态', level=2)
para('输出是分数还是判断，直接决定了落地时要不要额外调阈值。')
table(['形态', '代表', '能否直接卡阈值', '能否解释'],
      [['裸分数', 'PatchCore 类', '不能（每类阈值不同）', '不能'],
       ['softmax 概率', 'WinCLIP / AdaCLIP / AnomalyCLIP', '勉强（需校准）', '能'],
       ['margin', 'AA-CLIP', '不能（范围约 −50~+50）', '弱'],
       ['判断 + 自然语言', 'AnomalyGPT', '不需要阈值', '最强']],
      widths=[3.4, 5.4, 4.6, 3.2], font=8.5)
para('最有力的证据来自 AnomalyGPT 论文：PatchCore 在 MVTec 上图像 AUC 99.3%，'
     '但用统一阈值推理时准确率只有 79.76%。高 AUC 不等于可用。', size=9.5)

doc.add_heading('4.3 可复现性', level=2)
table(['方法', '代码', '权重', '超参', '复现难度'],
      [['CLIP', '✅', '✅', '✅', '中（数据未公开）'],
       ['WinCLIP', '❌（仅社区版）', '❌', '✅', '高（社区版差 20 点）'],
       ['AdaCLIP', '✅', '✅', '✅', '中（fp16 不稳）'],
       ['AnomalyCLIP', '✅', '✅', '✅', '低（开箱可跑）'],
       ['AA-CLIP', '✅', '❌', '✅', '高（需自己训练）'],
       ['AnomalyGPT', '✅', '✅', '✅', '高（7B 双卡）'],
       ['Dual Anchors', '❌', '❌', '❌', '极高（不可复现）'],
       ['MoECLIP', '✅', '未确认', '✅', '中'],
       ['FB-CLIP', '✅', '未确认', '部分', '中'],
       ['AnomalyVFM', '✅', '未确认', '部分', '中高（管线复杂）']],
      widths=[3.0, 3.4, 2.6, 2.4, 5.2], font=8.5)
para('一个重要规律：精度最高的几篇，复现难度也最高；而长期被当作标准 baseline 的 AnomalyCLIP，'
     '恰恰是唯一开箱即跑的那一篇。论文排名与「能用的排名」并不一致。', size=9.5)

doc.add_heading('4.4 2026 年的三个新方向', level=2)
para('从最新三篇可以看出一条共同趋势：单纯调整 CLIP 的文本侧已经接近上限，'
     '2026 年的工作开始在「适配的粒度」与「主干的来源」上做文章。')
bullet('更细的适配粒度（MoECLIP）：不再对所有 patch 用同一套适配，而是让每个 patch 路由到专门的专家。'
       '这假设「不同区域需要不同的异常先验」，与 AA-CLIP 的「统一 adapter」形成对比。')
bullet('更强的空间解耦（FB-CLIP）：把「前景缺陷」与「背景干扰」显式分开。'
       '这继承了 AnomalyCLIP 与 AA-CLIP 对「语义纠缠」的关注，但把战场从文本空间搬到了空间维度。')
bullet('更换主干（Dual Anchors / AnomalyVFM）：两篇都不约而同地指出 CLIP 的视觉表示已不够强。'
       'Dual Anchors 用 DINOv3 补视觉、仍保留 CLIP 文本；AnomalyVFM 干脆完全抛弃文本，纯视觉 + 合成数据。')
para('如果这条趋势延续，未来的 ZSAD 可能不再是「怎么用 CLIP」，而是「怎么用 VFM + 少量文本先验」。')
doc.add_page_break()

# ---------- 五、选型 ----------
doc.add_heading('五、选型建议', level=1)
table(['你的约束', '建议', '理由'],
      [['一个训练样本都没有', 'WinCLIP', '唯一零训练的方法，拿 CLIP 权重即可出分'],
       ['有 1~4 张正常图、无异常标注', 'WinCLIP+', '只有它用得上这几张图'],
       ['有辅助集（含 mask）、要开箱即用', 'AnomalyCLIP', '唯一自带权重，test.sh 直接跑'],
       ['辅助数据极少（每类 1 正 1 负）', 'AA-CLIP', '30 张图即可到 91.0 像素'],
       ['要最省推理', 'AnomalyCLIP / AA-CLIP', '都是一遍 ViT-L 前向'],
       ['不想设阈值、要对话解释', 'AnomalyGPT', '唯一输出判断 + 位置 + 对话'],
       ['追求最前沿数字、有工程能力', 'MoECLIP / AnomalyVFM', 'MVTec 94 以上，但需自行复现'],
       ['要最稳、最省事', 'AnomalyCLIP + AA-CLIP 组合思路', '一个能跑、一个有增量']],
      widths=[6.2, 4.6, 6.0], font=9)

para('选型时先问三个问题：')
bullet('你手上有异常图吗？没有 → 只剩 WinCLIP / WinCLIP+。')
bullet('你能给每个类写一句描述吗？不能 → AnomalyCLIP（唯一不需要物体名）。')
bullet('你接受手工阈值吗？不接受 → AnomalyGPT（唯一直接给判断）。')
doc.add_page_break()

# ---------- 六、复现避坑 ----------
doc.add_heading('六、复现避坑：代码与论文的差异', level=1)
para('以下差异均由实际阅读官方或社区代码发现，按严重程度排列。🔴 表示会显著影响复现结果。')
rows = [
    ['🔴', 'WinCLIP', '论文称三个尺度（含整图），代码默认只开 2×2 与 3×3', '缺 image-scale，图像级融合失效'],
    ['🔴', 'WinCLIP', '论文称跨尺度调和平均，代码用算术平均', '边界抑制被冲掉（0.034 → 0.446）'],
    ['🔴', 'WinCLIP', '论文式 (4) 的图像级融合未实现', 'AC 直接取热力图最大值'],
    ['🔴', 'AdaCLIP', 'HSF 从「取最异常簇」改为「top-k 聚类后所有簇心平均」', '异常簇被稀释'],
    ['🔴', 'AA-CLIP', '式 (1) 的 Norm 在代码中是「模长对齐」，论文未写', '不这么做模长会逐层漂移'],
    ['🔴', 'AnomalyGPT', '训练时以 50% 概率随机切换到 few-shot 路径', '不这么做 few-shot 分支会崩'],
    ['🟡', 'WinCLIP', '异常分用 1/p(normal) 而非 1−p', '排序不变、阈值全变'],
    ['🟡', 'WinCLIP', '非正方形图直接拉伸，未做 tiling', 'VisA 比例失真'],
    ['🟡', 'AdaCLIP', '默认辅助集 README 更正为 MVTec + ColonDB', '训练集不同，数字不可直接比'],
    ['🟡', 'AdaCLIP', '动态 prompt 生成需完整跑一遍 ViT-L', '单图两遍前向，论文未提成本'],
    ['🟡', 'AdaCLIP', '4 层损失求和而非平均', '像素损失权重是图像的 4 倍'],
    ['🟡', 'AnomalyCLIP', 'DPAM 只替换第 6~24 层（19 层）', '论文未写替换范围'],
    ['🟡', 'AA-CLIP', '阶段一实际没有分类损失（图中画了）', '图比代码多一条线'],
    ['🟡', 'AA-CLIP', '图像级特征来自 patch 平均而非 CLS', 'CLS 早被全局注意力稀释'],
    ['🟡', 'AA-CLIP', '分类 logits 无温度缩放', '输出是 margin，不是概率'],
    ['🟡', 'AnomalyGPT', '论文称三个损失，代码实际两项', '「图像描述 CE」靠数据混合实现'],
    ['🟡', 'AnomalyGPT', 'prompt learner 最后一层卷积为 107M 参数', '「轻量」仅指 decoder'],
    ['🟢', 'AnomalyCLIP', '(I−S_n)/2 + S_a/2 在二类 softmax 下恒等于 S_a', '写法对称，无额外信息'],
    ['🟢', 'CLIP', '论文称文本截断到 76，代码为 77（含起止符）', '计数方式不同'],
]
table(['级别', '论文', '差异', '影响'], rows, widths=[1.2, 2.6, 8.2, 5.2], font=8)
para('三条经验：① 差异集中在「最后一步怎么聚合」——论文对这类细节写得最含糊、对结果影响最大；'
     '② 论文里的「轻量」要自己算参数；③ 官方代码也会与论文打架，读代码比读论文可靠。')
doc.add_page_break()

# ---------- 附录 ----------
doc.add_heading('附录：材料与阅读方法', level=1)
para('每一篇论文都在工作区里有一个独立目录，包含以下材料：')
table(['文件 / 目录', '内容'],
      [['paper.pdf', '论文原文（含附录）'],
       ['source/', 'arXiv LaTeX 源码，可直接编译'],
       ['paper.tex', '源码拍平后的单文件（\\input 已展开），带行号'],
       ['outline.md', '章节与公式的行号索引'],
       ['figures/', '逐张裁出的图与表 PNG，含 caption 索引'],
       ['code/', '官方或社区代码（浅克隆）'],
       ['精读笔记.md', '六站式精读笔记：动机 / 架构 / 训练 / 数据 / 信息流 / 例子 / 实验']],
      widths=[4.0, 12.8], font=9)
para('阅读方法采用「六站式」：① 动机（要解决什么、以前卡在哪、押注什么）→ ② 架构（每个框是什么、数据怎么流）→ '
     '③ 训练（损失在逼模型学什么）→ ④ 数据（一条样本长什么样）→ ⑤ 信息流（张量怎么变）→ '
     '⑥ 拿一个具体例子对着代码走一遍 → ⑦ 实验与地位。')
para('本报告共整理十篇论文、约 70 个精读站点、28 处代码与论文的差异。所有原始笔记保留在各自的精读笔记中，'
     '可按需查阅。')
para('')
para('— 报告完 —', align=WD_ALIGN_PARAGRAPH.CENTER, size=10)

doc.save('零样本异常检测十篇精读报告.docx')
print('saved')

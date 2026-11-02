"""Draw the proposed VLM post-training design as editable SVG and vector PDF.

No model, data download, or training is performed. Coordinates use a top-left
origin; PDF is rendered from the same drawing operations as SVG.
"""
from pathlib import Path
from html import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import toColor as HexColor

OUT = Path(__file__).resolve().parent
W, H = 1700, 1100
SCALE = (180 / 25.4 * 72) / W
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
pdfmetrics.registerFont(TTFont('ArialFigure', str(FONT_DIR / 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('ArialFigureBold', str(FONT_DIR / 'Arial Bold.ttf')))
pdfmetrics.registerFont(TTFont('ArialFigureItalic', str(FONT_DIR / 'Arial Italic.ttf')))
pdfmetrics.registerFont(TTFont('ArialFigureCJK', str(FONT_DIR / 'Arial Unicode.ttf')))

INK = '#263346'
MUTED = '#526174'
BORDER = '#C7D0DA'
FROZEN = '#EEF1F5'
DATA = '#EAF5EF'
BLUE = '#EAF1FB'
ORANGE = '#FFF0DE'
TRAIN = '#B85B0D'
TEAL = '#367C67'


class Figure:
    def __init__(self, language):
        self.language = language
        self.stem = OUT / f'industrial_vlm_framework_{language}'
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{180*H/W:.4f}mm" viewBox="0 0 {W} {H}">',
                    '<title>Proposed industrial VLM instruction tuning framework</title>',
                    '<desc>Offline annotation-guided instruction construction, frozen Qwen3-VL visual modules and NF4 language weights with trainable attention LoRA, assistant-only SFT, and independent inference evaluation. Proposed design; no experimental results are implied.</desc>',
                    f'<rect width="{W}" height="{H}" fill="white"/>']
        self.pdf = canvas.Canvas(str(self.stem.with_suffix('.pdf')), pagesize=(W*SCALE, H*SCALE))
        self.pdf.setTitle('Proposed industrial VLM instruction tuning framework')
        self.pdf.setAuthor('Project design')
        self.warnings = []

    def rect(self, x, y, w, h, fill='white', stroke=BORDER, radius=12, dashed=False):
        dash = ' stroke-dasharray="8 6"' if dashed else ''
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"{dash}/>')
        p = self.pdf
        p.setFillColor(HexColor(fill)); p.setStrokeColor(HexColor(stroke)); p.setLineWidth(1.8*SCALE)
        p.setDash([8*SCALE,6*SCALE] if dashed else [])
        p.roundRect(x*SCALE, (H-y-h)*SCALE, w*SCALE, h*SCALE, radius*SCALE, stroke=1, fill=1)
        p.setDash([])

    def text(self, x, y, lines, size=23, bold=False, color=INK, anchor='middle', leading=None, maxwidth=None):
        if isinstance(lines, str): lines = [lines]
        leading = leading or size*1.28
        font = 'ArialFigureCJK' if self.language == 'zh' else ('ArialFigureBold' if bold else 'ArialFigure')
        family = 'Arial Unicode MS, Arial, sans-serif' if self.language == 'zh' else 'Arial, Helvetica, sans-serif'
        weight = '700' if bold else '400'
        for i, line in enumerate(lines):
            baseline = y + i*leading
            width = pdfmetrics.stringWidth(line, font, size)
            if maxwidth is not None and width > maxwidth:
                self.warnings.append(f'{line}: {width:.1f}>{maxwidth}')
            self.svg.append(f'<text x="{x}" y="{baseline}" fill="{color}" font-family="{family}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(line)}</text>')
            p = self.pdf
            p.setFillColor(HexColor(color)); p.setFont(font,size*SCALE)
            fn = {'middle':p.drawCentredString,'start':p.drawString,'end':p.drawRightString}[anchor]
            fn(x*SCALE,(H-baseline)*SCALE,line)

    def arrow(self, points, color=INK, dashed=False, width=2.6):
        pts = ' '.join(f'{x},{y}' for x,y in points)
        dash = ' stroke-dasharray="9 6"' if dashed else ''
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        p = self.pdf
        p.setStrokeColor(HexColor(color)); p.setLineWidth(width*SCALE)
        p.setDash([9*SCALE,6*SCALE] if dashed else [])
        path = p.beginPath(); path.moveTo(points[0][0]*SCALE,(H-points[0][1])*SCALE)
        for x,y in points[1:]: path.lineTo(x*SCALE,(H-y)*SCALE)
        p.drawPath(path); p.setDash([])
        import math
        x,y = points[-1]; x0,y0 = points[-2]
        theta = math.atan2(y-y0,x-x0)
        length, half = 11, 5
        b1 = (x-length*math.cos(theta)+half*math.sin(theta), y-length*math.sin(theta)-half*math.cos(theta))
        b2 = (x-length*math.cos(theta)-half*math.sin(theta), y-length*math.sin(theta)+half*math.cos(theta))
        self.svg.append(f'<polygon points="{x},{y} {b1[0]},{b1[1]} {b2[0]},{b2[1]}" fill="{color}"/>')
        tri = p.beginPath(); tri.moveTo(x*SCALE,(H-y)*SCALE); tri.lineTo(b1[0]*SCALE,(H-b1[1])*SCALE); tri.lineTo(b2[0]*SCALE,(H-b2[1])*SCALE); tri.close()
        p.setFillColor(HexColor(color)); p.drawPath(tri,fill=1,stroke=0)

    def math(self, center_x, y, pieces):
        """Render identical vector math labels using explicit subscript runs."""
        total = sum(pdfmetrics.stringWidth(s, 'ArialFigureItalic' if it else 'ArialFigure', sz)
                    for s,sz,dy,it in pieces)
        x = center_x-total/2
        for s,sz,dy,it in pieces:
            font = 'ArialFigureItalic' if it else 'ArialFigure'
            style = 'italic' if it else 'normal'
            self.svg.append(f'<text x="{x}" y="{y+dy}" fill="{INK}" font-family="Arial, Helvetica, sans-serif" font-size="{sz}" font-style="{style}">{escape(s)}</text>')
            self.pdf.setFillColor(HexColor(INK)); self.pdf.setFont(font,sz*SCALE)
            self.pdf.drawString(x*SCALE,(H-y-dy)*SCALE,s)
            x += pdfmetrics.stringWidth(s,font,sz)

    def box(self, x, y, w, h, lines, fill='white', size=23, bold=False, stroke=BORDER):
        self.rect(x,y,w,h,fill=fill,stroke=stroke)
        n = len(lines) if isinstance(lines,list) else 1
        lineheight = size*1.28
        baseline = y+h/2 - (n-1)*lineheight/2 + size*.34
        self.text(x+w/2,baseline,lines,size=size,bold=bold,maxwidth=w-20)

    def save(self):
        self.svg.append('</svg>')
        self.stem.with_suffix('.svg').write_text('\n'.join(self.svg))
        self.pdf.showPage(); self.pdf.save()
        if self.warnings:
            raise ValueError('Text overflow:\n'+'\n'.join(self.warnings))


def draw(language):
    z = language == 'zh'
    t = lambda en,zh: zh if z else en
    f = Figure(language)

    # (a) Offline data construction: split before deriving any instruction.
    f.rect(15,15,1670,265)
    f.text(40,53,t('(a) Annotation-guided instruction construction','(a) 标注约束的工业指令数据构建'),size=28,bold=True,anchor='start')
    boxes = [
        (45,90,220,110,t(['VisA images','Labels + masks'],['VisA 原始图像','标签与像素标注'])),
        (310,90,210,110,t(['Group-aware split','before derivation'],['按原图/关联组划分','先划分，再派生'])),
        (565,90,235,110,t(['Region extraction','Reference pairing'],['区域框与裁剪','正常参考图配对'])),
        (845,90,235,110,t(['QA construction','Annotation + rewriting'],['构建多任务问答','标注约束与表达改写'])),
        (1125,90,230,110,t(['Quality filtering','Rules + human audit'],['数据质量过滤','规则校验与人工抽查'])),
        (1400,90,240,110,t(['Instruction records','Images + request','Target answer'],['高质量指令记录','图像 / 指令 / 答案'])),
    ]
    for x,y,w,h,lines in boxes: f.box(x,y,w,h,lines,fill=DATA,size=22)
    for a,b in zip(boxes,boxes[1:]): f.arrow([(a[0]+a[2],145),(b[0],145)])
    f.text(1095,230,t('Training partition only: crops, pairs, rewrites and all derived QA share source split.',
                       '区域裁剪、图像配对、表达改写与派生问答仅在训练划分内构建。'),size=22,color=TEAL)
    f.text(415,258,t('Held-out partitions are locked for evaluation in (c).','验证 / 内部测试 / 外部测试独立锁定，供 (c) 评测。'),size=21,color=MUTED)

    # (b) Native multimodal forward; new trainable mechanism is attention LoRA.
    f.rect(15,300,1670,420)
    f.text(40,340,t('(b) Native Qwen3-VL-8B with parameter-efficient adaptation','(b) 保留 Qwen3-VL-8B 原生多模态架构，进行参数高效适配'),size=28,bold=True,anchor='start')
    f.box(45,385,225,120,t(['Query image','Optional normal ref.'],['待检图像','可选正常参考图']),size=22)
    f.box(45,600,225,70,t(['Inspection request'],['质检指令']),size=22)
    f.box(325,385,300,85,t(['Native vision encoder','Frozen'],['原生视觉编码器','冻结']),fill=FROZEN,size=24)
    f.box(325,505,300,85,t(['Merger + DeepStack path','Frozen'],['Merger 与 DeepStack 路径','冻结']),fill=FROZEN,size=23)
    f.box(325,600,300,70,t(['Request tokenization','+ answer prefix in SFT'],['文本模板与分词','训练时包含答案前缀']),size=21)
    f.arrow([(270,445),(325,445)])
    f.arrow([(475,470),(475,505)])
    f.arrow([(270,635),(325,635)])
    f.box(705,445,230,160,t(['Native fusion','Visual + text tokens','Grids + positions'],['原生多模态融合','视觉与文本 token','图像网格与位置编码']),fill=BLUE,size=22)
    f.arrow([(625,547),(705,547)])
    f.arrow([(625,635),(670,635),(670,580),(705,580)])
    f.rect(1000,385,340,285,fill='white',stroke='#8B99AB')
    f.text(1170,420,t('Language model','语言模型'),size=25,bold=True)
    f.box(1020,442,300,72,t(['NF4 base weights','Frozen'],['NF4 量化基座权重','冻结']),fill=FROZEN,size=23)
    f.box(1020,535,300,75,t(['Attention LoRA','Trainable'],['Attention LoRA','可训练']),fill=ORANGE,size=24,bold=True,stroke=TRAIN)
    f.math(1170,645,[('h = W',22,0,True),('NF4',14,5,False),('x + (α/r) BAx',22,0,True)])
    f.arrow([(935,525),(1000,525)])
    f.box(1405,400,235,70,t(['Next-token logits'],['下一 token logits']),fill=BLUE,size=23)
    f.box(1405,510,235,160,t(['Generated answer','Anomaly decision','Bounding boxes','Visible evidence'],['自回归生成答案','异常判断','缺陷位置框','可见视觉依据']),fill=BLUE,size=22)
    f.arrow([(1340,435),(1405,435)])
    f.arrow([(1522,470),(1522,510)])
    f.text(820,698,t('Native visual feature injection is preserved; LoRA adapts selected language attention projections.',
                    '保留原生视觉特征注入；LoRA 仅适配选定的语言侧注意力投影。'),size=21,color=MUTED)

    # (c) Left: deployment/evaluation; right: assistant-only training objective.
    f.rect(15,740,1670,295)
    f.text(40,780,t('(c) Inference and independent evaluation','(c) 推理与独立评测'),size=27,bold=True,anchor='start')
    f.text(810,780,t('Assistant-only supervised fine-tuning','仅对答案部分计算 SFT 损失'),size=27,bold=True,anchor='start')
    f.box(45,815,195,82,t(['New image(s)','User request'],['新图像','用户请求']),size=22)
    f.box(285,815,195,82,t(['Adapted model (b)','No weight updates'],['适配后的模型 (b)','推理时参数固定']),size=21)
    f.box(525,815,195,82,t(['Decision / boxes','Visible evidence'],['生成答案','判断 / 框 / 依据']),size=20)
    f.arrow([(240,856),(285,856)])
    f.arrow([(480,856),(525,856)])
    f.box(45,935,195,65,t(['Held-out labels'],['独立测试标注']),size=22)
    f.box(285,935,435,65,t(['Detection / grounding / evidence / efficiency'],['检测 / 框定位 / 证据一致性 / 效率']),fill=DATA,size=21)
    f.arrow([(240,968),(285,968)],color=TEAL)
    f.arrow([(622,897),(622,935)])
    f.box(810,815,270,90,t(['Target assistant tokens','From qualified records'],['目标答案 token','来自质检后的训练记录']),fill=DATA,size=22)
    f.box(1130,815,230,90,t(['Token cross-entropy','Answer tokens only'],['Token 交叉熵','仅监督答案位置']),fill=ORANGE,size=22,stroke=TRAIN)
    f.box(1410,815,230,90,t(['Token logits'],['预测 token logits']),fill=BLUE,size=22)
    f.arrow([(1080,860),(1130,860)],color=TEAL)
    f.arrow([(1410,860),(1360,860)],color=TEAL)
    f.arrow([(1640,435),(1660,435),(1660,860),(1640,860)])
    f.arrow([(1360,835),(1370,835),(1370,573),(1320,573)],color=TRAIN,dashed=True)
    f.text(1440,730,t('Update LoRA only','仅更新 LoRA'),size=20,color=TRAIN)
    f.math(1225,958,[('L',25,0,True),('SFT',15,6,False),(' = − Σ',25,0,False),
                     ('t ∈ A',14,7,False),(' log p',25,0,False),('θ,φ',15,6,True),
                     ('(y',25,0,False),('t',15,6,True),(' | I, q, y',25,0,False),
                     ('<t',15,6,True),(')',25,0,False)])
    f.text(1225,995,t('Box coordinates are supervised as answer tokens.',
                     '框坐标作为答案 token 监督。'),size=21,color=MUTED)

    # Explicit legend outside panels, useful at two-column print width.
    f.rect(35,1060,28,22,fill=FROZEN,radius=3)
    f.text(75,1079,t('Frozen parameters','冻结参数'),size=21,anchor='start')
    f.rect(290,1060,28,22,fill=ORANGE,stroke=TRAIN,radius=3)
    f.text(330,1079,t('Trainable parameters','可训练参数'),size=21,anchor='start')
    f.arrow([(600,1071),(650,1071)])
    f.text(665,1079,t('Forward / data','前向 / 数据'),size=21,anchor='start')
    f.arrow([(900,1071),(950,1071)],color=TEAL)
    f.text(965,1079,t('Supervision','监督'),size=21,anchor='start')
    f.arrow([(1175,1071),(1225,1071)],color=TRAIN,dashed=True)
    f.text(1240,1079,t('Gradient / update','梯度 / 更新'),size=21,anchor='start')
    f.text(1660,1079,t('Proposed design','设计方案'),size=20,color=MUTED,anchor='end')
    f.save()


if __name__ == '__main__':
    for language in ['en','zh']:
        draw(language)
    print('Created English and Chinese editable SVG + vector PDF.')

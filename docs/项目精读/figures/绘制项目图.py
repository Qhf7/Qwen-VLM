"""项目精读配图。只使用Pillow绘制说明图，不执行模型。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
FONT="/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
BOLD="/System/Library/Fonts/STHeiti Medium.ttc"
INK="#15243b"
MUTED="#52647b"
WHITE="#ffffff"

def font(size,bold=False):
    return ImageFont.truetype(BOLD if bold else FONT,size)

def base(title,subtitle,size=(2200,1260)):
    im=Image.new("RGB",size,"#f4f7fb")
    d=ImageDraw.Draw(im)
    d.text((65,45),title,font=font(64,True),fill=INK)
    d.text((68,132),subtitle,font=font(31),fill=MUTED)
    return im,d

def txt(d,xy,lines,size=32,fill=INK,bold=False,spacing=13):
    if isinstance(lines,str): lines=lines.split("\n")
    x,y=xy
    for line in lines:
        d.text((x,y),line,font=font(size,bold),fill=fill)
        y+=size+spacing

def box(d,xy,title,body,accent="#2369ad",fill="#ffffff",title_size=36,body_size=31):
    x0,y0,x1,y1=xy
    d.rounded_rectangle(xy,20,fill=fill,outline=accent,width=3)
    d.rectangle((x0+1,y0+23,x0+8,y1-23),fill=accent)
    txt(d,(x0+25,y0+22),title,title_size,accent,True)
    txt(d,(x0+25,y0+82),body,body_size)

def arrow(d,start,end,color="#52647b",width=7):
    d.line([start,end],fill=color,width=width)
    x,y=end
    if abs(end[0]-start[0])>=abs(end[1]-start[1]):
        sg=1 if end[0]>=start[0] else -1
        d.polygon([(x,y),(x-20*sg,y-12),(x-20*sg,y+12)],fill=color)
    else:
        sg=1 if end[1]>=start[1] else -1
        d.polygon([(x,y),(x-12,y-20*sg),(x+12,y-20*sg)],fill=color)

def finish(im,name):
    im.save(OUT/name,optimize=True)

im,d=base("工业异常检测项目：三条实现路线","先分别理解数据流，再讨论如何组合；图中形状省略 batch 维。")
rows=[
(220,"A  原生 Qwen","VisA · 领域适配","#2369ad",
 ("原生视觉通道",["图片＋指令","processor → 视觉塔","merger / DeepStack"]),
 ("语言注意力 LoRA",["底座与视觉冻结","q / k / v / o：r = 16","可训练 15,335,424 参数"]),
 ("两种输出协议",["生成 JSON：判断 / 框","候选评分：true 对 false","没有额外像素分割头"])),
(550,"B  AnomalyGPT","VisA · 桥接迁移","#227a66",
 ("ImageBind 缓存",["全局特征 1024","四层局部 patch 特征","类别正常 / 异常文本原型"]),
 ("异常图 → 软提示",["四层概率图平均 224×224","9 个基础＋9 个动态提示","全局投影＋手工文字前缀"]),
 ("语言回答＋热图",["Qwen / Vicuna 语言模块","Qwen 原生视觉已移除","可选语言 LoRA：r = 8"])),
(880,"C  跨零件分割","Real-IAD · 整类留出","#7854a1",
 ("冻结 DINOv2",["448×448 RGB","取第 3 / 6 / 9 / 12 层","每层 768×32×32"]),
 ("可训练分割解码器",["多层投影、卷积融合","可选文本锚点 / 缓存支路","Focal ＋ Dice 像素监督"]),
 ("像素异常图",["448×448 logits / probability","Pixel AP、AUROC、AUPRO","该路径不逐 token 生成文字"]))
]
for y,label,sub,c,a,b,out in rows:
    txt(d,(65,y+40),label,40,c,True)
    txt(d,(65,y+110),sub,27,MUTED)
    for x,w,node in [(405,470,a),(960,510,b),(1555,575,out)]:
        box(d,(x,y,x+w,y+260),node[0],node[1],c,body_size=29)
    arrow(d,(890,y+130),(945,y+130),c)
    arrow(d,(1485,y+130),(1540,y+130),c)
txt(d,(68,1192),"代码阅读顺序：任务 / 数据 → 模型输入 → 哪些参数更新 → 输出协议 → 指标与证据边界。",30,MUTED)
finish(im,"三条实验路线.png")

im,d=base("原生 Qwen：同一视觉通道，训练与两种推理","蓝色为共同输入；绿色为训练；橙色为推理。GT 答案只进入训练分支。",(2200,1470))
box(d,(70,245,620,480),"图片＋任务指令",["图片 / 文字 → native processor","训练先拼入 answer，屏蔽前缀标签","input_ids / pixel_values / grid"],body_size=28)
box(d,(750,245,1450,480),"Qwen3-VL 原生模型",["冻结视觉塔、merger、DeepStack、语言底座","仅语言 q/k/v/o 上 LoRA 可训练","r=16，alpha=32，dropout=0.05"],body_size=30)
arrow(d,(640,355),(730,355),"#2369ad")
arrow(d,(1470,355),(1630,355),"#2369ad")
box(d,(1650,245,2130,480),"词表 logits",["每个位置的候选 token 分数","因果注意力：只看历史","后续处理取决于协议"],body_size=29)
d.line([(1890,500),(1890,550),(350,550),(350,610)],fill="#52647b",width=5)
d.line([(1050,550),(1050,610)],fill="#52647b",width=5)
d.line([(1790,550),(1790,610)],fill="#52647b",width=5)
box(d,(65,630,685,1020),"训练：assistant-only CE",["加入真实 answer，问题 labels = −100","位置 t 预测目标 token t+1","累计 8 张，按监督 token 数归一","backward → AdamW → 保存 adapter"],"#227a66",body_size=29)
box(d,(790,630,1410,1020),"推理①：自由生成 JSON",["只给图片和指令，无真实 answer","greedy 逐 token 生成，使用 KV cache","parser 提取判断 / 区域框","评估 F1、格式、端到端框召回"],"#b86a21",body_size=29)
box(d,(1515,630,2135,1020),"推理②：候选续写评分",["固定公共 JSON 前缀，比较 true / false","只计布尔值 token 的 log likelihood","margin = logP(true) − logP(false)","评估 AUROC / AP，另选判别阈值"],"#b86a21",body_size=29)
box(d,(65,1100,2135,1370),"真实验证例子：cashew / Anomaly / 050.JPG",[
"1274×1176；缺陷 1,013 像素，占 0.0676%；GT 框 [705,573,749,608]（0–1000 坐标）。",
"同图候选 margin：NF4 底座 −4 → DD +3；sigmoid 分数约 0.018 → 0.953（未经概率校准）。",
"整体效果看全部 120 张图：60 异常 × 60 正常；DD AUROC 94.56%，不是生成准确率。"],"#52647b",body_size=30)
finish(im,"原生Qwen信息流.png")

im,d=base("AnomalyGPT 迁移：预测热图 → 连续软提示 → 语言模型","全局信息与局部信息分别进入语言前缀；GT mask 只参与训练损失。",(2200,1430))
box(d,(65,240,575,455),"全局缓存",["[B,1024]","整个图像的表示"],"#227a66")
box(d,(720,240,1300,455),"llama_proj",["Linear：1024 → 4096","输出 [B,1,4096]"],"#227a66")
arrow(d,(595,345),(700,345),"#227a66")
box(d,(65,575,575,860),"四层局部缓存＋文本",["4 × [257,B,1280]，含 CLS","去 CLS，投影 1280 → 1024","与正常 / 异常文本做相似度"],"#2369ad",body_size=29)
box(d,(720,575,1300,860),"四层概率图",["16×16 → 224×224 → softmax","每层取异常通道，再平均","最终 [B,1,224,224]"],"#2369ad",body_size=29)
arrow(d,(595,715),(700,715),"#2369ad")
box(d,(720,960,1300,1255),"PromptLearner",["CNN / pooling：224 → 7 → 3","9 个动态 ＋ 9 个基础向量","输出 [B,18,4096]"],"#7854a1",body_size=29)
arrow(d,(1000,880),(1000,940),"#2369ad")
box(d,(1500,395,2135,800),"语言 embedding 前缀",["1 个 global token","18 个 soft prompt","＋手工对话文字 embedding","Qwen 语言模块 / Vicuna","输出 Yes / No 或解释"],"#227a66",body_size=31)
d.line([(1320,345),(1430,345),(1430,490)],fill="#227a66",width=6)
arrow(d,(1430,490),(1480,490),"#227a66")
d.line([(1320,1110),(1430,1110),(1430,710)],fill="#7854a1",width=6)
arrow(d,(1430,710),(1480,710),"#7854a1")
box(d,(65,1000,575,1255),"像素损失：仅训练",["逐层预测与 GT mask 比较","四层各算 Focal ＋ Dice","GT 不进入语言前缀"],"#b86a21",body_size=29)
d.line([(750,880),(650,880),(650,920),(320,920)],fill="#b86a21",width=5)
arrow(d,(320,920),(320,980),"#b86a21")
box(d,(1500,925,2135,1255),"需要验证的研究假设",["CE 可穿过冻结 LLM 回传到 bridge","soft prompt 改变 ≠ 决策变正确","迁移 Qwen 不用原生 visual","r=8 LoRA 与原生 r=16 分开"],"#52647b",body_size=29)
txt(d,(70,1330),"bridge 训练参数约 1.17 亿；Qwen LoRA 臂另加 7,667,712。不能只统计 LoRA 而忽略 bridge。",30,MUTED)
finish(im,"AnomalyGPT桥接.png")
print("生成3张项目讲解图。")

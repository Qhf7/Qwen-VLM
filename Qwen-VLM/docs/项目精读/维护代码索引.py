"""读取AST，生成当前文档的代码链接与关键函数索引，不导入项目模型。"""
from __future__ import annotations
import ast
import json
import os
import re
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[1]
MANIFEST = DOC / "代码定位清单.json"

MODULES = {
"data.py": "数据读取、标注转换、去重与分层划分",
"modeling.py": "原生Qwen加载、精确LoRA注入与processor输入",
"train.py": "原生Qwen的CE训练、梯度累计、保存与重载",
"infer.py": "原生Qwen单图生成与JSON解析",
"score.py": "候选续写评分、排序指标与阈值",
"metrics.py": "原生输出解析、分类与外接框评测",
"compare.py": "原生同源生成结果及提示一致性比较",
"prepare_multitask_v2.py": "RR/DD/RD任务分配与匹配曝光构建",
"train_qwen_matched.py": "缓存bridge接入Qwen语言端及r8 LoRA",
"train_vicuna_matched.py": "共享缓存bridge、手工对话与Vicuna前向",
"evaluate_models.py": "bridge热图与语言输出的统一评测",
"prompt_map_ablation.py": "固定global/问题的预测map交换诊断",
"AnomalyGPT_models.py": "局部特征投影与热图软提示结构快照",
"loss.py": "AnomalyGPT Focal与逐图Dice损失快照",
"train_seg.py": "DINO分割的数据、主干、解码器与损失",
"a2_language.py": "CLIP/随机锚点及辅助logit分支",
"train_a3.py": "缓存视觉特征与DINO的五路融合",
"extract_vlm_feats.py": "视觉缓存提取与固定随机降维",
"align_dino_llm.py": "历史DINO→64个视觉token→LLM探索",
"metrics_pixel.py": "全像素直方图与区域覆盖计分",
"stats_pixel.py": "逐图统计档案及组bootstrap",
"bootstrap_pixel.py": "两模型的配对组bootstrap",
"common.py": "配置、split路径及协议自洽校验",
"winclip_b2.py": "WinCLIP思路改写：窗口文本与DINO正常参考",
}
DESCRIPTIONS = {
"_mask_info":"读取索引mask，统计缺陷像素与归一化全缺陷外接框。",
"_validate_record":"核对图像、mask、标签、尺寸及EXIF约定。",
"_deduplicate":"按内容去重；冲突标注拒收。",
"instruction":"把来源转为图片/指令/目标答案。",
"split_records":"按产品类别和异常标签分层划分。",
"training_subsets":"构建预算匹配的训练来源子集。",
"audit_trainable_parameters":"核对可训练参数名称、数量、dtype均符合语言LoRA约定。",
"load_model":"精确枚举模块并加载NF4语言底座、BF16视觉及LoRA。",
"make_inputs":"原生processor、chat template、image grid与assistant-only labels。",
"set_training_mode":"设置底座与LoRA的训练/eval行为。",
"_read_rows":"读取并校验训练样本与任务字段。",
"_save":"保存adapter及运行元数据，不等于完整optimizer恢复。",
"_validate":"teacher-forcing CE诊断，不替代自由生成指标。",
"_last_logits":"对一个受检输入取logits用于adapter重载核对。",
"candidate_plan":"验证固定候选、token offsets及布尔值区间。",
"boolean_token_span":"布尔值必须由完整、不跨边界的token覆盖。",
"_score_image":"同图true/false两候选前向，只计布尔token似然。",
"normalized_score":"margin及稳定二候选归一化；不是概率校准。",
"ranking_metrics":"按相同分数组计算AUROC和非插值AP。",
"threshold_metrics":"按严格大于阈值规则构建混淆矩阵。",
"calibrate_normal_threshold":"从验证正常图选经验FPR对应的order statistic。",
"_run":"评分运行、身份校验与已有文件续跑约定。",
"parse_prediction":"解析JSON，当前允许围栏/浮点坐标/多框。",
"_summary":"分类、格式有效性及端到端框召回。",
"enclosing_box":"多个框压成一个包围矩形。",
"union_box_iou":"两个外接矩形IoU，不是mask或多矩形并集IoU。",
"build_artifacts":"冻结来源顺序、两次曝光任务和确认来源隔离。",
"_instruction":"构造第二轮固定任务措辞与目标。",
"QwenCachedModel.__init__":"移除原生visual，建立ImageBind缓存bridge与冻结Qwen语言端。",
"QwenCachedModel.prompt_wrap":"拼global、18个soft prompt和手工对话embedding。",
"QwenCachedModel._forward_lm":"构造[3,B,S]位置，causal shift后按有效目标算CE。",
"read_schedule":"核验显式batch文件列表、步数与batch约束。",
"reset_rng":"模型构建/审计后重置随机数状态。",
"VicunaCachedModel.forward_cached":"四层异常图、Focal/Dice、soft prompt及语言CE完整闭环。",
"bridge_fingerprint":"对指定bridge张量的身份生成hash。",
"build_one_instance":"手工###对话格式和助手监督mask。",
"process_batch_instance":"右padding，按整体文字序列上限截取。",
"collate":"按token-first局部缓存布局组合batch。",
"LinearLayer.forward":"转batch-first、去CLS并对四层局部特征做投影。",
"PromptLearner.__init__":"五次pooling和末层5×5卷积，定义基础提示。",
"PromptLearner.forward":"将预测热图编码为9动态＋9基础embedding。",
"FocalLoss.forward":"gamma=2、含平滑的双通道Focal损失。",
"BinaryDiceLoss.forward":"smooth=1，逐图算Dice后batch平均。",
"decode_prediction":"解析开头Yes/No；invalid单独记录。",
"map_and_prefix":"四层概率图平均，再构造语言前缀。",
"language_predictions":"KV cache生成、停止符与候选续写评分。",
"SwapPromptLearner":"交换预测map，保持自身global与问题。",
"SegDataset.__init__":"设置RGB归一化与共享几何变换参数。",
"SegDataset._load_mask_pil":"Real-IAD >127二值mask及nearest resize。",
"SegDataset._geom":"RGB与mask共享裁剪/翻转；插值方法分别选择。",
"SegDataset.__getitem__":"输出RGB、GT、路径、产品类别及视角标签。",
"check_alignment":"亮块与GT经过同一几何变换后的IoU自检。",
"StubFeatures":"随机卷积冒烟接口；不代表DINO表征。",
"DinoFeatures.forward":"冻结DINO第3/6/9/12层，去CLS后恢复空间网格。",
"SegDecoder.__init__":"1×1投影、3×3融合、GroupNorm/GELU与head。",
"SegDecoder.forward":"四层融合→上采样→单通道像素logits。",
"SegModel":"冻结视觉特征与可训练分割头的组合。",
"SegModel.forward":"decoder logits与语言aux相加；没有调用modulate。",
"focal_dice":"0.5 Focal＋0.5全batch Dice，eps=1e-6。",
"train":"配置驱动的DINO分割训练循环。",
"build_anchors":"CLIP EOS隐藏向量或固定随机方向对照。",
"LanguageBranch._sim":"DINO投影后与归一化文本锚点做余弦匹配。",
"LanguageBranch.forward":"异常/正常logsumexp差，再上采样aux logit。",
"LanguageBranch.modulate":"定义f×(1+gate)，当前主分割前向未调用。",
"VLMFeatStore":"mmap数组加parquet路径索引。",
"VLMFeatStore.__init__":"按需映射缓存并建立路径→数组行索引。",
"FusionDataset.__getitem__":"组合RGB/GT与缓存，缺失用零；空间变换同步需复核。",
"FusionDecoder.forward":"四路DINO加一路缓存投影融合。",
"build_projection":"固定随机1152→512降维，不是训练的语义对齐。",
"Projector.forward":"两次stride2把32²网格变成64个4096维token。",
"Aligner.generate":"每次重算完整视觉前缀和文本历史的生成。",
"PixelEvaluator.add":"全像素分箱统计，图内区域平均覆盖。",
"PixelEvaluator.compute":"池化排序、全局FPR与图平均PRO的积分。",
"_label_regions":"GT连通域；依赖缺失fallback会改变区域定义。",
"pooled_metrics_from_hist":"从256-bin前景/背景分布近似计算排序指标。",
"bootstrap":"按指定来源组重采样估计指标区间。",
"paired":"两模型共同组使用同次重采样索引。",
"resolve_splits":"按A/B/C及fold选择split文件。",
"check_protocol":"校验协议名、类型、fold和类别隔离。",
"read_visa":"读取官方CSV，建立统一来源记录。",
"balanced_subset":"选择固定规模的类别/标签均衡开发子集。",
"prepare_visa":"串联VisA读取、校验、划分、指令文件输出。",
"prepare_mvtec":"MVTec数据准备入口；不是本次主结果的数据来源。",
"candidate_audit":"验证手工对话续写的token边界与候选分段。",
"continuation_logp":"在给定embedding前缀下计算候选续写log likelihood。",
"aggregate":"按固定产品、标签与评测定义聚合结果。",
"token_budget_audit":"核对真实调度的曝光、来源、目标token和截断。",
"VicunaCachedModel.prompt_wrap":"拼接global、soft prompt、问题与答案embedding，屏蔽前缀监督。",
"VicunaCachedModel._forward_lm":"位置构造、causal shift及有效目标CE。",
"select_sources":"用标签冻结每产品2正常＋2异常诊断来源。",
"image_hist":"一张图前景/背景分数的全像素直方图。",
"image_pro":"一张图内连通域覆盖曲线统计。",
"PixelStats.add":"保存逐图统计与组、类别、面积等身份。",
"PixelStats.metrics":"从已有统计档案重算指标，无需重新推理。",
"PixelStats.by_class":"按产品类别分别汇总。",
"evaluate":"分割推理、sigmoid概率图与评测统计。",
"auroc":"图像分数的排序AUROC。",
"aupr":"PR曲线积分，需与原生非插值AP区分。",
"ece":"将概率分箱，比较预测概率与观察频率。",
"brier":"预测概率与二值真值的均方误差。",
"ClipWindowScorer.text_prototypes":"模板×状态词编码集成，形成正常/异常原型。",
"ClipWindowScorer.maps":"局部窗口CLS与原型匹配，多尺度融合。",
"ClipWindowScorer._agg":"重叠窗口调和聚合；未覆盖格子填1需注意语义。",
"_harmonic":"调和平均，较小输入影响更强。",
"DinoRef.build_memory":"K张正常图的patch特征展平为记忆库。",
"DinoRef.score":"每query取最大余弦匹配，转成0.5×(1−相似度)。",
}
EXTRAS = {
"industrial_vlm/src/industrial_vlm/data.py":["read_visa","balanced_subset","prepare_visa","prepare_mvtec"],
"industrial_vlm/src/industrial_vlm/score.py":["boolean_token_span"],
"industrial_vlm/src/industrial_vlm/evaluate.py":["main"],
"industrial_vlm/src/industrial_vlm/report.py":["main"],
"AnomalyGPT/followup_20261026/evaluate_models.py":["candidate_audit","continuation_logp","aggregate","main"],
"AnomalyGPT/followup_20261026/train_vicuna_matched.py":["token_budget_audit","VicunaCachedModel.prompt_wrap","VicunaCachedModel._forward_lm","main"],
"AnomalyGPT/followup_20261026/prompt_map_ablation.py":["select_sources"],
"seg_cross_part/stats_pixel.py":["image_hist","image_pro","PixelStats.add","PixelStats.metrics","PixelStats.by_class"],
"seg_cross_part/train_seg.py":["evaluate"],
"seg_cross_part/metrics.py":["auroc","aupr","ece","brier"],
}

def resolve(path, symbol):
    nodes=ast.parse((ROOT/path).read_text()).body
    for part in symbol.split("."):
        node=next((n for n in nodes if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==part),None)
        if node is None:
            raise ValueError(f"找不到 {path}::{symbol}")
        nodes=node.body
    return node

def link(path,symbol):
    n=resolve(path,symbol)
    return f"[{Path(path).name} · {symbol}]({os.path.relpath(ROOT/path,DOC)}#L{n.lineno})"

def main():
    refs=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    for md in DOC.glob("*.md"):
        s=md.read_text().replace("stats_pixel.py::PixelStats.bootstrap","stats_pixel.py::bootstrap")
        def replace(m):
            path,symbol=m.groups()
            refs.append({"chapter":md.name,"path":path,"symbol":symbol,"line":resolve(path,symbol).lineno})
            return link(path,symbol)
        s=re.sub(r"\[\[([^\]]+?)::([^\]]+?)\]\]",replace,s)
        md.write_text(s)
    refs=list({(r["chapter"],r["path"],r["symbol"]):r for r in refs}.values())
    for r in refs:
        r["line"]=resolve(r["path"],r["symbol"]).lineno
    MANIFEST.write_text(json.dumps(refs,ensure_ascii=False,indent=2)+"\n")
    grouped={}
    for r in refs:
        grouped.setdefault(r["path"],set()).add(r["symbol"])
    for path,symbols in EXTRAS.items():
        grouped.setdefault(path,set()).update(symbols)
    lines=["# 逐函数代码索引","","> 本索引由AST读取生成，不导入或运行模型；以代码基线146748a对应的当前工作区定位。覆盖概念章节引用的关键符号与补充入口，不遍历全部上游依赖。","",
    "按问题找模块，再看函数及调用者。forward负责张量运算；main负责配置、数据、循环和输出。Qwen、DINO、CLIP与PEFT内部架构在依赖库中，本索引定位本项目对它们的调用与审计。",""]
    count=0
    def route_order(item):
        path=item[0]
        rank=0 if path.startswith("industrial_vlm/src/") else 1 if path.startswith("industrial_vlm/") else 2 if path.startswith("AnomalyGPT/") else 3
        return rank,path
    for path,symbols in sorted(grouped.items(),key=route_order):
        title=MODULES.get(Path(path).name,"已有结果的评测/报告入口")
        if path=="seg_cross_part/metrics.py":
            title="图像指标与概率校准辅助"
        lines += [f"## {title}","",f"文件：[{path}]({os.path.relpath(ROOT/path,DOC)})","",
        "| 函数/类（当前行号） | 负责什么 |","|---|---|"]
        for symbol in sorted(symbols,key=lambda x:resolve(path,x).lineno):
            node=resolve(path,symbol)
            desc=DESCRIPTIONS.get(symbol)
            if symbol=="main":
                desc=title+"的命令行入口；本次未执行。"
            if desc is None:
                doc=ast.get_docstring(node)
                desc=doc.splitlines()[0] if doc else title+"中的 "+symbol+"；沿调用链阅读。"
            desc=desc.replace("|","/").replace("\n"," ")
            lines.append(f"| {link(path,symbol)}（L{node.lineno}） | {desc} |")
            count+=1
        lines.append("")
    lines += ["## 四条闭环阅读顺序","",
    "原生训练：prepare_visa/build_artifacts → _read_rows → load_model → make_inputs → forward/loss → backward/optimizer → _save。","",
    "原生评分：candidate_plan → _score_image → normalized_score → ranking_metrics/calibrate_normal_threshold。","",
    "桥接：collate → image_decoder → 四层相似度/像素损失 → PromptLearner → prompt_wrap → _forward_lm；Qwen复用Vicuna的forward_cached。","",
    "分割：SegDataset → DinoFeatures → SegDecoder（可加LanguageBranch.forward）→ focal_dice → PixelEvaluator/PixelStats。","",
    f"共定位{count}个关键类、函数或方法。行号随代码编辑可能移动，函数名是第二定位依据。",""]
    (DOC/"06_逐函数代码索引.md").write_text("\n".join(lines))
    print(json.dumps({"indexed_symbols":count,"chapter_code_references":len(refs)},ensure_ascii=False))

if __name__=="__main__":
    main()

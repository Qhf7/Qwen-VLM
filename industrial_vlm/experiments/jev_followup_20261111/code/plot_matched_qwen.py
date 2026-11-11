#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,socket,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
j=read(R/'stage08_rescue/metrics.json');q=read(R/'stage09_qwen_matched/metrics.json');qc=read(R/'stage09_qwen_matched/contract.json');qf=read(R/'stage09_qwen_matched/prediction_freeze.json')
assert j['status']==q['status']=='complete' and sha(R/'stage09_qwen_matched/scores.jsonl')==qf['scores_sha256']
assert q['image_size']==[448,448] and q['merged_visual_tokens']==196 and q['fitted_thresholds'] is False
scores=[json.loads(l) for l in (R/'stage09_qwen_matched/scores.jsonl').open() if l.strip()];assert len(scores)==240
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(18,7.2),gridspec_kw={'width_ratios':[1.2,1,1]})
names=['Jev 0','Jev 1\nseed43','Jev 1\nseed44','Jev 4\nseed43','Jev 4\nseed44','Qwen\nbase','Qwen\nDD LoRA'];arms=['shared0','s43_1','s44_1','s43_4','s44_4']
pool=[j['arms'][a]['metrics']['auroc'] for a in arms]+[q['arms'][a]['overall']['auroc'] for a in ['base','decision_decision']]
macro=[j['category_macro'][a]['auroc'] for a in arms]+[float(np.mean([v['auroc'] for v in q['arms'][a]['by_category'].values()])) for a in ['base','decision_decision']]
x=np.arange(7);w=.34;ax[0].bar(x-w/2,pool,w,color='#337A96',label='Pooled120');ax[0].bar(x+w/2,macro,w,color='#A0C5B5',label='Macro12 categories')
ax[0].set_xticks(x,names,fontsize=8);ax[0].set(ylim=(0,1.09),ylabel='AUROC',title='A  Same query PNG bytes, raw rank scores');ax[0].legend(fontsize=8,loc='upper left')
for i,v in enumerate(pool):ax[0].text(i-w/2,v+.015,f'{v:.3f}',ha='center',fontsize=7)
for arm,c,marker,label in [('shared0','#555555','x','Jev0'),('s43_1','#2F6EA3','o','Jev1 seed43'),('s44_1','#DC8C3B','o','Jev1 seed44'),('s43_4','#2F6EA3','s','Jev4 seed43'),('s44_4','#DC8C3B','s','Jev4 seed44')]:
 v=j['arms'][arm]['metrics'];ax[1].scatter(v['fpr'],v['recall'],c=c,marker=marker,s=75,label=label)
for arm,c,marker,label in [('base','#9474B0','D','Qwen base'),('decision_decision','#AE5874','P','Qwen DD')]:
 v=q['arms'][arm]['default_equal_candidate_threshold'];ax[1].scatter(v['fpr'],v['recall'],c=c,marker=marker,s=90,label=label)
ax[1].set(xlim=(-.02,.4),ylim=(.15,1),xlabel='FPR (60 normal denominator)',ylabel='Recall (60 anomaly denominator)',title='B  Fixed native working points');ax[1].legend(fontsize=7.7,ncol=2,loc='lower right')
cats=sorted(q['arms']['base']['by_category']);xx=np.arange(12);ww=.24
series=[('Jev0',[j['by_category'][c]['shared0']['auroc'] for c in cats],'#A3AAB3'),('Qwen base',[q['arms']['base']['by_category'][c]['auroc'] for c in cats],'#8DA3C4'),('Qwen DD',[q['arms']['decision_decision']['by_category'][c]['auroc'] for c in cats],'#A95977')]
for i,(name,vs,c) in enumerate(series):ax[2].bar(xx+(i-1)*ww,vs,ww,color=c,label=name)
ax[2].set_xticks(xx,cats,rotation=60,ha='right',fontsize=7);ax[2].set(ylim=(0,1.15),ylabel='Per-category AUROC (5 normal/5 anomaly)',title='C  Category heterogeneity remains');ax[2].legend(fontsize=7.5,loc='upper right')
fig.subplots_adjust(left=.055,right=.985,top=.85,bottom=.31,wspace=.29)
fig.text(.055,.97,'Matched query PNG control: Jev and Qwen3-VL base versus DD LoRA',fontsize=15,weight='bold',va='top')
cap='All120 previously viewed development sources; Qwen uses exact Jev448x448 query PNG, one image, zero normal references.\nJev1/4shot additionally receive normal images. Qwen:196 merged visual tokens, native language-only NF4/BF16vision; two candidate forward passes.\nPixels/images are matched only for Jev0 vs Qwen; internal tokens,FLOPs,question protocol and normalized-score semantics still differ.\nPanelB:Qwen strict logodds>0, Jev p>=0.5. No fitted threshold/calibration. Native original-image Qwen94.56% is separate historical evidence.'
fig.text(.055,.125,cap,va='top',fontsize=9,color='#444444',linespacing=1.6)
fd=R/'figures';png=fd/'support_qwen_matched_comparison.png';svg=fd/'support_qwen_matched_comparison.svg';fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
prov={'schema':'jev-support-matched-qwen-figure-v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'script_sha256':sha(Path(__file__)),'inputs_sha256':{str(p.relative_to(R)):sha(p) for p in [R/'stage08_rescue/metrics.json',R/'stage09_qwen_matched/metrics.json',R/'stage09_qwen_matched/contract.json',R/'stage09_qwen_matched/prediction_freeze.json',R/'stage09_qwen_matched/scores.jsonl']},'outputs_sha256':{p.name:sha(p) for p in [png,svg]},'pooled_auroc':dict(zip(names,pool)),'category_macro_auroc':dict(zip(names,macro)),'qwen_prompt_tokens':{'minimum':min(r['prompt_tokens'] for r in scores),'maximum':max(r['prompt_tokens'] for r in scores),'merged_visual_tokens':196},'same_query_png_note':'exact bytes, not internal token/FLOP equality; references addimages for Jevfewshot','historical_native_control_separate':True,'confirmation_opened':False,'new_api_requests':0}
p=fd/'support_qwen_matched_comparison.provenance.json';assert not p.exists()
with p.open('x') as f:f.write(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
md='''# 匹配448 PNG的Jev/Qwen对照：图表说明

本图来自已冻结stage08完整600与stage09完整Qwen240。Qwen base/DD LoRA逐样本使用与Jev0相同的448×448 PNG字节，单图零参考；统一来源并核原图与PNG哈希。Jev1/4shot还加入正常参考，因此它们不属于与Qwen相同图像数的控制。两套模型仍有不同提示、概率/似然语义、图像编码器、tokens和计算量；相同像素不等于相同FLOPs。Qwen在此固定196 merged视觉tokens，语言层NF4/BF16视觉、每图true/false两次候选forward。

| 模型/输入 | pooled AUROC | 类别macro AUROC | 原生Recall | 原生FPR |
|---|---:|---:|---:|---:|
'''
for arm,n in [('base','Qwen base matched448'),('decision_decision','Qwen DD LoRA matched448')]:
 v=q['arms'][arm];mac=np.mean([x['auroc'] for x in v['by_category'].values()]);t=v['default_equal_candidate_threshold'];md+=f"| {n} | {v['overall']['auroc']:.6f} | {mac:.6f} | {t['recall']:.6f} | {t['fpr']:.6f} |\n"
md+='''
匹配448后DD比base AUROC高11.04个百分点；原生严格logodds>0工作点DD Recall71.67%、FPR5%，base Recall23.33%、FPR0。它说明模型更新改善排序，但固定工作点和置信度仍需要独立校准检验；不能把FPR5%写为未来5%保证。

历史native原图控制是base82.39%/DD94.56%，stage09匹配448控制是82.44%/93.49%。两者输入预算不同，应并列清楚，不能用一个实验的base减另一个实验的adapter。类别macro与pooled也有不同定义，不互换。pcb3两臂在本5+5小样本上表现差，显示类别异质性；不据此在当前集调提示或挑类别。

原生Qwen决策严格logodds>0，Jev使用p>=0.5；等号处理不同已明示。所有120为已查看开发来源，不是confirmation或盲测。图为support_qwen_matched_comparison.png/.svg，完整哈希见同名provenance JSON；历史native图保留为support_qwen_comparison。
'''
p=R/'stage09_qwen_matched/FIGURE_ANALYSIS.md';assert not p.exists()
with p.open('x') as f:f.write(md)
print(json.dumps({'status':'complete','png':str(png),'pooled':dict(zip(names,pool)),'confirmation_opened':False},ensure_ascii=False))


#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,socket,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111');S=R/'stage10_resolution';m=json.loads((S/'metrics.json').read_text());a=json.loads((R/'stages/10_postexecution_audit.json').read_text())
assert socket.gethostname()=='autodl-container-132444827f-9c594038' and a['failed']==0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert a['metrics_sha256']==sha(S/'metrics.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(17.4,7),gridspec_kw={'width_ratios':[1,1.2,1]});arms=['size320_0shot','shared0_448'];colors=['#DA8B40','#2D7794'];x=np.arange(2)
for j,(kind,label,c) in enumerate([('pooled','Pooled120','#2D7794'),('macro','Category macro','#95BDAA')]):
 vals=[m['arms'][arm]['metrics']['auroc'] if kind=='pooled' else m['category_macro'][arm]['auroc'] for arm in arms]
 ax[0].bar(x+(j-.5)*.32,vals,.32,label=label,color=c)
 for xi,v in zip(x,vals):ax[0].text(xi+(j-.5)*.32,v+.015,f'{v:.3f}',ha='center',fontsize=9)
ax[0].set(xticks=x,xticklabels=['320 pixels','448 pixels'],ylabel='AUROC',ylim=(0,1.08),title='A  Ranking under a smaller image budget');ax[0].legend(fontsize=8,loc='upper left')
ks=['auroc','recall','fpr','brier'];labels=['AUROC','Recall','FPR','Brier'];yy=np.arange(4)
for i,k in enumerate(ks):
 v=m['paired_bootstrap']['pooled'][k];lo,hi=v['ci95'];ax[1].errorbar(v['delta'],i,xerr=[[v['delta']-lo],[hi-v['delta']]],fmt='o',capsize=4,color='#4F667F')
 ax[1].text(.98,i,f"{v['delta']:+.4f} [{lo:+.4f},{hi:+.4f}]",transform=ax[1].get_yaxis_transform(),ha='right',va='center',fontsize=8)
ax[1].axvline(0,color='#888',linestyle='--',linewidth=1);ax[1].set(yticks=yy,yticklabels=labels,xlim=(-.25,.22),ylim=(3.6,-.7),xlabel='320 minus448 (raw metric units)',title='B  Paired source bootstrap95% intervals')
for i,arm in enumerate(arms):
 v=m['arms'][arm]['metrics'];ax[2].scatter(v['fpr'],v['recall'],s=120,color=colors[i],label=arm)
 ax[2].annotate('320' if i==0 else '448',(v['fpr'],v['recall']),xytext=(7,8),textcoords='offset points')
ax[2].set(xlim=(.1,.42),ylim=(.5,.9),xlabel='FPR (60 normal denominator)',ylabel='Recall (60 anomaly denominator)',title='C  Fixed p>=0.5 operating tradeoff')
fig.subplots_adjust(left=.06,right=.985,top=.85,bottom=.31,wspace=.35)
fig.text(.06,.97,'320 versus448: lower token use with a recall tradeoff on the same development sources',fontsize=15,weight='bold',va='top')
b=m['arms']['size320_0shot']['budget'];r=m['arms']['shared0_448']['budget'];ratio=b['observed_usage']['input_tokens']['observed_sum']/r['observed_usage']['input_tokens']['observed_sum']
cap=f'Approved viewed development120, no confirmation. Both whole-image0shot; identical source SHA/model/question and Pillow12.3 renderer.\nNew320120 accepted/120started/0fail;448shared0120 reused from amended complete600, excluding support480.\n320/448 canvas ratio0.5102; observed input-token ratio{ratio:.4f} (86,785 /113,663); different ratios, no exact pixel-to-token or currency formula.\n448 collected earlier and502rescue later; provider/time randomness unmeasured. AUROC95% interval crosseszero; no pure resolution-causality claim.'
fig.text(.06,.125,cap,va='top',fontsize=9,color='#444444',linespacing=1.6)
fd=R/'figures';png=fd/'resolution_tradeoff.png';svg=fd/'resolution_tradeoff.svg';fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
prov={'schema':'jev-resolution-figure-provenance-v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'script_sha256':sha(Path(__file__)),'inputs_sha256':{str(p.relative_to(R)):sha(p) for p in [S/'metrics.json',S/'prediction_freeze.json',S/'evaluation_contract_amended_reference.json',R/'stages/10_postexecution_audit.json']},'outputs_sha256':{p.name:sha(p) for p in [png,svg]},'input_token_ratio320_over448':ratio,'canvas_pixel_ratio':320**2/448**2,'confirmation_opened':False,'new_api_requests':0}
p=fd/'resolution_tradeoff.provenance.json';assert not p.exists()
with p.open('x') as f:f.write(json.dumps(prov,indent=2)+'\n')
v=m['paired_bootstrap']['pooled']['auroc'];doc=f'''# Stage10独立分析：320/448零参考分辨率对照

独立审计4498项通过、0失败，包含完整stage08救援来源链、共享query对齐、仅shared0参考120、独立指标与2000次配对分层bootstrap。新320共120 accepted/120 started，无失败与unknown；448参考来自完整救援证书，其他480支持臂不混入。

320 AUROC=79.04%，448=83.46%，差-4.42个百分点；95%配对区间[{100*v['ci95'][0]:+.2f},{100*v['ci95'][1]:+.2f}]个百分点跨0，不能宣称该AUROC差异已被稳定确认。固定0.5阈值下，320 Recall由78.33%降至66.67%（少检出7/60异常），FPR由35%降至21.67%（少误报8/60正常）。这是漏检/误报同时变化，不是无代价提升。Brier320=0.18009，比448的0.17294更高；macroF1略升主要反映当前固定工作点的权衡。

320画布像素约448的51.02%；实际返回input_tokens总量86,785/113,663，比例{ratio:.4f}，约减少{100*(1-ratio):.2f}%。两种比例不同，证明不能把像素缩放直接当精确token或账单节省。120响应均有token usage；输出token总量两臂均2520。平均延时约3.71s/4.09s，但非交错采集，服务时间变化与额外502救援可能影响差异，不能归因于纯分辨率因果。

这是已查看开发120的消融，不是盲测或跨域泛化结论。下一步按既定系统选择规则维持448，完成固定4shot的来源隔离校准/留出研究；不从这张图临时调阈值或重选新样本。confirmation继续封存。

图：figures/resolution_tradeoff.png / .svg；溯源JSON同名.provenance.json。
'''
p=S/'INDEPENDENT_ANALYSIS.md';assert not p.exists()
with p.open('x') as f:f.write(doc)
print(json.dumps({'status':'complete','png':str(png),'analysis':str(p),'input_token_ratio':ratio,'auroc_delta_ci':v,'confirmation_opened':False},ensure_ascii=False,indent=2))


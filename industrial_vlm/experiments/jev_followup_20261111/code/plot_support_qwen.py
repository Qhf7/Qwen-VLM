#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,socket,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
m=read(R/'stage08_rescue/metrics.json');audit=read(R/'stages/08_postexecution_audit.json');cal=read(R/'stages/07_calibration_results.json');ev=read(R/'stages/05_evaluation.json')
assert m['status']=='complete' and audit['status']=='complete' and audit['failed']==0 and audit['metrics_sha256']==sha(R/'stage08_rescue/metrics.json')
native={};qp=[]
for a,n in [('qwen_base','base'),('qwen_decision_decision','decision_decision')]:
 p=R/ev['qwen_scorer_control']['models'][n]['file'];qp.append(p);assert sha(p)==ev['qwen_scorer_control']['models'][n]['sha256']
 rr=[json.loads(l) for l in p.open() if l.strip()]
 native[a]={'strict_tp':sum(r['label'] and r['anomaly_log_odds']>0 for r in rr),'strict_fp':sum(not r['label'] and r['anomaly_log_odds']>0 for r in rr),'ties':sum(r['anomaly_log_odds']==0 for r in rr),'n':len(rr),'prompt_tokens':{'min':min(r['prompt_tokens'] for r in rr),'max':max(r['prompt_tokens'] for r in rr),'mean':float(np.mean([r['prompt_tokens'] for r in rr]))},'candidate_input_tokens_example':rr[0]['candidate_input_tokens']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(18,7.4),gridspec_kw={'width_ratios':[1.1,1.25,1.]})
color={'43':'#256AA6','44':'#DC8D39'}
base=m['arms']['shared0']['metrics']['auroc'];bm=m['category_macro']['shared0']['auroc']
for seed in ['43','44']:
 vals=[base]+[m['arms']['s'+seed+'_'+str(n)]['metrics']['auroc'] for n in [1,4]]
 mac=[bm]+[m['category_macro']['s'+seed+'_'+str(n)]['auroc'] for n in [1,4]]
 axs[0].plot([0,1,4],vals,'o-',color=color[seed],label='Seed'+seed+' pooled',linewidth=2)
 axs[0].plot([0,1,4],mac,'s--',color=color[seed],label='Seed'+seed+' category macro',linewidth=1.4)
axs[0].set(xlabel='Normal-reference shots (nested 1 within 4)',ylabel='AUROC',xticks=[0,1,4],ylim=(.75,1.005),title='A  Two fixed banks, same query sources')
axs[0].legend(loc='lower right',fontsize=8)
axs[0].text(.04,.89,'0-shot shared by both seeds\nY-axis shown from 0.75',transform=axs[0].transAxes,fontsize=8,color='#444444')
names=['Jev 0','Jev 1\nseed43','Jev 1\nseed44','Jev 4\nseed43','Jev 4\nseed44','Qwen\nbase','Qwen\nDD LoRA']
arms=['shared0','s43_1','s44_1','s43_4','s44_4']
pool=[m['arms'][a]['metrics']['auroc'] for a in arms]+[cal['models'][a]['raw']['pooled']['auroc'] for a in ['qwen_base','qwen_decision_decision']]
macro=[m['category_macro'][a]['auroc'] for a in arms]+[cal['models'][a]['raw']['category_macro']['auroc'] for a in ['qwen_base','qwen_decision_decision']]
x=np.arange(len(names));w=.34
axs[1].bar(x-w/2,pool,w,color='#2D718C',label='Pooled120');axs[1].bar(x+w/2,macro,w,color='#91B9AA',label='Macro across12 categories')
axs[1].set_xticks(x,names,fontsize=8);axs[1].set(ylabel='AUROC',ylim=(0,1.1),title='B  Same sources, different input budgets')
axs[1].legend(fontsize=8,loc='upper left')
for i,v in enumerate(pool):axs[1].text(i-w/2,v+.016,f'{v:.3f}',ha='center',fontsize=7)
for seed,marker in [('43','o'),('44','s')]:
 for n in [1,4]:
  v=m['arms']['s'+seed+'_'+str(n)]['metrics'];axs[2].scatter(v['fpr'],v['recall'],c=color[seed],marker=marker,s=95 if n==4 else 45,edgecolor='white',label=seed+'/'+str(n)+'shot')
v=m['arms']['shared0']['metrics'];axs[2].scatter(v['fpr'],v['recall'],color='#555555',marker='x',s=80,label='0shot')
for a,marker,c,label in [('qwen_base','D','#8C72B0','Qwen base'),('qwen_decision_decision','P','#A14F6C','Qwen DD')]:
 n=native[a];axs[2].scatter(n['strict_fp']/60,n['strict_tp']/60,marker=marker,c=c,s=85,label=label)
axs[2].set(xlim=(-.025,.4),ylim=(.2,1.02),xlabel='FPR (normal denominator60)',ylabel='Recall (anomaly denominator60)',title='C  Native fixed decision rules')
axs[2].legend(loc='lower right',fontsize=8,ncol=2)
fig.subplots_adjust(left=.055,right=.985,top=.86,bottom=.31,wspace=.29)
fig.text(.055,.97,'Whole-image anomaly judgment: support robustness and source-matched Qwen controls',fontsize=15,weight='bold',va='top')
cap='Viewed development120 (60 normal/60 anomaly), never confirmation. Jev:448x448, same renderer/question; shared0 + fixed seeds43/44,1shot nested.\nQwen raw conditional true/false likelihood (base / supervised DD LoRA): native max_pixels524288 and tokenizer-specific prompt/candidate lengths.\nVisual/token/compute budgets differ; this is not a shots-only ablation. PanelC Qwen uses strict logodds>0; Jev uses p>=0.5. AUROC includes score ties.\nSupport600 accepted/699 HTTP starts/99 explicit502; three rescue successes. Fixed-bank intervals exclude provider and support-selection variance.'
fig.text(.055,.13,cap,va='top',fontsize=9,linespacing=1.6,color='#444444')
fd=R/'figures';png=fd/'support_qwen_comparison.png';svg=fd/'support_qwen_comparison.svg'
fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
inputs=[R/'stage08_rescue/metrics.json',R/'stage08_rescue/completion_certificate.json',R/'stages/08_postexecution_audit.json',R/'stages/07_calibration_results.json',R/'stages/05_evaluation.json']+qp
prov={'schema':'jev-support-qwen-figure-provenance-v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'script_sha256':sha(Path(__file__)),'inputs_sha256':{str(p.relative_to(R)):sha(p) for p in inputs},'outputs_sha256':{p.name:sha(p) for p in [png,svg]},'pooled_auroc':dict(zip(names,pool)),'category_macro_auroc':dict(zip(names,macro)),'native_qwen_rules':native,'qwen_internal_budget_caveat':'Qwen maxpixels524288 and conditional sequence lengths vs Jev448 perasset/provider tokens are not equivalent; differing internal patch/token encoders','confirmation_opened':False,'new_api_requests':0,'rgb_opened':False,'gt_masks_opened':False}
p=fd/'support_qwen_comparison.provenance.json';assert not p.exists()
with p.open('x') as f:f.write(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'complete','png':str(png),'native_qwen_rules':native,'figure_sha256':sha(png),'new_api_requests':0},ensure_ascii=False,indent=2))


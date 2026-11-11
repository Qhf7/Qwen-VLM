#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,socket,datetime,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111');S=R/'stage12/run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
m=read(S/'metrics.json');a=read(R/'stages/12_postexecution_audit.json');f=read(S/'calibrator.json');sup=read(S/'STATISTICAL_SUPPLEMENT.json');c=read(S/'contract.json')
assert m['status']==a['status']=='complete' and a['failed']==0 and a['metrics_sha256']==sha(S/'metrics.json')
pred={r['source_id']:r for r in [json.loads(l) for l in (S/'predictions.jsonl').open() if l.strip()]};rr=[r for r in c['records'] if r['role']=='heldout_eval'];labels={r['source_id']:r['is_anomalous'] for r in [json.loads(l) for l in (R/'stage12_calibration_plan/heldout_eval_labels.jsonl').open() if l.strip()]}
y=np.array([int(labels[r['source_id']]) for r in rr]);raw=np.array([pred[r['source_id']]['probability'] for r in rr]);clip=np.clip(raw,1e-6,1-1e-6);ca=f['coefficient'][0][0];cb=f['intercept'][0];cal=1/(1+np.exp(-(ca*np.log(clip/(1-clip))+cb)))
threshold=1/(1+math.exp(cb/ca))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(17.6,7.2),gridspec_kw={'width_ratios':[1.1,1.1,1.]});colors={'raw':'#46779F','calibrated':'#269986'};reliability={}
ax[0].plot([0,1],[0,1],'--',color='#999999',linewidth=1)
for name,p in [('raw',raw),('calibrated',cal)]:
 bins=np.minimum((p*10).astype(int),9);cells=[]
 for b in range(10):
  ix=bins==b
  if ix.any():cells.append(dict(bin=b,n=int(ix.sum()),mean_probability=float(p[ix].mean()),anomaly_fraction=float(y[ix].mean())))
 reliability[name]=cells
 ax[0].plot([v['mean_probability'] for v in cells],[v['anomaly_fraction'] for v in cells],'o-',color=colors[name],label=name)
 for v in cells:ax[0].scatter(v['mean_probability'],v['anomaly_fraction'],s=15+v['n']*3,color=colors[name],alpha=.55)
ax[0].set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean model probability in bin',ylabel='Observed anomaly fraction',title='A  Heldout reliability (10 fixed bins)');ax[0].legend(fontsize=9,loc='upper left')
ks=['brier','ece_10_equal_width','binary_logloss'];xx=np.arange(3);w=.34
for j,name in enumerate(['raw','calibrated']):
 vals=[m['heldout'][name]['metrics']['brier'],m['heldout'][name]['metrics']['ece_10_equal_width'],sup['methods'][name]['binary_logloss']]
 bars=ax[1].bar(xx+(j-.5)*w,vals,w,color=colors[name],label=name)
 for bar,v in zip(bars,vals):ax[1].text(bar.get_x()+bar.get_width()/2,v+.01,f'{v:.3f}',ha='center',fontsize=8)
ax[1].set(xticks=xx,xticklabels=['Brier','ECE10','Logloss'],ylabel='Loss / calibration error (lower is better)',ylim=(0,.48),title='B  Proper losses improve; ECE CI crosses0');ax[1].legend(fontsize=9,loc='upper left')
for name in ['raw','calibrated']:
 cov=[.5,.75,1.];risk=[m['heldout'][name]['coverage'][str(x)]['error_rate'] for x in cov]
 ax[2].plot(cov,risk,'o-',color=colors[name],label=name)
 for x,v in zip(cov,risk):ax[2].annotate(f'{100*v:.1f}%',(x,v),textcoords='offset points',xytext=(0,8 if name=='raw' else -14),ha='center',fontsize=8)
ax[2].set(xlim=(.45,1.05),ylim=(0,.2),xticks=[.5,.75,1.],xlabel='Predeclared retained coverage',ylabel='Classification error among retained',title='C  Fixed coverage; abstention is not normal');ax[2].legend(loc='upper left',fontsize=9)
fig.subplots_adjust(left=.06,right=.985,top=.85,bottom=.31,wspace=.3)
fig.text(.06,.97,'Source-disjoint calibration120 / heldout120: fixed Jev4-shot seed43 at448',fontsize=15,weight='bold',va='top')
cap='System selected before new responses. All240 predictions frozen, then logistic fit only calibration120; coefficients frozen before heldout labels.\nHeldout:AUROC92.82% unchanged; Recall53/60 unchanged; FP11/60 to6/60. Brier delta -0.01649 (95%CI -0.02366,-0.00872).\nECE delta -0.04791 (95%CI -0.07082,+0.00843) crosses0; FPR6/60 Wilson95%[4.66%,20.15%]. No low-FPR deployment guarantee.\nNative-validation remainder may have historical exposure, not blind/confirmation. Balanced50% anomaly prior; fit/provider uncertainty excluded.'
fig.text(.06,.13,cap,va='top',fontsize=9,color='#444444',linespacing=1.6)
fd=R/'figures';png=fd/'calibration_heldout_analysis.png';svg=fd/'calibration_heldout_analysis.svg';fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
prov={'schema':'jev-stage12-calibration-figure-v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'script_sha256':sha(Path(__file__)),'inputs_sha256':{str(p.relative_to(R)):sha(p) for p in [S/'metrics.json',S/'calibrator.json',S/'STATISTICAL_SUPPLEMENT.json',R/'stages/12_postexecution_audit.json',S/'predictions.jsonl',S/'contract.json',R/'stage12_calibration_plan/heldout_eval_labels.jsonl']},'outputs_sha256':{p.name:sha(p) for p in [png,svg]},'reliability_bins':reliability,'raw_equivalent_threshold_for_calibrated_half':threshold,'confirmation_opened':False,'new_api_requests':0}
p=fd/'calibration_heldout_analysis.provenance.json';assert not p.exists()
with p.open('x') as f:f.write(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
md=f'''# Stage12留出结果独立解释

指定服务器完成240预测/251started/11纯502，unknown0；独立审计1550项通过、0失败。先冻结预测，使用calibration120拟合固定logit-logistic，冻结系数后才访问heldout120标签。参数a={ca:.12f}、b={cb:.12f}，映射为p_cal=sigmoid(a*logit(clip(p_raw,1e-6,1-1e-6))+b)。这是单调映射，所以本数据AUROC/AP不变：92.82%/94.24%。它调整概率与固定决策边界，而非改模型排序。

校准后p_cal>=0.5等价于p_raw约>={threshold:.6f}，该数值由calibration拟合系数推导，没有在heldout调阈值。heldout异常53/60检出不变；正常误报由11/60降到6/60。FPR配对差-8.33pp，95%区间[-15,-3.33]pp。calibrated FPR自身Wilson95%区间[4.66%,20.15%]，不能保证未来10%或5%。

Brier从0.12423降到0.10774，配对95%差区间[-0.02366,-0.00872]；Logloss从0.40479降到0.34859，差区间[-0.07614,-0.03383]。这两项支持固定映射在本留出来源上改善概率质量。ECE10从0.15833降到0.11042，但差区间[-0.07082,+0.00843]跨0，不能把点估计下降直接写成稳定ECE收益。ECE分箱对120小样本敏感，应与proper loss一起看。

固定coverage50%/75%下raw和calibrated条件错误率分别同为3.33%/8.89%；100%覆盖错误率由15%降到10.83%。拒答不当正常，coverage是在这份留出集上按预声明置信排序取得固定比例，不保证未来部署coverage，也没有重新选择最优拒答阈值。

这是来源互斥的新采样native validation remainder研究，不是盲测或最终confirmation。每类只有5正常/5异常，平衡50%异常先验不同于产线；bootstrap固定本次calibrator，未包含calibration拟合或API/provider变化。后续最终确认必须保持唯一system/映射/阈值/拒答规则冻结，不能依据这轮留出再换prompt、参考、size或shots。

图为figures/calibration_heldout_analysis.png/.svg，逐bin/provenance和完整STATISTICAL_SUPPLEMENT均已保存。confirmation在本阶段未读。
'''
p=S/'INDEPENDENT_ANALYSIS.md';assert not p.exists()
with p.open('x') as f:f.write(md)
print(json.dumps({'status':'complete','png':str(png),'threshold':threshold,'analysis':str(p),'confirmation_opened':False},ensure_ascii=False))


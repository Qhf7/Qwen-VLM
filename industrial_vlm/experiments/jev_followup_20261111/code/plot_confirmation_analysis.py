#!/usr/bin/env python3
"""Server-only plot of complete, audited, fixed confirmation aggregates."""
from pathlib import Path
import json,hashlib,socket,datetime,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111');S=R/'stage13_confirmation';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
m=read(S/'metrics.json');a=read(R/'stages/13_postexecution_audit.json');sup=read(S/'STATISTICAL_SUPPLEMENT.json');bundle=read(S/'final_system_bundle.json');cert=read(S/'completion_certificate.json')
assert m['status']==a['status']==cert['status']=='complete' and a['failed']==0 and a['metrics_sha256']==sha(S/'metrics.json') and cert['metrics_sha256']==sha(S/'metrics.json')
names=['secondary_raw','primary_calibrated'];labels={'secondary_raw':'raw secondary','primary_calibrated':'calibrated primary'};colors={'secondary_raw':'#46779F','primary_calibrated':'#269986'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(17.6,7.5),gridspec_kw={'width_ratios':[1.1,1.1,1.]});x=np.arange(4);width=.35
keys=['auroc','ap','recall','fpr']
for j,name in enumerate(names):
 vals=[m['results'][name]['metrics'][k] for k in keys];bars=ax[0].bar(x+(j-.5)*width,vals,width,color=colors[name],label=labels[name]);cis=[m['bootstrap']['intervals'][name][k]['ci95'] for k in keys];ax[0].errorbar(x+(j-.5)*width,vals,yerr=np.array([[v-ci[0] for v,ci in zip(vals,cis)],[ci[1]-v for v,ci in zip(vals,cis)]]),fmt='none',ecolor='#333333',capsize=3,linewidth=.8)
 for bar,v in zip(bars,vals):ax[0].text(bar.get_x()+bar.get_width()/2,min(v+.04,1.045),f'{v:.3f}',ha='center',fontsize=8)
ax[0].set(xticks=x,xticklabels=['AUROC','AP','Recall','FPR'],ylim=(0,1.13),ylabel='Metric (fixed threshold for Recall/FPR)',title='A  Confirmation240; primary remains fixed');ax[0].legend(loc='upper left',fontsize=8)
ks=['brier','ece_10_equal_width','logloss'];d=sup['paired_primary_minus_secondary']['pooled'];dv=[d[k]['delta'] for k in ks[:2]]+[sup['paired_logloss_difference']['pooled']['delta']];ci=[d[k]['ci95'] for k in ks[:2]]+[sup['paired_logloss_difference']['pooled']['ci95']]
ax[1].axvline(0,color='#999999',linestyle='--',linewidth=1)
for i,(v,z) in enumerate(zip(dv,ci)):
 ax[1].errorbar(v,i,xerr=[[v-z[0]],[z[1]-v]],fmt='o',color='#269986',capsize=4,linewidth=2);ax[1].annotate(f'{v:+.4f}',(v,i),textcoords='offset points',xytext=(0,11),ha='center',fontsize=8)
ax[1].set(yticks=np.arange(3),yticklabels=['Brier','ECE10','Logloss'],ylim=(2.5,-.5),xlabel='Primary minus raw; negative favors mapping',title='B  Paired95% intervals; fixed calibrator')
for name in names:
 cov=[.5,.75,1.];v=[sup['coverage'][name][str(c)]['conditional']['error'] for c in cov];risk=[z['estimate'] for z in v];ax[2].plot(cov,risk,'o-',color=colors[name],label=labels[name]);ax[2].errorbar(cov,risk,yerr=[[z['estimate']-z['ci95'][0] for z in v],[z['ci95'][1]-z['estimate'] for z in v]],fmt='none',ecolor=colors[name],capsize=3,alpha=.6)
 for c,z in zip(cov,risk):ax[2].annotate(f'{100*z:.1f}%',(c,z),textcoords='offset points',xytext=(0,8 if name=='secondary_raw' else -14),ha='center',fontsize=8)
ax[2].set(xlim=(.45,1.05),xticks=[.5,.75,1.],ylim=(0,min(1,max(.2,max(z['ci95'][1] for name in names for z in [sup['coverage'][name][str(c)]['conditional']['error'] for c in [.5,.75,1.]])+0.04))),xlabel='Predeclared retained coverage',ylabel='Conditional error (Wilson95%)',title='C  Fixed coverage; abstention is not normal');ax[2].legend(loc='upper left',fontsize=8)
fig.subplots_adjust(left=.06,right=.985,top=.84,bottom=.31,wspace=.34)
fig.text(.06,.97,'One-time confirmation240: Jev4-shot seed43 at448, frozen calibration120 mapping',fontsize=15,weight='bold',va='top')
primary=m['results']['primary_calibrated']['metrics'];raw=m['results']['secondary_raw']['metrics'];cm=sup['methods']['primary_calibrated']['confusion'];fpr=sup['methods']['primary_calibrated']['fpr'];rec=sup['methods']['primary_calibrated']['recall'];op=m['operations']
cap=f"Final policy before stage12 outcomes; exact bundle before one-time input opening. All240 predictions and mapped outputs frozen before final label evaluation.\nPrimary:AUROC{100*primary['auroc']:.2f}%, Recall{cm['TP']}/120, FP{cm['FP']}/120; primary FPR Wilson95%[{100*fpr['ci95'][0]:.2f}%,{100*fpr['ci95'][1]:.2f}%].\n{op['accepted']}accepted/{op['started']}started/{op['failed']}failed;unknown0. Pairedbootstrap2000: category x label, fixed calibrator; no fit/provider uncertainty.\nBalanced50% anomaly prior; one fixed sample, no crossdomain or deployment guarantee. Source/byte disjointness does not prove physical-part independence."
fig.text(.06,.13,cap,va='top',fontsize=9,color='#444444',linespacing=1.6)
fd=R/'figures';png=fd/'confirmation_fixed_system_analysis.png';svg=fd/'confirmation_fixed_system_analysis.svg';assert not png.exists() and not svg.exists();fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
prov=dict(schema='jev-stage13-fixed-confirmation-figure-v1',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),script_sha256=sha(Path(__file__)),inputs_sha256={str(p.relative_to(R)):sha(p) for p in [S/'metrics.json',S/'STATISTICAL_SUPPLEMENT.json',R/'stages/13_postexecution_audit.json',S/'final_system_bundle.json',S/'completion_certificate.json']},outputs_sha256={p.name:sha(p) for p in [png,svg]},protected_labels_or_legacy_read=False,new_api_requests=0)
with (fd/'confirmation_fixed_system_analysis.provenance.json').open('x') as f:f.write(json.dumps(prov,ensure_ascii=False,indent=2)+chr(10))
ca=bundle['calibrator']['coefficient'][0][0];cb=bundle['calibrator']['intercept'][0];threshold=1/(1+math.exp(cb/ca));lines=['# Stage13最终确认结果独立解释','',f"指定服务器在最终系统冻结后一次性打开原固定240输入，{op['accepted']}accepted/{op['started']}started/{op['failed']}failed、unknown0。全部预测和固定映射冻结后才由最终评价器读取boolean标签；独立审计{a['passed']}项通过、0失败。",'',f"主系统在stage12结果前预声明为Jev1.14、4正常参考、seed43、448像素与calibration120固定logistic映射。a={ca:.12f}、b={cb:.12f}；p_cal≥0.5相当于raw约≥{threshold:.6f}，不在本确认集优化阈值。raw只作同一次输出的次要诊断，不因结果优劣切换主系统。",'', '| 指标 | raw次要 | calibrated主系统 | 主−raw及95%配对区间 |','|---|---:|---:|---|']
for k in ['auroc','ap','macro_f1','recall','fpr','brier','ece_10_equal_width']:
 v=d[k];lines.append(f"| {k} | {raw[k]:.6f} | {primary[k]:.6f} | {v['delta']:+.6f} [{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}] |")
v=sup['paired_logloss_difference']['pooled'];lines.append(f"| logloss | {sup['methods']['secondary_raw']['binary_logloss']:.6f} | {sup['methods']['primary_calibrated']['binary_logloss']:.6f} | {v['delta']:+.6f} [{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}] |")
lines+=['',f"主系统混淆矩阵TP={cm['TP']}、FP={cm['FP']}、TN={cm['TN']}、FN={cm['FN']}。Recall={100*rec['estimate']:.2f}%、Wilson95%[{100*rec['ci95'][0]:.2f}%,{100*rec['ci95'][1]:.2f}%]；FPR={100*fpr['estimate']:.2f}%、Wilson95%[{100*fpr['ci95'][0]:.2f}%,{100*fpr['ci95'][1]:.2f}%]。有限样本FPR点估计不能当成未来部署上界。",'','正斜率单调映射保持排序，因此应将分类阈值/可靠性收益与AUROC/AP区分。配对区间包含0的差异不能仅由点估计宣称稳定改善；负结果不用于换系统。完整逐类与category-macro差异及区间保留在STATISTICAL_SUPPLEMENT.json。','','固定coverage50%/75%/100%是预声明置信排序的回顾分析；拒答不是正常，条件错误率不代表全部240错误率，更不保证未来coverage。主评价分母始终全部240，不拒答。','','结果只对应一个均衡固定同域样本与本次API响应。来源与原始字节隔离不等于物理零件/近重复独立，public pretraining和teacher历史暴露需另行界定。区间条件于固定calibrator，未包含拟合、provider或支持选择变异。确认集只用于检验既定系统，本轮此处结束；后续新假设须新协议和新的留出来源。','','图为figures/confirmation_fixed_system_analysis.png/.svg；本图仅使用已完成审核的聚合统计，不重新打开protectedlabels/legacy或GT。','']
with (S/'INDEPENDENT_ANALYSIS.md').open('x') as f:f.write(chr(10).join(lines))
print(json.dumps(dict(status='complete',figure=str(png),analysis=str(S/'INDEPENDENT_ANALYSIS.md'),labels_read=False),ensure_ascii=False))

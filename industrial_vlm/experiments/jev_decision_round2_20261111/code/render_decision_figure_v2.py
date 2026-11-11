#!/usr/bin/env python3
"""Display-only second render; preserve first render, fixed numbers unchanged."""
from pathlib import Path
import json,hashlib,socket,datetime,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert socket.gethostname()=='autodl-container-132444827f-9c594038'
result=read(ROOT/'results.json');grid=[json.loads(l) for l in (ROOT/'EXPLORATORY_HELDOUT_GRID.jsonl').open() if l.strip()];audit=read(ROOT/'independent_audit.json');assert audit['failed']==0 and audit['results_sha256']==sha(ROOT/'results.json')
rev=ROOT/'revisions/first_render';rev.mkdir(parents=True,exist_ok=True)
for n in ['decision_routing_analysis.png','decision_routing_analysis.svg','decision_routing_analysis.provenance.json']:
 assert not (rev/n).exists();(ROOT/n).rename(rev/n)
primary=next(v for v in result['scenarios'] if v['policy']['primary']);hm=primary['heldout'];binary=primary['heldout_baselines']['calibrated_binary_half']['metrics'];cm=hm['confusion'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,ax=plt.subplots(2,2,figsize=(14.5,10.8));colors=['#269986','#D47945','#7C88A0']
# A: outcome actions preserve true-class denominators.
for i,(truth,prefix) in enumerate([('Normal (N=60)','normal'),('Anomaly (N=60)','anomaly')]):
 vals=[cm[prefix+'_auto_normal'],cm[prefix+'_auto_anomaly'],cm[prefix+'_human']];left0=0
 for j,(name,z) in enumerate(zip(['Auto normal','Auto anomaly','Human'],vals)):
  ax[0,0].barh(i,z,left=left0,color=colors[j],label=name if i==0 else None)
  if z:ax[0,0].text(left0+z/2,i,str(z),ha='center',va='center',color='white',fontsize=11,weight='bold')
  left0+=z
ax[0,0].set(yticks=[0,1],yticklabels=['Normal (60)','Anomaly (60)'],xlim=(0,60),xlabel='Heldout image count',title='A  Cal-selected primary: lower0.07 / upper0.80');ax[0,0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
ks=['normal_fpr','unsafe_release','auto_normal_contamination'];labs=['Normal FPR\nFP / all normal','Unsafe release\nFN / all anomaly','Auto-normal contamination\nFN / auto normal']
for j,(name,v,col) in enumerate([('Calibrated binary .5',binary,'#46779F'),('Three-way primary',hm,'#269986')]):
 yy=np.arange(3)+(j-.5)*.15;est=[v[k]['estimate'] for k in ks];ci=[v[k]['ci95'] for k in ks];ax[0,1].errorbar(est,yy,xerr=[[a-z[0] for a,z in zip(est,ci)],[z[1]-a for a,z in zip(est,ci)]],fmt='o',color=col,capsize=4,label=name)
 for x,y,k in zip(est,yy,ks):ax[0,1].annotate(f"{v[k]['count']}/{v[k]['denominator']}",(x,y),textcoords='offset points',xytext=(4,5),fontsize=8)
ax[0,1].set(yticks=np.arange(3),yticklabels=labs,xlim=(-.01,.5),ylim=(2.6,-.5),xlabel='Rate and Wilson95% (different denominators)',title='B  Zero observed release is not a guarantee');ax[0,1].legend(fontsize=8,loc='lower right')
# C: all 12 cost scenarios, never select by heldout performance.
sc=result['scenarios'];xx=np.arange(len(sc));bottom=np.zeros(len(sc))
for j,key in enumerate(['auto_normal','auto_anomaly','human']):
 vv=np.array([v['heldout']['actions'][key]/120 for v in sc]);ax[1,0].bar(xx,vv,bottom=bottom,color=colors[j],label=key.replace('_',' '));bottom+=vv
ax[1,0].set(xticks=xx,xticklabels=[f"{v['policy']['fn_to_fp']}\n{v['policy']['human_penalty']}" for v in sc],ylim=(0,1),xlabel='Assumed FN:FP ratio (top) / human penalty h (bottom)',ylabel='Heldout fraction',title='C  All12 cal-selected scenarios; no heldout choice')
# D: full existing heldout threshold surface is explicitly exploratory.
z=np.full((101,101),np.nan)
for v in grid:z[round(v['lower']*100),round(v['upper']*100)]=(10*v['fn']+v['fp']+.25*v['human'])/120
im=ax[1,1].imshow(z,origin='lower',extent=[-.005,1.005,-.005,1.005],aspect='auto',cmap='viridis');p=primary['policy'];ax[1,1].plot(p['upper'],p['lower'],'*',color='red',markersize=14,markeredgecolor='white',label='Cal-only frozen primary');ax[1,1].set(xlabel='Upper threshold',ylabel='Lower threshold',title='D  EXPLORATORY heldout cost surface; not optimized');ax[1,1].legend(fontsize=8,loc='upper left');fig.colorbar(im,ax=ax[1,1],fraction=.045,pad=.02,label='Assumed surrogate cost, r10 h0.25')
fig.subplots_adjust(left=.08,right=.97,top=.87,bottom=.19,wspace=.48,hspace=.61);fig.text(.06,.97,'Three-way decision development analysis: frozen stage12 probabilities, no new inference',fontsize=15,weight='bold',va='top');fig.text(.06,.93,'Primary assumptions FN:FP10:1 / human penalty0.25; calibration-only thresholds; previously seen heldout120',fontsize=10,color='#444444')
cap='Primary automatic coverage41/120=34.17%; human79/120=65.83%. Unsafe release0/60: Wilson95% upper6.02%.\nAuto-normal contamination0/5: Wilson95% upper43.45%; only5 images released. Different denominators must not be interchanged.\nCosts are illustrative assumptions; human eventual accuracy and end-to-end costs are unmeasured. Fixed policy uncertainty excludes fitting/selection.\nHeldout was previously evaluated: development analysis, not a new blind test. No stage13 confirmation data read or used for this routing.'
fig.text(.06,.115,cap,fontsize=9,va='top',linespacing=1.6,color='#444444');png=ROOT/'decision_routing_analysis.png';svg=ROOT/'decision_routing_analysis.svg';fig.savefig(png,dpi=280,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
prov=dict(schema='jev-three-way-figure-provenance-v1',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),script_sha256=sha(Path(__file__)),inputs_sha256={n:sha(ROOT/n) for n in ['protocol.json','selected_policies.json','results.json','independent_audit.json','EXPLORATORY_HELDOUT_GRID.jsonl']},outputs_sha256={n:sha(ROOT/n) for n in ['decision_routing_analysis.png','decision_routing_analysis.svg']},statistics_on_server=True,stage13_data_read=False,api_calls=0)
with (ROOT/'decision_routing_analysis.provenance.json').open('x') as f:f.write(json.dumps(prov,ensure_ascii=False,indent=2)+chr(10))
print(json.dumps(dict(status='display_revision_complete',new_api_calls=0,numeric_results_unchanged=True)))

#!/usr/bin/env python3
"""Independent fixed-policy audit and aggregate figures; never reads confirmation."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,socket,datetime,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111');OLD=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(l) for l in p.open() if l.strip()]
checks=[]
def check(name,v):
 checks.append(dict(name=name,passed=bool(v)))
 if not v:raise AssertionError(name)
def close(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(close(v,b[k]) for k,v in a.items())
 if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
 return a==b
def rate(k,n):
 if not n:return dict(count=int(k),denominator=0,estimate=None,ci95=None,method='Wilson two-sided95; undefined denominator0')
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return dict(count=int(k),denominator=int(n),estimate=float(p),ci95=[float(max(0,mid-half)),float(min(1,mid+half))],method='Wilson two-sided95')
def recompute(y,p,l,u):
 action=np.where(p<l,0,np.where(p>=u,1,2));n=len(y);nn=int((y==0).sum());na=n-nn;cm={(truth,a):int(((y==truth)&(action==a)).sum()) for truth in [0,1] for a in [0,1,2]};an=int((action==0).sum());aa=int((action==1).sum());hn=n-an-aa
 return dict(n=n,normal=nn,anomalous=na,actions=dict(auto_normal=an,auto_anomaly=aa,human=hn),confusion=dict(anomaly_auto_normal=cm[1,0],anomaly_auto_anomaly=cm[1,1],anomaly_human=cm[1,2],normal_auto_normal=cm[0,0],normal_auto_anomaly=cm[0,1],normal_human=cm[0,2]),normal_fpr=rate(cm[0,1],nn),unsafe_release=rate(cm[1,0],na),auto_normal_contamination=rate(cm[1,0],an),automatic_coverage=rate(an+aa,n),human_rate=rate(hn,n))
def cost(v,r,h):return (r*v['confusion']['anomaly_auto_normal']+v['confusion']['normal_auto_anomaly']+h*v['actions']['human'])/v['n']
check('server',socket.gethostname()=='autodl-container-132444827f-9c594038')
proto=read(ROOT/'protocol.json');result=read(ROOT/'results.json');pol=read(ROOT/'selected_policies.json');table=rows(ROOT/'analysis_source_table.jsonl');grid=rows(ROOT/'EXPLORATORY_HELDOUT_GRID.jsonl');fit=read(OLD/'stage12/run/calibrator.json');c=read(OLD/'stage12/run/contract.json');pred=rows(OLD/'stage12/run/predictions.jsonl');pr={r['source_id']:r for r in pred};src={r['source_id']:r for r in c['records']}
check('protocol_code',sha(ROOT/'code/decision_routing_analysis.py')==proto['analysis_code_sha256'])
check('bind_frozen_protocol',result['protocol_sha256']==pol['protocol_sha256']==sha(ROOT/'protocol.json') and result['selected_policy_sha256']==sha(ROOT/'selected_policies.json'))
check('source_hash_bindings',all(sha(OLD/n)==h for n,h in proto['input_sha256'].items()))
check('table_sha',sha(ROOT/'analysis_source_table.jsonl')==result['source_table_sha256'])
check('table240_unique',len(table)==len({r['source_id'] for r in table})==240)
for i,r in enumerate(table):
 source=src[r['source_id']];p=min(1-1e-6,max(1e-6,r['raw_probability']));z=fit['coefficient'][0][0]*math.log(p/(1-p))+fit['intercept'][0];q=1/(1+math.exp(-z))
 check('source_probability_'+str(i),r['role']==source['role'] and r['category']==source['category'] and r['raw_probability']==pr[r['source_id']]['probability'] and math.isclose(q,r['calibrated_probability'],abs_tol=1e-14))
roles={role:[r for r in table if r['role']==role] for role in ['calibration','heldout_eval']}
for role,rr in roles.items():
 ll={r['source_id']:r for r in rows(OLD/'stage12_calibration_plan'/(role+'_labels.jsonl'))}
 check('labels_'+role,len(rr)==120 and all(r['label']==int(ll[r['source_id']]['is_anomalous']) for r in rr) and Counter((r['category'],r['label']) for r in rr)==Counter({(cat,y):5 for cat in {r['category'] for r in rr} for y in [0,1]}))
yc=np.array([r['label'] for r in roles['calibration']]);pc=np.array([r['calibrated_probability'] for r in roles['calibration']]);yh=np.array([r['label'] for r in roles['heldout_eval']]);ph=np.array([r['calibrated_probability'] for r in roles['heldout_eval']]);rh=np.array([r['raw_probability'] for r in roles['heldout_eval']]);q=np.linspace(0,1,101)
# Independent compact grid count implementation, rather than importing selector.
fn=np.array([int(((yc==1)&(pc<t)).sum()) for t in q]);fp=np.array([int(((yc==0)&(pc>=t)).sum()) for t in q]);left=np.array([(pc<t).sum() for t in q]);right=np.array([(pc>=t).sum() for t in q]);gridcal=[(float(l),float(u),int(fn[i]),int(fp[j]),int(120-left[i]-right[j])) for i,l in enumerate(q) for j,u in enumerate(q) if j>=i]
check('5151pair_grid',len(gridcal)==5151 and len(grid)==5151 and result['exploratory_grid_sha256']==sha(ROOT/'EXPLORATORY_HELDOUT_GRID.jsonl'))
strata=defaultdict(list)
for i,r in enumerate(roles['heldout_eval']):strata[r['category'],r['label']].append(i)
rng=np.random.default_rng(42);w=np.zeros((2000,120))
for _,idx in sorted(strata.items()):w[:,idx]=rng.multinomial(len(idx),np.ones(len(idx))/len(idx),size=2000)
for i,v in enumerate(result['scenarios']):
 p=v['policy'];r=p['fn_to_fp'];h=p['human_penalty'];l=p['lower'];u=p['upper'];key=lambda z:(round((r*z[2]+z[3]+h*z[4])/120,14),z[2],-z[4],z[3],z[0],-z[1]);best=min(gridcal,key=key)
 check('cal_only_threshold_'+str(i),l==best[0] and u==best[1] and close(key(best),tuple(p['calibration_selection']['tie_tuple'])))
 cal=recompute(yc,pc,l,u);held=recompute(yh,ph,l,u);check('all_metrics_'+str(i),close(cal,v['calibration']) and close(held,v['heldout']))
 check('cost_'+str(i),math.isclose(cost(cal,r,h),v['calibration_surrogate_cost']) and math.isclose(cost(held,r,h),v['heldout_surrogate_cost']))
 for name,pp,ll,uu in [('raw_binary_half',rh,.5,.5),('calibrated_binary_half',ph,.5,.5),('all_human_review',ph,0.,1.0000000001)]:
  b=recompute(yh,pp,ll,uu);check('baseline_'+str(i)+'_'+name,close(b,v['heldout_baselines'][name]['metrics']) and math.isclose(cost(b,r,h),v['heldout_baselines'][name]['surrogate_cost']))
 act=np.where(ph<l,0,np.where(ph>=u,1,2));loss=np.where(act==2,h,np.where((act==0)&(yh==1),r,np.where((act==1)&(yh==0),1.,0.)));binary=(ph>=.5).astype(int);bloss=np.where((binary==0)&(yh==1),r,np.where((binary==1)&(yh==0),1.,0.));d=loss-bloss
 check('pairedcost_'+str(i),math.isclose(d.mean(),v['paired_cost_minus_calibrated_binary']['delta']) and close(np.quantile(w@d/120,[.025,.975]).tolist(),v['paired_cost_minus_calibrated_binary']['ci95']))
 for cat,expected in v['by_category'].items():
  idx=np.array([j for j,z in enumerate(roles['heldout_eval']) if z['category']==cat]);check('category_'+str(i)+'_'+cat,close(recompute(yh[idx],ph[idx],l,u),expected))
for i,v in enumerate(grid):
 expected=recompute(yh,ph,v['lower'],v['upper']);check('exploratory_grid_'+str(i),v['fn']==expected['confusion']['anomaly_auto_normal'] and v['fp']==expected['confusion']['normal_auto_anomaly'] and v['human']==expected['actions']['human'] and v['exploratory_only'])
check('no_confirmation_or_api',result['stage13_inputs_predictions_labels_read'] is False and result['no_new_api_calls'] and result['no_model_forward'] and result['no_model_refit'] and all('stage13' not in p and 'confirmation' not in p for p in result['file_reads']))
check('frozen_before_result',proto['timestamp_utc']<pol['timestamp_utc']<result['timestamp_utc'])
audit=dict(schema='jev-three-way-independent-audit-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),passed=sum(v['passed'] for v in checks),failed=sum(not v['passed'] for v in checks),checks=checks,auditor_sha256=sha(Path(__file__)),results_sha256=sha(ROOT/'results.json'),protocol_sha256=sha(ROOT/'protocol.json'),selected_policy_sha256=sha(ROOT/'selected_policies.json'),independent='Algebraic mapping, vector compact calgrid selector, action-coded routing, all5151 exploratory counts, Wilson denominators, fixed paired bootstrap2000',confirmation_data_read=False,api_calls=0)
with (ROOT/'independent_audit.json').open('x') as f:f.write(json.dumps(audit,ensure_ascii=False,indent=2)+chr(10))
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
ax[1,0].set(xticks=xx,xticklabels=[f"{v['policy']['fn_to_fp']}\n{v['policy']['human_penalty']}" for v in sc],ylim=(0,1),xlabel='Assumed FN:FP ratio (top) / human penalty h (bottom)',ylabel='Heldout fraction',title='C  All12 cal-selected scenarios; no heldout choice');ax[1,0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
# D: full existing heldout threshold surface is explicitly exploratory.
z=np.full((101,101),np.nan)
for v in grid:z[round(v['lower']*100),round(v['upper']*100)]=(10*v['fn']+v['fp']+.25*v['human'])/120
im=ax[1,1].imshow(z,origin='lower',extent=[-.005,1.005,-.005,1.005],aspect='auto',cmap='viridis');p=primary['policy'];ax[1,1].plot(p['upper'],p['lower'],'*',color='red',markersize=14,markeredgecolor='white',label='Cal-only frozen primary');ax[1,1].set(xlabel='Upper threshold',ylabel='Lower threshold',title='D  EXPLORATORY heldout cost surface; not optimized');ax[1,1].legend(fontsize=8,loc='upper left');fig.colorbar(im,ax=ax[1,1],fraction=.045,pad=.02,label='Assumed surrogate cost, r10 h0.25')
fig.subplots_adjust(left=.08,right=.97,top=.87,bottom=.19,wspace=.48,hspace=.61);fig.text(.06,.97,'Three-way decision development analysis: frozen stage12 probabilities, no new inference',fontsize=15,weight='bold',va='top');fig.text(.06,.93,'Primary assumptions FN:FP10:1 / human penalty0.25; calibration-only thresholds; previously seen heldout120',fontsize=10,color='#444444')
cap='Primary automatic coverage41/120=34.17%; human79/120=65.83%. Unsafe release0/60: Wilson95% upper6.02%.\nAuto-normal contamination0/5: Wilson95% upper43.45%; only5 images released. Different denominators must not be interchanged.\nCosts are illustrative assumptions; human eventual accuracy and end-to-end costs are unmeasured. Fixed policy uncertainty excludes fitting/selection.\nHeldout was previously evaluated: development analysis, not a new blind test. No stage13 confirmation data read or used for this routing.'
fig.text(.06,.115,cap,fontsize=9,va='top',linespacing=1.6,color='#444444');png=ROOT/'decision_routing_analysis.png';svg=ROOT/'decision_routing_analysis.svg';fig.savefig(png,dpi=280,facecolor='white');fig.savefig(svg,facecolor='white');plt.close(fig)
prov=dict(schema='jev-three-way-figure-provenance-v1',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),script_sha256=sha(Path(__file__)),inputs_sha256={n:sha(ROOT/n) for n in ['protocol.json','selected_policies.json','results.json','independent_audit.json','EXPLORATORY_HELDOUT_GRID.jsonl']},outputs_sha256={n:sha(ROOT/n) for n in ['decision_routing_analysis.png','decision_routing_analysis.svg']},statistics_on_server=True,stage13_data_read=False,api_calls=0)
with (ROOT/'decision_routing_analysis.provenance.json').open('x') as f:f.write(json.dumps(prov,ensure_ascii=False,indent=2)+chr(10))
with (ROOT/'AUDIT.md').open('x') as f:f.write('# 三态分析独立复核'+chr(10)*2+f"指定服务器独立复算{audit['passed']}项通过、0失败。复核cal-only5151候选的全部12情景最优和tie、每个风险分母/Wilson、逐类统计、基线、2000配对成本区间及全部heldout探索网格计数。"+chr(10)*2+'未读任何stage13确认输入、概率、标签或GT；无API/模型forward/refit。图仅用已固定的聚合结果，主情景不按heldout修改。'+chr(10))
print(json.dumps(dict(status='complete',audit_passed=audit['passed'],audit_failed=audit['failed'],audit_sha256=sha(ROOT/'independent_audit.json'),png_sha256=sha(png),svg_sha256=sha(svg),provenance_sha256=sha(ROOT/'decision_routing_analysis.provenance.json')),ensure_ascii=False,indent=2))

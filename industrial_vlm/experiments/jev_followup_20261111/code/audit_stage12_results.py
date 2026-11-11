#!/usr/bin/env python3
"""Stage12 completed-only independent audit and fixed supplementary statistics."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,socket,datetime,math
import numpy as np
from sklearn.linear_model import LogisticRegression
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(ROOT/'code'))
import isolated_calibration as iso
import audit_followup_metrics as stats
checks=[]
def check(n,v):
 checks.append({'name':n,'passed':bool(v)})
 if not v:raise AssertionError(n)
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def close(a,b):return stats.close(a,b)
def logit(p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
 return np.log(p/(1-p)).reshape(-1,1)
def losses(y,p):
 p=np.clip(np.asarray(p,float),1e-15,1-1e-15);y=np.asarray(y,int)
 return -(y*np.log(p)+(1-y)*np.log1p(-p))
def wilson(k,n):
 if n==0:return None
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return dict(count=int(k),denominator=int(n),estimate=p,ci95=[max(0.,mid-half),min(1.,mid+half)],method='Wilson two-sided95')
def extra(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=.5
 tp=int(sum((y==1)&pred));fp=int(sum((y==0)&pred));tn=int(sum((y==0)&~pred));fn=int(sum((y==1)&~pred))
 return dict(n=len(y),threshold=.5,comparison='p>=0.5',confusion=dict(TP=tp,FP=fp,TN=tn,FN=fn),recall=wilson(tp,tp+fn),fpr=wilson(fp,fp+tn),accuracy=wilson(tp+tn,len(y)),error=wilson(fp+fn,len(y)),binary_logloss=float(losses(y,p).mean()),logloss_clip_epsilon=1e-15,exact_half_ties=int(sum(p==.5)))
def point(y,p):
 if len(np.unique(y))<2:return dict(n=len(y),normal=int(sum(y==0)),anomalous=int(sum(y==1)),**{k:None for k in stats.METRICS})
 return stats.point(y,p)
def coverage(rr,y,p):
 conf=np.maximum(p,1-p);order=sorted(range(len(rr)),key=lambda i:(-float(conf[i]),rr[i]['source_id']));out={}
 for cov in [.5,.75,1.]:
  ix=np.array(order[:math.ceil(cov*len(rr))],int);pred=p[ix]>=.5
  out[str(cov)]=dict(requested_coverage=cov,accepted=len(ix),total=len(rr),actual_coverage=len(ix)/len(rr),error_rate=float(np.mean(pred!=y[ix])),confidence_cutoff=float(conf[ix[-1]]),accepted_source_ids=[rr[i]['source_id'] for i in ix],metrics=point(y[ix],p[ix]),note='Fixed fraction retrospective risk; confidence tie break source_id ascending; no heldout threshold optimization')
 return out

def main():
 check('designated_server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 stage=ROOT/'stage12/run';plan=ROOT/'stage12_calibration_plan';required=['contract.json','status.json','predictions.jsonl','attempts.jsonl','prediction_freeze.json','calibrator.json','metrics.json']
 if any(not (stage/n).exists() for n in required):print(json.dumps({'status':'pending','reason':'complete240/freeze/coeff/metrics not all present'}));return
 status=read(stage/'status.json')
 if status['status']!='complete':print(json.dumps({'status':'pending','completed':status.get('completed'),'reason':'formal240notcomplete'}));return
 c=read(stage/'contract.json');s=read(ROOT/'stage12/system_contract.json');freeze=read(stage/'prediction_freeze.json');fit=read(stage/'calibrator.json');m=read(stage/'metrics.json')
 check('runner_source_frozen',sha(ROOT/'code/isolated_calibration.py')==c['runner_sha256']==s['runner_sha256']==freeze['runner_sha256']=='d8489c953ec4cb5e9d7804643d35f43d5ad0801211abc118bcc8e0003fec37d1')
 check('systemandcontract_frozen',freeze['contract_sha256']==sha(stage/'contract.json') and freeze['system_contract_sha256']==sha(ROOT/'stage12/system_contract.json')==c['system_contract_sha256'])
 check('formal240_complete',status['completed']==status['expected']==len(c['records'])==240 and c['mode']=='formal_source_disjoint_calibration')
 check('predictionfreeze_hashes',freeze['predictions_sha256']==sha(stage/'predictions.jsonl') and freeze['attempts_sha256']==sha(stage/'attempts.jsonl') and freeze['predictions_before_labels'] is True and m['prediction_freeze']==freeze)
 check('finalsystem4shot43_448',s['shots']==m['system_shots']==4 and s['support_seed']==m['support_seed']==43 and s['size']==448)
 check('metrics_complete_noheldoutfit',m['status']=='complete' and m['heldout_used_for_fit'] is False and fit['heldout_labels_used_for_fit'] is False)
 check('sourceplan_hashes',all(sha(plan/n)==v for n,v in c['source_plan_files'].items()))
 records={r['source_id']:r for r in c['records']};check('240_unique',len(records)==240)
 images={sid:(stage/r['asset']['path']).read_bytes() for sid,r in records.items()}
 for rr in s['supports'].values():
  for r in rr:images[r['source_id']]=(ROOT/'stage08_support'/r['asset']['path']).read_bytes()
 ch=hashlib.sha256(canon(c)).hexdigest();hashes={sid:hashlib.sha256(canon(iso.payload(r,s,images,ch))).hexdigest() for sid,r in records.items()}
 check('canonical_status_contract',status['contract_sha256']==ch)
 complete={}
 for i,v in enumerate(rows(stage/'predictions.jsonl')):
  sid=v['source_id'];check('prediction_'+str(i),sid in records and sid not in complete and v['contract_sha256']==ch and v['request_sha256']==hashes[sid] and v['role']==records[sid]['role'] and v['category']==records[sid]['category'] and v['response_model']=='jev-1.14.0' and math.isfinite(v['probability']) and 0<=v['probability']<=1)
  complete[sid]=v
 check('full240_predictions',set(complete)==set(records))
 starts=Counter();intent={};term={};events=Counter()
 for i,v in enumerate(rows(stage/'attempts.jsonl')):
  sid=v['source_id'];k=(sid,v['attempt']);events[v['event']]+=1
  check('attempt_'+str(i),sid in records and v['role']==records[sid]['role'] and v['contract_sha256']==ch and v['request_sha256']==hashes[sid] and 1<=v['attempt']<=3)
  if v['event']=='started':
   starts[sid]+=1;check('startseq_'+str(i),starts[sid]==v['attempt'] and k not in intent);intent[k]=v
  else:
   check('terminal_'+str(i),v['event'] in ['succeeded','failed'] and k in intent and k not in term and v['invocation_id']==intent[k]['invocation_id']);term[k]=v
   if v['event']=='succeeded':check('success_link_'+str(i),complete[sid]['attempt']==v['attempt'])
 check('no_unknown_intents',set(intent)==set(term))
 check('acceptance_intents',all((sid,r['attempt']) in intent for sid,r in complete.items()) and events['succeeded']==240)
 check('bounded_started',max(starts.values())<=3 and sum(starts.values())<=720)
 byrole={role:[r for r in c['records'] if r['role']==role] for role in ['calibration','heldout_eval']}
 check('role120disjoint',len(byrole['calibration'])==len(byrole['heldout_eval'])==120 and not {r['source_id'] for r in byrole['calibration']}&{r['source_id'] for r in byrole['heldout_eval']})
 # This guard is before any label access: all240, durable attempts, freeze andcoefficients are complete.
 check('labels_gate_satisfied',len(complete)==240 and not set(intent)-set(term) and freeze['predictions_before_labels'] and fit['heldout_labels_used_for_fit'] is False)
 def getlabel(role):
  path=plan/(role+'_labels.jsonl');check('labelsha_'+role,sha(path)==iso.LABEL_HASHES[role])
  rr=rows(path);lookup={r['source_id']:r for r in rr};target=byrole[role]
  check('labelroles_'+role,len(lookup)==len(rr)==120 and set(lookup)=={r['source_id'] for r in target} and all(r['role']==role and type(r['is_anomalous']) is bool for r in rr))
  y=np.array([int(lookup[r['source_id']]['is_anomalous']) for r in target]);cnt=Counter((r['category'],int(v)) for r,v in zip(target,y));check('labels5x2x12_'+role,len(cnt)==24 and set(cnt.values())=={5})
  return y,sha(path)
 ycal,clsha=getlabel('calibration');pcal=np.array([complete[r['source_id']]['probability'] for r in byrole['calibration']])
 model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,random_state=42).fit(logit(pcal),ycal)
 check('fixed_logistic_coefficients',close(model.coef_.tolist(),fit['coefficient']) and close(model.intercept_.tolist(),fit['intercept']) and model.classes_.tolist()==fit['classes'] and model.n_iter_.tolist()==fit['n_iter'])
 check('fit_frozen_calibration_only',fit['C']==1 and fit['penalty']=='l2' and fit['solver']=='lbfgs' and fit['max_iter']==1000 and fit['random_state']==42 and fit['calibration_label_sha256']==clsha and fit['calibration_prediction_records_sha256']==hashlib.sha256(canon([complete[r['source_id']] for r in byrole['calibration']])).hexdigest() and m['calibrator']==fit)
 # Replay fit is completed before opening the heldout label file.
 yh,hlsha=getlabel('heldout_eval');hr=byrole['heldout_eval'];raw=np.array([complete[r['source_id']]['probability'] for r in hr]);calibrated=model.predict_proba(logit(raw))[:,1];probs={'raw':raw,'calibrated':calibrated}
 check('result_label_hashes',m['calibration_label_sha256']==clsha and m['heldout_label_sha256']==hlsha)
 for name,p in [('raw',pcal),('calibrated',model.predict_proba(logit(pcal))[:,1])]:check('calibration_fit_metric_'+name,close(point(ycal,p),m['calibration_fit_metrics'][name]))
 cats=sorted({r['category'] for r in hr});ix={cat:np.array([i for i,r in enumerate(hr) if r['category']==cat]) for cat in cats};points={name:point(yh,p) for name,p in probs.items()}
 coverages={name:coverage(hr,yh,p) for name,p in probs.items()}
 for name,p in probs.items():
  check('heldout_metrics_'+name,close(points[name],m['heldout'][name]['metrics']))
  check('coverage_source_order_metrics_'+name,close(coverages[name],m['heldout'][name]['coverage']))
  bycat={cat:point(yh[idx],p[idx]) for cat,idx in ix.items()}
  for cat in cats:check('category_'+cat+'_'+name,close(bycat[cat],m['by_category'][cat][name]))
  check('category_macro_'+name,close({k:float(np.mean([bycat[cat][k] for cat in cats])) for k in stats.METRICS},m['category_macro'][name]))
 w=stats.bootstrap_weights(hr,yh);boot={name:stats.wm(yh,p,w) for name,p in probs.items()}
 check('bootstrap_direction',m['paired_bootstrap']['direction']=='calibrated-minus-raw')
 for k in stats.METRICS:
  v=m['paired_bootstrap']['pooled'][k];delta=points['calibrated'][k]-points['raw'][k];ci=np.quantile(boot['calibrated'][k]-boot['raw'][k],[.025,.975]).tolist()
  check('pairedbootstrap_'+k,close(float(delta),v['delta']) and close(ci,v['ci95']) and v['replicates']==2000 and v['seed']==42 and v['paired'] is True and v['calibrator_refitted'] is False)
 fields=sorted({k for v in complete.values() for k in v['usage']});usage={k:dict(observed_sum=float(sum(v['usage'][k] for v in complete.values() if k in v['usage'])),observed_calls=sum(k in v['usage'] for v in complete.values()),missing_calls=sum(k not in v['usage'] for v in complete.values())) for k in fields}
 operations=dict(accepted=240,started=sum(starts.values()),failed=events['failed'],unknown=0,usage=usage);check('operations_counts_usage',close(operations,m['operations']))
 audit={'schema':'jev-stage12-independent-postexecution-audit-v1','status':'complete','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'passed':sum(v['passed'] for v in checks),'failed':sum(not v['passed'] for v in checks),'checks':checks,'auditor_sha256':sha(Path(__file__)),'contract_sha256':sha(stage/'contract.json'),'system_sha256':sha(ROOT/'stage12/system_contract.json'),'prediction_freeze_sha256':sha(stage/'prediction_freeze.json'),'calibrator_sha256':sha(stage/'calibrator.json'),'metrics_sha256':sha(stage/'metrics.json'),'point_metrics':points,'operations':operations,'max_started_per_target':max(starts.values()),'fit_replay':'Onlycalibration120 labels; fixedC1clip1e-6 L2lbfgs; heldout labelsopened after refit andexistingfreeze guard','coverage_checked':[.5,.75,1.],'gt_masks_opened':False,'confirmation_opened':False,'new_api_requests':0}
 out=ROOT/'stages/12_postexecution_audit.json';check('auditnotexists',not out.exists())
 with out.open('x') as f:f.write(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 llraw=losses(yh,raw);llcal=losses(yh,calibrated);ldelta=float((llcal-llraw).mean());lboot=w@(llcal-llraw)/w.sum(axis=1)
 supp={'schema':'jev-stage12-fixed-statistical-supplement-v1','status':'complete','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'inputs_sha256':{'metrics.json':sha(stage/'metrics.json'),'calibrator.json':sha(stage/'calibrator.json'),'prediction_freeze.json':sha(stage/'prediction_freeze.json'),'12_postexecution_audit.json':sha(out)},'methods':{name:extra(yh,p) for name,p in probs.items()},'by_category':{cat:{name:extra(yh[idx],p[idx]) for name,p in probs.items()} for cat,idx in ix.items()},'coverage':{},'paired_logloss_difference':dict(direction='calibrated-minus-raw',delta=ldelta,ci95=np.quantile(lboot,[.025,.975]).tolist(),replicates=2000,seed=42,strata='category x label',paired=True,calibrator_refitted=False,logloss_clip_epsilon=1e-15),'definitions':{'Wilson':'two-sided95 z1.959963984540054','confusion':'p>=0.5, anomalypositive','logloss':'-(ylogp+(1-y)log(1-p)), numerical clip1e-15, prespecified bysupplementary auditcode beforecomplete results','coverage':'same frozen descendingmax(p,1-p), sourceIDties; conditional subsetintervals and full120 coverage/abstentions reported; no retuning'},'limitations':['Supplemental computation does not modify runner metrics or modelselection','Labelsbalanced50percent, notproductionprior','Nativevalidationremainder notblind/confirmation','Intervals conditiononfittedcalibrator, exclude calibration/provideruncertainty'],'gt_masks_opened':False,'confirmation_opened':False,'new_api_requests':0}
 for name,p in probs.items():
  conf=np.maximum(p,1-p);order=sorted(range(len(hr)),key=lambda i:(-float(conf[i]),hr[i]['source_id']));supp['coverage'][name]={}
  for cov in [.5,.75,1.]:
   sel=np.array(order[:math.ceil(cov*len(hr))],int);supp['coverage'][name][str(cov)]=dict(accepted=len(sel),abstained=120-len(sel),coverage=len(sel)/120,conditional=extra(yh[sel],p[sel]),accepted_source_set_sha256=hashlib.sha256(canon([hr[i]['source_id'] for i in sel])).hexdigest())
 p=stage/'STATISTICAL_SUPPLEMENT.json';check('supplementnotexists',not p.exists())
 with p.open('x') as f:f.write(json.dumps(supp,ensure_ascii=False,indent=2)+'\n')
 lines=['# Stage12独立审计与补充统计','','独立审计通过'+str(audit['passed'])+'项，0失败。240预测及完整attempt账本冻结后，先仅calibration120复拟固定logistic并核系数，再读heldout120标签。此补充不改runner/metrics，不选新阈值或方法。','','| 方法 | TP | FP | TN | FN | Recall及Wilson95% | FPR及Wilson95% | Logloss |','|---|---:|---:|---:|---:|---|---|---:|']
 for name,v in supp['methods'].items():
  cm=v['confusion'];fmt=lambda x:f"{x['estimate']:.4f} [{x['ci95'][0]:.4f},{x['ci95'][1]:.4f}]"
  lines.append('| '+name+' | '+' | '.join(str(cm[k]) for k in ['TP','FP','TN','FN'])+' | '+fmt(v['recall'])+' | '+fmt(v['fpr'])+f" | {v['binary_logloss']:.6f} |")
 lines+=['',f"校准−raw Logloss差{ldelta:+.6f}，95%配对来源bootstrap区间[{supp['paired_logloss_difference']['ci95'][0]:+.6f},{supp['paired_logloss_difference']['ci95'][1]:+.6f}]；固定calibrator，不包含校准拟合不确定性。Logloss数值clip1e-15，与拟合featureclip1e-6区别已明确。",'', '| 方法 | 覆盖率 | 保留 | 拒答 | 条件错误率及Wilson95% |','|---|---:|---:|---:|---|']
 for name,values in supp['coverage'].items():
  for cov,v in values.items():
   err=v['conditional']['error'];lines.append(f"| {name} | {cov} | {v['accepted']} | {v['abstained']} | {err['estimate']:.4f} [{err['ci95'][0]:.4f},{err['ci95'][1]:.4f}] |")
 lines+=['','拒答不是正常预测；固定coverage是事后风险描述，不保证部署coverage。条件subset的FPR/Recall与全120分母不可混用。完整混淆矩阵、逐类区间、coverage子集聚合与logloss差异见同名JSON。','','来源隔离新采样仍属于nativeval remainder，可能历史验证已看，不称blind或confirmation；平衡异常先验50%不代表产线缺陷率。校准有改善也有退化可能，保留完整负结果与区间。confirmation仍未读取。','',f"- runner metrics SHA：{sha(stage/'metrics.json')}",f"- calibrator SHA：{sha(stage/'calibrator.json')}",f"- independent audit SHA：{sha(out)}",f"- supplement JSON SHA：{sha(p)}",'']
 with (stage/'STATISTICAL_SUPPLEMENT.md').open('x') as f:f.write('\n'.join(lines))
 print(json.dumps({k:v for k,v in audit.items() if k!='checks'},ensure_ascii=False,indent=2));print(json.dumps({'supplement':str(p),'logloss_delta':supp['paired_logloss_difference'],'confusion':supp['methods']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()


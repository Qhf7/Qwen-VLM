#!/usr/bin/env python3
"""Completed-only independent online-gate pilot statistics. No confirmation or masks."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,hashlib,socket,datetime,math,argparse
import numpy as np
R=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111');S=R/'online_route_pilot';OLD=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(OLD/'code'))
import online_route_pilot as online
import audit_followup_metrics as stats
import audit_stage12_results as common
checks=[]
def check(n,v):
 checks.append(dict(name=n,passed=bool(v)))
 if not v:raise AssertionError(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def put(p,v):
 with p.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def point(y,p):
 if len(np.unique(y))<2:return dict(n=len(y),normal=int(sum(y==0)),anomalous=int(sum(y==1)),**{k:None for k in stats.METRICS})
 return stats.point(y,p)
def freeze_method():
 assert socket.gethostname()=='autodl-container-132444827f-9c594038'
 check('not_final_results',not (S/'metrics.json').exists() and not (S/'prediction_freeze.json').exists())
 v=dict(schema='jev-online-independent-methodology-freeze-v1',status='frozen_before_complete_pilot_results',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),auditor_sha256=sha(Path(__file__)),runner_sha256=sha(R/'code/online_route_pilot.py'),protocol_sha256=sha(S/'protocol.json'),contract_sha256=sha(S/'contract.json'),methods=dict(metrics=list(stats.METRICS),binary_threshold=.5,logloss_clip=1e-15,confusion='TP FP TN FN anomalypositive',wilson='Two-sided95 z1.959963984540054; explicit denominators',bootstrap='2000 seed42 multinomial within observed binary-label strata, paired sources, fixed routes; per-category descriptive only (sparse)',comparison='Four frozen K-select policies vs same newly collected duplicate24; all use same new downstream potential-outcome table',costs='Actualpilot allgate+allnewdup+unionROI; perpolicy marginal acquisition includingdupbaseline; ideal minimal policy chosenfinal24 plusgate24 when needed, historical-control prepass unknown separately',gatescore='Needforcloseup only, never final anomaly probability'),label_gate='Complete24 gate + all24newdup + unionROI settled, all hashes durable prediction_freeze; then only approved_labels24.json subset',stage13_data_read=False,GT_masks_read=False,new_api_calls=0,development_pilot=True,no_deployment_or_agent_guarantee=True)
 put(S/'independent_methodology_freeze.json',v);print(json.dumps(v,ensure_ascii=False,indent=2))

def main():
 check('server',socket.gethostname()=='autodl-container-132444827f-9c594038');method=read(S/'independent_methodology_freeze.json');amend=read(S/'independent_interface_amendment.json');check('method_unchanged',amend['updated_auditor_sha256']==sha(Path(__file__)) and amend['original_auditor_sha256']==method['auditor_sha256'] and amend['methodology_sha256']==sha(S/'independent_methodology_freeze.json') and method['runner_sha256']==sha(R/'code/online_route_pilot.py'))
 c,images=online.verify(S);routing=read(S/'routing.json');gf=read(S/'gate/prediction_freeze.json');freeze=read(S/'prediction_freeze.json');records=c['records'];sids=[r['source_id'] for r in records];check('24unique',len(sids)==len(set(sids))==24)
 allcomplete={};startcount=0;failcount=0;usage_by_phase={};attempts_by_phase={}
 for name in ['gate','final']:
  phase,(complete,starts,attempts,hashes,unknown)=online.phase_data(S,name,c,images);status=read(S/name/'status.json');check('complete_'+name,status['status']=='complete' and len(complete)==len(phase['schedule']) and not unknown)
  allcomplete[name]=complete;startcount+=sum(starts.values());failcount+=sum(r['event']=='failed' for r in attempts);attempts_by_phase[name]=attempts
  for target,r in complete.items():check('response_'+name+'_'+target[0][:16]+'_'+target[1],r['request_sha256']==hashes[target] and r['response_model']=='jev-1.14.0' and math.isfinite(r['probability']) and 0<=r['probability']<=1)
  usage_by_phase[name]={k:dict(observed_sum=float(sum(r['usage'][k] for r in complete.values() if k in r['usage'])),observed_calls=sum(k in r['usage'] for r in complete.values()),missing_calls=sum(k not in r['usage'] for r in complete.values())) for k in sorted({k for r in complete.values() for k in r['usage']})}
 check('gatefreeze_hash',gf['predictions_sha256']==sha(S/'gate/predictions.jsonl') and gf['attempts_sha256']==sha(S/'gate/attempts.jsonl'))
 # Versioned evaluator freeze must bind every source file it declares; guard before any labels.
 bindings=freeze.get('files_sha256',freeze.get('input_sha256',freeze.get('artifacts_sha256',{})))
 if not bindings:
  known={'parent_contract_sha256':'contract.json','routing_sha256':'routing.json','gate_contract_sha256':'gate/contract.json','gate_predictions_sha256':'gate/predictions.jsonl','gate_attempts_sha256':'gate/attempts.jsonl','final_contract_sha256':'final/contract.json','final_predictions_sha256':'final/predictions.jsonl','final_attempts_sha256':'final/attempts.jsonl'};bindings={path:freeze[field] for field,path in known.items()};check('freeze_runner',freeze['runner_sha256']==sha(R/'code/online_route_pilot.py') and freeze['predictions_before_pilot_GT'])
 check('predictionfreeze_has_bindings',bool(bindings))
 for name,h in bindings.items():
  p=Path(name);p=p if p.is_absolute() else S/name;check('predictionfreeze_'+name,sha(p)==h)
 check('full_set_before_labels',len(allcomplete['gate'])==24 and len(allcomplete['final'])==24+len(routing['union_roi']) and len(routing['union_roi'])<=24 and startcount<=216 and failcount<=32)
 labpath=S/'approved_labels24.json';labelsdoc=read(labpath);labels=labelsdoc.get('labels',labelsdoc);check('approved24_label_freeze',labelsdoc['prediction_freeze_sha256']==sha(S/'prediction_freeze.json') and labelsdoc['stage13_data_read'] is False);check('approved24_label_identity',len(labels)==24 and set(labels)==set(sids) and all(type(y) in (bool,int) and int(y) in [0,1] for y in labels.values()));y=np.array([int(labels[sid]) for sid in sids]);check('bothlabels_for_metrics',len(np.unique(y))==2)
 final=allcomplete['final'];raw=np.array([final[sid,'duplicate']['probability'] for sid in sids]);probs={'new_duplicate':raw}
 for name,selected in routing['routes'].items():probs[name]=np.array([final[sid,'roi' if sid in selected else 'duplicate']['probability'] for sid in sids])
 resultpoints={name:point(y,p) for name,p in probs.items()};extras={name:common.extra(y,p) for name,p in probs.items()};cats=sorted({r['category'] for r in records});bycat={cat:{name:point(y[np.array([i for i,r in enumerate(records) if r['category']==cat])],p[np.array([i for i,r in enumerate(records) if r['category']==cat])]) for name,p in probs.items()} for cat in cats}
 rng=np.random.default_rng(42);w=np.zeros((2000,24))
 for label in [0,1]:
  idx=np.flatnonzero(y==label);w[:,idx]=rng.multinomial(len(idx),np.ones(len(idx))/len(idx),size=2000)
 boot={name:stats.wm(y,p,w) for name,p in probs.items()};deltas={}
 for name in probs:
  if name=='new_duplicate':continue
  deltas[name]={k:dict(delta=resultpoints[name][k]-resultpoints['new_duplicate'][k],ci95=np.quantile(boot[name][k]-boot['new_duplicate'][k],[.025,.975],method='linear').tolist()) for k in stats.METRICS};ld=common.losses(y,probs[name])-common.losses(y,raw);deltas[name]['logloss']=dict(delta=float(ld.mean()),ci95=np.quantile(w@ld/24,[.025,.975],method='linear').tolist())
 runner_metrics=read(S/'metrics.json');check('runner_complete',runner_metrics['status']=='complete' and runner_metrics['n']==24 and runner_metrics['normal']==int(sum(y==0)) and runner_metrics['anomalous']==int(sum(y==1)) and runner_metrics['K']==c['K'] and runner_metrics['stage13_data_read'] is False)
 for name in probs:
  check('runner_point_'+name,stats.close({k:resultpoints[name][k] for k in runner_metrics['results'][name]['classification']},runner_metrics['results'][name]['classification']))
  if name!='new_duplicate':
   for k in runner_metrics['paired_bootstrap_vs_newduplicate'][name]:check('runner_paired_'+name+'_'+k,stats.close(deltas[name][k],runner_metrics['paired_bootstrap_vs_newduplicate'][name][k]))
 # Descriptive costs separate collection experiment from idealized deployment graph.
 cost={}
 for name,selected in routing['routes'].items():
  finals=[final[sid,'roi' if sid in selected else 'duplicate'] for sid in sids];gatecalls=list(allcomplete['gate'].values()) if name=='learned_gate' else [];used=finals+gatecalls
  cost[name]=dict(K=len(selected),chosen_final_predictions=24,gate_calls=24 if gatecalls else 0,ideal_minimal_logical_calls=len(used),pilot_policy_acquisition_with_new_duplicate_baseline=24+len(selected)+(24 if gatecalls else 0),historical_uncertainty_prepass='Required prior duplicate probabilities; outsidepilot cost unknown, cannot claim free' if name=='historical_uncertainty' else 'not needed',observed_chosen_success_usage={k:dict(observed_sum=float(sum(r['usage'][k] for r in used if k in r['usage'])),observed_calls=sum(k in r['usage'] for r in used),missing_calls=sum(k not in r['usage'] for r in used)) for k in sorted({k for r in used for k in r['usage']})},note='Ideal counts inferred from potentialoutcome table, not directly timed deployment; paid failures/unknownusage excluded only as unknown, notzero')
 supplement=dict(schema='jev-online-independent-statistical-supplement-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),methodology_sha256=sha(S/'independent_methodology_freeze.json'),contract_sha256=sha(S/'contract.json'),prediction_freeze_sha256=sha(S/'prediction_freeze.json'),approved24_labels_sha256=sha(labpath),n=24,normal=int(sum(y==0)),anomalous=int(sum(y==1)),K=c['K'],eligible=c['eligible_count'],union_roi=len(routing['union_roi']),metric_semantics='All values compare regional-candidate question outputs with image-level binary labels: image-label proxy diagnostics; Brier/ECE/logloss are NOT regional calibration or calibrated all-image anomaly probabilities',metrics=resultpoints,extras=extras,by_category=bycat,paired_minus_newduplicate=deltas,bootstrap=dict(replicates=2000,seed=42,strata='image_label',fixed_routes=True,route_selection_uncertainty=False,provider_uncertainty=False),costs=cost,actualpilot=dict(accepted=24+len(final),started=startcount,failed=failcount,unknown=0,usage=usage_by_phase,missing_or_failed_usage='unknown',currency_bill='unavailable'),limitations=['Exploratory previously viewed24 source subset, not newblind test or crossdomain','24 source sample may be imbalanced; sparsecategory not macrocomparable','All policies share same measured newpotentialoutcomes; no perpolicyindependent provider replicate','Cropneedgate is not trained anomalyrouter, not complete Agent','Ideal minimal counts and surrogate usage are inferred, not measured latency/bill','Historicaluncertainty prepass notfree and different collection time','No GTmasks or stage13 data accessed'],stage13_data_read=False,GT_masks_read=False,new_api_calls=0)
 put(S/'INDEPENDENT_STATISTICAL_SUPPLEMENT.json',supplement);audit=dict(schema='jev-online-independent-postexecution-audit-v1',status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),passed=sum(x['passed'] for x in checks),failed=sum(not x['passed'] for x in checks),checks=checks,auditor_sha256=sha(Path(__file__)),supplement_sha256=sha(S/'INDEPENDENT_STATISTICAL_SUPPLEMENT.json'),prediction_freeze_sha256=sha(S/'prediction_freeze.json'),api_calls=0,stage13_data_read=False,GT_masks_read=False)
 put(S/'independent_postexecution_audit.json',audit)
 print(json.dumps(dict(status='complete',passed=audit['passed'],failed=audit['failed'],metrics=resultpoints,deltas=deltas,actualpilot=supplement['actualpilot']),ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze-method',action='store_true');a=ap.parse_args()
 if a.freeze_method:freeze_method()
 else:main()

#!/usr/bin/env python3
"""Independent completed-stage08/10 metrics audit. Never opens confirmation."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,sys,json,hashlib,socket,datetime,math
import numpy as np
ROOT=Path('/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111')
sys.path.insert(0,str(ROOT/'code'))
import support_robustness as support
import resolution_ablation as res
METRICS=('auroc','ap','macro_f1','recall','fpr','brier','ece_10_equal_width')
checks=[]
def check(name,value):
 checks.append({'name':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open() if l.strip()]
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def close(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(close(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=2e-12)
 return a==b

def wm(y,p,w):
 y=np.asarray(y,int);p=np.asarray(p,float);w=np.asarray(w,float)
 wp=w[:,y==1];wn=w[:,y==0];np_=wp.sum(axis=1);nn=wn.sum(axis=1);n=np_+nn
 pos=p[y==1];neg=p[y==0]
 pair=(pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])
 auc=np.sum((wp@pair)*wn,axis=1)/(np_*nn)
 order=np.argsort(-p,kind='stable');ps=p[order];ys=y[order];ws=w[:,order]
 tp=np.cumsum(ws*ys[None,:],axis=1);fp=np.cumsum(ws*(1-ys)[None,:],axis=1)
 end=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(p)-1]
 group_tp=np.diff(np.c_[np.zeros(len(w)),tp[:,end]],axis=1)
 ap=np.sum(group_tp*np.divide(tp[:,end],tp[:,end]+fp[:,end],out=np.zeros_like(tp[:,end]),where=(tp[:,end]+fp[:,end])>0),axis=1)/np_
 pred=p>=.5;tps=w@(y*pred);fps=w@((1-y)*pred);fns=np_-tps;tns=nn-fps
 f1p=np.divide(2*tps,2*tps+fps+fns,out=np.zeros_like(tps),where=(2*tps+fps+fns)>0)
 f1n=np.divide(2*tns,2*tns+fps+fns,out=np.zeros_like(tns),where=(2*tns+fps+fns)>0)
 bins=np.minimum(np.floor(p*10).astype(int),9);ece=np.zeros(len(w))
 for b in range(10):
  z=bins==b
  if z.any():ece+=np.abs(w[:,z]@(y[z]-p[z]))/n
 return {'auroc':auc,'ap':ap,'macro_f1':.5*(f1p+f1n),'recall':tps/np_,'fpr':fps/nn,'brier':w@((y-p)**2)/n,'ece_10_equal_width':ece}
def point(y,p):
 v=wm(y,p,np.ones((1,len(y))))
 return dict(n=len(y),normal=int(sum(y==0)),anomalous=int(sum(y==1)),**{k:float(a[0]) for k,a in v.items()})
def bootstrap_weights(qs,y):
 st=defaultdict(list)
 for i,q in enumerate(qs):st[q['category'],int(y[i])].append(i)
 rng=np.random.default_rng(42);w=np.zeros((2000,len(y)))
 for key,ix in sorted(st.items()):w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=2000)
 return w

def accepted(stage,contract,mode):
 if mode==8:
  query={r['source_id']:r for r in contract['queries']};assets={r['source_id']:(stage/r['asset']['path']).read_bytes() for r in contract['queries']}
  for groups in contract['supports'].values():
   for rs in groups.values():
    for r in rs:assets[r['source_id']]=(stage/r['asset']['path']).read_bytes()
  ch=hashlib.sha256(canon(contract)).hexdigest()
  expected={(q,a) for q in query for a in support.ARMS}
  req={(sid,a):hashlib.sha256(canon(support.payload(query[sid],a,contract,assets,ch))).hexdigest() for sid,a in expected}
 else:
  query={r['source_id']:r for r in contract['queries']};assets={r['source_id']:(stage/r['asset320']['path']).read_bytes() for r in contract['queries']};ch=hashlib.sha256(canon(contract)).hexdigest();expected={(q,res.ARM) for q in query}
  req={(sid,res.ARM):hashlib.sha256(canon(res.payload(q,assets,ch))).hexdigest() for sid,q in query.items()}
 rs=rows(stage/'predictions.jsonl');by={}
 for row in rs:
  k=(row['source_id'],row['arm']);check('prediction_identity_'+str(mode)+'_'+str(len(by)),k in expected and k not in by and row['contract_sha256']==ch and row['request_sha256']==req[k] and row['response_model']=='jev-1.14.0' and type(row['probability']) in (int,float) and math.isfinite(row['probability']) and 0<=row['probability']<=1 and row['category']==query[k[0]]['category'])
  by[k]=row
 check('complete_unique_targets_'+str(mode),set(by)==expected)
 starts=Counter();intent={};terminal=set();counts=Counter()
 for i,row in enumerate(rows(stage/'attempts.jsonl')):
  k=(row['source_id'],row['arm']);identity=k,row['attempt'];counts[row['event']]+=1
  check('ledger_identity_'+str(mode)+'_'+str(i),k in expected and row['contract_sha256']==ch and row['request_sha256']==req[k])
  if row['event']=='started':
   starts[k]+=1;check('ledger_start_'+str(mode)+'_'+str(i),starts[k]==row['attempt'] and starts[k]<=3 and identity not in intent);intent[identity]=row
  else:
   check('ledger_terminal_'+str(mode)+'_'+str(i),row['event'] in ['succeeded','failed'] and identity in intent and identity not in terminal and row['invocation_id']==intent[identity]['invocation_id']);terminal.add(identity)
   if row['event']=='succeeded':check('success_link_'+str(mode)+'_'+str(i),by[k]['attempt']==row['attempt'])
 check('no_unknown_intents_'+str(mode),set(intent)==terminal)
 check('accepted_success_count_'+str(mode),counts['succeeded']==len(by))
 return by,starts,counts

def run(stage_id):
 check('server',socket.gethostname()=='autodl-container-132444827f-9c594038')
 stage=ROOT/('stage08_support' if stage_id==8 else 'stage10_resolution')
 needed=['status.json','contract.json','prediction_freeze.json','metrics.json','predictions.jsonl','attempts.jsonl']
 if any(not (stage/n).exists() for n in needed):
  print(json.dumps({'status':'pending','stage':stage_id,'reason':'complete/frozen artifacts not all present'}));return
 status=read(stage/'status.json')
 if status['status']!='complete':print(json.dumps({'status':'pending','stage':stage_id,'completed':status.get('completed')}));return
 c=read(stage/'contract.json');freeze=read(stage/'prediction_freeze.json');metric=read(stage/'metrics.json')
 check('status_complete',status['completed']==status['expected']==(600 if stage_id==8 else 120))
 check('prediction_freeze_contract',freeze['contract_sha256']==sha(stage/'contract.json'))
 check('prediction_freeze_content',freeze['predictions_sha256' if stage_id==8 else 'predictions320_sha256']==sha(stage/'predictions.jsonl'))
 if stage_id==8:check('attempt_freeze',freeze['attempts_sha256']==sha(stage/'attempts.jsonl'))
 check('metric_freeze_binding',metric['prediction_freeze']==freeze and metric['status']=='complete' and metric['confirmation_opened'] is False)
 by,starts,events=accepted(stage,c,stage_id)
 qs=c['queries'];sids=[q['source_id'] for q in qs]
 # Approved development labels become accessible only after all complete/freeze guards above.
 config=read(ROOT/'source_baseline/frozen/config.json');labels=read(ROOT/'source_baseline/frozen/labels.json')
 check('approved_label_hash',hashlib.sha256(canon(labels)).hexdigest()==config['labels_sha256'])
 y=np.array([labels[sid] for sid in sids],int)
 check('balanced120',len(y)==120 and int(y.sum())==60)
 if stage_id==8:
  probs={a:np.array([by[sid,a]['probability'] for sid in sids]) for a in support.ARMS};pairs=list(support.PAIRS)
 else:
  ref=ROOT/'stage08_support';rc=read(ref/'contract.json');rf=read(ref/'prediction_freeze.json')
  check('448_reference_freeze',freeze['reference_contract_sha256']==sha(ref/'contract.json') and freeze['reference_predictions_sha256']==sha(ref/'predictions.jsonl')==rf['predictions_sha256'])
  rb,_,_=accepted(ref,rc,8)
  probs={res.ARM:np.array([by[sid,res.ARM]['probability'] for sid in sids]),res.REFERENCE:np.array([rb[sid,'shared0']['probability'] for sid in sids])};pairs=[(res.ARM,res.REFERENCE)]
  check('only120_reused448_no480support',metric['reused448_accepted']==120 and metric['excluded_support_arm_observations']==480)
  check('canvas_pixel_ratio',close(metric['canvas_pixel_ratio_320_over448'],320**2/448**2))
 cats=sorted({q['category'] for q in qs});idx={cat:np.array([i for i,q in enumerate(qs) if q['category']==cat]) for cat in cats}
 points={a:point(y,p) for a,p in probs.items()};category={cat:{a:point(y[ix],p[ix]) for a,p in probs.items()} for cat,ix in idx.items()}
 macro={a:{k:float(np.mean([category[cat][a][k] for cat in cats])) for k in METRICS} for a in probs}
 for a in probs:
  check('point_metrics_'+a,close(points[a],metric['arms'][a]['metrics']))
  for cat in cats:check('category_metrics_'+cat+'_'+a,close(category[cat][a],metric['by_category'][cat][a]))
  check('macro_metrics_'+a,close(macro[a],metric['category_macro'][a]))
 w=bootstrap_weights(qs,y);boot={a:wm(y,p,w) for a,p in probs.items()};cb={cat:{a:wm(y[ix],p[ix],w[:,ix]) for a,p in probs.items()} for cat,ix in idx.items()}
 for a,b in pairs:
  target=metric['paired_bootstrap'][a+'-minus-'+b] if stage_id==8 else metric['paired_bootstrap']
  if stage_id==10:check('paired_direction',target['direction']=='320-minus448')
  for k in METRICS:
   for name,delta,values in [('pooled',points[a][k]-points[b][k],boot[a][k]-boot[b][k]),('category_macro',np.mean([category[cat][a][k]-category[cat][b][k] for cat in cats]),np.mean([cb[cat][a][k]-cb[cat][b][k] for cat in cats],axis=0))]:
    stored=target[name][k]
    check('bootstrap_'+a+'_'+b+'_'+name+'_'+k,close(float(delta),stored['delta']) and close(np.quantile(values,[.025,.975],method='linear').tolist(),stored['ci95']) and stored['replicates']==2000 and stored['seed']==42 and stored['paired'] is True)
 if stage_id==8:
  check('operations',metric['operations']==dict(accepted=600,started=sum(starts.values()),failed=events['failed'],unknown=0))
 else:check('operations320',metric['operations320']==dict(accepted=120,started=sum(starts.values()),failed=events['failed'],unknown=0))
 check('metric_file_unchanged',metric==read(stage/'metrics.json'))
 result={'schema':'jev-independent-completed-stage-audit-v1','status':'complete','stage':stage_id,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'passed':sum(v['passed'] for v in checks),'failed':sum(not v['passed'] for v in checks),'checks':checks,'auditor_sha256':sha(Path(__file__)),'contract_sha256':sha(stage/'contract.json'),'prediction_freeze_sha256':sha(stage/'prediction_freeze.json'),'metrics_sha256':sha(stage/'metrics.json'),'operations':{'accepted':len(by),'started':sum(starts.values()),'failed':events['failed'],'max_target_started':max(starts.values()),'unknown':0},'independent_metric_implementation':'AUC by pairwise weighted positive-negative comparison, AP grouped-threshold cumulative precision; confusion/Brier/ECE independently recomputed; paired multinomial source bootstrap2000 and per-category macro intervals','point_metrics':points,'query_label_source':'frozen approved current120 after complete prediction-freeze checks','gt_masks_opened':False,'confirmation_opened':False,'new_api_requests':0}
 out=ROOT/'stages'/f'{stage_id:02d}_postexecution_audit.json';assert not out.exists(),'Do not overwrite audit'
 with out.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[8,10],required=True);run(ap.parse_args().stage)


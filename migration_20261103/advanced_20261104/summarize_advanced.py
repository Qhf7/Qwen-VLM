"""Recompute all saved prediction metrics and report paired advanced results."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from migration_metrics import evaluate_predictions

KEYS=('auroc','ap','gen_f1','recall','fpr','invalid')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):return dict(mean=statistics.mean(x),sample_sd=statistics.stdev(x) if len(x)>1 else None,n=len(x))
def same(a,b,path=''):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:same(a[k],b[k],path+'/'+k)
 elif isinstance(a,float):assert abs(a-b)<1e-12,(path,a,b)
 else:assert a==b,(path,a,b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).parent);ap.add_argument('--historical',type=Path,default=Path(__file__).resolve().parents[1]/'lora_20261104');args=ap.parse_args()
 rows=[];identities={};recomputed=[]
 for seed in (42,43,44):
  for arm in ('D','Cpair','R'):
   name=('arm_D' if arm=='D' else arm)+'_s'+str(seed);run=args.root/name
   if not (run/'complete.json').exists():continue
   done=json.loads((run/'complete.json').read_text());cfg=json.loads((run/'run.json').read_text());identity=cfg['identity'];identities[arm,seed]=identity
   assert done['steps']==500 and not done['preflight'] and done['test_queries']==0
   assert done['reload_max_abs_score_error']==0 and done['zero_adapter_max_abs_logit_error']==0 and done['frozen_parameter_versions_unchanged']
   schedule=json.loads((run/'sample_indices.json').read_text());assert hashlib.sha256(json.dumps(schedule).encode()).hexdigest()==identity['sample_schedule_sha256']
   assert len(schedule)==2000
   for mode in ('best','final'):
    pp=run/('best_val_generated_predictions.json' if mode=='best' else 'final_val_predictions.json');mp=run/(mode+'_val_metrics.json')
    pred=json.loads(pp.read_text());metrics=json.loads(mp.read_text());assert len(pred)==380 and len({p['path'] for p in pred})==380
    computed=evaluate_predictions(pred,[p['score'] for p in pred],[p['generated'] for p in pred]);same(computed,metrics)
    recomputed.append(dict(file=str(pp.relative_to(args.root)),n=len(pred),sha256=digest(pp),matches=True))
    s=metrics['overall']['scores'];g=metrics['overall']['generated']
    rows.append(dict(arm=arm,seed=seed,mode=mode,step=done['best_step'] if mode=='best' else 500,auroc=s['image_auroc'],ap=s['image_ap'],gen_f1=g['macro_f1'],recall=g['recall'],fpr=g['fpr'],invalid=g['invalid_rate'],category_auroc=metrics['category_macro']['scores']['image_auroc'],peak_gib=done['peak_memory_bytes']/2**30,exposures_per_second=done['exposures_per_second']))
 pairs=[]
 for seed in (42,43,44):
  if ('Cpair',seed) in identities and ('R',seed) in identities:
   a,b=identities['Cpair',seed],identities['R',seed]
   for k in ('seed','bridge_sha256','cache_sha256','decoder_sha256','base_manifest_sha256','sample_schedule_sha256'):assert a[k]==b[k],(seed,k)
   for mode in ('best','final'):
    c=next(x for x in rows if x['arm']=='Cpair' and x['seed']==seed and x['mode']==mode);r=next(x for x in rows if x['arm']=='R' and x['seed']==seed and x['mode']==mode)
    pairs.append(dict(contrast='R_minus_Cpair',seed=seed,mode=mode,delta={k:r[k]-c[k] for k in KEYS}))
  if ('D',seed) in identities:
   cdir=args.historical/('arm_C_s'+str(seed))
   if (cdir/'complete.json').exists():
    prior=json.loads((cdir/'run.json').read_text())['identity']
    for k in ('seed','bridge_sha256','cache_sha256','decoder_sha256','base_manifest_sha256','sample_schedule_sha256'):assert identities['D',seed][k]==prior[k],(seed,k)
    for mode in ('best','final'):
     m=json.loads((cdir/(mode+'_val_metrics.json')).read_text());s=m['overall']['scores'];g=m['overall']['generated'];c=dict(auroc=s['image_auroc'],ap=s['image_ap'],gen_f1=g['macro_f1'],recall=g['recall'],fpr=g['fpr'],invalid=g['invalid_rate']);d=next(x for x in rows if x['arm']=='D' and x['seed']==seed and x['mode']==mode)
     pairs.append(dict(contrast='D_minus_C',seed=seed,mode=mode,delta={k:d[k]-c[k] for k in KEYS}))
 aggregate={};paired={}
 for arm in ('D','Cpair','R'):
  for mode in ('best','final'):
   group=[r for r in rows if r['arm']==arm and r['mode']==mode]
   if len(group)==3:aggregate[arm+'_'+mode]={k:stats([r[k] for r in group]) for k in KEYS+('category_auroc','peak_gib','exposures_per_second')}
 for contrast in ('D_minus_C','R_minus_Cpair'):
  for mode in ('best','final'):
   group=[p for p in pairs if p['contrast']==contrast and p['mode']==mode]
   if len(group)==3:paired[contrast+'_'+mode]={k:dict(**stats([p['delta'][k] for p in group]),positive_seeds=sum(p['delta'][k]>0 for p in group)) for k in KEYS}
 result=dict(state='complete' if len(rows)==18 else 'partial',completed_runs=len(rows)//2,expected_runs=9,rows=rows,aggregate=aggregate,paired_deltas=pairs,paired_aggregate=paired,prediction_recomputations=recomputed,external_results_opened=False)
 (args.root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 lines=['# D 与配对排序实验结果','','状态：'+result['state']+'，仅纳入完成500步并通过工程门槛的正式运行。VisA val380同时用于选点和报告，属于开发验证。best允许step0；final固定500。外部结果未用于本表。','','| Arm | Seed | Model | Step | AUROC | AP | Gen F1 | Recall | FPR | Invalid |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append('| {arm} | {seed} | {mode} | {step} | {auroc:.6f} | {ap:.6f} | {gen_f1:.6f} | {recall:.6f} | {fpr:.6f} | {invalid:.6f} |'.format(**r))
 lines+=['','| 三seed模型 | AUROC均值±样本SD | AP | 生成F1 | Recall | FPR |','|---|---:|---:|---:|---:|---:|']
 for name,m in aggregate.items():lines.append('| %s | %.6f ± %.6f | %.6f | %.6f | %.6f | %.6f |'%(name,m['auroc']['mean'],m['auroc']['sample_sd'],m['ap']['mean'],m['gen_f1']['mean'],m['recall']['mean'],m['fpr']['mean']))
 lines+=['','同seed先相减，再均值；以下单位为百分点。D与历史C同样本序列；R与Cpair具有相同成对采样、batch和CE归约。','','| 配对差 | ΔAUROC均值±SD | ΔAP | ΔGenF1 | ΔRecall | ΔFPR | AUROC上升seed |','|---|---:|---:|---:|---:|---:|---:|']
 for name,m in paired.items():lines.append('| %s | %+.4f ± %.4f | %+.4f | %+.4f | %+.4f | %+.4f | %d/3 |'%(name,100*m['auroc']['mean'],100*m['auroc']['sample_sd'],100*m['ap']['mean'],100*m['gen_f1']['mean'],100*m['recall']['mean'],100*m['fpr']['mean'],m['auroc']['positive_seeds']))
 lines+=['','已逐条重算%d份预测报告；不依据训练loss推断性能，不将开发验证改进称为独立泛化或推理能力证明。'%len(recomputed),'']
 (args.root/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(dict(state=result['state'],completed_runs=result['completed_runs'],recomputed=len(recomputed))))
if __name__=='__main__':main()

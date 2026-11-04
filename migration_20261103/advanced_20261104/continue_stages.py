"""Run S and native ROI stages after completed D/ranking and fresh audits."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path

def write(value,path):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('/root/autodl-tmp/AnomalyGPT'));ap.add_argument('--root',type=Path,default=Path('runs/migration_20261103/advanced_20261104'));args=ap.parse_args()
 os.chdir(args.repo);root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
 state=dict(state='waiting_for_D_rank_and_reload',jobs=[],test_queries=0,external_queries=0);sp=root/'continuation_status.json';write(state,sp)
 for name in ('status.json','verification_status.json'):
  while True:
   value=json.loads((root/name).read_text()) if (root/name).exists() else {}
   if value.get('state') in ('failed','blocked_by_training_failure'):raise RuntimeError('Prior stage failed: '+name)
   if value.get('state')=='complete':break
   time.sleep(15)
 os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('MKL_NUM_THREADS','8');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
 cache=args.repo/'runs/migration_20261103/teacher_cache.pt';targets=root/'structured_targets.jsonl';manifest=args.repo/'runs/migration_20261103/lora_20261104/base_manifest.json';lockpath=root.parent/'gpu.lock'
 def job(name,cmd,marker):
  if marker.exists():
   state['jobs'].append(dict(name=name,state='already_complete'));write(state,sp);return
  state.update(state='running',stage=name,command=[str(x) for x in cmd]);write(state,sp)
  with open(root/(name+'.log'),'a') as log:
   p=subprocess.Popen([str(x) for x in cmd],stdout=log,stderr=subprocess.STDOUT);state['pid']=p.pid;write(state,sp);code=p.wait()
  if code:raise RuntimeError('%s exit=%s; preserve partial directory'% (name,code))
  assert marker.exists(),('missing completion marker',marker)
  state['jobs'].append(dict(name=name,state='complete'));write(state,sp)
 try:
  with open(lockpath,'a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not targets.exists():job('structured_targets',[sys.executable,'structured_geometry.py','--cache',cache,'--out',targets],targets)
   for preflight,seed in [(True,42)]+[(False,s) for s in (42,43,44)]:
    name=('preflight_S' if preflight else 'arm_S')+'_s'+str(seed);out=root/name
    if not preflight:
     check=json.loads((root/'preflight_S_s42/complete.json').read_text());assert check['preflight'] and check['steps']==40
     assert check['reload_max_abs_score_error']==0. and check['frozen_parameter_versions_unchanged'] and check['bridge_unchanged']
    cmd=[sys.executable,'-u','structured_train.py','--cache',cache,'--targets',targets,'--source-c',args.repo/('runs/migration_20261103/lora_20261104/arm_C_s%d'%seed),'--base-manifest',manifest,'--out',out,'--seed',seed]
    if preflight:cmd+=['--preflight','--steps',40,'--val-every',20]
    job(name,cmd,out/'complete.json')
    if not preflight:
     for mode in ('best','final'):job(name+'_verify_'+mode,[sys.executable,'structured_verify.py','--run',out,'--mode',mode],out/('fresh_'+mode+'_reload.json'))
   for name,limit in [('tool_preflight',8),('tool_val',0)]:
    out=root/name
    if name=='tool_val':assert json.loads((root/'tool_preflight/run.json').read_text())['status']=='complete'
    if not (out/'run.json').exists() or json.loads((out/'run.json').read_text()).get('status')!='complete':
     job(name,[sys.executable,'-u','tool_workflow.py','--run','--limit',limit,'--out',out],out/'run.json')
    else:state['jobs'].append(dict(name=name,state='already_complete'));write(state,sp)
    assert json.loads((out/'run.json').read_text())['status']=='complete'
    job(name+'_audit',[sys.executable,'tool_audit.py','--run',out,'--out',out/'audit.json','--with-masks'],out/'audit.json')
   state.update(state='complete',stage='complete',finished=time.time())
 except Exception as error:
  state.update(state='failed',error=str(error),traceback=traceback.format_exc());raise
 finally:write(state,sp)
if __name__=='__main__':main()

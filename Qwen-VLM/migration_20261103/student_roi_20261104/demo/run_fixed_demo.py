from pathlib import Path
import subprocess,json,sys,fcntl,hashlib,time
r=Path('/root/autodl-tmp/AnomalyGPT');b=r/'runs/migration_20261103/student_roi_20261104';d=b/'demo';d.mkdir(exist_ok=True)
lock=(r/'runs/migration_20261103/gpu.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
expected='f500c9c7cbf27c2ce28e923babf7e9c50eeed90401f331c101a25b6137b91ee4'
assert hashlib.sha256((r/'migration_infer_student_roi.py').read_bytes()).hexdigest()==expected
for tag,kind,name in [('normal0784','Normal','0784'),('anomaly010','Anomaly','010')]:
 image='/root/autodl-tmp/industrial_vlm/datasets/VisA/candle/Data/Images/'+kind+'/'+name+'.JPG'
 out=d/(tag+'.json')
 if out.exists():
  print(json.dumps({'tag':tag,'status':'already_exists'}),flush=True);continue
 args=[sys.executable,str(r/'migration_infer_student_roi.py'),'--model','/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct','--head',str(r/'runs/migration_20261103/dense_pre_gt_s42/best.pt'),'--image',image,'--outjson',str(out),'--device','cuda:0','--max-new-tokens','32','--heatmap','--run']
 print(json.dumps({'tag':tag,'status':'starting'}),flush=True)
 subprocess.run(args,cwd=r,check=True)
 print(json.dumps({'tag':tag,'status':'complete'}),flush=True)
print(json.dumps({'status':'complete','n':2}),flush=True)

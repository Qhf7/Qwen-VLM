import argparse,csv,json,random
from pathlib import Path

CLASSES=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--train_per_class',type=int,default=4); p.add_argument('--test_per_class',type=int,default=2); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 rows=list(csv.DictReader(open(a.csv))); rng=random.Random(a.seed)
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
 records={}
 for split,n,out_name in [('train',a.train_per_class,'balanced_train.csv'),('test',a.test_per_class,'balanced_test.csv')]:
  selected=[]
  for cls in CLASSES:
   pool=[r for r in rows if r['object']==cls and r['split']==split and r['label']=='normal']
   if len(pool)<n: raise RuntimeError(f'{cls} {split}: {len(pool)}<{n}')
   pool=sorted(pool,key=lambda r:r['image']); rng.shuffle(pool); chosen=sorted(pool[:n],key=lambda r:r['image'])
   selected += chosen; records[f'{split}:{cls}']=[r['image'] for r in chosen]
  path=out/out_name
  with open(path,'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['object','split','label','image','mask']); w.writeheader(); w.writerows(selected)
  print(path,len(selected))
 json.dump({'seed':a.seed,'classes':CLASSES,'train_per_class':a.train_per_class,'test_per_class':a.test_per_class,'records':records},open(out/'balanced_manifest.json','w'),indent=2)
if __name__=='__main__': main()

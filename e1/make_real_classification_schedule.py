"""Build a deterministic, class-balanced real-view classification schedule."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,default=Path('artifacts/splits/fold1/train.parquet')); p.add_argument('--cache-manifest',type=Path,default=Path('artifacts/e1/train448_cache_manifest.csv')); p.add_argument('--cache-root',type=Path,default=Path('Train448')); p.add_argument('--out-dir',type=Path,default=Path('artifacts/e1_real_classification')); p.add_argument('--steps',type=int,default=1000); p.add_argument('--batch-size',type=int,default=8); args=p.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(args.train).copy()
    cached=set(pd.read_csv(args.cache_manifest).image_path)
    missing=df[~df.image_path.isin(cached) | ~df.image_path.map(lambda x: (args.cache_root / x).is_file())]
    if len(missing):
        raise RuntimeError(f'Full training cache required: {len(missing)} missing images; first={missing.iloc[0].image_path}')
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError('batch-size must be positive and even')
    if args.steps <= 0 or not df.view_label.isin([0,1]).all() or df.image_path.duplicated().any():
        raise ValueError('Invalid steps, image labels, or duplicate training images')
    # view_label is the target: an invisible angle of an NG part is normal.
    real=df[df.view_label.eq(1)].copy(); normal=df[df.view_label.eq(0)].copy()
    pools={'real_anomaly':real,'normal':normal}; rows=[]
    rng=np.random.default_rng(20260901)
    for source,part in pools.items():
        by_class={}
        for cls,g in part.groupby('class_name',sort=True):
            vals=g.index.to_numpy().copy(); rng.shuffle(vals); by_class[cls]=df.loc[vals].reset_index(drop=True)
        pools[source]=by_class
        if set(by_class) != set(df.class_name):
            raise ValueError(f'{source} must contain every component class')
    cursors={(source,cls):0 for source,by_class in pools.items() for cls in by_class}
    for step in range(1,args.steps+1):
        for pos in range(args.batch_size):
            source='real_anomaly' if pos < args.batch_size//2 else 'normal'
            classes=sorted(pools[source]); cls=classes[((step-1)*(args.batch_size//2)+(pos % (args.batch_size//2))) % len(classes)]
            pool=pools[source][cls]
            key=(source,cls); cursor=cursors[key]
            if cursor and cursor % len(pool) == 0:
                pool=pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
                pools[source][cls]=pool
            row=pool.iloc[cursor % len(pool)]; cursors[key]+=1
            rows.append({'step':step,'batch_position':pos,'image_path':row.image_path,'source_type':source})
    # pool index is the row identity consumed by the training Dataset.
    pool=pd.concat([real.assign(source_type='real_anomaly'),normal.assign(source_type='normal')],ignore_index=True); pool.insert(0,'pool_index',np.arange(len(pool)))
    # rebuild dataset indices against concatenated pool
    lookup={(r.image_path,r.source_type):int(i) for i,r in pool.iterrows()}
    schedule=pd.DataFrame(rows); schedule['dataset_index']=[lookup[(p,s)] for p,s in zip(schedule.image_path,schedule.source_type)]
    pool.to_parquet(args.out_dir/'real_classification_pool.parquet',index=False); pool.to_csv(args.out_dir/'real_classification_pool.csv',index=False); schedule.to_parquet(args.out_dir/'real_classification_schedule.parquet',index=False); schedule.to_csv(args.out_dir/'real_classification_schedule.csv',index=False)
    print({'real_views':len(real),'normal_views':len(normal),'pool':len(pool),'steps':args.steps,'batch_size':args.batch_size})
if __name__=='__main__': main()

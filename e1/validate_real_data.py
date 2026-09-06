"""Audit full cached training data and deterministic single-image sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--pool-dir',type=Path,required=True)
    p.add_argument('--train',type=Path,default=Path('artifacts/splits/fold1/train.parquet'))
    p.add_argument('--dev',type=Path,default=Path('artifacts/splits/fold1/dev.parquet'))
    args=p.parse_args()
    train=pd.read_parquet(args.train); dev=pd.read_parquet(args.dev)
    manifest=pd.read_csv(args.manifest)
    pool=pd.read_parquet(args.pool_dir/'real_classification_pool.parquet')
    schedule=pd.read_parquet(args.pool_dir/'real_classification_schedule.parquet')
    assert len(train)==len(pool)==len(manifest)==18775
    assert set(train.image_path)==set(pool.image_path)==set(manifest.image_path)
    assert not pool.image_path.duplicated().any() and not manifest.image_path.duplicated().any()
    assert set(train.class_name)==set(dev.class_name) and train.class_name.nunique()==18
    assert not set(train.group_path)&set(dev.group_path)
    assert not set(train.visual_group_id)&set(dev.visual_group_id)
    assert pool.view_label.eq(pool.image_path.map(train.set_index('image_path').view_label)).all()
    assert pool.source_type.map({'normal':0,'real_anomaly':1}).eq(pool.view_label).all()
    assert len(schedule)==8000 and set(schedule.step)==set(range(1,1001))
    assert schedule.groupby(['step','source_type']).size().eq(4).all()
    expected=pool.iloc[schedule.dataset_index].reset_index(drop=True)
    assert expected.image_path.eq(schedule.image_path).all()
    assert expected.source_type.eq(schedule.source_type).all()
    checks=[]
    for step in (250,500,750,1000):
        part=expected[schedule.step<=step]
        counts=part.groupby(['source_type','class_name']).size()
        assert len(counts)==36 and counts.max()-counts.min()<=1
        checks.append(dict(step=step,unique_images=part.image_path.nunique(),
                           min_class_label_count=int(counts.min()),max_class_label_count=int(counts.max())))
    for row in manifest.itertuples(index=False):
        with Image.open(args.cache_root/row.image_path) as image:
            assert image.size==(448,448),row.image_path
            pixels=np.asarray(image.convert('RGB'))
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==row.image_pixel_sha256,row.image_path
    report=dict(passed=True,train_images=len(train),normal=int(train.view_label.eq(0).sum()),
                anomaly=int(train.view_label.eq(1).sum()),classes=18,steps=checks,
                cached_image_hashes_verified=len(manifest),group_overlap=0,visual_group_overlap=0)
    (args.pool_dir/'data_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

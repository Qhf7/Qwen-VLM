"""Real-view pools and deterministic eight-example Patch9 batches."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from e1.patch9_v2_geometry import (make_patch_specs, crop_image, crop_mask,
    resize_image, resize_mask, read_rgb, read_mask)

KINDS = ('anomaly_full', 'anomaly_full', 'positive_patch', 'positive_patch',
         'anomaly_empty_patch', 'normal_empty_patch', 'normal_full', 'normal_full')

def relative(root, name):
    p = Path(str(name))
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Invalid relative path: {name}')
    return Path(root) / p

def check_splits(train, dev):
    for name, df in [('train', train), ('dev', dev)]:
        for col in ('image_path', 'mask_path', 'view_label', 'class_name', 'group_path', 'visual_group_id'):
            if col not in df:
                raise ValueError(f'{name}: missing {col}')
        if df.image_path.duplicated().any() or not df.view_label.isin([0, 1]).all():
            raise ValueError(f'{name}: duplicate images or invalid view_label')
        if set(df.view_label.unique()) != {0, 1}:
            raise ValueError(f'{name}: both normal and anomaly views required; use full fold1 split')
        if df[['class_name','group_path','visual_group_id','image_path']].isna().any().any():
            raise ValueError(f'{name}: missing split identity')
    if set(train.class_name) != set(dev.class_name):
        raise ValueError('train/dev category mismatch')
    for key in ('image_path', 'group_path', 'visual_group_id'):
        if set(train[key]) & set(dev[key]):
            raise ValueError(f'train/dev overlap: {key}')

def audit_frame(df, root, gt_root):
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        with Image.open(relative(root, row.image_path)) as im:
            width, height = im.size
        mask = None
        if pd.notna(row.mask_path) and str(row.mask_path):
            mask = read_mask(relative(gt_root, row.mask_path))
            if mask.shape != (height, width):
                raise ValueError(f'Image/mask mismatch: {row.image_path}')
        label = int(row.view_label)
        if label and (mask is None or not mask.any()):
            raise ValueError(f'Anomaly requires nonempty real mask: {row.image_path}')
        if not label and mask is not None and mask.any():
            raise ValueError(f'Normal view has nonempty mask: {row.image_path}')
        positive, negative, vanished = [], [], []
        for spec in make_patch_specs(width, height):
            raw_positive = mask is not None and mask[spec.y0:spec.y1,spec.x0:spec.x1].any()
            if not raw_positive:
                negative.append(spec.index)
            elif crop_mask(mask, spec).any():
                positive.append(spec.index)
            else:
                vanished.append(spec.index)
        full_positive = bool(mask is not None and resize_mask(mask).any())
        if label and not full_positive:
            raise ValueError(f'Full mask disappears at448: {row.image_path}')
        records.append(dict(index=i, positive=positive, negative=negative, vanished=vanished))
        if (i+1) % 1000 == 0:
            print(f'[data audit] {i+1}/{len(df)}', flush=True)
    return records

def make_schedule(df, records, steps, seed):
    rng = np.random.default_rng(seed)
    pools = {k: {} for k in set(KINDS)}
    for rec, row in zip(records, df.itertuples(index=False)):
        kinds = ['anomaly_full'] if int(row.view_label) else ['normal_full', 'normal_empty_patch']
        if row.view_label and rec['positive']:
            kinds.append('positive_patch')
        if row.view_label and rec['negative']:
            kinds.append('anomaly_empty_patch')
        for kind in kinds:
            pools[kind].setdefault(row.class_name, {}).setdefault(row.group_path, []).append(rec['index'])
    cycles = {}
    for kind, pool in pools.items():
        if not pool:
            raise ValueError(f'Empty sampling pool: {kind}')
        if set(pool) != set(df.class_name):
            raise ValueError(f'Category missing in sampling pool: {kind}')
        cycles[kind] = []
    result = []
    for step in range(1, steps+1):
        for position, kind in enumerate(KINDS):
            if not cycles[kind]:
                cycles[kind] = list(rng.permutation(sorted(pools[kind])))
            cls = cycles[kind].pop()
            groups = pools[kind][cls]
            group = rng.choice(sorted(groups))
            index = int(rng.choice(groups[group]))
            choices = records[index]['positive' if kind == 'positive_patch' else 'negative']
            patch = int(rng.choice(choices)) if 'patch' in kind else -1
            result.append(dict(step=step, position=position, kind=kind, index=index,
                               patch=patch, image_path=df.iloc[index].image_path, class_name=cls))
    return pd.DataFrame(result)

def example(df, selection, root, gt_root):
    row = df.iloc[int(selection.index)]
    image = read_rgb(relative(root, row.image_path))
    mask = read_mask(relative(gt_root, row.mask_path)) if row.view_label else np.zeros(image.shape[:2], bool)
    if image.shape[:2] != mask.shape:
        raise ValueError('Image/mask shape mismatch')
    if selection.patch == -1:
        x, y = resize_image(image), resize_mask(mask)
    else:
        spec = make_patch_specs(image.shape[1], image.shape[0])[int(selection.patch)]
        x, y = crop_image(image, spec), crop_mask(mask, spec)
    expected_positive = selection.kind in ('anomaly_full', 'positive_patch')
    if bool(y.any()) != expected_positive:
        raise ValueError(f'Target mismatch: {selection.kind} {row.image_path}')
    return x, y

"""Full-image and nine-patch Dev evaluation, with separate normal metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from e1.patch9_v2_geometry import (read_rgb, read_mask, resize_image, resize_mask,
    make_patch_specs, crop_image, fuse_probabilities)
from e1.patch9_training_data import relative

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)

def measure(prob, gt):
    if prob.shape != gt.shape or not np.isfinite(prob).all():
        raise ValueError('Invalid prediction shape or finite check')
    pred = prob >= .5
    tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
    return dict(tp=tp, fp=fp, fn=fn, dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                pixel_precision=tp/(tp+fp) if tp+fp else 0.,
                pixel_recall=tp/(tp+fn) if tp+fn else 0.,
                predicted_area=float(pred.mean()), probability_mean=float(prob.mean()))

def summarize(frame):
    positive = frame[frame.view_label.eq(1)]
    normal = frame[frame.view_label.eq(0)]
    if positive.empty:
        raise ValueError('Evaluation requires anomaly views')
    per_class = positive.groupby('class_name')[['dice','iou']].mean()
    tp, fp, fn = (int(positive[k].sum()) for k in ('tp','fp','fn'))
    return dict(n=len(frame), anomaly_n=len(positive), normal_n=len(normal),
        macro_dice=float(per_class.dice.mean()), macro_iou=float(per_class.iou.mean()),
        anomaly_dice=float(positive.dice.mean()), anomaly_iou=float(positive.iou.mean()),
        # Explicitly distinguish per-image means from pooled pixel counts.
        pixel_precision_image_mean=float(positive.pixel_precision.mean()),
        pixel_recall_image_mean=float(positive.pixel_recall.mean()),
        pixel_precision_pooled=tp/(tp+fp) if tp+fp else 0.,
        pixel_recall_pooled=tp/(tp+fn) if tp+fn else 0.,
        empty_prediction_rate_anomaly=float(positive.predicted_area.eq(0).mean()),
        normal_nonempty_rate=float(normal.predicted_area.gt(0).mean()) if len(normal) else None,
        normal_area_mean=float(normal.predicted_area.mean()) if len(normal) else None,
        empty_mask_dice=float(normal.dice.mean()) if len(normal) else None,
        anomaly_area_mean=float(positive.predicted_area.mean()),
        anomaly_area_p95=float(positive.predicted_area.quantile(.95)),
        all_area_p95=float(frame.predicted_area.quantile(.95)))

def evaluate(predict, dev, root, gt_root, out, step):
    """Predict is a callable RGB448 -> FP32 probability448; no GT-based gating."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(dev.itertuples(index=False), 1):
        image = read_rgb(relative(root, row.image_path))
        gt = resize_mask(read_mask(relative(gt_root,row.mask_path))) if row.view_label else np.zeros((448,448),bool)
        full = predict(resize_image(image))
        specs = make_patch_specs(image.shape[1], image.shape[0])
        patches = {s.index: predict(crop_image(image,s)) for s in specs}
        stitched = fuse_probabilities(full, patches, specs, image.shape[:2], patch_weight=.5)
        ratio = float(gt.mean())
        area = 'normal' if not row.view_label else ('tiny' if ratio <= .001 else 'small' if ratio <= .01 else 'large')
        for variant, prob in [('full',full), ('mean_w0.5',stitched)]:
            rows.append(dict(image_path=row.image_path,class_name=row.class_name,view_label=int(row.view_label),
                             variant=variant,area_bucket=area,gt_ratio=ratio,**measure(prob,gt)))
        if i % 100 == 0:
            print(f'[eval {step}] {i}/{len(dev)}',flush=True)
    frame = pd.DataFrame(rows)
    tag = f'step{step:05d}'
    frame.to_csv(out/f'{tag}_views.csv',index=False)
    columns = ['dice','iou','pixel_precision','pixel_recall','predicted_area']
    for name,keys in [('per_class',['variant','view_label','class_name']),('per_area',['variant','area_bucket'])]:
        grouped=frame.groupby(keys)
        table=grouped[columns].mean()
        table['n']=grouped.size()
        table.to_csv(out/f'{tag}_{name}.csv')
    result=dict(step=step,threshold=.5,area_definition='GT448 ratio: tiny<=0.001, small<=0.01, large>0.01',
                variants={name:summarize(g) for name,g in frame.groupby('variant')})
    write_json(out/f'{tag}_metrics.json',result)
    print(f'[eval complete {step}] {json.dumps(result)}',flush=True)
    return result

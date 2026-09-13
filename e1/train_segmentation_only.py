"""Frozen Qwen3 features: decoder-only segmentation on anomalous single views."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from b0.single_view import candidate_ids
from e1.model import QwenE1

SIDE = 448
EVAL_STEPS = (250, 500, 750, 1000, 1500, 2000, 2500, 3000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def relative_path(root, value):
    path = Path(str(value))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected dataset-relative path: {value}')
    return root / path


def load_mask(row, root):
    path = relative_path(root, row.mask_path)
    with Image.open(path) as image:
        original = np.asarray(image.convert('L')) > 0
    if not original.any():
        raise ValueError(f'Empty original anomaly mask: {path}')
    mask = np.asarray(Image.fromarray(original.astype(np.uint8)).resize(
        (SIDE, SIDE), Image.Resampling.NEAREST)).copy() > 0
    if not mask.any():
        raise ValueError(f'Anomaly disappears after nearest resize: {path}')
    return mask


def audit_data(args, train, dev):
    report = {}
    for split, frame, root, expected in (
        ('train', train, args.root, 15782), ('dev', dev, args.dev_root, 1775)
    ):
        if len(frame) != expected or not frame.view_label.eq(1).all():
            raise ValueError(f'{split}: expected {expected} anomalous images')
        if frame.image_path.duplicated().any() or frame.mask_path.isna().any():
            raise ValueError(f'{split}: duplicate images or missing mask paths')
        for n, row in enumerate(frame.itertuples(index=False), 1):
            with Image.open(relative_path(root, row.image_path)) as image:
                image.load()
                if image.size != (SIDE, SIDE):
                    raise ValueError(f'Image is not direct448: {row.image_path}')
            with Image.open(relative_path(args.gt_root, row.image_path)) as image:
                original_size = image.size
            with Image.open(relative_path(args.gt_root, row.mask_path)) as image:
                if image.size != original_size:
                    raise ValueError(f'Original image/mask size mismatch: {row.image_path}')
            load_mask(row, args.gt_root)
            if n % 2000 == 0:
                print(f'[audit {split}] {n}/{len(frame)}', flush=True)
        report[split] = {'images': len(frame), 'classes': int(frame.class_name.nunique()),
                         'nonempty_masks': len(frame), 'image_root': str(root)}
    if set(train.class_name) != set(dev.class_name) or train.class_name.nunique() != 18:
        raise ValueError('Expected the same 18 categories in train and dev')
    for column in ('image_path', 'group_path', 'visual_group_id'):
        if column in train and column in dev:
            if set(train[column].dropna()) & set(dev[column].dropna()):
                raise ValueError(f'Train/dev overlap: {column}')
    report.update(passed=True, train_sha256=digest(args.train), dev_sha256=digest(args.dev))
    return report


def make_schedule(train, steps, accumulation, seed):
    """Shuffle class cycles, then sample a group and a view uniformly."""
    rng = np.random.default_rng(seed)
    classes = sorted(train.class_name.unique())
    groups = {c: [g.index.to_numpy() for _, g in train[train.class_name.eq(c)].groupby('group_path')]
              for c in classes}
    order = []
    while len(order) < steps * accumulation:
        order.extend(rng.permutation(classes).tolist())
    rows = []
    for i, category in enumerate(order[:steps * accumulation]):
        group = groups[category][int(rng.integers(len(groups[category])))]
        index = int(rng.choice(group))
        rows.append({'step': i // accumulation + 1, 'position': i % accumulation,
                     'dataset_index': index, 'class_name': category,
                     'image_path': str(train.iloc[index].image_path)})
    return pd.DataFrame(rows)


def encode(proc, row, root):
    with Image.open(relative_path(root, row.image_path)) as image:
        batch = proc.image_processor(images=image.convert('RGB'), return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in batch.items()}
    if inputs['image_grid_thw'].tolist() != [[1, 28, 28]]:
        raise ValueError(f'Unexpected Qwen3 grid: {inputs["image_grid_thw"]}')
    return inputs


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def check_gradients(model):
    names = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not name.startswith('decoder.') or not p.requires_grad:
                raise RuntimeError(f'Unexpected gradient: {name}')
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f'Nonfinite gradient: {name}')
            names.append(name)
        elif p.requires_grad:
            raise RuntimeError(f'Disconnected trainable parameter: {name}')
    if not names:
        raise RuntimeError('No decoder gradients')
    return names


def validate_model(model, proc, train, args):
    row = train.iloc[0]
    inputs = encode(proc, row, args.root)
    features, merged, deepstack = model._vision_features(inputs['pixel_values'], inputs['image_grid_thw'])
    shapes = [list(f.shape) for f in features]
    if shapes != [[1, 1024, 28, 28]] * 3:
        raise RuntimeError(f'Unexpected feature dimensions: {shapes}')
    target = torch.from_numpy(load_mask(row, args.gt_root)[None, None].astype(np.float32)).cuda()
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.segmentation_only(inputs, target)
    if tuple(result.mask_logits.shape) != (1, 1, 448, 448) or not torch.isfinite(result.loss):
        raise RuntimeError('Invalid output shape/loss')
    result.loss.backward()
    names = check_gradients(model)
    norm = float(torch.nn.utils.clip_grad_norm_(trainable(model), 1.0))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError('Zero or nonfinite decoder gradient')
    report = {'passed': True, 'feature_shapes': shapes, 'merged_shape': list(merged.shape),
              'deepstack_shapes': [list(f.shape) for f in deepstack], 'language_model_used': False,
              'mask_shape': list(result.mask_logits.shape), 'loss': float(result.loss.detach()),
              'gradient_norm': norm, 'gradient_parameters': names,
              'trainable_parameters': sum(p.numel() for p in trainable(model)),
              'base_trainable': any(p.requires_grad for p in model.base_model.parameters()),
              'base_has_gradient': any(p.grad is not None for p in model.base_model.parameters()),
              'prompt_trainable': any(p.requires_grad for p in model.prompt_learner.parameters()),
              'prompt_has_gradient': any(p.grad is not None for p in model.prompt_learner.parameters())}
    model.zero_grad(set_to_none=True)
    return report


@torch.inference_mode()
def evaluate(model, proc, dev, args, step):
    model.eval()
    rows = []
    mask_dir = args.out / f'step{step:04d}_masks'
    mask_dir.mkdir()
    for n, row in enumerate(dev.itertuples(index=False), 1):
        gt = load_mask(row, args.gt_root)
        target = torch.from_numpy(gt[None, None].astype(np.float32)).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model.segmentation_only(encode(proc, row, args.dev_root), target)
        probability = torch.sigmoid(result.mask_logits.float())[0, 0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise RuntimeError('Nonfinite evaluation prediction')
        pred = probability >= .5
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        rows.append({'image_path': row.image_path, 'class_name': row.class_name, 'label': 1,
                     'gt_area': int(gt.sum()), 'pred_area': int(pred.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
                     'dice': 2*tp/(2*tp+fp+fn), 'iou': tp/(tp+fp+fn),
                     'pixel_precision': tp/(tp+fp) if tp+fp else 0., 'pixel_recall': tp/(tp+fn),
                     'predicted_area': float(pred.mean()), 'mask_probability_mean': float(probability.mean()),
                     'gt_ratio': float(gt.mean()), 'loss': float(result.loss)})
        destination = mask_dir / str(row.image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred.astype(np.uint8)*255).save(destination)
        if n % 250 == 0:
            print(f'[eval {step}] {n}/{len(dev)}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / f'step{step:04d}_views.csv', index=False)
    metrics = ['dice', 'iou', 'pixel_precision', 'pixel_recall', 'predicted_area', 'loss']
    per_class = frame.groupby('class_name')[metrics].mean()
    per_class['n'] = frame.groupby('class_name').size()
    per_class.to_csv(args.out / f'step{step:04d}_per_class.csv')
    buckets = pd.cut(frame.gt_ratio, [0, .001, .01, 1], labels=['tiny_0_0.1pct','small_0.1_1pct','large_1_100pct'])
    by_area = frame.groupby(buckets, observed=True)[metrics].mean()
    by_area['n'] = frame.groupby(buckets, observed=True).size()
    by_area.to_csv(args.out / f'step{step:04d}_per_area.csv')
    summary = {'step': step, 'n': len(frame), 'nonempty_n': len(frame), 'threshold': .5,
               **{k: float(frame[k].mean()) for k in metrics}, 'macro_dice': float(per_class.dice.mean()),
               'macro_iou': float(per_class.iou.mean()), 'area_p95': float(frame.predicted_area.quantile(.95))}
    write_json(args.out / f'step{step:04d}_metrics.json', summary)
    print(f'[eval complete] {json.dumps(summary)}', flush=True)
    return summary


def save_checkpoint(path, state):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-id', default='models/Qwen3-VL-2B-Instruct')
    p.add_argument('--train', type=Path, default=Path('artifacts/segmentation_real_anomaly/train.parquet'))
    p.add_argument('--dev', type=Path, default=Path('artifacts/segmentation_real_anomaly/dev.parquet'))
    p.add_argument('--root', type=Path, default=Path('Train448_verified_20260905'))
    p.add_argument('--dev-root', type=Path, default=Path('Dataset448'))
    p.add_argument('--gt-root', type=Path, default=Path('Dataset'))
    p.add_argument('--out', type=Path, default=Path('artifacts/segmentation_qwen3_decoder_only'))
    p.add_argument('--steps', type=int, default=3000, choices=[1000, 3000])
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'training_history.csv').exists():
        raise FileExistsError('An existing experiment must not be overwritten')
    seed = 20260901; accumulation = 8
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is required')
    train = pd.read_parquet(args.train).reset_index(drop=True)
    dev = pd.read_parquet(args.dev).reset_index(drop=True)
    write_json(args.out / 'data_validation.json', audit_data(args, train, dev))
    proc = AutoProcessor.from_pretrained(args.model_id, min_pixels=SIDE**2, max_pixels=SIDE**2)
    base = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, dtype=torch.bfloat16,
                       attn_implementation='eager').to('cuda').eval()
    layers = tuple(base.config.vision_config.deepstack_visual_indexes)
    if layers != (5, 11, 17):
        raise ValueError(f'Unexpected Qwen3 layers: {layers}')
    model = QwenE1(base, candidate_ids(proc.tokenizer,'Yes'), candidate_ids(proc.tokenizer,'No'),
                   feature_layers=layers, decoder_channels=288).cuda()
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.decoder.region_head.requires_grad_(False)
    model.train()
    report = validate_model(model, proc, train, args)
    write_json(args.out / 'forward_validation.json', report)
    print('[validation] passed: decoder gradients only, 448x448, BF16', flush=True)
    preflight = argparse.Namespace(**vars(args))
    preflight.out = args.out / 'preflight'
    preflight.out.mkdir()
    evaluate(model, proc, dev.groupby('class_name', sort=True).head(1), preflight, 0)
    print('[validation] evaluation and mask export passed on one Dev image per class', flush=True)
    if args.validate_only:
        return
    schedule = make_schedule(train, args.steps, accumulation, seed)
    schedule.to_parquet(args.out / 'sampling_schedule.parquet', index=False)
    protocol = {'model': args.model_id, 'feature_layers': list(layers), 'decoder_channels': 288,
                'seed': seed, 'steps': args.steps, 'evaluation_steps': list(EVAL_STEPS),
                'micro_batch_size': 1, 'gradient_accumulation': accumulation, 'logical_batch_size': 8,
                'precision': 'BF16 autocast, FP32 decoder parameters and loss',
                'optimizer': 'AdamW', 'learning_rate': .0001, 'weight_decay': .01,
                'warmup_steps': 50, 'scheduler': 'cosine', 'gradient_clip_norm': 1.,
                'loss': 'existing focal(gamma=2,positive_alpha=0.25)+soft_Dice(nonempty)',
                'sampler': 'shuffled class cycles; uniform group and view within class',
                'checkpoint_selection': 'Dev category-macro Dice at threshold 0.5',
                'train_sha256': digest(args.train), 'dev_sha256': digest(args.dev),
                'source_sha256': digest(Path(__file__)), 'model_code_sha256': digest(Path(__file__).with_name('model.py')),
                'train_images': len(train), 'dev_images': len(dev),
                'trainable_parameter_names': [n for n,p in model.named_parameters() if p.requires_grad]}
    write_json(args.out / 'frozen_training_protocol.json', protocol)
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/50 if s<50 else
                    .5*(1+math.cos(math.pi*(s-50)/max(1,args.steps-50))))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out / 'tensorboard'))
    fields = ['step','loss','focal_loss','dice_loss','grad_norm','lr','step_s','elapsed_s','predicted_area']
    best = -1.; started = time.perf_counter()
    with (args.out / 'training_history.csv').open('x', newline='') as stream:
        history = csv.DictWriter(stream, fieldnames=fields); history.writeheader(); stream.flush()
        for step, group in schedule.groupby('step', sort=True):
            model.train(); optimizer.zero_grad(set_to_none=True); begin = time.perf_counter()
            values = np.zeros(4)
            for index in group.dataset_index:
                row = train.iloc[int(index)]
                target = torch.from_numpy(load_mask(row, args.gt_root)[None,None].astype(np.float32)).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model.segmentation_only(encode(proc,row,args.root), target)
                if not torch.isfinite(result.loss):
                    raise RuntimeError('Nonfinite segmentation loss')
                (result.loss / accumulation).backward()
                values += [float(result.loss.detach()), float(result.focal_loss.detach()),
                           float(result.dice_loss.detach()), float(result.predicted_area.detach())]
            check_gradients(model)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable(model), 1.0))
            if not math.isfinite(norm):
                raise RuntimeError('Nonfinite accumulated gradient')
            lr = optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step()
            values /= accumulation
            record = dict(zip(fields, [int(step), *values[:3], norm, lr,
                          time.perf_counter()-begin, time.perf_counter()-started, values[3]]))
            history.writerow(record); stream.flush()
            for key in ('loss','focal_loss','dice_loss','grad_norm','predicted_area'):
                writer.add_scalar('train/'+key, record[key], step)
            writer.flush()
            if step == 1 or step % 10 == 0:
                print(f'[train] {step}/{args.steps} loss={values[0]:.5f} area={values[3]:.5f} step_s={record["step_s"]:.2f}', flush=True)
            if step % 50 == 0:
                state = {'step': int(step), 'decoder': model.decoder.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'protocol': protocol}
                save_checkpoint(args.out/'latest.pt', state)
            if step in EVAL_STEPS:
                save_checkpoint(args.out/f'step{step:04d}.pt', state)
                metrics = evaluate(model,proc,dev,args,step)
                for key in ('macro_dice','macro_iou','pixel_recall'):
                    writer.add_scalar('dev/'+key,metrics[key],step)
                if metrics['macro_dice'] > best:
                    best = metrics['macro_dice']; save_checkpoint(args.out/'best.pt',state)
    writer.close()
    write_json(args.out/'training_summary.json', {'completed_steps':args.steps,'best_macro_dice':best,
               'elapsed_s':time.perf_counter()-started,'evaluation_steps':list(EVAL_STEPS)})
    print('[complete] 1000 optimizer steps and four full Dev evaluations',flush=True)


if __name__ == '__main__':
    main()

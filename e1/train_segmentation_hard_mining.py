"""Independent v4/v5 hard-mining ablations; original models are read-only."""
import argparse
import csv
import copy
import importlib
import json
import random
import time
from pathlib import Path
from e1.dataset_config import BASE_DEV, BASE_TRAIN, assert_original_paths, assert_original_split

import numpy as np
import pandas as pd
import torch

from e1.hard_example_sampler import HardSampler, difficulty
from e1.patch9 import crop_resize, crop_resize_mask, grid_boxes, positive_boxes
from e1.patch9_v2_geometry import resize_mask
from e1.train_segmentation_patch9_v2_geometry import predictor


def save(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def positive_pair(base, row, root, rng, version):
    prepared = base.prepare_examples(row, root, root, rng)
    crops, ratio = prepared if version == 'v5' else (prepared, 0.)
    valid = [(im, m, tag) for im, m, tag in crops if m.any()]
    repaired = len(crops) != 2 or len(valid) != 2
    if not repaired:
        return valid, ratio, False
    # (a) Preserve every surviving target; the row is never erased.
    # (c) Fill unusable slots with another standard nine-grid window (0.4 of the
    # image, stride 0.3) that still contains the defect, so the replacement
    # shares the training magnification. The previous 33px positive-centred crop
    # magnified about 13x and was far outside the training input distribution.
    image, mask = base.load_original(row, root, root)
    if not mask.any():
        raise ValueError(f'Anomaly has empty original mask: {row.image_path}')
    boxes = grid_boxes(image.shape[1], image.shape[0])
    positive = positive_boxes(mask, boxes)
    if not len(positive):
        raise ValueError(f'Defect survives in no grid window: {row.image_path}')
    used = {int(tag[5:]) for _, _, tag in crops
            if tag.startswith('patch') and tag[5:].isdigit()}
    # Original-mask intersection alone does not guarantee a nonempty resized mask.
    candidates = []
    for i in positive:
        m = crop_resize_mask(mask, boxes[i])
        if m.any():
            candidates.append((i, m))
    if not candidates:
        raise ValueError(f'No standard patch preserves GT at 448: {row.image_path}')
    unused = [(i,m) for i,m in candidates if i not in used] or candidates
    while len(valid) < 2:
        index, m = unused[int(rng.integers(len(unused)))]
        valid.append((crop_resize(image, boxes[index]), m, f'patch_recover{index}'))
    assert len(valid) == 2 and all(m.any() for _,m,_ in valid)
    return valid, ratio, True


def mine(base, model, processor, anomaly, args, step):
    """Score all training anomalies with full-image inference, without classifier gating.

    This ranks rows, not individual patches. Training keeps the original random
    positive-patch policy. No Dev/Test labels or leaderboard feedback are used.
    """
    model.eval()
    predict = predictor(model, processor)
    records = []
    for i, row in anomaly.iterrows():
        image, mask = base.load_original(row, args.root, args.root)
        from e1.patch9_v2_geometry import resize_image
        target = resize_mask(mask)
        disappeared = not target.any()
        pred = predict(resize_image(image)) >= .5
        tp = int((pred & target).sum())
        fp = int((pred & ~target).sum())
        fn = int((~pred & target).sum())
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.
        recall = tp / (tp + fn) if (tp + fn) else 0.
        records.append(dict(index=int(i), image_path=row.image_path,
                            class_name=row.class_name, ratio=float(mask.mean()),
                            dice=dice, recall=recall, empty=not pred.any(), full_target_disappeared=disappeared,
                            difficulty=difficulty(dice, recall, not pred.any())))
        if (i + 1) % 500 == 0:
            print(f'[mine {step}] {i+1}/{len(anomaly)}', flush=True)
    pd.DataFrame(records).to_csv(args.out / f'hard_pool_step{step:05d}.csv', index=False)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', choices=['v4', 'v5'], required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--root', type=Path, default=Path('Dataset'))
    p.add_argument('--train', type=Path, default=BASE_TRAIN)
    p.add_argument('--dev', type=Path, default=BASE_DEV)
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--pool', type=Path,
                   help='reuse an existing hard_pool_step*.csv instead of re-mining every row')
    args = p.parse_args()
    base = importlib.import_module(f'e1.train_segmentation_patch9_{args.base}')
    if args.out.exists() and any(path.name != 'run.log' for path in args.out.iterdir()):
        raise FileExistsError('Choose a fresh output directory')
    if not (args.teacher.parent / 'training_summary.json').exists():
        raise ValueError('Teacher experiment must finish first')
    state = torch.load(args.teacher, map_location='cpu', weights_only=False)
    experiment = state['protocol'].get('experiment','')
    if f'_{args.base}_' not in experiment:
        raise ValueError('Teacher version does not match requested loss branch')
    if state['step'] != 24000:
        raise ValueError('Use the completed step24000 teacher')
    train_path = args.train
    dev_path = args.dev
    train = pd.read_parquet(train_path).reset_index(drop=True)
    dev = pd.read_parquet(dev_path)
    assert_original_split(train, "train")
    assert_original_split(dev, "dev")
    assert_original_paths(train, "train")
    assert_original_paths(dev, "dev")
    if set(train.image_path) & set(dev.image_path):
        raise ValueError('Train/Dev image overlap')
    for column in ('group_path', 'sample_group'):
        if column in train and column in dev:
            if set(zip(train.class_name,train[column])) & set(zip(dev.class_name,dev[column])):
                raise ValueError('Train/Dev group overlap')
    if state['protocol']['train_sha256'] != base.digest(train_path):
        raise ValueError('Teacher and mining training splits differ')
    if state['protocol']['accumulation'] != 16:
        raise ValueError('Expected the actual 16-input baseline recipe')
    if args.base == 'v5':
        assert 'alpha=0.4,beta=0.6' in state['protocol']['loss']
        assert state['protocol']['area_weight_buckets']['weights'] == {
            'tiny': 2., 'small': 1.5, 'large': 1., 'empty_crop': 1.}
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = open('artifacts/hard_mining_gpu.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed = 20260921
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, processor, _ = base.build_model('models/Qwen3-VL-2B-Instruct')
    initial = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    model.decoder.load_state_dict(state['decoder'])
    anomaly = train[train.view_label.eq(1)].reset_index(drop=True)
    normal = train[train.view_label.eq(0)].reset_index(drop=True)
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resume:
        assert resume['protocol']['base'] == args.base
        assert resume['protocol']['train_sha256'] == base.digest(train_path)
        pools = sorted(args.resume.parent.glob('hard_pool_step*.csv'))
        pool = max((x for x in pools if int(x.stem[-5:]) <= resume['step']), key=lambda x: x.name)
        records = pd.read_csv(pool).to_dict('records')
        pd.DataFrame(records).to_csv(args.out/pool.name,index=False)
    elif args.pool:
        records = pd.read_csv(args.pool).to_dict('records')
        pd.DataFrame(records).to_csv(args.out/'hard_pool_step00000.csv', index=False)
        print(f'[hard] reused pool {args.pool} rows={len(records)}', flush=True)
    else:
        records = mine(base, model, processor, anomaly, args, 0)
    model.decoder.load_state_dict(initial)
    del state, initial
    sampler = HardSampler(records, seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(base.trainable(model), lr=1e-4, weight_decay=.01)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/50 if s < 50 else
        .5 * (1 + math.cos(math.pi * (s-50)/23950)))
    protocol = dict(base=args.base, steps=args.steps, loss='unchanged from '+args.base,
                    teacher=str(args.teacher), teacher_sha256=base.digest(args.teacher),
                    train_sha256=base.digest(train_path), dev_sha256=base.digest(dev_path),
                    mining='full-image training anomalies only; top 25%; weights capped at 4',
                    refresh_steps=4000, source_ratio=[.60,.25,.15], small_ratio_lt=.005,
                    initialization='fresh decoder; teacher only for initial sampling scores',
                    accumulation=16, learning_rate=1e-4, warmup=50, scheduler='cosine',
                    seed=seed, trainer_sha256=base.digest(Path(__file__)),
                    base_code_sha256=base.digest(Path(base.__file__)))
    start = 1
    if resume:
        model.decoder.load_state_dict(resume['decoder'])
        opt.load_state_dict(resume['optimizer']); sched.load_state_dict(resume['scheduler'])
        start = int(resume['step']) + 1
        protocol['resumed_from'] = str(args.resume)
        protocol['resume_step'] = start-1
        protocol['legacy_rng_restart'] = 'rng' not in resume
        if 'rng' in resume:
            rng.bit_generator.state = resume['rng']
            sampler = resume['sampler']
            torch.set_rng_state(resume['torch_rng'])
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
    protocol['tiny_recovery'] = ('keep surviving inputs; fill unusable slots with another '
                                 'standard 0.4 nine-grid window that still contains the defect')
    base.write_json(args.out/'frozen_training_protocol.json', protocol)
    fields = ['step','loss','focal','overlap','area_weight','grad_norm','input_count',
              'ordinary','hard','small','fallback']
    begun = time.time()
    with (args.out/'training_history.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for step in range(start, args.steps + 1):
            if step > 1 and (step-1) % 4000 == 0:
                records = mine(base, model, processor, anomaly, args, step-1)
                sampler = HardSampler(records, seed+step)
            model.train(); model.base_model.eval(); opt.zero_grad(set_to_none=True)
            examples = []
            selected = sampler.draw()
            for index, _ in selected:
                row = anomaly.iloc[index]
                crops, ratio, repaired = positive_pair(base, row, args.root, rng, args.base)
                if repaired:
                    with (args.out/'tiny_recoveries.jsonl').open('a') as audit:
                        audit.write(json.dumps(dict(step=step,image_path=row.image_path,tags=[c[2] for c in crops]))+'\n')
                examples.extend((im,m,ratio) for im,m,_ in crops)
            for _ in range(2):
                row = normal.iloc[int(rng.integers(len(normal)))]
                prepared = base.prepare_examples(row,args.root,args.root,rng)
                crops = prepared[0] if args.base == 'v5' else prepared
                examples.extend((im,m,0.) for im,m,_ in crops)
            made = 0
            for _ in range(40):
                if made == 2: break
                pool = anomaly if rng.random() < .5 else normal
                row = pool.iloc[int(rng.integers(len(pool)))]
                try: im,m,_ = base.prepare_negative_patch(row,args.root,args.root,rng)
                except LookupError: continue
                examples.append((im,m,0.)); made += 1
            if len(examples) != 16: raise ValueError('Invalid logical batch')
            values = np.zeros(4)
            for im, mask, ratio in examples:
                target = base.target_from(mask)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    r = model.segmentation_only(base.encode(processor,im),target,allow_empty=True)
                if args.base == 'v5':
                    overlap = base.tversky_loss(r.mask_logits,target,.4,.6)
                    weight = base.area_weight(ratio) if mask.any() else 1.
                    loss = weight*(r.focal_loss+overlap)
                else: loss,overlap,weight = r.loss,r.dice_loss,1.
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                (loss/16).backward()
                values += [loss.detach().item(),r.focal_loss.detach().item(),overlap.detach().item(),weight]
            base.check_gradients(model)
            norm = float(torch.nn.utils.clip_grad_norm_(base.trainable(model),1.))
            if not math.isfinite(norm): raise ValueError('Invalid gradient norm')
            opt.step(); sched.step(); values /= 16
            sources = [s for _,s in selected]
            writer.writerow(dict(zip(fields,[step,*values,norm,16,
                sources.count('ordinary'),sources.count('hard'),sources.count('small'),
                sources.count('ordinary_fallback')]))); stream.flush()
            if step % 50 == 0:
                snapshot = dict(step=step,decoder=model.decoder.state_dict(),optimizer=opt.state_dict(),
                                scheduler=sched.state_dict(),protocol=protocol, rng=copy.deepcopy(rng.bit_generator.state),
                                sampler=copy.deepcopy(sampler),torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all())
                save(args.out/'latest.pt',snapshot)
                print(f'[hard {args.base}] {step}/{args.steps} loss={values[0]:.5f}',flush=True)
                if step in (8000,12000,16000,20000,24000):
                    save(args.out/f'step{step:05d}.pt',snapshot)
    base.write_json(args.out/'training_summary.json',dict(completed_steps=args.steps,elapsed_s=time.time()-begun))


if __name__ == '__main__': main()

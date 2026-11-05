"""Matched frozen-language migration using one cached repaired visual teacher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from dec_big import atomic_json, atomic_save
from migration_language import FrozenLanguageBridge
from migration_metrics import evaluate_predictions, donor_map_indices, spatial_shuffle_maps, validate_cache
from visa_pipeline import PIPELINE_VERSION


YES = 'Yes, there is an anomaly in the image.'
NO = 'No, there is no anomaly in the image.'


def sample_schedule(items, seed, exposures):
    pools = [[i for i, x in enumerate(items) if x['split'] == 'train' and x['label'] == label]
             for label in (0, 1)]
    rng = random.Random(seed)
    labels = [0] * (exposures // 2) + [1] * (exposures - exposures // 2)
    rng.shuffle(labels)
    return [rng.choice(pools[label]) for label in labels]


def fixed_subset(ids, items, n):
    if not n or n >= len(ids):
        return ids
    rng = random.Random(42)
    pools = [[i for i in ids if items[i]['label'] == label] for label in (0, 1)]
    n0 = min(len(pools[0]), n // 2)
    n1 = min(len(pools[1]), n - n0)
    return rng.sample(pools[0], n0) + rng.sample(pools[1], n1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['vicuna', 'qwen'], required=True)
    ap.add_argument('--cache', default='runs/migration_20261103/teacher_cache.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--val-every', type=int, default=250)
    ap.add_argument('--val-limit', type=int, default=0)
    ap.add_argument('--eval-batch-size', type=int, default=8)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--eval-split', choices=['val', 'test'], default='val')
    ap.add_argument('--generate-limit', type=int, default=0,
                    help='0 means every image, positive values are smoke diagnostics only')
    ap.add_argument('--skip-map-controls', action='store_true', help='Smoke tests only')
    args = ap.parse_args()
    if min(args.steps, args.batch_size, args.accum, args.val_every, args.eval_batch_size) < 1:
        raise ValueError('Steps, batch sizes, accumulation and validation interval must be positive')
    if min(args.val_limit, args.generate_limit) < 0 or args.lr <= 0:
        raise ValueError('Limits must be nonnegative and learning rate positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.eval_only and any((out / f).exists() for f in ('best.pt', 'final.pt')):
        raise FileExistsError('Choose a fresh training directory: ' + str(out))
    if (out / (args.eval_split + '_metrics.json')).exists():
        raise FileExistsError('Existing evaluation is preserved; choose a fresh output directory')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cache = torch.load(args.cache, map_location='cpu')
    validate_cache(cache, expected_pipeline_version=PIPELINE_VERSION)
    sidecar = json.load(open(Path(args.cache).with_suffix('.json')))
    cache_hash = hashlib.sha256()
    with open(args.cache, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            cache_hash.update(block)
    assert cache_hash.hexdigest() == sidecar['cache_sha256'], 'Cache file hash changed'
    items = cache['items']
    assert cache['maps'].shape == (len(items), 224, 224)
    assert cache['cls'].shape == (len(items), 1024)
    indices = {s: [i for i, x in enumerate(items) if x['split'] == s]
               for s in ('train', 'val', 'test')}
    val_ids = fixed_subset(indices['val'], items, args.val_limit)
    schedule = sample_schedule(items, args.seed, args.steps * args.batch_size * args.accum)
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    if not args.eval_only:
        atomic_json(schedule, str(out / 'sample_indices.json'))
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model = FrozenLanguageBridge(args.backend, device='cuda', init_seed=args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    params_n = sum(p.numel() for p in params)
    initial_hash = hashlib.sha256()
    for name, module in [('prompt_learner', model.prompt_learner), ('cls_proj', model.cls_proj)]:
        for key, value in module.state_dict().items():
            initial_hash.update((name + '.' + key).encode())
            initial_hash.update(value.detach().cpu().numpy().tobytes())
    print('[load] %.1fs trainable=%d allocated=%.2fGB' % (
        time.time()-started, params_n, torch.cuda.memory_allocated()/1e9), flush=True)
    config = dict(pipeline_version=PIPELINE_VERSION,
                  training_protocol='supervised_custom_60_20_20', backend=args.backend,
                  decoder_sha256=cache['decoder_sha256'], cache_manifest_sha256=cache['manifest_sha256'],
                  cache_sha256=cache_hash.hexdigest(),
                  counts={s: len(v) for s, v in indices.items()}, val_n=len(val_ids),
                  selection='maximum full validation image AUROC; earliest wins ties',
                  sample_schedule_sha256=schedule_hash, trainable_parameters=params_n,
                  precision='frozen BF16 language weights; FP32 bridge and AdamW',
                  initialization='fresh same-seed PromptLearner and CLS projection for both backends',
                  initialization_sha256=initial_hash.hexdigest(),
                  initialization_audit=model.initialization_audit,
                  loss='answer-template token mean CE including EOS; frozen teacher has no segmentation gradient',
                  config=vars(args))
    if not args.eval_only:
        atomic_json(config, str(out / 'run.json'))
    print('[config]', json.dumps(config), flush=True)

    def batch(ids, answer=False, maps=None):
        amap = cache['maps'][ids] if maps is None else maps
        return model.pack(cache['cls'][ids].cuda(), amap.cuda(),
                          [items[i]['category'] for i in ids],
                          [YES if items[i]['label'] else NO for i in ids] if answer else None)

    @torch.no_grad()
    def evaluate(ids, condition='real_map', generate=False):
        model.eval()
        subset = [items[i] for i in ids]
        cmaps = cache['maps'][ids]
        if condition == 'zero_map':
            cmaps = torch.zeros_like(cmaps)
        elif condition == 'donor_opposite_label_map':
            donors = donor_map_indices(subset, seed=42, opposite_label=True)
            cmaps = cmaps[torch.as_tensor(donors)]
        elif condition == 'spatial_shuffle_map':
            cmaps = spatial_shuffle_maps(cmaps, seed=42)
        elif condition != 'real_map':
            raise ValueError(condition)
        scores, texts = [], []
        for start in range(0, len(ids), args.eval_batch_size):
            chunk = ids[start:start+args.eval_batch_size]
            packed = batch(chunk, maps=cmaps[start:start+len(chunk)])
            scores.extend(model.score(packed).detach().float().cpu().tolist())
            if generate:
                texts.extend(model.generate(packed, max_new_tokens=32))
        records = [dict(**item, score=float(score), **({'generated': text} if generate else {}))
                   for item, score, text in zip(subset, scores, texts if generate else [None]*len(ids))]
        metrics = evaluate_predictions(subset, scores=scores, generated_texts=texts if generate else None)
        model.train()
        return metrics, records

    def checkpoint(step, val_auroc):
        return dict(prompt_learner=model.prompt_learner.state_dict(),
                    cls_proj=model.cls_proj.state_dict(), step=step, val_auroc=val_auroc,
                    **config)

    def load(path):
        state = torch.load(path, map_location='cpu')
        assert state['backend'] == args.backend
        assert state['decoder_sha256'] == cache['decoder_sha256']
        assert state['pipeline_version'] == PIPELINE_VERSION
        assert state['cache_manifest_sha256'] == cache['manifest_sha256']
        assert state['cache_sha256'] == cache_hash.hexdigest()
        model.prompt_learner.load_state_dict(state['prompt_learner'], strict=True)
        model.cls_proj.load_state_dict(state['cls_proj'], strict=True)
        return state

    if not args.eval_only:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        if args.steps < 40:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        best, best_step, last_val_auc = -1., 0, None
        history = open(out / 'training.jsonl', 'w')
        initial, _ = evaluate(val_ids)
        atomic_json(initial, str(out / 'initial_val.json'))
        print('[initial val]', json.dumps(initial['overall']), flush=True)
        model.train()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.
            for k in range(args.accum):
                offset = (step * args.accum + k) * args.batch_size
                packed = batch(schedule[offset:offset+args.batch_size], answer=True)
                loss = model.loss(packed)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss at step %d' % (step+1))
                (loss / args.accum).backward()
                loss_total += float(loss.detach()) / args.accum
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient at step %d' % (step+1))
            if step == 0:
                groups = {name: sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                          for name, module in [('prompt_learner', model.prompt_learner), ('cls_proj', model.cls_proj)]}
                if not all(np.isfinite(v) and v > 0 for v in groups.values()):
                    raise ValueError('Missing bridge gradient: ' + str(groups))
                print('[gradient]', json.dumps(groups), flush=True)
            optimizer.step()
            scheduler.step()
            record = dict(step=step+1, loss=loss_total, gradient_norm=float(norm),
                          lr=scheduler.get_last_lr()[0], elapsed_seconds=time.time()-started,
                          peak_gb=torch.cuda.max_memory_allocated()/1e9)
            if (step+1) % 10 == 0 or step == 0:
                history.write(json.dumps(record)+'\n'); history.flush()
                print('[step]', json.dumps(record), flush=True)
            if (step+1) % args.val_every == 0 or step+1 == args.steps:
                metrics, predictions = evaluate(val_ids)
                val_auc = metrics['overall']['scores']['image_auroc']
                last_val_auc = val_auc
                val_record = dict(step=step+1, metrics=metrics)
                history.write(json.dumps(val_record)+'\n'); history.flush()
                print('[val]', json.dumps(dict(step=step+1, **metrics['overall'])), flush=True)
                atomic_json(metrics, str(out / ('val_step_%d.json' % (step+1))))
                if val_auc > best:
                    best, best_step = val_auc, step+1
                    atomic_save(checkpoint(best_step, best), str(out / 'best.pt'))
                    atomic_json(predictions, str(out / 'best_val_predictions.json'))
        atomic_save(dict(**checkpoint(args.steps, last_val_auc), best_val_auroc=best,
                         best_step=best_step), str(out / 'final.pt'))
        history.close()
        # Reload must recover exactly the selected model's validation scores.
        selected = load(str(out / 'best.pt'))
        reloaded, reloaded_records = evaluate(val_ids)
        previous = json.load(open(out / 'best_val_predictions.json'))
        max_err = max(abs(a['score']-b['score']) for a, b in zip(previous, reloaded_records))
        assert max_err < 1e-5, ('save/reload score mismatch', max_err)
        atomic_json(dict(best_step=best_step, best_val_auroc=best,
                         elapsed_seconds=time.time()-started,
                         peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                         reload_max_abs_score_error=max_err), str(out / 'train_complete.json'))
        print('[train done]', json.dumps(dict(best_step=best_step, best_val_auroc=best, reload_error=max_err)), flush=True)
        del optimizer, scheduler
        torch.cuda.empty_cache()
    else:
        if not args.ckpt:
            raise ValueError('--eval-only requires --ckpt')
        selected = load(args.ckpt)
    eval_ids = indices[args.eval_split]
    if args.eval_split == 'val':
        eval_ids = val_ids
    result = dict(selected_step=selected['step'], split=args.eval_split,
                  decoder_sha256=cache['decoder_sha256'], conditions={})
    conditions = ('real_map',) if args.skip_map_controls else (
        'real_map', 'zero_map', 'donor_opposite_label_map', 'spatial_shuffle_map')
    for condition in conditions:
        metrics, predictions = evaluate(eval_ids, condition=condition)
        result['conditions'][condition] = metrics
        atomic_json(predictions, str(out / ('%s_%s_predictions.json' % (args.eval_split, condition))))
        atomic_json(result, str(out / ('%s_metrics.json' % args.eval_split)))
        print('[evaluation]', condition, json.dumps(metrics['overall']), flush=True)
    gen_ids = fixed_subset(eval_ids, items, args.generate_limit)
    metrics, predictions = evaluate(gen_ids, generate=True)
    result['generated_real_map'] = metrics
    result['generated_n'] = len(gen_ids)
    result['generated_is_subset'] = len(gen_ids) != len(eval_ids)
    atomic_json(predictions, str(out / ('%s_generated_predictions.json' % args.eval_split)))
    atomic_json(result, str(out / ('%s_metrics.json' % args.eval_split)))
    print('[done]', json.dumps(result['generated_real_map']['overall']), flush=True)


if __name__ == '__main__':
    main()

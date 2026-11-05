"""Advanced joint bridge/LoRA and paired CE/ranking experiments.

Uses the repaired Phase-1 cache and the matching seed's best bridge. Test
predictions are never queried. Checkpoints contain bridge weights for B or
only PEFT adapter weights for C, plus the exact source bridge/cache identity.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time
import torch.nn.functional as F

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from dec_big import atomic_json, atomic_save
from migration_language import FrozenLanguageBridge, YES, NO
from migration_metrics import evaluate_predictions, validate_cache
from migration_train import sample_schedule, fixed_subset
from visa_pipeline import PIPELINE_VERSION


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def bridge_state(model):
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in [('prompt_learner', model.prompt_learner), ('cls_proj', model.cls_proj)]}


def bridge_digest(model):
    return tensor_digest({n + '.' + k: v for n, module in
                          [('prompt_learner', model.prompt_learner), ('cls_proj', model.cls_proj)]
                          for k, v in module.state_dict().items()})


def load_bridge(model, state):
    for name in ('prompt_learner', 'cls_proj'):
        getattr(model, name).load_state_dict(state[name], strict=True)


def adapter_state(model):
    return {k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(model.language_model).items()}


def configure_arm(model, arm, rank, alpha, lr_bridge, lr_lora):
    """Only the retained text model is handed to PEFT; no vision targets."""
    target_names = []
    if arm in ('C', 'D', 'R'):
        target_names = [n for n, m in model.language_model.named_modules()
                        if n.endswith(('.q_proj', '.v_proj')) and isinstance(m, torch.nn.Linear)]
        expected = 2 * model.language_model.config.num_hidden_layers
        if len(target_names) != expected:
            raise AssertionError(('Unexpected attention targets', len(target_names), expected))
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias='none',
                         target_modules=['q_proj', 'v_proj'], init_lora_weights=True)
        model.language_model = get_peft_model(model.language_model, cfg)
    for n, p in model.named_parameters():
        bridge_allowed = arm in ('B', 'D') and n.startswith(('prompt_learner.', 'cls_proj.'))
        lora_allowed = arm in ('C', 'D', 'R') and n.startswith('language_model.') and (
            '.lora_A.' in n or '.lora_B.' in n)
        allowed = bridge_allowed or lora_allowed
        p.requires_grad_(allowed)
        if allowed and p.dtype != torch.float32:
            p.data = p.data.float()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert params
    assert all(p.dtype == torch.float32 for _, p in params), 'Trainable tensors must be FP32'
    # Base model remains in eval. Dropout is explicitly zero; eval still permits
    # gradients through the adapters and custom inputs_embeds interface.
    if arm == 'D':
        groups = [dict(params=[p for n, p in params if n.startswith(('prompt_learner.', 'cls_proj.'))],
                       lr=lr_bridge, name='bridge'),
                  dict(params=[p for n, p in params if '.lora_A.' in n or '.lora_B.' in n],
                       lr=lr_lora, name='language_lora')]
    else:
        lr = lr_bridge if arm == 'B' else lr_lora
        groups = [dict(params=[p for _, p in params], lr=lr, name=arm)]
    assert all(group['params'] for group in groups)
    return params, target_names, groups


def paired_schedule(items, seed, exposures):
    """Balanced normal/anomaly pairs sampled within the same training category."""
    if exposures % 2:
        raise ValueError('Paired ranking needs an even number of exposures')
    pools = {}
    for label in (0, 1):
        for index, item in enumerate(items):
            if item['split'] == 'train' and int(item['label']) == label:
                pools.setdefault(item['category'], {0: [], 1: []})[label].append(index)
    categories = sorted(category for category, values in pools.items() if values[0] and values[1])
    if not categories:
        raise ValueError('No training category has both normal and anomalous images')
    rng = random.Random(seed)
    result = []
    for _ in range(exposures // 2):
        category = rng.choice(categories)
        result.extend((rng.choice(pools[category][0]), rng.choice(pools[category][1])))
    return result


def load_delta(model, path, identity):
    state = torch.load(path, map_location='cpu')
    for k in ('arm', 'seed', 'bridge_sha256', 'cache_sha256', 'base_manifest_sha256',
              'sample_schedule_sha256', 'pipeline_version'):
        assert state['identity'][k] == identity[k], ('Checkpoint identity mismatch', k)
    if identity['arm'] in ('B', 'D'):
        load_bridge(model, state['bridge'])
    if identity['arm'] in ('C', 'D', 'R'):
        result = set_peft_model_state_dict(model.language_model, state['adapter'])
        assert not result.unexpected_keys, result.unexpected_keys
        actual = adapter_state(model)
        assert actual.keys() == state['adapter'].keys()
        assert all(torch.equal(actual[k], state['adapter'][k]) for k in actual)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['B', 'C', 'D', 'R'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bridge', required=True)
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--microbatch', type=int, default=1)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--eval-batch-size', type=int, default=8)
    ap.add_argument('--val-every', type=int, default=100)
    ap.add_argument('--lr-bridge', type=float, default=5e-4)
    ap.add_argument('--lr-lora', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--objective', choices=['ce', 'ce_pair_rank'], default='ce')
    ap.add_argument('--rank-weight', type=float, default=0.0)
    ap.add_argument('--paired-sampling', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    if not args.preflight:
        assert (args.steps, args.val_every) == (500, 100)
        if args.paired_sampling:
            assert (args.microbatch, args.accum) == (2, 2)
        else:
            assert (args.microbatch, args.accum) == (1, 4)
    if args.objective == 'ce_pair_rank':
        assert args.paired_sampling and args.rank_weight > 0
    else:
        assert args.rank_weight == 0
    assert min(args.steps, args.microbatch, args.accum, args.eval_batch_size, args.val_every) > 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cache = torch.load(args.cache, map_location='cpu')
    validate_cache(cache, expected_pipeline_version=PIPELINE_VERSION)
    cache_sha = sha256_file(args.cache)
    assert cache_sha == json.load(open(Path(args.cache).with_suffix('.json')))['cache_sha256']
    source = torch.load(args.bridge, map_location='cpu')
    for k, v in [('backend', 'qwen'), ('pipeline_version', PIPELINE_VERSION),
                 ('cache_sha256', cache_sha), ('decoder_sha256', cache['decoder_sha256']),
                 ('cache_manifest_sha256', cache['manifest_sha256'])]:
        assert source[k] == v, ('Source bridge identity mismatch', k)
    assert source['config']['seed'] == args.seed
    items = cache['items']
    val_ids = [i for i, x in enumerate(items) if x['split'] == 'val']
    assert len(val_ids) == 380
    val_ids = fixed_subset(val_ids, items, 8) if args.preflight else val_ids
    schedule_builder = paired_schedule if args.paired_sampling else sample_schedule
    schedule = schedule_builder(items, args.seed + 100000,
                                args.steps * args.microbatch * args.accum)
    if args.preflight:
        # Repeated fixed balanced train examples are used only for overfit gating.
        eight = schedule_builder(items, args.seed + 100000, 8)
        schedule = (eight * ((len(schedule) + 7) // 8))[:len(schedule)]
    assert all(items[i]['split'] == 'train' for i in schedule)
    atomic_json(schedule, str(out / 'sample_indices.json'))
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = FrozenLanguageBridge('qwen', device='cuda', init_seed=args.seed)
    load_bridge(model, source)
    initial_bridge_hash = bridge_digest(model)

    def pack(ids, answers=False):
        return model.pack(cache['cls'][ids].cuda(), cache['maps'][ids].cuda(),
                          [items[i]['category'] for i in ids],
                          [YES if items[i]['label'] else NO for i in ids] if answers else None)

    @torch.no_grad()
    def evaluate(ids, generate=False):
        model.eval()
        scores, texts = [], []
        for j in range(0, len(ids), args.eval_batch_size):
            chunk = ids[j:j + args.eval_batch_size]
            packed = pack(chunk)
            scores.extend(model.score(packed).float().cpu().tolist())
            if generate:
                texts.extend(model.generate(packed, max_new_tokens=32))
        subset = [items[i] for i in ids]
        metrics = evaluate_predictions(subset, scores, texts if generate else None)
        records = [dict(**item, score=float(score),
                        **({'generated': text} if generate else {}))
                   for item, score, text in zip(subset, scores, texts if generate else [None] * len(ids))]
        model.train()
        return metrics, records

    probe_ids = val_ids[:args.eval_batch_size]
    model.eval()
    with torch.no_grad():
        before = model.forward_batch(pack(probe_ids)).detach().cpu()
    named_params, targets, optimizer_groups = configure_arm(model, args.arm, args.rank, args.alpha,
                                                            args.lr_bridge, args.lr_lora)
    model.eval()
    with torch.no_grad():
        after = model.forward_batch(pack(probe_ids)).detach().cpu()
    zero_error = float((before.float() - after.float()).abs().max())
    assert zero_error == 0., ('Untrained adapter changed logits', zero_error)
    del before, after
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    base_manifest = json.load(open(args.base_manifest))
    assert str(model.model_path) == base_manifest['model_path']
    for name, entry in base_manifest['files'].items():
        assert (model.model_path / name).stat().st_size == entry['bytes']
    identity = dict(arm=args.arm, seed=args.seed, pipeline_version=PIPELINE_VERSION,
                    bridge_path=str(Path(args.bridge).resolve()), bridge_sha256=sha256_file(args.bridge),
                    bridge_step=source['step'], bridge_tensor_sha256=initial_bridge_hash,
                    cache_sha256=cache_sha, decoder_sha256=cache['decoder_sha256'],
                    cache_manifest_sha256=cache['manifest_sha256'],
                    base_manifest_sha256=sha256_file(args.base_manifest),
                    sample_schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest())
    params = [p for _, p in named_params]
    config = dict(identity=identity, config=vars(args), targets=targets,
                  trainable_parameters=sum(p.numel() for p in params),
                  trainable_names=[n for n, _ in named_params],
                  train_images=9985, validation_images=len(val_ids), test_queries=0,
                  exposures=len(schedule), precision='BF16 base; FP32 trainable tensors and AdamW',
                  loss=('answer-token mean CE including EOS per microbatch; mean over %d accumulation batches; objective=%s' % (args.accum, args.objective)),
                  selection='max full val image AUROC, step0 eligible, earliest wins ties',
                  lora=dict(r=args.rank, alpha=args.alpha, dropout=0., bias='none', targets=['q_proj', 'v_proj'])
                  if args.arm in ('C', 'D', 'R') else None,
                  objective=dict(name=args.objective, rank_weight=args.rank_weight,
                                 pair_order='normal_then_anomaly_within_category' if args.paired_sampling else None),
                  gradient_checkpointing=False, zero_adapter_max_abs_logit_error=zero_error,
                  source_sha256={n: sha256_file(Path(__file__).parent / n) for n in [Path(__file__).name]},
                  runtime_versions=dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
                                        peft=__import__('peft').__version__))
    atomic_json(config, str(out / 'run.json'))
    print('[config]', json.dumps(config), flush=True)

    def checkpoint(step, auc):
        state = dict(identity=identity, config=config, step=step, val_auroc=auc)
        if args.arm in ('B', 'D'):
            state['bridge'] = bridge_state(model)
        if args.arm in ('C', 'D', 'R'):
            state['adapter'] = adapter_state(model)
        return state

    def check_frozen():
        for n, p in model.named_parameters():
            if n in frozen_versions:
                assert p._version == frozen_versions[n], ('Frozen tensor mutated', n)
                assert p.grad is None, ('Frozen tensor has gradient', n)
        if args.arm in ('C', 'R'):
            assert bridge_digest(model) == initial_bridge_hash, 'Frozen bridge changed'

    # A is already fully evaluated by the paired Phase-1 run. Validate all
    # scores bit-for-bit, audit eight generations, then reuse its raw texts.
    # This changes no model computation or reported predictions.
    def audit_and_reuse_texts(records, reference):
        assert [r['path'] for r in reference] == [r['path'] for r in records]
        assert all(a['score'] == b['score'] for a, b in zip(reference, records))
        lookup = {x['path']: x['generated'] for x in reference}
        model.eval()
        with torch.no_grad():
            check = model.generate(pack(val_ids[:8]), max_new_tokens=32)
        assert check == [lookup[items[i]['path']] for i in val_ids[:8]]
        model.train()
        updated = [dict(r, generated=lookup[r['path']]) for r in records]
        return evaluate_predictions([items[i] for i in val_ids],
                                    [r['score'] for r in updated],
                                    [r['generated'] for r in updated]), updated

    if args.preflight:
        initial, records = evaluate(val_ids, generate=True)
    else:
        initial, records = evaluate(val_ids)
        previous_A = json.load(open(Path(args.bridge).parent / 'val_generated_predictions.json'))
        initial, records = audit_and_reuse_texts(records, previous_A)
    atomic_json(initial, str(out / 'initial_val.json'))
    atomic_json(records, str(out / 'initial_val_predictions.json'))
    if not args.preflight:
        reference = json.load(open(Path(args.bridge).parent / 'best_val_predictions.json'))
        assert [r['path'] for r in reference] == [r['path'] for r in records]
        reference_error = max(abs(a['score'] - b['score']) for a, b in zip(reference, records))
        assert reference_error == 0., ('Start does not recover paired A', reference_error)
    else:
        reference_error = None
    initial_auc = initial['overall']['scores']['image_auroc']
    best, best_step = initial_auc, 0
    atomic_save(checkpoint(0, best), str(out / 'best.pt'))
    atomic_json(records, str(out / 'best_val_predictions.json'))
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in params}
    max_lrs = [group['lr'] for group in optimizer_groups]
    scheduler = (torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=args.steps)
                 if args.preflight else torch.optim.lr_scheduler.OneCycleLR(
                     optimizer, max_lr=max_lrs, total_steps=args.steps, pct_start=0.05))
    train_seconds, losses = 0., []
    train_peak = torch.cuda.memory_allocated()
    gradient_records = []
    history = open(out / 'training.jsonl', 'w')
    last_auc = initial_auc
    for step in range(1, args.steps + 1):
        model.train()
        torch.cuda.synchronize()
        tick = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_total, rank_total = 0., 0.
        for k in range(args.accum):
            offset = ((step - 1) * args.accum + k) * args.microbatch
            batch = pack(schedule[offset:offset + args.microbatch], answers=True)
            if args.objective == 'ce_pair_rank':
                logits = model.forward_batch(batch)
                labels = batch['labels'][:, 1:]
                ce_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                          labels.reshape(-1), ignore_index=-100)
                rows = torch.arange(2, device=logits.device)
                first = logits[rows, batch['prefix_lengths'] - 1].float()
                scores = first[:, model.yes_id] - first[:, model.no_id]
                rank_loss = F.softplus(-(scores[1] - scores[0]))
                loss = ce_loss + args.rank_weight * rank_loss
            else:
                ce_loss = model.loss(batch)
                loss = ce_loss
                rank_loss = torch.zeros((), device=loss.device)
            assert torch.isfinite(loss), ('Nonfinite loss', step)
            (loss / args.accum).backward()
            loss_total += float(loss.detach()) / args.accum
            rank_total += float(rank_loss.detach()) / args.accum
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm) and float(norm) > 0., ('Invalid gradient norm', step)
        if step <= 2:
            groups = {}
            for n, p in named_params:
                assert p.grad is not None and torch.isfinite(p.grad).all(), ('Missing/invalid gradient', n)
                group = 'lora_A' if '.lora_A.' in n else 'lora_B' if '.lora_B.' in n else n.split('.')[0]
                groups[group] = groups.get(group, 0.) + float(p.grad.float().norm())
            assert all(v > 0. for k, v in groups.items() if k != 'lora_A' or step > 1), groups
            gradient_records.append(dict(step=step, norms=groups))
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        train_seconds += time.time() - tick
        train_peak = max(train_peak, torch.cuda.max_memory_allocated())
        losses.append(loss_total)
        row = dict(step=step, loss=loss_total, ranking_loss=rank_total,
                   gradient_norm=float(norm), learning_rates=scheduler.get_last_lr(),
                   train_seconds=train_seconds, peak_memory_bytes=torch.cuda.max_memory_allocated())
        if step <= 2 or step % 10 == 0:
            history.write(json.dumps(row) + '\n'); history.flush()
            print('[step]', json.dumps(row), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            check_frozen()
            metrics, records = evaluate(val_ids)
            last_auc = metrics['overall']['scores']['image_auroc']
            atomic_json(metrics, str(out / ('val_step_%d.json' % step)))
            history.write(json.dumps(dict(step=step, validation=metrics)) + '\n'); history.flush()
            print('[val]', json.dumps(dict(step=step, **metrics['overall'])), flush=True)
            if last_auc > best:
                best, best_step = last_auc, step
                atomic_save(checkpoint(step, best), str(out / 'best.pt'))
                atomic_json(records, str(out / 'best_val_predictions.json'))
    check_frozen()
    atomic_save(checkpoint(args.steps, last_auc), str(out / 'final.pt'))
    history.close()
    del optimizer, scheduler, batch, loss
    torch.cuda.empty_cache()
    final_metrics, final_records = evaluate(val_ids, generate=True)
    atomic_json(final_metrics, str(out / 'final_val_metrics.json'))
    atomic_json(final_records, str(out / 'final_val_predictions.json'))
    if args.arm == 'D':
        with model.language_model.disable_adapter():
            final_bridge_only_metrics, final_bridge_only_records = evaluate(val_ids)
        atomic_json(final_bridge_only_metrics, str(out / 'final_adapter_disabled_val_metrics.json'))
        atomic_json(final_bridge_only_records, str(out / 'final_adapter_disabled_val_predictions.json'))
    # Change trainable values before loading so a no-op loader cannot pass.
    with torch.no_grad():
        params[0].add_(0.001)
    selected = load_delta(model, out / 'best.pt', identity)
    if not args.preflight and selected['step'] in (0, args.steps):
        metrics, records = evaluate(val_ids)
        reuse = json.load(open(out / ('initial_val_predictions.json' if selected['step'] == 0
                                      else 'final_val_predictions.json')))
        metrics, records = audit_and_reuse_texts(records, reuse)
    else:
        metrics, records = evaluate(val_ids, generate=True)
    previous = json.load(open(out / 'best_val_predictions.json'))
    reload_error = max(abs(a['score'] - b['score']) for a, b in zip(previous, records))
    assert reload_error == 0., ('Save/reload score mismatch', reload_error)
    check_frozen()
    if args.arm == 'D':
        with model.language_model.disable_adapter():
            bridge_only_metrics, bridge_only_records = evaluate(val_ids)
        atomic_json(bridge_only_metrics, str(out / 'best_adapter_disabled_val_metrics.json'))
        atomic_json(bridge_only_records, str(out / 'best_adapter_disabled_val_predictions.json'))
    atomic_json(metrics, str(out / 'best_val_metrics.json'))
    atomic_json(records, str(out / 'best_val_generated_predictions.json'))
    if args.arm in ('C', 'D', 'R'):
        model.language_model.save_pretrained(out / 'best_adapter', safe_serialization=True)
        # Verify PEFT disk export against the exact adapter used for scoring.
        from safetensors.torch import load_file
        exported = load_file(str(out / 'best_adapter/adapter_model.safetensors'))
        expected = adapter_state(model)
        assert exported.keys() == expected.keys()
        assert all(torch.equal(exported[k], expected[k]) for k in expected)
    if args.preflight:
        assert np.mean(losses[-5:]) < np.mean(losses[:5]), ('Fixed-example fit did not improve', losses)
    complete = dict(arm=args.arm, seed=args.seed, steps=args.steps, best_step=selected['step'],
                    initial_val_auroc=initial_auc, best_val_auroc=best, final_val_auroc=last_auc,
                    paired_A_max_abs_score_error=reference_error, reload_max_abs_score_error=reload_error,
                    zero_adapter_max_abs_logit_error=zero_error, gradient_audit=gradient_records,
                    frozen_parameter_versions_unchanged=True,
                    frozen_bridge_unchanged=args.arm in ('C', 'R'), trainable_parameters=config['trainable_parameters'],
                    peak_memory_bytes=torch.cuda.max_memory_allocated(), train_peak_memory_bytes=train_peak,
                    train_seconds=train_seconds, exposures_per_second=len(schedule) / train_seconds,
                    elapsed_seconds=time.time() - started, first5_mean_loss=float(np.mean(losses[:5])),
                    last5_mean_loss=float(np.mean(losses[-5:])), preflight=args.preflight, test_queries=0)
    atomic_json(complete, str(out / 'complete.json'))
    print('[complete]', json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()

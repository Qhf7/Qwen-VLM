"""LoRA-only structured target continuation from original C best, never test."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from structured_model import StructuredLanguageBridge, load_fixed_c
from structured_geometry import SCHEMA, sha256_file, localization_metrics, parse_structured, target_answer, mask_to_target
import numpy as np
import torch
from peft import set_peft_model_state_dict
from migration_advanced_train import adapter_state, bridge_digest
from migration_train import sample_schedule, fixed_subset
from migration_metrics import evaluate_predictions, validate_cache
from dec_big import atomic_json, atomic_save
from visa_pipeline import PIPELINE_VERSION
from advanced_generation import freeze_generation_policy, force_generation_policy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--source-c', type=Path, required=True)
    ap.add_argument('--base-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--val-every', type=int, default=100)
    ap.add_argument('--microbatch', type=int, default=1)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--eval-batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-new-tokens', type=int, default=80)
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    if min(args.steps, args.val_every, args.microbatch, args.accum, args.eval_batch_size, args.max_new_tokens) < 1 or args.lr <= 0:
        ap.error('All sizes and learning rate must be positive')
    if not args.preflight and (args.steps, args.val_every, args.microbatch, args.accum, args.lr, args.max_new_tokens) != (500, 100, 1, 4, 1e-4, 80):
        ap.error('Formal protocol is 500 steps, val100, micro1 accum4, LR1e-4, max80')
    return args


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    started = time.time()
    cache = torch.load(args.cache, map_location='cpu')
    validate_cache(cache, expected_pipeline_version=PIPELINE_VERSION)
    cache_sha = sha256_file(args.cache)
    assert cache_sha == json.loads(args.cache.with_suffix('.json').read_text())['cache_sha256']
    target_meta = json.loads(args.targets.with_suffix('.json').read_text())
    assert target_meta['schema'] == SCHEMA and target_meta['manifest_sha256'] == sha256_file(args.targets)
    assert target_meta['cache_sha256'] == cache_sha and target_meta['test_mask_reads'] == 0
    targets = {int(t['index']): t for t in map(json.loads, args.targets.read_text().splitlines()) if t['split'] in ('train', 'val')}
    items = cache['items']
    expected_ids = {i for i, item in enumerate(items) if item['split'] in ('train', 'val')}
    assert targets.keys() == expected_ids
    for index, target in targets.items():
        assert all(target[k] == items[index][k] for k in ('path', 'category', 'split', 'label'))
        parsed = parse_structured(target['answer'])
        assert parsed is not None and all(parsed[k] == target[k] for k in ('label', 'bbox', 'location'))
    val_ids = [i for i, item in enumerate(items) if item['split'] == 'val']
    assert len(val_ids) == 380
    val_ids = fixed_subset(val_ids, items, 8) if args.preflight else val_ids
    schedule = sample_schedule(items, args.seed + 100000, args.steps * args.microbatch * args.accum)
    if args.preflight:
        eight = sample_schedule(items, args.seed + 100000, 8)
        schedule = (eight * ((len(schedule)+7)//8))[:len(schedule)]
    assert all(items[i]['split'] == 'train' for i in schedule)
    atomic_json(schedule, str(args.out / 'sample_indices.json'))
    torch.cuda.reset_peak_memory_stats()
    model = StructuredLanguageBridge('qwen', device='cuda', init_seed=args.seed)
    named, target_names, groups, source_run, source_state = load_fixed_c(model, args.source_c)
    assert source_run['identity']['cache_sha256'] == cache_sha
    manifest = json.loads(args.base_manifest.read_text())
    assert source_run['identity']['base_manifest_sha256'] == sha256_file(args.base_manifest)
    assert str(model.model_path) == manifest['model_path']
    for name, entry in manifest['files'].items():
        assert (model.model_path / name).stat().st_size == entry['bytes']
    params = [p for _, p in named]
    assert all('.lora_A.' in n or '.lora_B.' in n for n, _ in named)
    answer_lengths = [len(model._check_answer(t['answer'])) for t in targets.values()]
    if max(answer_lengths) > args.max_new_tokens:
        raise ValueError(('Target exceeds frozen generation budget', max(answer_lengths), args.max_new_tokens))
    bridge_hash = bridge_digest(model)
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}

    def pack(ids, answers=False):
        # The only visual inputs are pre-existing CLS and teacher maps. Targets
        # are attached after the prompt solely as masked teacher-forcing labels.
        return model.pack(cache['cls'][ids].cuda(), cache['maps'][ids].cuda(),
                          [items[i]['category'] for i in ids],
                          [targets[i]['answer'] for i in ids] if answers else None)

    @torch.no_grad()
    def evaluate(ids, generate=False):
        model.eval(); scores, texts = [], []
        for j in range(0, len(ids), args.eval_batch_size):
            chunk = ids[j:j+args.eval_batch_size]; packed = pack(chunk)
            scores.extend(model.score(packed).float().cpu().tolist())
            if generate:
                texts.extend(model.generate(packed, max_new_tokens=args.max_new_tokens))
        subset = [items[i] for i in ids]
        metrics = evaluate_predictions(subset, scores, texts if generate else None)
        if generate:
            metrics['localization'] = localization_metrics([targets[i] for i in ids], texts)
            metrics['localization_per_category'] = {
                c: localization_metrics([targets[i] for i in ids if items[i]['category'] == c],
                                        [t for i, t in zip(ids, texts) if items[i]['category'] == c])
                for c in sorted({items[i]['category'] for i in ids})}
        records = [dict(**items[i], index=i, score=float(score), target=targets[i],
                        **({'generated': text, 'parsed': parse_structured(text)} if generate else {}))
                   for i, score, text in zip(ids, scores, texts if generate else [None]*len(ids))]
        model.train(); return metrics, records

    identity = dict(schema=SCHEMA, seed=args.seed, source_c=str(args.source_c.resolve()),
                    source_c_best_sha256=sha256_file(args.source_c/'best.pt'),
                    source_c_run_sha256=sha256_file(args.source_c/'run.json'), source_c_step=source_state['step'],
                    source_bridge_sha256=bridge_hash, cache_sha256=cache_sha,
                    targets_sha256=sha256_file(args.targets), base_manifest_sha256=sha256_file(args.base_manifest),
                    schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest())
    config = dict(identity=identity, config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  trainable_parameters=sum(p.numel() for p in params), target_names=target_names,
                  loss='answer token mean CE including EOS, microbatch mean, accum%d; structured targets add annotation and tokens' % args.accum,
                  target_tokens=dict(minimum=min(answer_lengths), maximum=max(answer_lengths), mean=float(np.mean(answer_lengths))),
                  selection='maximum validation image AUROC, initial C with structured prompt eligible, earliest ties',
                  prompt='fixed structured schema v1; categories match original VisA prompt; no mask/bbox at inference',
                  generation='greedy max80; exact-gated leftpad batch candidate on fixed VisA val8; failure fixes scalar per initial/final/best checkpoint condition; policy serialized',
                  source_sha256={n:sha256_file(Path(__file__).parent/n) for n in ('structured_train.py','structured_model.py','structured_geometry.py','migration_advanced_train.py','advanced_generation.py')},
                  test_queries=0, external_queries=0)
    atomic_json(config, str(args.out/'run.json'))
    print('[config]', json.dumps(config), flush=True)

    def checkpoint(step, auc):
        return dict(identity=identity, config=config, step=step, val_auroc=auc, adapter=adapter_state(model))

    def check_frozen():
        assert bridge_digest(model) == bridge_hash
        for name, p in model.named_parameters():
            if name in frozen_versions:
                assert p._version == frozen_versions[name] and p.grad is None, name

    policy = freeze_generation_policy(model, cache, args.max_new_tokens, args.eval_batch_size,
                                      context=dict(condition='initial', seed=args.seed, source_c_best_sha256=identity['source_c_best_sha256']))
    atomic_json(policy, str(args.out/'initial_generation_policy.json'))
    initial, initial_records = evaluate(val_ids, generate=True)
    atomic_json(initial, str(args.out/'initial_val_metrics.json'))
    atomic_json(initial_records, str(args.out/'initial_val_predictions.json'))
    best = last_auc = initial['overall']['scores']['image_auroc']; best_step = 0
    atomic_save(checkpoint(0, best), str(args.out/'best.pt'))
    atomic_json(initial_records, str(args.out/'best_val_predictions.json'))
    # Simple baseline: fixed upper-5%-of-map region, no fitting or GT at inference.
    # Separate oracle-label and model-decision-gated views expose dependency on
    # the existing localization teacher and on detection quality.
    baseline_texts, gated_texts = [], []
    for i, record in zip(val_ids, initial_records):
        probability = cache['maps'][i].numpy(); region = probability >= np.quantile(probability, .95)
        target = mask_to_target(region, 1)
        baseline_texts.append(target_answer(target))
        gated_texts.append(target_answer(target) if record['score'] > 0 else target_answer(dict(label=0,bbox=None,location='none')))
    atomic_json(dict(rule='union box of teacher map >= its imagewise 95th percentile; no GT/calibration; box on every image',
                     unconditional_box=localization_metrics([targets[i] for i in val_ids], baseline_texts),
                     initial_score_gated=localization_metrics([targets[i] for i in val_ids], gated_texts)), str(args.out/'teacher_map_bbox_baseline.json'))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=args.steps) if args.preflight else torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=.05)
    losses, gradient_audit, train_seconds = [], [], 0.
    with (args.out/'training.jsonl').open('w') as history:
        for step in range(1, args.steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); tick=time.time(); total=0.
            for k in range(args.accum):
                offset = ((step-1)*args.accum+k)*args.microbatch
                batch = pack(schedule[offset:offset+args.microbatch], answers=True)
                loss = model.loss(batch)
                assert torch.isfinite(loss)
                (loss/args.accum).backward(); total += float(loss.detach())/args.accum
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            assert torch.isfinite(norm) and float(norm) > 0.
            if step <= 2:
                audited = {g:sum(float(p.grad.float().norm()) for n,p in named if g in n and p.grad is not None) for g in ('.lora_A.', '.lora_B.')}
                assert all(v>0 for v in audited.values()); gradient_audit.append(dict(step=step,norms=audited))
            optimizer.step(); scheduler.step(); torch.cuda.synchronize(); train_seconds += time.time()-tick; losses.append(total)
            row=dict(step=step,loss=total,gradient_norm=float(norm),train_seconds=train_seconds,learning_rate=scheduler.get_last_lr()[0])
            if step<=2 or step%10==0:
                history.write(json.dumps(row)+'\n'); history.flush(); print('[step]',json.dumps(row),flush=True)
            if step%args.val_every==0 or step==args.steps:
                check_frozen(); metrics, records = evaluate(val_ids)
                last_auc=metrics['overall']['scores']['image_auroc']
                atomic_json(metrics,str(args.out/('val_step_%d.json'%step)))
                print('[val]',step,last_auc,flush=True)
                if last_auc>best:
                    best,best_step=last_auc,step
                    atomic_save(checkpoint(step,best),str(args.out/'best.pt'))
                    atomic_json(records,str(args.out/'best_val_predictions.json'))
    check_frozen(); atomic_save(checkpoint(args.steps,last_auc),str(args.out/'final.pt'))
    del optimizer,scheduler,batch,loss
    torch.cuda.empty_cache()
    policy = freeze_generation_policy(model, cache, args.max_new_tokens, args.eval_batch_size,
                                      context=dict(condition='final', seed=args.seed, checkpoint_sha256=sha256_file(args.out/'final.pt')))
    atomic_json(policy, str(args.out/'final_generation_policy.json'))
    final_metrics,final_records=evaluate(val_ids,generate=True)
    atomic_json(final_metrics,str(args.out/'final_val_metrics.json')); atomic_json(final_records,str(args.out/'final_val_predictions.json'))
    with torch.no_grad():
        params[0].add_(.001)
    selected=torch.load(args.out/'best.pt',map_location='cpu')
    result=set_peft_model_state_dict(model.language_model,selected['adapter']); assert not result.unexpected_keys
    reloaded=adapter_state(model); assert all(torch.equal(reloaded[k],v) for k,v in selected['adapter'].items())
    best_generation_reused = selected['step'] in (0, args.steps)
    if best_generation_reused:
        reference_mode = 'initial' if selected['step'] == 0 else 'final'
        reference_path = args.out / (reference_mode + '_val_predictions.json')
        reference = json.loads(reference_path.read_text())
        policy_path = args.out / (reference_mode + '_generation_policy.json')
        policy = json.loads(policy_path.read_text())
        # The implementation decision is copied, not reselected after reload.
        # Keeping the file bytes identical makes the lineage auditable.
        force_generation_policy(model, policy)
        (args.out/'best_generation_policy.json').write_bytes(policy_path.read_bytes())
        scored_metrics, scored_records = evaluate(val_ids)
        assert [r['path'] for r in reference] == [r['path'] for r in scored_records]
        assert [r['index'] for r in reference] == [r['index'] for r in scored_records]
        assert [r['target'] for r in reference] == [r['target'] for r in scored_records]
        full_score_error = max(abs(left['score']-right['score']) for left,right in zip(reference,scored_records))
        assert full_score_error == 0., ('Endpoint-reuse full validation scores differ', full_score_error)
        # Reproduce complete original evaluation batches. The first and last
        # batches are fixed without labels/model outputs and span normal and
        # anomaly groups in the source order. Formal batch4 checks exactly8.
        all_batches = [val_ids[j:j+args.eval_batch_size] for j in range(0,len(val_ids),args.eval_batch_size)]
        if len(val_ids) < 8:
            raise ValueError('Endpoint-reuse engineering audit needs at least eight validation images')
        chosen_batches = [all_batches[0]]
        if len(all_batches)>1:
            chosen_batches.append(all_batches[-1])
        for chunk in all_batches[1:-1]:
            if sum(len(batch) for batch in chosen_batches) >= 8:
                break
            chosen_batches.append(chunk)
        check_ids = [index for chunk in chosen_batches for index in chunk]
        reference_lookup = {record['index']:record for record in reference}
        generated = []
        model.eval()
        with torch.no_grad():
            for chunk in chosen_batches:
                generated.extend(model.generate(pack(chunk), max_new_tokens=args.max_new_tokens))
        assert generated == [reference_lookup[index]['generated'] for index in check_ids], 'Endpoint-reuse generation audit differs'
        model.train()
        best_records = [dict(record, generated=old['generated'], parsed=old['parsed']) for record,old in zip(scored_records,reference)]
        best_metrics = evaluate_predictions([items[i] for i in val_ids],
                                           [r['score'] for r in best_records],
                                           [r['generated'] for r in best_records])
        best_metrics['localization'] = localization_metrics([targets[i] for i in val_ids], [r['generated'] for r in best_records])
        best_metrics['localization_per_category'] = {
            c:localization_metrics([targets[i] for i in val_ids if items[i]['category']==c],
                                   [r['generated'] for r in best_records if r['category']==c])
            for c in sorted({items[i]['category'] for i in val_ids})}
        reuse_audit = dict(state='complete', identity=identity, selected_step=selected['step'],
                           selected_checkpoint_sha256=sha256_file(args.out/'best.pt'),
                           reference_mode=reference_mode, reference_predictions_sha256=sha256_file(reference_path),
                           reference_policy_sha256=sha256_file(policy_path),
                           copied_best_policy_sha256=sha256_file(args.out/'best_generation_policy.json'),
                           full_score_n=len(val_ids), full_score_max_abs_error=full_score_error,
                           generation_n=len(check_ids), generation_indices=check_ids,
                           generation_paths=[items[i]['path'] for i in check_ids],
                           original_evaluation_batch_composition_preserved=True,
                           generation_exact_match=True, generation_policy=policy['selected'],
                           reused_generation_n=len(best_records), model_outputs_changed=False,
                           training_or_selection_changed=False, test_queries=0, external_queries=0)
    else:
        policy = freeze_generation_policy(model, cache, args.max_new_tokens, args.eval_batch_size,
                                          context=dict(condition='best', seed=args.seed, checkpoint_sha256=sha256_file(args.out/'best.pt')))
        atomic_json(policy, str(args.out/'best_generation_policy.json'))
        best_metrics,best_records=evaluate(val_ids,generate=True)
        reuse_audit = dict(state='not_applicable', selected_step=selected['step'],
                           reason='Intermediate selected checkpoint receives a full generation evaluation',
                           reused_generation_n=0, test_queries=0, external_queries=0)
    atomic_json(reuse_audit, str(args.out/'best_generation_reuse_audit.json'))
    prior=json.loads((args.out/'best_val_predictions.json').read_text())
    reload_error=max(abs(a['score']-b['score']) for a,b in zip(prior,best_records)); assert reload_error==0.
    atomic_json(best_metrics,str(args.out/'best_val_metrics.json')); atomic_json(best_records,str(args.out/'best_val_generated_predictions.json'))
    model.language_model.save_pretrained(args.out/'best_adapter',safe_serialization=True)
    check_frozen()
    if args.preflight:
        assert np.mean(losses[-5:])<np.mean(losses[:5]), 'Fixed train example fit did not improve'
    complete=dict(state='complete',schema=SCHEMA,seed=args.seed,steps=args.steps,exposures=len(schedule),preflight=args.preflight,
                  best_step=best_step,initial_val_auroc=initial['overall']['scores']['image_auroc'],best_val_auroc=best,final_val_auroc=last_auc,
                  reload_max_abs_score_error=reload_error,bridge_unchanged=True,frozen_parameter_versions_unchanged=True,
                  gradient_audit=gradient_audit,first5_mean_loss=float(np.mean(losses[:5])),last5_mean_loss=float(np.mean(losses[-5:])),
                  train_seconds=train_seconds,exposures_per_second=len(schedule)/train_seconds,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-started,test_queries=0,external_queries=0,
                  generation_policies={mode:json.loads((args.out/(mode+'_generation_policy.json')).read_text())['selected'] for mode in ('initial','best','final')},
                  best_generation_reused=best_generation_reused,
                  best_generation_reuse_audit_sha256=sha256_file(args.out/'best_generation_reuse_audit.json'))
    atomic_json(complete,str(args.out/'complete.json')); print('[complete]',json.dumps(complete),flush=True)


if __name__=='__main__':
    main()

"""Fresh-process best/final reload audit with a single immutable base load."""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from migration_language import FrozenLanguageBridge
from migration_advanced_train import (adapter_state, bridge_digest, configure_arm,
                                      load_bridge, load_delta, sha256_file)
from dec_big import atomic_json


def choose_batches(records, batch_size, n):
    """Spread complete original batches across validation, retaining companions."""
    starts = list(range(0, len(records), batch_size))
    count = min(len(starts), max(1, (n + batch_size - 1) // batch_size))
    chosen = sorted(set(starts[round(k * (len(starts) - 1) / max(1, count - 1))]
                        for k in range(count)))
    return [records[start:start + batch_size] for start in chosen]


def audit_mode(model, cache, run_dir, run, done, mode, n):
    cfg, identity = run['config'], run['identity']
    seed, arm = identity['seed'], identity['arm']
    # D restores BOTH bridge and adapter each time. No trained final bridge is
    # reused when best differs, including best step0. C/R bridge remains fixed.
    state = load_delta(model, run_dir / (mode + '.pt'), identity)
    assert state['step'] == (done['best_step'] if mode == 'best' else done['steps'])
    reference_name = 'final_val_predictions.json' if mode == 'final' else 'best_val_generated_predictions.json'
    all_records = json.loads((run_dir / reference_name).read_text())
    assert len(all_records) == 380 and len({r['path'] for r in all_records}) == 380
    chunks = choose_batches(all_records, cfg['eval_batch_size'], n)
    chosen = [record for chunk in chunks for record in chunk]
    lookup = {item['path']: index for index, item in enumerate(cache['items'])}
    id_chunks = [[lookup[record['path']] for record in chunk] for chunk in chunks]
    ids = [index for chunk in id_chunks for index in chunk]
    assert all(cache['items'][index]['split'] == 'val' for index in ids)
    model.eval()
    scores, texts, adapter_disabled = [], [], []
    for batch_ids in id_chunks:
        packed = model.pack(cache['cls'][batch_ids].cuda(), cache['maps'][batch_ids].cuda(),
                            [cache['items'][index]['category'] for index in batch_ids])
        with torch.no_grad():
            scores.extend(model.score(packed).float().cpu().tolist())
            texts.extend(model.generate(packed, max_new_tokens=32))
            with model.language_model.disable_adapter():
                adapter_disabled.extend(model.score(packed).float().cpu().tolist())
        del packed
    error = max(abs(record['score'] - score) for record, score in zip(chosen, scores))
    assert error == 0., ('Reload score mismatch', mode, error)
    assert texts == [record['generated'] for record in chosen], ('Reload generation mismatch', mode)
    disabled_reference = ('initial_val_predictions.json' if arm in ('C', 'R')
                          else mode + '_adapter_disabled_val_predictions.json')
    initial = json.loads((run_dir / disabled_reference).read_text())
    initial_lookup = {record['path']: record['score'] for record in initial}
    disabled_error = max(abs(initial_lookup[record['path']] - score)
                         for record, score in zip(chosen, adapter_disabled))
    assert disabled_error == 0., ('Disabled adapter reference mismatch', mode, disabled_error)
    adapter = adapter_state(model)
    nonzero = any(torch.count_nonzero(value).item() for key, value in adapter.items() if 'lora_B' in key)
    assert nonzero or (mode == 'best' and state['step'] == 0), 'Trained adapter is zero'
    return dict(state='complete', arm=arm, seed=seed, checkpoint=mode,
                step=state['step'], n=len(chosen), split='existing_val',
                original_batch_composition_preserved=True,
                audited_categories=sorted(set(cache['items'][index]['category'] for index in ids)),
                audited_labels=sorted(set(cache['items'][index]['label'] for index in ids)),
                fresh_reload_max_abs_score_error=error, generated_exact_match=True,
                adapter_B_nonzero=nonzero, adapter_disabled_reference=('D_bridge_only' if arm == 'D' else 'A'),
                adapter_disabled_max_abs_score_error=disabled_error,
                bridge_tensor_sha256=bridge_digest(model),
                checkpoint_sha256=sha256_file(run_dir / (mode + '.pt')),
                reference_predictions_sha256=sha256_file(run_dir / reference_name),
                verifier_sha256=sha256_file(__file__), test_queries=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--mode', choices=['best', 'final', 'both'], default='final')
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()
    assert args.n > 0
    root, run_dir = args.root.resolve(), args.run.resolve()
    with open(root.parent / 'gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = json.loads((run_dir / 'complete.json').read_text())
        assert not done['preflight'] and done['steps'] == 500
        run = json.loads((run_dir / 'run.json').read_text())
        cfg, identity = run['config'], run['identity']
        seed, arm = identity['seed'], identity['arm']
        assert arm in ('C', 'D', 'R') and done['arm'] == arm and done['seed'] == seed
        assert sha256_file(identity['bridge_path']) == identity['bridge_sha256']
        assert sha256_file(cfg['cache']) == identity['cache_sha256']
        assert sha256_file(cfg['base_manifest']) == identity['base_manifest_sha256']
        torch.set_num_threads(8)
        torch.manual_seed(seed)
        # One base/source/config/cache load for both checkpoints in this fresh
        # process. Every checkpoint's complete delta is reloaded before scoring.
        model = FrozenLanguageBridge('qwen', device='cuda', init_seed=seed)
        load_bridge(model, torch.load(identity['bridge_path'], map_location='cpu'))
        configure_arm(model, arm, cfg['rank'], cfg['alpha'], cfg['lr_bridge'], cfg['lr_lora'])
        cache = torch.load(cfg['cache'], map_location='cpu')
        modes = ('final', 'best') if args.mode == 'both' else (args.mode,)
        results = [audit_mode(model, cache, run_dir, run, done, mode, args.n) for mode in modes]
        # Do not publish a successful dual-mode gate before both audits finish.
        for result in results:
            result.update(both_modes_verified=args.mode == 'both',
                          immutable_base_loads_in_process=1, verification_invocation=args.mode)
            atomic_json(result, str(run_dir / ('fresh_' + result['checkpoint'] + '_reload.json')))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Fresh-process reload audit for D and paired CE/ranking checkpoints."""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from migration_language import FrozenLanguageBridge
from migration_advanced_train import (adapter_state, bridge_digest, configure_arm,
                                  load_bridge, load_delta, sha256_file)
from dec_big import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--mode', choices=['best', 'final'], default='final')
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()
    root, run_dir = args.root.resolve(), args.run.resolve()
    with open(root.parent / 'gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert (run_dir / 'complete.json').exists()
        run = json.loads((run_dir / 'run.json').read_text())
        cfg, identity = run['config'], run['identity']
        seed, arm = identity['seed'], identity['arm']
        assert arm in ('C', 'D', 'R')
        assert sha256_file(identity['bridge_path']) == identity['bridge_sha256']
        assert sha256_file(cfg['cache']) == identity['cache_sha256']
        assert sha256_file(cfg['base_manifest']) == identity['base_manifest_sha256']
        torch.set_num_threads(8)
        torch.manual_seed(seed)
        model = FrozenLanguageBridge('qwen', device='cuda', init_seed=seed)
        load_bridge(model, torch.load(identity['bridge_path'], map_location='cpu'))
        configure_arm(model, arm, cfg['rank'], cfg['alpha'], cfg['lr_bridge'], cfg['lr_lora'])
        state = load_delta(model, run_dir / (args.mode + '.pt'), identity)
        cache = torch.load(cfg['cache'], map_location='cpu')
        reference_name = 'final_val_predictions.json' if args.mode == 'final' else 'best_val_generated_predictions.json'
        all_records = json.loads((run_dir / reference_name).read_text())
        batch_size = cfg['eval_batch_size']
        # Keep original eval-batch companions, but spread audited batches across all val.
        starts = list(range(0, len(all_records), batch_size))
        count = min(len(starts), max(1, (args.n + batch_size - 1) // batch_size))
        chosen_starts = sorted(set(starts[round(k*(len(starts)-1)/max(1,count-1))] for k in range(count)))
        chosen = [record for start in chosen_starts for record in all_records[start:start+batch_size]]
        lookup = {x['path']: i for i, x in enumerate(cache['items'])}
        ids = [lookup[x['path']] for x in chosen]
        assert all(cache['items'][i]['split'] == 'val' for i in ids)
        model.eval()
        scores, texts = [], []
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start+batch_size]
            packed = model.pack(cache['cls'][batch_ids].cuda(), cache['maps'][batch_ids].cuda(),
                                [cache['items'][i]['category'] for i in batch_ids])
            with torch.no_grad():
                scores.extend(model.score(packed).float().cpu().tolist())
                texts.extend(model.generate(packed, max_new_tokens=32))
        error = max(abs(x['score']-score) for x, score in zip(chosen, scores))
        assert error == 0., ('Reload score mismatch', error)
        assert texts == [x['generated'] for x in chosen], 'Reload generation mismatch'
        with torch.no_grad():
            adapter_disabled = []
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start+batch_size]
                packed = model.pack(cache['cls'][batch_ids].cuda(), cache['maps'][batch_ids].cuda(),
                                    [cache['items'][i]['category'] for i in batch_ids])
                with model.language_model.disable_adapter():
                    adapter_disabled.extend(model.score(packed).float().cpu().tolist())
        if arm in ('C', 'R'):
            initial = json.loads((run_dir / 'initial_val_predictions.json').read_text())
            initial_lookup = {x['path']: x['score'] for x in initial}
            disabled_error = max(abs(initial_lookup[x['path']]-score)
                                 for x, score in zip(chosen, adapter_disabled))
            assert disabled_error == 0., ('Disabled adapter did not recover A', disabled_error)
        else:
            initial = json.loads((run_dir / (args.mode + '_adapter_disabled_val_predictions.json')).read_text())
            initial_lookup = {x['path']: x['score'] for x in initial}
            disabled_error = max(abs(initial_lookup[x['path']]-score)
                                 for x, score in zip(chosen, adapter_disabled))
            assert disabled_error == 0., ('Disabled adapter did not recover D bridge-only output', disabled_error)
        adapter = adapter_state(model)
        nonzero = any(torch.count_nonzero(value).item() for key, value in adapter.items()
                      if 'lora_B' in key)
        assert nonzero or state['step'] == 0, 'Selected trained adapter is zero'
        result = dict(state='complete', arm=arm, seed=seed, checkpoint=args.mode,
                      step=state['step'], n=len(chosen), split='existing_val',
                      audited_categories=sorted(set(cache['items'][i]['category'] for i in ids)),
                      audited_labels=sorted(set(cache['items'][i]['label'] for i in ids)),
                      fresh_reload_max_abs_score_error=error, generated_exact_match=True,
                      adapter_B_nonzero=nonzero,
                      adapter_disabled_reference=('D_bridge_only' if arm == 'D' else 'A'),
                      adapter_disabled_max_abs_score_error=disabled_error,
                      bridge_tensor_sha256=bridge_digest(model),
                      verifier_sha256=sha256_file(__file__), test_queries=0)
        atomic_json(result, str(run_dir / ('fresh_' + args.mode + '_reload.json')))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

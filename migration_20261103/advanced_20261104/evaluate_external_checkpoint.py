"""Evaluate a frozen bridge/LoRA checkpoint once on the locked MVTec residual set."""
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import torch

from migration_language import FrozenLanguageBridge
from migration_advanced_train import configure_arm, load_bridge, load_delta, sha256_file
from migration_metrics import evaluate_predictions
from advanced_generation import attach_generation, freeze_generation_policy
from dec_big import atomic_json as _atomic_json

def atomic_json(value, path):
    _atomic_json(value, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--external-cache', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--mode', choices=['best'], default='best')
    ap.add_argument('--baseline-a', action='store_true', help='Use matching source bridge with no language LoRA')
    args = ap.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    cache_path, manifest_path = args.external_cache.resolve(), args.manifest.resolve()
    manifest_sha = sha256_file(manifest_path)
    results_dir = run / ('external_mvtec_residual_v1_A' if args.baseline_a else 'external_mvtec_residual_v1')
    results_dir.mkdir(exist_ok=False)
    with open(root.parent/'gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = json.loads((run/'run.json').read_text())
        identity, cfg = config['identity'], config['config']
        torch.set_num_threads(8)
        torch.manual_seed(identity['seed'])
        model = FrozenLanguageBridge('qwen', device='cuda', init_seed=identity['seed'])
        assert sha256_file(identity['bridge_path']) == identity['bridge_sha256']
        assert sha256_file(cfg['base_manifest']) == identity['base_manifest_sha256']
        source = torch.load(identity['bridge_path'], map_location='cpu')
        load_bridge(model, source)
        if args.baseline_a:
            state = dict(step=source['step'])
        else:
            configure_arm(model, identity['arm'], cfg['rank'], cfg['alpha'], cfg['lr_bridge'], cfg['lr_lora'])
            state = load_delta(model, run/'best.pt', identity)
        if not args.baseline_a and state['step'] != json.loads((run/'complete.json').read_text())['best_step']:
            raise AssertionError('Selected checkpoint does not match completed validation selection')
        sidecar = json.loads(cache_path.with_suffix('.json').read_text())
        assert sha256_file(cache_path) == sidecar['cache_sha256']
        visa_cache = torch.load(cfg['cache'], map_location='cpu')
        assert sha256_file(cfg['cache']) == identity['cache_sha256']
        attach_generation(model)
        policy = freeze_generation_policy(model, visa_cache, 32, cfg['eval_batch_size'], context=('A' if args.baseline_a else run.name)+'_external_frozen')
        atomic_json(policy, results_dir/'generation_policy.json')
        del visa_cache
        torch.cuda.empty_cache()
        cache = torch.load(cache_path, map_location='cpu')
        assert cache['decoder_sha256'] == identity['decoder_sha256']
        assert cache['pipeline_version'] == identity['pipeline_version']
        if cache['manifest_sha256'] != manifest_sha:
            raise AssertionError('External cache is bound to another manifest')
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
        if len(records) != len(cache['items']) or len(records) != 1575:
            raise AssertionError(('External manifest count changed', len(records), len(cache['items'])))
        for record, item in zip(records, cache['items']):
            if record['source_id'] != item['source_id'] or record['label'] != item['label']:
                raise AssertionError('External record/cache order mismatch')
        scores, texts, latencies = [], [], []
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        model.eval()
        batch_size = cfg['eval_batch_size']
        for start in range(0, len(records), batch_size):
            end = min(start+batch_size, len(records))
            cls, maps = cache['cls'][start:end].cuda(), cache['maps'][start:end].cuda()
            categories = [item['category'] for item in cache['items'][start:end]]
            packed = model.pack(cls, maps, categories)
            tick = time.time()
            with torch.no_grad():
                scores.extend(model.score(packed).float().cpu().tolist())
                texts.extend(model.generate(packed, max_new_tokens=32))
            latencies.extend([(time.time()-tick)/(end-start)]*(end-start))
            del packed, cls, maps
            if end % 128 < batch_size or end == len(records):
                partial = dict(run=run.name, processed=end, total=len(records), elapsed=time.time()-started)
                atomic_json(partial, results_dir/'progress.json')
                print(json.dumps(partial), flush=True)
        items = [dict(source_id=r['source_id'], path=r['path'], category=r['category'],
                      label=int(r['label']), split='external') for r in records]
        metrics = evaluate_predictions(items, scores, texts)
        output = [dict(**item, score=float(score), generated=text, latency_seconds=float(latency))
                  for item,score,text,latency in zip(items,scores,texts,latencies)]
        atomic_json(metrics, results_dir/'metrics.json')
        prediction_path = results_dir/'predictions.jsonl'
        prediction_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output))
        summary = dict(state='complete', run=run.name, arm=('A' if args.baseline_a else identity['arm']), seed=identity['seed'],
                       checkpoint_step=state['step'], n=len(items), category_count=len({x['category'] for x in items}),
                       manifest_sha256=manifest_sha, cache_sha256=sha256_file(cache_path),
                       metrics=metrics, mean_latency_seconds=sum(latencies)/len(latencies),
                       peak_memory_bytes=torch.cuda.max_memory_allocated(),
                       checkpoint_sha256=sha256_file(identity['bridge_path'] if args.baseline_a else run/'best.pt'),
                       evaluator_sha256=sha256_file(__file__),
                       elapsed_seconds=time.time()-started, mask_opened=False, historical_visa_test_queries=0,
                       external_queries=len(items),
                       limitation='Public cross-dataset benchmark transfer audit; foundation pretraining exposure unknown.')
        atomic_json(summary, results_dir/'summary.json')
        (results_dir/'predictions.sha256').write_text(sha256_file(prediction_path)+'  predictions.jsonl\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

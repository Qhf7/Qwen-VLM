"""Choose a fixed ranking-loss scale using train-only gradient norms."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import torch.nn.functional as F

from migration_advanced_train import configure_arm, paired_schedule, sha256_file, load_bridge
from migration_language import FrozenLanguageBridge, YES, NO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bridge', required=True)
    ap.add_argument('--base-manifest', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(8)
    random.seed(100042)
    torch.manual_seed(42)
    cache = torch.load(args.cache, map_location='cpu')
    items = cache['items']
    schedule = paired_schedule(items, 100042, args.pairs * 2)
    model = FrozenLanguageBridge('qwen', device='cuda', init_seed=42)
    bridge = torch.load(args.bridge, map_location='cpu')
    load_bridge(model, bridge)
    named, _, _ = configure_arm(model, 'C', 8, 16, 1e-5, 1e-4)
    params = [p for _, p in named]
    measurements = []
    model.eval()
    for offset in range(0, len(schedule), 2):
        ids = schedule[offset:offset+2]
        packed = model.pack(cache['cls'][ids].cuda(), cache['maps'][ids].cuda(),
                            [items[i]['category'] for i in ids],
                            [YES if items[i]['label'] else NO for i in ids])
        logits = model.forward_batch(packed)
        labels = packed['labels'][:, 1:]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             labels.reshape(-1), ignore_index=-100)
        rows = torch.arange(2, device=logits.device)
        first = logits[rows, packed['prefix_lengths'] - 1].float()
        score = first[:, model.yes_id] - first[:, model.no_id]
        rank = F.softplus(-(score[1] - score[0]))
        ce_grad = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
        rank_grad = torch.autograd.grad(rank, params, allow_unused=True)
        ce_norm = sum(float(g.float().pow(2).sum()) for g in ce_grad if g is not None) ** 0.5
        rank_norm = sum(float(g.float().pow(2).sum()) for g in rank_grad if g is not None) ** 0.5
        if not (ce_norm > 0 and rank_norm > 0):
            raise RuntimeError(('Nonpositive objective gradient norm', offset, ce_norm, rank_norm))
        measurements.append(dict(pair_index=offset//2, category=items[ids[0]]['category'],
                                 ce=float(ce.detach()), rank=float(rank.detach()),
                                 ce_gradient_norm=ce_norm, rank_gradient_norm=rank_norm,
                                 equal_gradient_weight=ce_norm/rank_norm))
    proposed = statistics.median(x['equal_gradient_weight'] for x in measurements)
    weight = min(1.0, max(0.01, proposed))
    result = dict(state='complete', seed=42, split='train_only', pairs=len(measurements),
                  schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                  base_manifest_sha256=sha256_file(args.base_manifest),
                  source_bridge_sha256=sha256_file(args.bridge),
                  median_equal_gradient_weight=proposed, fixed_rank_weight=weight,
                  clamp=[0.01, 1.0], measurements=measurements,
                  validation_or_test_reads=0,
                  rule='median paired gradient-norm ratio ||grad CE||/||grad rank|| at LoRA initialization; clamp to [0.01,1.0]')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k!='measurements'}), flush=True)


if __name__ == '__main__':
    main()

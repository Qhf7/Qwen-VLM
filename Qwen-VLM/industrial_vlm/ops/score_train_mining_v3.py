"""Use the frozen scorer for train-only mining without editing v2 source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    from industrial_vlm.data import DECISION_PROMPT
    from industrial_vlm.score import _run
    def rows(p):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    inventory = {r['source_id']: r for r in rows(root/'prepared/visa/manifest.jsonl')}
    data = root/'prepared/curation_v3/mining_normal_candidates.jsonl'
    candidates = rows(data)
    if len(candidates) != 1152 or len({r['source_id'] for r in candidates}) != 1152:
        raise ValueError('Expected exactly 1152 unique frozen mining sources')
    for row in candidates:
        original = inventory[row['source_id']]
        if original['split'] != 'train' or original['is_anomalous'] is not False:
            raise ValueError('Mining source is not a train normal')
        if any(row[k] != original[k] for k in ('image','category','is_anomalous','boxes')):
            raise ValueError('Candidate metadata differs from original inventory')
        if row['prompt'] != DECISION_PROMPT or json.loads(row['answer']) != {'is_anomalous':False}:
            raise ValueError('Mining requires the exact label-free decision scoring policy')
    from collections import Counter
    if set(Counter(r['category'] for r in candidates).values()) != {96}:
        raise ValueError('Mining category quotas differ')
    # Preparation owns this immutable manifest. The runner separately freezes
    # plan, preparation code, and candidate bytes before any model stage.
    _run(argparse.Namespace(model=str(root/'models/Qwen3-VL-8B-Instruct'), adapter=None,
        data=data, image_root=root/'datasets', output=root/'evaluations/curation_v3/mining_base_train_normal',
        data_role='train_mining', max_pixels=524288, max_length=2048, limit=None,
        run_name='curation_v3_base_train_normal_mining', resume=args.resume, calibration=None))


if __name__ == '__main__':
    main()

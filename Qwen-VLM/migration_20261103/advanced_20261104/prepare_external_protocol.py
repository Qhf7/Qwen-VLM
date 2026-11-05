"""Freeze an MVTec cross-dataset audit set excluding previously scored images."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=Path('AnomalyGPT/data/mvtec_anomaly_detection'))
    ap.add_argument('--pilot-manifest', type=Path,
                    default=Path('industrial_vlm/prepared/mvtec/external_test.jsonl'))
    ap.add_argument('--pilot-predictions', type=Path,
                    default=Path('industrial_vlm/evaluations/external_base_region/predictions.jsonl'))
    ap.add_argument('--output', type=Path,
                    default=Path('migration_20261103/advanced_20261104/external_mvtec'))
    args = ap.parse_args()
    root = args.data_root.resolve()
    assert root.is_dir(), root
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    prior_records = read_rows(args.pilot_predictions)
    prior_ids = {str(row.get('source_id') or row.get('id', '').split('__', 1)[0])
                 for row in prior_records}
    prior_ids.discard('')
    pilot_rows = read_rows(args.pilot_manifest)
    manifest_ids = {str(row['source_id']) for row in pilot_rows}
    if prior_ids != manifest_ids:
        raise AssertionError(('Prior scored-image inventory differs from pilot manifest',
                              len(prior_ids), len(manifest_ids), len(prior_ids ^ manifest_ids)))

    previous_source_manifest = {
        'path': str(args.pilot_predictions), 'sha256': sha256(args.pilot_predictions),
        'n_predictions': len(prior_records), 'n_unique_source_ids': len(prior_ids),
        'selection_manifest': str(args.pilot_manifest),
        'selection_manifest_sha256': sha256(args.pilot_manifest),
    }
    candidates, excluded_path = [], []
    decode_failures, duplicate_hashes = [], []
    seen = {}
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        test_dir = category_dir / 'test'
        if not test_dir.is_dir():
            continue
        for image_path in sorted(p for p in test_dir.rglob('*') if p.is_file()
                                 and p.suffix.lower() in IMAGE_SUFFIXES):
            rel = image_path.relative_to(root).as_posix()
            image_hash = sha256(image_path)
            category = category_dir.name
            is_normal = image_path.parent.name.lower() == 'good'
            label = 0 if is_normal else 1
            source_id = 'mvtec_' + image_hash
            if source_id in prior_ids or image_hash in prior_ids:
                excluded_path.append(rel)
                continue
            if image_hash in seen:
                previous = seen[image_hash]
                if previous['label'] != label:
                    raise AssertionError(('Conflicting duplicate labels', previous['path'], rel))
                duplicate_hashes.append([previous['path'], rel, image_hash])
                continue
            seen[image_hash] = {'path': rel, 'label': label}
            mask_path = None
            mask_hash = None
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    image.convert('RGB')
                if label:
                    candidate = root / category / 'ground_truth' / image_path.parent.name / (image_path.stem + '_mask.png')
                    if not candidate.is_file():
                        raise FileNotFoundError(candidate)
                    with Image.open(candidate) as mask:
                        mask.load()
                        if mask.size != (width, height):
                            raise ValueError(('Image/mask geometry mismatch', rel, mask.size, (width, height)))
                        values = np.asarray(mask.convert('L'))
                        if not np.isin(values, (0, 255)).all() or not np.any(values):
                            raise ValueError(('Unexpected/empty anomaly mask', rel))
                    mask_path = candidate.relative_to(root).as_posix()
                    mask_hash = sha256(candidate)
                candidates.append(dict(source_id=source_id, path=rel, category=category,
                                       label=label, width=width, height=height,
                                       image_sha256=image_hash, mask_path=mask_path,
                                       mask_sha256=mask_hash,
                                       prompt_category='generic_industrial_component'))
            except Exception as exc:
                decode_failures.append(dict(path=rel, error=repr(exc)))
    if decode_failures:
        raise RuntimeError(('External-test inventory has invalid records', decode_failures[:5], len(decode_failures)))
    if not candidates:
        raise RuntimeError('No eligible external test images remained')
    counts = Counter((r['category'], r['label']) for r in candidates)
    categories = sorted({r['category'] for r in candidates})
    for category in categories:
        if counts[(category, 0)] == 0 or counts[(category, 1)] == 0:
            raise AssertionError(('Each category needs both labels for category AUROC', category))
    rows = sorted(candidates, key=lambda r: (r['category'], r['path']))
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows)
    manifest_path = out / 'manifest.jsonl'
    manifest_path.write_text(payload)
    category_counts = {category: {
        'normal': counts[(category, 0)], 'anomaly': counts[(category, 1)],
        'n': counts[(category, 0)] + counts[(category, 1)]}
        for category in categories}
    protocol = dict(
        protocol='visa_to_mvtec_residual_official_test_v1',
        status='frozen_before_D_and_ranking_experiments',
        role='method-specific cross-dataset evaluation for the AnomalyGPT language-migration branch',
        source_dataset='MVTec AD official test split',
        source_root_label='AnomalyGPT/data/mvtec_anomaly_detection',
        source_root_path_not_published=True,
        previous_project_evaluation_excluded=previous_source_manifest,
        excluded_prior_scored_paths=sorted(excluded_path),
        excluded_prior_scored_n=len(excluded_path),
        excluded_duplicate_groups=duplicate_hashes,
        retained_n=len(rows), n_normal=sum(r['label'] == 0 for r in rows),
        n_anomaly=sum(r['label'] == 1 for r in rows), categories=category_counts,
        manifest='manifest.jsonl', manifest_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        selection='all valid unique official test images except exact images scored in the existing 150-image Qwen pilot; no model-dependent sampling',
        model_selection='Use only VisA val380; freeze selected checkpoints/prompts before opening this external result',
        prompt='one shared generic industrial-component prompt across all 15 categories',
        training='No MVTec image, label, mask, or normal reference is used for training or hyperparameter selection',
        output_metrics=['image AUROC/AP', 'generated macro-F1/recall/FPR/invalid',
                        'category-macro AUROC/AP/F1', 'bbox IoU and pointing only for structured-output stage'],
        limits=['The 150 excluded pilot images were evaluated by another Qwen route in this repository.',
                'The retained images were not present in the AnomalyGPT VisA training cache or that pilot prediction manifest.',
                'Foundation-model pretraining exposure to the public benchmark is unknown.',
                'This is an external transfer audit, not a newly collected private blind benchmark.'],
        new_model_queries=0, test_queries=0)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(state='frozen', n=len(rows), normal=protocol['n_normal'],
                          anomaly=protocol['n_anomaly'], categories=len(categories),
                          excluded_prior=len(excluded_path), duplicate_groups=len(duplicate_hashes),
                          manifest_sha256=protocol['manifest_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()

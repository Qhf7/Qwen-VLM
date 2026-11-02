"""Prepare auditable VisA instruction splits; optionally prepare external MVTec AD.

Only Pillow is required beyond the Python standard library. No model is loaded.
Run ``python -m industrial_vlm.data --help`` with ``src`` on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

REGION_PROMPT = (
    'Inspect this industrial image. Return JSON only: '
    '{"is_anomalous": true or false, "boxes": [[x1,y1,x2,y2]]}. '
    'Use integer coordinates from 0 to 1000 relative to the full image. '
    'For an anomaly, give one box enclosing all defective pixels; '
    'for a normal image, use an empty boxes list.'
)
DECISION_PROMPT = (
    'Inspect this industrial image for defects. Return JSON only: '
    '{"is_anomalous": true or false}.'
)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _json_write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _jsonl_write(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(root: Path, value: str) -> Path:
    value = value.strip().replace('\\', '/')
    if not value:
        raise ValueError('empty_path')
    given = Path(value)
    choices = [given] if given.is_absolute() else [root / given]
    # Official mirrors sometimes retain the archive's top directory in CSV paths.
    if not given.is_absolute() and given.parts and given.parts[0].lower() == root.name.lower():
        choices.append(root.joinpath(*given.parts[1:]))
    for candidate in choices:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            continue
        if resolved.is_file():
            return resolved
    raise ValueError(f'file_missing_or_outside_data_root:{value}')


def _label(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {'normal', 'good', '0', 'false'}:
        return False
    if normalized in {'anomaly', 'anomalous', 'abnormal', 'defect', 'defective', 'bad', '1', 'true'}:
        return True
    raise ValueError(f'unknown_label:{value}')


def _mask_info(path: Path | None, image_size: tuple[int, int], anomalous: bool,
               dataset: str = 'mvtec') -> dict:
    width, height = image_size
    if path is None:
        if anomalous:
            raise ValueError('anomalous_image_without_mask')
        return {'boxes': [], 'mask_area_pixels': 0, 'mask_area_fraction': 0.0,
                'mask_sha256_binary': None, 'mask_values': [],
                'mask_label_semantics': 'no_mask_normal_image'}
    with Image.open(path) as mask:
        mask.load()
        if mask.size != image_size:
            raise ValueError(f'mask_size_mismatch:{mask.size}!={image_size}')
        if mask.mode in {'RGB', 'RGBA'}:
            red, green, blue = mask.convert('RGB').split()
            if red.tobytes() != green.tobytes() or red.tobytes() != blue.tobytes():
                raise ValueError('mask_is_not_grayscale')
        if mask.mode not in {'1', 'L', 'P', 'RGB', 'RGBA'}:
            raise ValueError(f'unsupported_mask_mode:{mask.mode}')
        # VisA's original masks contain defect class IDs, e.g. [0, 1, 2, 7].
        # Preserve palette indices instead of mapping them through display colors.
        gray = mask.copy() if mask.mode == 'L' else mask.convert('L')
        if dataset == 'visa' and mask.mode == 'P':
            gray = Image.frombytes('L', mask.size, mask.tobytes())
        histogram = gray.histogram()
        values = [i for i, count in enumerate(histogram) if count]
        if dataset != 'visa' and not (set(values).issubset({0, 1}) or set(values).issubset({0, 255})):
            raise ValueError(f'mask_is_not_binary:{values[:12]}')
        binary = gray.point(lambda value: 255 if value else 0)
        area = width * height - binary.histogram()[0]
        if anomalous and area == 0:
            raise ValueError('anomalous_mask_is_empty')
        if not anomalous and area:
            raise ValueError('normal_mask_has_foreground')
        bounds = binary.getbbox()  # Pixel edges: left, top, right-exclusive, bottom-exclusive.
        boxes = []
        if bounds is not None:
            x1, y1, x2, y2 = bounds
            boxes = [[math.floor(x1 * 1000 / width), math.floor(y1 * 1000 / height),
                      math.ceil(x2 * 1000 / width), math.ceil(y2 * 1000 / height)]]
        return {'boxes': boxes, 'mask_area_pixels': area,
                'mask_area_fraction': area / (width * height),
                'mask_sha256_binary': hashlib.sha256(binary.tobytes()).hexdigest(),
                'mask_values': values,
                'mask_label_semantics': ('visa_class_index_0_background_positive_foreground'
                                         if dataset == 'visa' else 'binary_0_background_positive_foreground')}


def _validate_record(raw: dict, root: Path, dataset: str) -> dict:
    category = str(raw['category']).strip()
    if not category:
        raise ValueError('empty_category')
    image_path = _resolve_relative(root, str(raw['image']))
    anomalous = _label(raw['label'])
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        orientation = image.getexif().get(274, 1)
        if orientation not in {None, 1}:
            raise ValueError(f'nontrivial_exif_orientation:{orientation}')
        if width < 1 or height < 1:
            raise ValueError('empty_image')
    mask_value = str(raw.get('mask') or '').strip()
    mask_path = None if mask_value.lower() in {'', 'none', 'nan', 'null'} else _resolve_relative(root, mask_value)
    annotation = _mask_info(mask_path, (width, height), anomalous, dataset=dataset)
    image_hash = _sha256(image_path)
    return {'source_id': f'{dataset}_{image_hash}', 'dataset': dataset,
            'image_sha256': image_hash, 'image': str(image_path),
            'image_relative': image_path.relative_to(root).as_posix(),
            'category': category, 'is_anomalous': anomalous,
            'width': width, 'height': height,
            'exif_orientation': orientation, 'coordinate_frame': 'stored_pixels_no_exif_transpose',
            'mask': str(mask_path) if mask_path else None,
            'official_split': str(raw.get('official_split', '')),
            'input_row': raw.get('input_row'), **annotation}


def _deduplicate(records: list[dict], rejects: list[dict], *, external: bool = False) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record['image_sha256']].append(record)
    unique, duplicates = [], []
    for image_hash, members in sorted(grouped.items()):
        members.sort(key=lambda r: (r['category'], r['image_relative'], str(r['input_row'])))
        # Identical image bytes with inconsistent labels/masks are ambiguous supervision.
        signatures = {(r['is_anomalous'], tuple(tuple(b) for b in r['boxes']),
                       r['mask_sha256_binary'] if r['is_anomalous'] else None) for r in members}
        conflict = len(signatures) > 1
        crosses_official = external and len({r['official_split'] for r in members}) > 1
        if conflict or crosses_official:
            reason = 'duplicate_image_conflicting_annotations' if conflict else 'duplicate_across_official_splits'
            for member in members:
                rejects.append({'reason': reason, 'source_id': member['source_id'],
                                'image': member['image'], 'input_row': member['input_row']})
            continue
        canonical = dict(members[0])
        canonical['aliases'] = [{'image': r['image'], 'category': r['category'],
                                 'official_split': r['official_split'], 'input_row': r['input_row']}
                                for r in members]
        canonical['duplicate_count'] = len(members)
        unique.append(canonical)
        if len(members) > 1:
            duplicates.append({'source_id': canonical['source_id'],
                               'canonical_image': canonical['image'], 'aliases': canonical['aliases']})
    return unique, duplicates


def _visa_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / 'split_csv' / '1cls.csv').is_file():
        return root
    matches = sorted(root.glob('*/split_csv/1cls.csv'))
    if len(matches) == 1:
        return matches[0].parent.parent.resolve()
    raise ValueError('Expected DATA_ROOT/split_csv/1cls.csv, or exactly one archive directory containing it')


def read_visa(root: Path) -> tuple[Path, list[dict], list[dict], list[dict]]:
    root = _visa_root(root)
    records, rejects = [], []
    with (root / 'split_csv' / '1cls.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'object', 'split', 'label', 'image', 'mask'}.issubset(reader.fieldnames or []):
            raise ValueError('VisA CSV must contain object,split,label,image,mask columns')
        for row_number, row in enumerate(reader, start=2):
            raw = {'category': row['object'], 'official_split': row['split'], 'label': row['label'],
                   'image': row['image'], 'mask': row['mask'], 'input_row': row_number}
            try:
                records.append(_validate_record(raw, root, 'visa'))
            except (ValueError, OSError, SyntaxError) as error:
                rejects.append({'input_row': row_number, 'image': row.get('image'), 'reason': str(error)})
    unique, duplicates = _deduplicate(records, rejects)
    return root, unique, rejects, duplicates


def _rounded_split_sizes(count: int) -> list[int]:
    exact = [count * 0.70, count * 0.15, count * 0.15]
    sizes = [math.floor(value) for value in exact]
    priority = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))
    for index in priority[:count - sum(sizes)]:
        sizes[index] += 1
    return sizes


def split_records(records: list[dict], seed: int) -> dict[str, list[dict]]:
    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for record in records:
        strata[(record['category'], record['is_anomalous'])].append(record)
    splits: dict[str, list[dict]] = {name: [] for name in ('train', 'val', 'test')}
    for key, members in sorted(strata.items()):
        members = sorted(members, key=lambda r: r['source_id'])
        random.Random(f'{seed}:split:{key}').shuffle(members)
        sizes = _rounded_split_sizes(len(members))
        start = 0
        for name, count in zip(splits, sizes):
            for record in members[start:start + count]:
                splits[name].append({**record, 'split': name})
            start += count
    for members in splits.values():
        members.sort(key=lambda r: r['source_id'])
    return splits


def _quotas(capacities: dict[Any, int], size: int) -> dict[Any, int]:
    """Equal allocation with capacity redistribution; deterministic key order."""
    if size < 0 or size > sum(capacities.values()):
        raise ValueError(f'Requested {size} samples, but only {sum(capacities.values())} are available')
    result = dict.fromkeys(sorted(capacities), 0)
    while sum(result.values()) < size:
        for key in result:
            if result[key] < capacities[key]:
                result[key] += 1
                if sum(result.values()) == size:
                    break
    return result


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def training_subsets(records: list[dict], size: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for record in records:
        strata[(record['category'], record['is_anomalous'])].append(record)
    quotas = _quotas({key: len(rows) for key, rows in strata.items()}, size)
    quality, random_rows, thresholds = [], [], {}
    for key, members in sorted(strata.items()):
        members = sorted(members, key=lambda r: r['source_id'])
        random_order = members.copy()
        random.Random(f'{seed}:random:{key}').shuffle(random_order)
        selected_random = sorted(random_order[:quotas[key]], key=lambda r: r['source_id'])
        random_rows.extend(selected_random)
        if not key[1]:
            # Hold normal membership fixed: only anomalous area coverage varies.
            quality.extend({**row, 'area_stratum': 'normal'} for row in selected_random)
            continue
        bins: dict[str, list[dict]] = defaultdict(list)
        areas = [r['mask_area_fraction'] for r in members]
        q1, q2 = _quantile(areas, 1 / 3), _quantile(areas, 2 / 3)
        thresholds[key[0]] = {'q1': q1, 'q2': q2, 'computed_from': 'train_anomalies_only'}
        for record in members:
            area = record['mask_area_fraction']
            bucket = 'small' if area <= q1 else ('medium' if area <= q2 else 'large')
            bins[bucket].append(record)
        area_quotas = _quotas({bucket: len(rows) for bucket, rows in bins.items()}, quotas[key])
        selected_quality = []
        for bucket, rows in sorted(bins.items()):
            random.Random(f'{seed}:quality:{key}:{bucket}').shuffle(rows)
            selected_quality.extend({**row, 'area_stratum': bucket} for row in rows[:area_quotas[bucket]])
        quality.extend(sorted(selected_quality, key=lambda r: r['source_id']))
    # Each arm begins with equal-sized sorted category/label blocks. An identical
    # permutation matches the category/label at every position across the arms.
    random.Random(f'{seed}:shared_order').shuffle(quality)
    random.Random(f'{seed}:shared_order').shuffle(random_rows)
    normal_ids_equal = ({r['source_id'] for r in quality if not r['is_anomalous']} ==
                        {r['source_id'] for r in random_rows if not r['is_anomalous']})
    sequence_equal = ([(r['category'], r['is_anomalous']) for r in quality] ==
                      [(r['category'], r['is_anomalous']) for r in random_rows])
    if not normal_ids_equal or not sequence_equal:
        raise AssertionError('Sampling control failed: normal sources or category/label sequence differ')
    details = {'category_label_quotas': {f'{k[0]}|{int(k[1])}': v for k, v in quotas.items()},
               'area_thresholds': thresholds,
               'normal_source_ids_equal': normal_ids_equal,
               'category_label_sequence_equal': sequence_equal,
               'presentation_order': f'{seed}:shared_order',
               'quality_random_overlap': len({r['source_id'] for r in quality} &
                                             {r['source_id'] for r in random_rows})}
    return quality, random_rows, details


def balanced_subset(records: list[dict], requested_size: int, seed: int, tag: str) -> list[dict]:
    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for record in records:
        strata[(record['category'], record['is_anomalous'])].append(record)
    quotas = _quotas({key: len(rows) for key, rows in strata.items()}, min(requested_size, len(records)))
    chosen = []
    for key, rows in sorted(strata.items()):
        rows = sorted(rows, key=lambda r: r['source_id'])
        random.Random(f'{seed}:{tag}:{key}').shuffle(rows)
        chosen.extend(rows[:quotas[key]])
    return sorted(chosen, key=lambda r: r['source_id'])


def instruction(record: dict, *, decision_only: bool = False) -> dict:
    answer = {'is_anomalous': record['is_anomalous']}
    if not decision_only:
        answer['boxes'] = record['boxes']
    return {'id': record['source_id'] + ('__decision' if decision_only else '__region'),
            'source_id': record['source_id'], 'image': record['image'],
            'category': record['category'], 'is_anomalous': record['is_anomalous'],
            'anomaly_area_fraction': record['mask_area_fraction'],
            # Ground truth metadata; only image/prompt/answer may enter model inputs.
            'boxes': record['boxes'], 'prompt': DECISION_PROMPT if decision_only else REGION_PROMPT,
            'answer': json.dumps(answer, separators=(',', ':'))}


def _counts(records: list[dict]) -> dict:
    return dict(sorted(Counter(f"{r['category']}|{'anomaly' if r['is_anomalous'] else 'normal'}"
                               for r in records).items()))


def _assert_disjoint(splits: dict[str, list[dict]]) -> None:
    seen = set()
    for name, rows in splits.items():
        ids = {row['source_id'] for row in rows}
        if len(ids) != len(rows) or ids & seen:
            raise AssertionError(f'Duplicate or overlapping source_id in split {name}')
        seen.update(ids)


def prepare_visa(data_root: Path, output: Path, train_size: int = 768,
                 eval_size: int = 120, seed: int = 42) -> dict:
    root, records, rejects, duplicates = read_visa(data_root)
    if not records:
        raise ValueError('No valid VisA images remain after annotation validation')
    splits = split_records(records, seed)
    _assert_disjoint(splits)
    quality, random_rows, sampling = training_subsets(splits['train'], train_size, seed)
    if _counts(quality) != _counts(random_rows):
        raise AssertionError('Quality and random category/label quotas differ')
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = sorted((r for rows in splits.values() for r in rows), key=lambda r: r['source_id'])
    _jsonl_write(output / 'manifest.jsonl', manifest)
    _json_write(output / 'split.json', {name: [r['source_id'] for r in rows] for name, rows in splits.items()})
    _jsonl_write(output / 'train_quality_region.jsonl', [instruction(r) for r in quality])
    _jsonl_write(output / 'train_random_region.jsonl', [instruction(r) for r in random_rows])
    _jsonl_write(output / 'train_quality_decision.jsonl', [instruction(r, decision_only=True) for r in quality])
    eval_selections = {}
    for name in ('val', 'test'):
        rows = splits[name]
        _jsonl_write(output / f'{name}.jsonl', [instruction(r) for r in rows])
        _jsonl_write(output / f'{name}_decision.jsonl', [instruction(r, decision_only=True) for r in rows])
        balanced = balanced_subset(rows, eval_size, seed, name)
        _jsonl_write(output / f'{name}_balanced_{eval_size}.jsonl', [instruction(r) for r in balanced])
        _jsonl_write(output / f'{name}_balanced_{eval_size}_decision.jsonl',
                     [instruction(r, decision_only=True) for r in balanced])
        eval_selections[name] = {'requested': eval_size, 'actual': len(balanced), 'counts': _counts(balanced)}
    _jsonl_write(output / 'rejects.jsonl', rejects)
    _jsonl_write(output / 'duplicates.jsonl', duplicates)
    report = {'dataset': 'visa', 'protocol': 'custom_supervised_exact_hash_stratified_70_15_15',
              'data_root': str(root), 'seed': seed, 'image_dedup': 'sha256_original_file_bytes',
              'valid_unique_images': len(records), 'rejected_rows': len(rejects),
              'reject_reasons': dict(Counter(r['reason'].split(':')[0] for r in rejects)),
              'duplicate_groups': len(duplicates),
              'duplicates_removed': sum(len(r['aliases']) - 1 for r in duplicates),
              'splits': {name: {'n': len(rows), 'counts': _counts(rows)} for name, rows in splits.items()},
              'train_requested': train_size, 'quality_train_n': len(quality), 'random_train_n': len(random_rows),
              'sampling': sampling, 'balanced_evaluation': eval_selections,
              'checks': {'source_id_disjoint': True, 'exact_hash_disjoint': True,
                         'quality_random_same_category_label_quotas': True,
                         'normal_source_ids_equal': sampling['normal_source_ids_equal'],
                         'category_label_sequence_equal': sampling['category_label_sequence_equal'],
                         'region_decision_same_quality_sources': True},
              'warning': 'Custom supervised re-split; not the official normal-only VisA protocol. '
                         'Area coverage is a sampling ablation, not a label-correctness ablation.'}
    _json_write(output / 'audit_report.json', report)
    return report


def _mvtec_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if any((p / 'test').is_dir() for p in root.iterdir() if p.is_dir()):
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and
                  any((c / 'test').is_dir()
                      for c in p.iterdir() if c.is_dir())]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError('Expected MVTec category/train/good, category/test/<defect>, category/ground_truth/<defect>')


def prepare_mvtec(data_root: Path, output: Path, eval_size: int = 120, seed: int = 42) -> dict:
    """Prepare independent external evaluation; official train/good is reference-only."""
    root = _mvtec_root(data_root)
    valid, rejects = [], []
    for category in sorted(p for p in root.iterdir() if p.is_dir()):
        for official_split, image_dir in [('official_train_normal', category / 'train' / 'good'),
                                          ('official_test', category / 'test')]:
            if not image_dir.is_dir():
                continue
            for image in sorted(p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
                defect = 'good' if official_split == 'official_train_normal' else image.parent.name
                anomalous = defect.lower() not in {'good', 'normal'}
                mask = ''
                if anomalous:
                    mask_dir = category / 'ground_truth' / defect
                    candidates = [mask_dir / (image.stem + '_mask.png'), mask_dir / (image.stem + '.png')]
                    found = next((p for p in candidates if p.is_file()), None)
                    mask = str(found.relative_to(root)) if found else str(candidates[0].relative_to(root))
                raw = {'category': category.name, 'label': str(int(anomalous)),
                       'image': str(image.relative_to(root)), 'mask': mask,
                       'official_split': official_split, 'input_row': str(image.relative_to(root))}
                try:
                    valid.append(_validate_record(raw, root, 'mvtec'))
                except (ValueError, OSError, SyntaxError) as error:
                    rejects.append({'input_row': raw['input_row'], 'image': raw['image'], 'reason': str(error)})
    records, duplicates = _deduplicate(valid, rejects, external=True)
    train_normal = [r for r in records if r['official_split'] == 'official_train_normal']
    test = [r for r in records if r['official_split'] == 'official_test']
    if not test:
        raise ValueError('No valid official MVTec test images found')
    _assert_disjoint({'reference_only': train_normal, 'external_test': test})
    balanced = balanced_subset(test, eval_size, seed, 'mvtec_external')
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _jsonl_write(output / 'external_manifest.jsonl', records)
    _jsonl_write(output / 'external_train_normal_manifest.jsonl', train_normal)
    _jsonl_write(output / 'external_test.jsonl', [instruction(r) for r in test])
    _jsonl_write(output / 'external_test_decision.jsonl', [instruction(r, decision_only=True) for r in test])
    _jsonl_write(output / f'external_test_balanced_{eval_size}.jsonl', [instruction(r) for r in balanced])
    _jsonl_write(output / f'external_test_balanced_{eval_size}_decision.jsonl',
                 [instruction(r, decision_only=True) for r in balanced])
    _json_write(output / 'split.json', {'reference_only': [r['source_id'] for r in train_normal],
                                      'external_test': [r['source_id'] for r in test]})
    _jsonl_write(output / 'rejects.jsonl', rejects)
    _jsonl_write(output / 'duplicates.jsonl', duplicates)
    report = {'dataset': 'mvtec_ad', 'protocol': 'official_external_test_no_training',
              'data_root': str(root), 'seed': seed, 'reference_only_n': len(train_normal),
              'external_test_n': len(test), 'external_test_counts': _counts(test),
              'balanced_requested': eval_size, 'balanced_actual': len(balanced),
              'balanced_counts': _counts(balanced), 'rejected_rows': len(rejects),
              'coverage': 'external_partial_or_complete_inventory_not_verified',
              'reject_reasons': dict(Counter(r['reason'].split(':')[0] for r in rejects)),
              'duplicate_groups': len(duplicates),
              'checks': {'source_id_disjoint': True, 'training_generated': False},
              'warning': 'Inventory may be a partial external subset; this command does not verify '
                         'the complete official benchmark. Official train/good, when present, is '
                         'reference-only and must not enter VisA training. '
                         'Use external test only after model and prompt selection.'}
    _json_write(output / 'audit_report.json', report)
    return report


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = 'prepare_visa'
    if argv and argv[0] in {'prepare_visa', 'prepare_mvtec'}:
        mode = argv.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-size', type=int, default=768,
                        help='Total examples per VisA training arm; unused for prepare_mvtec')
    parser.add_argument('--eval-size', type=int, default=120,
                        help='Balanced evaluation subset size, capped by available holdout images')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_size < 1 or args.eval_size < 1:
        parser.error('--train-size and --eval-size must be positive')
    try:
        if mode == 'prepare_mvtec':
            report = prepare_mvtec(args.data_root, args.output, args.eval_size, args.seed)
        else:
            report = prepare_visa(args.data_root, args.output, args.train_size, args.eval_size, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(2, f'Data preparation failed: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""CPU-only mask targets and strict structured localization metrics."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

SCHEMA = 'visa-yesno-bbox1000-grid3-v1'
LOCATIONS = tuple(v + '_' + h for v in ('top', 'middle', 'bottom') for h in ('left', 'center', 'right'))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def location_for_box(box):
    x = min(2, int(((box[0] + box[2]) / 2) * 3 / 1000))
    y = min(2, int(((box[1] + box[3]) / 2) * 3 / 1000))
    return LOCATIONS[y * 3 + x]


def mask_to_target(mask, label):
    if label not in (0, 1):
        raise ValueError('Explicit binary label is required')
    if label == 0:
        if mask is not None and np.asarray(mask).any():
            raise ValueError('Normal image cannot have foreground')
        return dict(label=0, bbox=None, location='none', foreground_pixels=0)
    binary = np.asarray(mask) > 0
    if binary.ndim == 3:
        binary = binary.any(axis=2)
    if binary.ndim != 2 or not binary.any():
        raise ValueError('Anomalous image requires a nonempty binary mask')
    ys, xs = np.nonzero(binary)
    h, w = binary.shape
    box = [int(math.floor(xs.min() * 1000 / w)), int(math.floor(ys.min() * 1000 / h)),
           int(math.ceil((xs.max() + 1) * 1000 / w)), int(math.ceil((ys.max() + 1) * 1000 / h))]
    return dict(label=1, bbox=box, location=location_for_box(box),
                foreground_pixels=int(binary.sum()), mask_width=w, mask_height=h)


def target_answer(target):
    return ('Yes' if target['label'] else 'No') + '\n' + json.dumps(
        dict(bbox=target['bbox'], location=target['location']), separators=(',', ':'))


def parse_structured(text):
    if not isinstance(text, str):
        return None
    found = re.fullmatch(r'\s*(Yes|No)\s*\n\s*(\{.*\})\s*', text, flags=re.DOTALL)
    if not found:
        return None
    try:
        def unique_object(pairs):
            if len({k for k, _ in pairs}) != len(pairs):
                raise ValueError('Duplicate JSON key')
            return dict(pairs)
        payload = json.loads(found.group(2), object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {'bbox', 'location'}:
        return None
    label = int(found.group(1) == 'Yes')
    box, loc = payload['bbox'], payload['location']
    if label == 0:
        return dict(label=0, bbox=None, location='none') if box is None and loc == 'none' else None
    if (not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box)
            or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
            or loc not in LOCATIONS or loc != location_for_box(box)):
        return None
    return dict(label=1, bbox=box, location=loc)


def bbox_iou(first, second):
    if first is None or second is None:
        return 0.0
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    union = (first[2]-first[0])*(first[3]-first[1]) + (second[2]-second[0])*(second[3]-second[1]) - intersection
    return intersection / union if union else 0.0


def bbox_center_hit(predicted, target):
    if predicted is None or target is None:
        return False
    cx, cy = (predicted[0]+predicted[2])/2, (predicted[1]+predicted[3])/2
    return target[0] <= cx < target[2] and target[1] <= cy < target[3]


def localization_metrics(targets, texts):
    if len(targets) != len(texts) or not targets:
        raise ValueError('Nonempty aligned targets and predictions required')
    parsed = [parse_structured(x) for x in texts]
    anomaly = [i for i, t in enumerate(targets) if t['label'] == 1]
    valid_positive = [i for i in anomaly if parsed[i] and parsed[i]['label'] == 1]
    def average(values):
        return sum(values)/len(values) if values else None
    normal = [i for i, t in enumerate(targets) if t['label'] == 0]
    anomaly_valid = [parsed[i] is not None for i in anomaly]
    normal_valid = [parsed[i] is not None for i in normal]
    ious = [bbox_iou(parsed[i]['bbox'] if parsed[i] else None, targets[i]['bbox']) for i in anomaly]
    hits = [bbox_center_hit(parsed[i]['bbox'] if parsed[i] else None, targets[i]['bbox']) for i in anomaly]
    locations = [bool(parsed[i] and parsed[i]['label'] == 1 and parsed[i]['location'] == targets[i]['location']) for i in anomaly]
    return dict(schema=SCHEMA, n=len(targets), n_anomaly=len(anomaly), n_normal=len(targets)-len(anomaly),
                n_valid=sum(x is not None for x in parsed), format_valid_rate=average([x is not None for x in parsed]),
                n_valid_positive_on_anomaly=len(valid_positive),
                format_valid_rate_anomaly=average(anomaly_valid), format_valid_rate_normal=average(normal_valid),
                anomaly_bbox_iou_mean_all=average(ious), anomaly_iou_at_05_all=average([v >= .5 for v in ious]),
                anomaly_box_center_inside_gt_box_all=average(hits), anomaly_grid3_accuracy_all=average(locations),
                anomaly_bbox_iou_mean_given_valid_positive=average([bbox_iou(parsed[i]['bbox'], targets[i]['bbox']) for i in valid_positive]),
                anomaly_iou_at_05_given_valid_positive=average([bbox_iou(parsed[i]['bbox'], targets[i]['bbox']) >= .5 for i in valid_positive]),
                note='All-anomaly denominators count malformed, No, or missing boxes as zero. Conditional metrics require coverage. Center hit is box-center-inside-GT-box, not pixel-mask point accuracy; union box covers all components.')


def mask_path_for_item(item, data_root=None):
    if int(item['label']) == 0:
        return None
    image = Path(item['path']); parts = list(image.parts)
    try:
        offset = next(i for i in range(len(parts)-2) if parts[i:i+3] == ['Data', 'Images', 'Anomaly'])
    except StopIteration:
        raise ValueError('Not a VisA anomaly path: ' + str(image))
    if data_root is not None:
        image = Path(data_root).joinpath(*parts[offset-1:]); parts = list(image.parts)
        offset = next(i for i in range(len(parts)-2) if parts[i:i+3] == ['Data', 'Images', 'Anomaly'])
    parts[offset+1] = 'Masks'
    expected = Path(*parts).with_suffix('.png')
    if expected.is_file():
        return expected
    candidates = sorted(p for p in expected.parent.glob(expected.stem + '.*') if p.suffix.lower() == '.png')
    if len(candidates) != 1:
        raise FileNotFoundError('Missing or ambiguous VisA mask: ' + str(expected))
    return candidates[0]


def build_targets(items, data_root=None):
    records = []
    for index, item in enumerate(items):
        if item['split'] not in ('train', 'val'):
            continue
        path = mask_path_for_item(item, data_root)
        if path is None:
            target = mask_to_target(None, 0); mask_sha = None
        else:
            with Image.open(path) as image:
                pixels = np.array(image)
            target = mask_to_target(pixels, 1); mask_sha = sha256_file(path)
        records.append(dict(index=index, path=item['path'], category=item['category'], split=item['split'],
                            **target, mask_path=str(path) if path else None, mask_sha256=mask_sha,
                            answer=target_answer(target)))
    assert len({x['path'] for x in records}) == len(records)
    return records


def self_test():
    mask = np.zeros((10, 20), np.uint8); mask[2:4, 3:8] = 7
    target = mask_to_target(mask, 1)
    assert target['bbox'] == [150, 200, 400, 400]
    assert parse_structured(target_answer(target))['bbox'] == target['bbox']
    normal = mask_to_target(None, 0)
    assert parse_structured(target_answer(normal))['label'] == 0
    malformed = ['Yes\n{"bbox":[0,0,0,1],"location":"top_left"}',
                 'No\n{"bbox":[0,0,1000,1000],"location":"middle_center"}',
                 'Yes\n{"bbox":[0.0,0,1,1],"location":"top_left"}',
                 'Yes\n{"bbox":[0,0,100,100],"location":"bottom_right"}',
                 'No\n{"bbox":null,"bbox":null,"location":"none"}',
                 target_answer(target) + ' extra', 'No', '{}']
    assert all(parse_structured(x) is None for x in malformed)
    results = localization_metrics([target, target, normal], [target_answer(target), 'No', target_answer(normal)])
    assert results['anomaly_bbox_iou_mean_all'] == .5 and results['format_valid_rate'] == 2/3
    assert results['anomaly_bbox_iou_mean_given_valid_positive'] == 1.
    one = np.zeros((10000, 10000), dtype=np.uint8); one[9900, 9999] = 1
    assert mask_to_target(one, 1)['bbox'] == [999, 990, 1000, 991]
    assert bbox_iou([0,0,500,500], [0,0,1000,1000]) == .25
    return dict(state='passed', checks='mask palette/nonempty/coordinates, strict schema/decision/geometry, invalid-denominator, single-pixel lesion')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--items-jsonl', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    if bool(args.cache) == bool(args.items_jsonl) or not args.out:
        parser.error('Specify exactly one --cache/--items-jsonl and --out')
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.cache:
        import torch
        cache = torch.load(args.cache, map_location='cpu')
        items = cache['items']; cache_sha = sha256_file(args.cache)
    else:
        items = [json.loads(x) for x in args.items_jsonl.read_text().splitlines() if x.strip()]
        cache_sha = None
    records = build_targets(items, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
    metadata = dict(schema=SCHEMA, manifest_sha256=sha256_file(args.out), cache_sha256=cache_sha,
                    records=len(records), counts=dict(Counter((x['split'] + '_' + str(x['label'])) for x in records)),
                    mask_reads_by_split=dict(Counter(x['split'] for x in records if x['mask_path'])), test_mask_reads=0,
                    rule='full-resolution binary mask union box, outward-rounded normalized 0..1000; grid by box center; no masks passed as model input',
                    source_sha256=sha256_file(__file__))
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()

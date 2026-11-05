"""CPU verification of saved standalone student inference against the frozen batch."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    base = root / 'student_roi_20261104'
    manifest = json.loads((base / 'maps/manifest.json').read_text())
    full = json.loads((base / 'roi_full/run.json').read_text())
    require(manifest['status'] == full['status'] == 'complete', 'Batch/map run incomplete')
    predictions = [json.loads(line) for line in (base / 'roi_full/pre_gt_predictions.jsonl').read_text().splitlines()]
    require(len(predictions) == len({x['path'] for x in predictions}) == 380, 'Batch denominator or uniqueness differs')
    by_path = {row['path']: row for row in predictions}
    head = manifest['heads']['pre_gt']
    require(head['distill_weight'] == 0 and head['features'] == 'pre' and head['size'] == 448,
            'Frozen selected GT head changed')
    result = {'schema_version': 'student-single-image-batch-exact-verification-v1', 'status': 'complete',
              'gpu_used': False, 'scope': 'Saved artifacts only; two fixed val images, not a new performance estimate',
              'batch_run_sha256': sha(base / 'roi_full/run.json'),
              'maps_manifest_sha256': sha(base / 'maps/manifest.json'), 'cases': []}
    for tag, label, kind, stem in [('normal0784', 0, 'Normal', '0784'), ('anomaly010', 1, 'Anomaly', '010')]:
        path = base / 'demo' / (tag + '.json')
        demo = json.loads(path.read_text())
        payload = dict(demo)
        payload.pop('payload_sha256')
        require(canonical(payload) == demo['payload_sha256'], tag + ': payload changed')
        require(demo['status'] == 'complete' and demo['gpu_model_loaded'] is True, tag + ': not complete')
        image = demo['provenance']['image']
        require(image.endswith('/candle/Data/Images/' + kind + '/' + stem + '.JPG'), tag + ': wrong fixed input')
        row = by_path[image]
        require(row['label'] == label and row['category'] == 'candle', tag + ': fixed label mismatch')
        record = head['maps'][image]
        require(demo['provenance']['head']['sha256'] == head['sha256']
                and demo['provenance']['head']['state_sha256'] == head['state_sha256'], tag + ': head mismatch')
        require(demo['provenance']['image_sha256'] == record['input_sha256'], tag + ': input mismatch')
        require(all(demo[key] is False for key in ('teacher_model_used', 'teacher_cache_read', 'imagebind_used',
                                                   'gt_masks_read', 'train_fit')), tag + ': inference dependency differs')
        require('label' not in demo['config'] and 'category' not in demo['config'], tag + ': unexpected label input')
        require(demo['dtype_audit']['passed'] is True and demo['native_source_audit']['passed'] is True,
                tag + ': source/dtype audit failed')
        source = demo['script_sha256']
        require(source == sha(base / 'code/migration_infer_student_roi.py'), tag + ': CLI source differs')
        plan = json.loads(path.with_suffix('.plan.json').read_text())
        plan_sha = plan.pop('plan_sha256')
        require(canonical(plan) == plan_sha == demo['plan_sha256'], tag + ': plan identity differs')
        status = json.loads(path.with_suffix('.status.json').read_text())
        require(status['status'] == 'complete' and status['outjson_sha256'] == sha(path), tag + ': final status differs')
        artifacts = demo['artifacts']
        local_map = path.with_suffix('.map.npy')
        require(sha(local_map) == artifacts['probability_npy_sha256'] == record['sha256'], tag + ': npy differs')
        array = np.load(local_map, allow_pickle=False)
        require(array.shape == (224, 224) and array.dtype == np.float32 and np.isfinite(array).all(), tag + ': invalid map')
        require(0 <= array.min() <= array.max() <= 1, tag + ': probability range')
        require(hashlib.sha256(array.tobytes(order='C')).hexdigest() == artifacts['array_sha256'] == record['array_sha256'],
                tag + ': raw map array differs')
        require(float(array.min()) == artifacts['min'] == record['min']
                and float(array.max()) == artifacts['max'] == record['max'], tag + ': range audit differs')
        require(sha(path.with_suffix('.map.heatmap.png')) == artifacts['heatmap_png_sha256'], tag + ': heatmap differs')
        exact_fields = ('yes_logit', 'no_logit', 'score_yes_minus_no', 'generated_text', 'generated_token_ids', 'roi',
                        'source_image_wh', 'view_sources', 'view_roles', 'image_grid_thw', 'image_tokens_per_view',
                        'image_tokens_total', 'prompt_tokens_total', 'n_images')
        differences = [key for key in exact_fields if demo[key] != row[key]]
        require(not differences, tag + ': batch mismatch: ' + str(differences))
        require(demo['image_tokens_total'] == 392 and demo['prompt_tokens_total'] == 470, tag + ': token budget differs')
        require(demo['numerical_protocol']['dense_cudnn'] == dict(allow_tf32=False, benchmark=False, deterministic=True),
                tag + ': dense numerical settings differ')
        result['cases'].append({'tag': tag, 'path': image, 'label_for_evaluation_only': label, 'passed': True,
             'exact_fields': list(exact_fields), 'array_sha256': artifacts['array_sha256'],
             'max_abs_score_error': abs(demo['score_yes_minus_no'] - row['score_yes_minus_no']),
             'score_yes_minus_no': demo['score_yes_minus_no'], 'generated_text': demo['generated_text'],
             'generated_decision': demo['generated_decision'], 'missed_anomaly': label == 1 and demo['generated_decision'] == 'No',
             'elapsed_seconds_including_audit_and_loads': demo['elapsed_seconds'],
             'peak_gpu_allocated_bytes': demo['peak_gpu_allocated_bytes'], 'demo_json_sha256': sha(path)})
    result['n'] = len(result['cases'])
    result['passed'] = all(c['passed'] for c in result['cases'])
    result['script_sha256'] = sha(__file__)
    out = base / 'demo/verification.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

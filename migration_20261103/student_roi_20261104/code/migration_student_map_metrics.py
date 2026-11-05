"""CPU-only post-export metrics; GT masks are used only here, after predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from migration_roi_eval import atomic_json, sha256_file
from migration_student_roi import map_roi
from visa_pipeline import load_mask_tensor, mask_path_for_image, pixel_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest_path = Path(args.maps)
    manifest = json.loads(manifest_path.read_text())
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Existing offline metrics are preserved')
    if manifest['status'] != 'complete' or any(x['split'] != 'val' for x in manifest['items']):
        raise ValueError('Complete validation predictions required')
    truths = {x['path']: load_mask_tensor(mask_path_for_image(x['path'], x['label']), device='cpu', size=224)[0].numpy()
              for x in manifest['items']}
    result = {'pipeline_version': 'qwen-student-offline-export-map-metrics-v1',
              'scope': 'GT masks only for post-prediction CPU metrics; not part of inference or crop selection',
              'manifest_sha256': sha256_file(manifest_path), 'n_images': len(truths),
              'test_set_used': False, 'gpu_used': False, 'heads': {}}
    for name, head in manifest['heads'].items():
        predictions, hits, coverage = [], [], []
        for item in manifest['items']:
            record = head['maps'][item['path']]
            path = manifest_path.parent / record['path']
            if sha256_file(path) != record['sha256']:
                raise ValueError('Map changed after export')
            array = np.load(path, allow_pickle=False)
            truth = truths[item['path']]
            predictions.append((item['category'], item['label'], truth, array))
            if item['label'] == 1:
                spec = map_roi(array, 1 / 9)
                row, col = spec['student_peak_row_col']
                hits.append(bool(truth[row, col]))
                left, top, right, bottom = spec['bbox_xyxy_relative']
                axis = (np.arange(224) + .5) / 224
                chosen = ((axis[:, None] >= top) & (axis[:, None] < bottom)
                          & (axis[None, :] >= left) & (axis[None, :] < right))
                coverage.append(float(truth[chosen].sum() / truth.sum()))
        metrics = pixel_metrics(predictions)
        result['heads'][name] = {'metrics': metrics, 'peak_hits': sum(hits), 'anomaly_n': len(hits),
            'peak_in_gt_rate': float(np.mean(hits)), 'mean_gt_foreground_in_crop': float(np.mean(coverage)),
            'crop_contains_any_gt_n': sum(x > 0 for x in coverage),
            'crop_coverage_definition': '224 binary GT pixel centers inside normalized bbox; metric only, not crop construction'}
    atomic_json(result, out)
    print(json.dumps({'status': 'complete', 'out': str(out),
                      'heads': {n: {'pixel_ap': h['metrics']['pixel_ap_macro'],
                                     'pixel_auroc': h['metrics']['pixel_auroc_macro'],
                                     'peak_hits': h['peak_hits'], 'anomaly_n': h['anomaly_n']}
                                for n, h in result['heads'].items()}}))


if __name__ == '__main__':
    main()

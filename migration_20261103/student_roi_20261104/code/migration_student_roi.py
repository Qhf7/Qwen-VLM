"""Native Qwen ROI evaluation using exported Qwen dense-head probabilities.

No teacher cache, ImageBind model, teacher decoder, GT mask, fitting, or test
inference is used. A few recorded teacher crop boxes are replayed only to
check continuity with the earlier fixed-protocol control. Labels enter only
metrics and the explicitly separate replay sampling, never ROI selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from migration_roi_eval import (DEFAULT_MODEL, QUESTION, atomic_json, crop_raw_rgb,
                                prepare_processor_inputs, score_and_generate,
                                sha256_file, shifted_square_bbox, single_token_id)
from migration_metrics import evaluate_predictions


PIPELINE_VERSION = 'qwen3vl-student-map-native-roi-v1'


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_maps(path, selected_heads=None):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get('schema_version') != 'qwen-dense-map-export-v1' or value.get('status') != 'complete':
        raise ValueError('A complete dense map export is required')
    payload = dict(value)
    digest = payload.pop('payload_sha256', None)
    if digest != json_sha(payload):
        raise ValueError('Dense export manifest payload SHA mismatch')
    for key in ('no_teacher', 'no_imagebind', 'no_gt_masks'):
        if value.get(key) is not True:
            raise ValueError('Required student export contract missing: ' + key)
    for key in ('test_set_used', 'train_fit', 'language_model_instantiated'):
        if value.get(key) is not False:
            raise ValueError('Unsupported student export contract: ' + key)
    items = value['items']
    paths = [x['path'] for x in items]
    if not items or len(set(paths)) != len(paths) or any(x['split'] != 'val' for x in items):
        raise ValueError('Expected unique validation paths')
    if value.get('items_sha256') != json_sha(items) or value.get('n_images') != len(items):
        raise ValueError('Item identity mismatch')
    if value['geometry']['coordinate_system'] != 'direct-square-224' or value['geometry']['map_hw'] != [224, 224]:
        raise ValueError('ROI map coordinates must be direct-square-224')
    if value['geometry']['requested_image_hw'] != [448, 448] or value['geometry']['processor_do_resize'] is not False:
        raise ValueError('This controlled experiment requires actual dense square448')
    heads = list(selected_heads or value['heads'])
    if not heads or len(set(heads)) != len(heads) or any(h not in value['heads'] for h in heads):
        raise ValueError('Unknown or duplicate selected head')
    arrays, audits = {}, {}
    expected = {'pre_gt': ('dense_pre_gt_s42', 0.0), 'pre_kd': ('dense_pre_kd_s42', 0.5)}
    if set(heads) != set(expected):
        raise ValueError('This prespecified run requires pre_gt and pre_kd together')
    for name in heads:
        head = value['heads'][name]
        dirname, weight = expected[name]
        checkpoint = Path(head['checkpoint'])
        if (checkpoint.parent.name != dirname or checkpoint.name != 'best.pt' or head['features'] != 'pre'
                or head['size'] != 448 or head['step'] != 1200 or head['distill_weight'] != weight):
            raise ValueError('Wrong selected pre448 checkpoint identity for ' + name)
        training = json.loads((checkpoint.parent / 'run.json').read_text())
        completion = json.loads((checkpoint.parent / 'train_complete.json').read_text())
        if (training['compatibility'] != head['compatibility'] or training['config']['seed'] != 42
                or training['config']['steps'] != 1200 or completion['selected_best_step'] != 1200
                or completion['reload_max_abs_metric_error'] != 0):
            raise ValueError('Dense training/selected reload identity failed')
        config = training['config']
        if (config['bs'] != 4 or config['lr'] != .001 or config['val_every'] != 200
                or config['distill_weight'] != weight
                or head['matched_run_key'] != training['matched_run_key']
                or head['matched_run_key'] != 'e79f550e88f58e8a51829608f716fb0049e057cea8beadbb73f6b5ccb827dacf'):
            raise ValueError('GT/KD training budget or matched identity differs')
        training_plan = json.loads((checkpoint.parent / 'training_plan.json').read_text())
        if (training_plan['sha256'] != 'c057c9cf45a2fc1048ca042742da08d145a705ac83dc66b47507a96fd717821b'
                or json_sha(training_plan['batches']) != training_plan['sha256']):
            raise ValueError('Dense training sampling plan differs')
        if sha256_file(head['checkpoint']) != head['sha256']:
            raise ValueError('Student head checkpoint changed after map export')
        if set(head['maps']) != set(paths):
            raise ValueError('Map/image path keys do not match exactly for ' + name)
        arrays[name] = {}
        for item in items:
            record = head['maps'][item['path']]
            relative = Path(record['path'])
            source = (path.parent / relative).resolve()
            if relative.is_absolute() or path.parent not in source.parents:
                raise ValueError('Map file must be inside export directory')
            if sha256_file(source) != record['sha256']:
                raise ValueError('Map file hash mismatch')
            array = np.load(source, allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (224, 224) or not np.isfinite(array).all():
                raise ValueError('Map must be finite float32 [224,224]')
            if array.min() < 0 or array.max() > 1:
                raise ValueError('Map is not a probability')
            if record['shape'] != [224, 224] or record['dtype'] != 'float32':
                raise ValueError('Map metadata geometry/dtype differs')
            if record['source_sha256'] != head['compatibility']['source_sha256']:
                raise ValueError('Map feature source identity mismatch')
            geom = record['geometry']
            if (geom['requested_image_hw'] != [448, 448] or geom['actual_image_hw'] != [448, 448]
                    or geom['processor_do_resize'] is not False or geom['image_grid_thw'] != [1, 28, 28]):
                raise ValueError('Actual per-image dense geometry differs')
            if hashlib.sha256(array.tobytes(order='C')).hexdigest() != record['array_sha256']:
                raise ValueError('Map array hash mismatch')
            if record['head_sha256'] != head['sha256'] or record['input_sha256'] != sha256_file(item['path']):
                raise ValueError('Map checkpoint or input image identity mismatch')
            arrays[name][item['path']] = array
        audits[name] = {'maps_verified': len(items), 'checkpoint_sha256': head['sha256'],
                        'features': head['features'], 'size': head['size'], 'step': head['step'],
                        'distill_weight': head['distill_weight']}
    if value['heads']['pre_gt']['compatibility'] != value['heads']['pre_kd']['compatibility']:
        raise ValueError('GT/KD feature compatibility differs')
    return value, arrays, audits


def map_roi(array, area):
    if array.shape != (224, 224) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Invalid student probability geometry')
    row, col = np.unravel_index(int(array.argmax()), array.shape)
    center = [(int(col) + .5) / 224, (int(row) + .5) / 224]
    return {'kind': 'student', 'requested_center_xy_relative': center,
            'bbox_xyxy_relative': shifted_square_bbox(center, area), 'area_fraction': area,
            'student_peak_row_col': [int(row), int(col)], 'student_peak_value': float(array[row, col]),
            'student_map_hw': [224, 224], 'argmax_tie_rule': 'first in row-major order',
            'coordinate_source': 'student direct-square-224 map coordinates linearly to raw RGB x/y'}


def views_for_roi(image_path, spec, size=448):
    """Only image path and predicted/recorded box are inputs, never an item label."""
    with Image.open(image_path) as handle:
        raw = handle.convert('RGB')
    crop, pixel_box = crop_raw_rgb(raw, spec['bbox_xyxy_relative'], size)
    metadata = {'source_image_wh': list(raw.size), 'view_sources': [image_path, image_path],
                'view_roles': ['target_original', 'target_roi'], 'roi': {**spec, 'bbox_xyxy_raw_pixels_float': pixel_box}}
    return [raw.resize((size, size), Image.Resampling.BICUBIC), crop], metadata


def control_contract(control, items, area, size, max_tokens):
    cfg = control['config']
    if control.get('status') != 'complete' or control.get('n') != len(items):
        raise ValueError('Previous full control is not complete or has a different denominator')
    if control['selected_paths'] != [x['path'] for x in items] or control['question'] != QUESTION:
        raise ValueError('Previous control has different image identity/order or question')
    original_items = [{k: x[k] for k in ('path', 'category', 'label', 'split')} for x in items]
    if hashlib.sha256(json.dumps(original_items, sort_keys=True).encode()).hexdigest() != control['selected_manifest_sha256']:
        raise ValueError('Validation item metadata changed from previous control')
    if cfg['image_size'] != size or cfg['max_new_tokens'] != max_tokens or cfg['roi_area'] != area:
        raise ValueError('Previous ROI geometry/generation protocol differs')
    previous = control['results']['original_teacher_roi']
    if previous['visual_tokens_total_min_max'] != [392, 392] or previous['prompt_tokens_total_min_max'] != [470, 470]:
        raise ValueError('Previous teacher ROI token budget differs')
    return previous


def replay_cases(rows):
    categories = sorted({x['category'] for x in rows})[:3]
    result = []
    for category in categories:
        for label in (0, 1):
            result.append(next(x for x in rows if x['category'] == category and x['label'] == label))
    return result


def cpu_test():
    array = np.zeros((224, 224), np.float32)
    array[3, 9] = array[40, 90] = .8
    spec = map_roi(array, 1 / 9)
    assert spec['student_peak_row_col'] == [3, 9]
    left, top, right, bottom = spec['bbox_xyxy_relative']
    assert min(left, top) >= 0 and max(right, bottom) <= 1
    assert math.isclose((right - left) * (bottom - top), 1 / 9)
    assert spec['requested_center_xy_relative'] == [9.5 / 224, 3.5 / 224]
    print(json.dumps({'passed': True, 'argmax_tie': True, 'geometry_area': True, 'label_free_roi': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps')
    parser.add_argument('--head', action='append')
    parser.add_argument('--control-run')
    parser.add_argument('--control-predictions')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--out')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cpu-test-only', action='store_true')
    args = parser.parse_args()
    if args.cpu_test_only:
        cpu_test()
        return
    if not all((args.maps, args.out, args.control_run, args.control_predictions)):
        parser.error('--maps --out --control-run --control-predictions are required')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Use a fresh student ROI output directory')
    export, arrays, map_audits = checked_maps(args.maps, args.head)
    items = export['items']
    control = json.loads(Path(args.control_run).read_text())
    area, size, max_tokens = 1 / 9, 448, 32
    control_contract(control, items, area, size, max_tokens)
    rows = [json.loads(x) for x in Path(args.control_predictions).read_text().splitlines()]
    if [x['path'] for x in rows] != [x['path'] for x in items] or any(x['condition'] != 'original_teacher_roi' for x in rows):
        raise ValueError('Control predictions do not match validation path order')
    if any(any(row[k] != item[k] for k in ('category', 'label', 'split', 'cache_index')) for row, item in zip(rows, items)):
        raise ValueError('Control prediction metadata differs from validation manifest')
    if args.model != control['config']['model']:
        raise ValueError('Use the same model path as the earlier native ROI control')
    out.mkdir(parents=True)
    plan = {'pipeline_version': PIPELINE_VERSION, 'status': 'planned', 'n': len(items), 'split': 'val',
            'script_sha256': sha256_file(__file__),
            'items_sha256': json_sha(items), 'selected_paths': [x['path'] for x in items],
            'map_manifest': str(Path(args.maps).resolve()), 'map_manifest_sha256': sha256_file(args.maps),
            'map_audits': map_audits, 'question': QUESTION, 'config': vars(args),
            'image_size': size, 'roi_area': area, 'max_new_tokens': max_tokens,
            'teacher_model_used': False, 'teacher_cache_read': False, 'imagebind_used': False,
            'gt_masks_read': False, 'train_fit': False, 'test_set_used': False,
            'control_run': str(Path(args.control_run).resolve()), 'control_run_sha256': sha256_file(args.control_run),
            'control_predictions_sha256': sha256_file(args.control_predictions),
            'control_role': 'recorded teacher boxes replayed for continuity only; no teacher map/model loaded',
            'conditions': list(arrays), 'results': {}, 'replay': {}}
    atomic_json(plan, out / 'plan.json')
    if not args.run:
        print(json.dumps({'status': 'planned', 'n': len(items), 'heads': list(arrays)}))
        return
    run = {**plan, 'status': 'running'}
    atomic_json(run, out / 'run.json')
    try:
        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
        from migration_native_parity import source_audit, audit_dtypes
        if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16-capable CUDA device required')
        audit = source_audit(args.model)
        if not audit['passed'] or audit['source_sha256'] != control['source_audit']['source_sha256']:
            raise ValueError('Current native source differs from earlier control')
        if any(audit[k] != control['source_audit'][k] for k in ('model_config', 'processor_config', 'api')):
            raise ValueError('Current model/processor/API differs from earlier control')
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        run['source_audit'] = audit
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
            device_map={'': args.device}, attn_implementation='sdpa', low_cpu_mem_usage=True, local_files_only=True).eval()
        model.requires_grad_(False)
        dtype_audit = audit_dtypes(model)
        if not dtype_audit['passed']:
            raise ValueError('All native model modules must be BF16')
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        yes_id, no_id = (single_token_id(processor.tokenizer, x) for x in ('Yes', 'No'))
        cfg = GenerationConfig(max_new_tokens=max_tokens, do_sample=False, num_beams=1, use_cache=True,
            bos_token_id=model.generation_config.bos_token_id, eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id)
        run.update(dtype_audit=dtype_audit, runtime={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(args.device),
                   'precision': 'entire model BF16; no quantization', 'attention_backend': 'sdpa',
                   'yes_id': yes_id, 'no_id': no_id})
        with torch.inference_mode():
            cases = []
            for old in replay_cases(rows):
                views, _ = views_for_roi(old['path'], old['roi'], size)
                batch, tokens = prepare_processor_inputs(processor, model, views, args.device, size)
                actual = score_and_generate(model, processor, batch, yes_id, no_id, cfg, args.device)
                exact = (all(actual[k] == old[k] for k in ('yes_logit', 'no_logit', 'score_yes_minus_no', 'generated_token_ids', 'generated_text'))
                         and actual['score_yes_minus_no'] == actual['yes_logit'] - actual['no_logit']
                         and all(tokens[k] == old[k] for k in tokens))
                cases.append({'path': old['path'], 'passed': exact, 'max_abs_score_error': abs(actual['score_yes_minus_no'] - old['score_yes_minus_no']),
                              'token_budget': tokens, 'expected': {k: old[k] for k in actual}, 'actual': actual})
                # Timings are recorded but intentionally excluded from equality.
                del batch, views
            run['replay'] = {'passed': all(x['passed'] for x in cases), 'n': len(cases), 'cases': cases}
            atomic_json(run, out / 'run.json')
            if not run['replay']['passed']:
                raise AssertionError('Recorded teacher ROI replay changed; do not compare new arms to stale control')
            for name, maps in arrays.items():
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                baseline = torch.cuda.memory_allocated(args.device)
                started = time.perf_counter()
                predictions, scores, texts = [], [], []
                with (out / (name + '_predictions.jsonl')).open('x') as stream:
                    for position, item in enumerate(items):
                        began = time.perf_counter()
                        spec = map_roi(maps[item['path']], area)
                        views, metadata = views_for_roi(item['path'], spec, size)
                        batch, tokens = prepare_processor_inputs(processor, model, views, args.device, size)
                        prepared = time.perf_counter()
                        if tokens['image_tokens_per_view'] != [196, 196] or tokens['prompt_tokens_total'] != 470:
                            raise ValueError('Student ROI token budget differs from two-image control')
                        prediction = score_and_generate(model, processor, batch, yes_id, no_id, cfg, args.device)
                        row = {**item, 'condition': name, **metadata, **tokens, **prediction,
                               'prepare_seconds': prepared - began,
                               'elapsed_seconds': time.perf_counter() - began}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        predictions.append(row)
                        scores.append(prediction['score_yes_minus_no'])
                        texts.append(prediction['generated_text'])
                        del batch, views
                        if (position + 1) % 20 == 0 or position + 1 == len(items):
                            print(json.dumps({'condition': name, 'done': position + 1, 'n': len(items),
                                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
                summary = {'n': len(items), 'metrics': evaluate_predictions(items, scores=scores, generated_texts=texts),
                           'elapsed_seconds': time.perf_counter() - started,
                           'map_extraction_seconds_included': False,
                           'map_export_elapsed_seconds_all_heads': export.get('elapsed_seconds'),
                           'map_inference_seconds_total_shared_heads': export.get('inference_seconds_total'),
                           'map_export_seconds_shared_heads': export.get('export_map_seconds'),
                           'score_seconds_total': sum(x['score_seconds'] for x in predictions),
                           'generate_seconds_total': sum(x['generate_seconds'] for x in predictions),
                           'prepare_seconds_total': sum(x['prepare_seconds'] for x in predictions),
                           'baseline_gpu_allocated_bytes': baseline,
                           'visual_tokens_total_min_max': [min(x['image_tokens_total'] for x in predictions), max(x['image_tokens_total'] for x in predictions)],
                           'prompt_tokens_total_min_max': [min(x['prompt_tokens_total'] for x in predictions), max(x['prompt_tokens_total'] for x in predictions)],
                           'actual_image_grids': sorted({tuple(g) for x in predictions for g in x['image_grid_thw']}),
                           'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(args.device),
                           'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(args.device)}
                run['results'][name] = summary
                atomic_json(summary, out / (name + '_metrics.json'))
                atomic_json(run, out / 'run.json')
        run['status'] = 'complete'
    except Exception as error:
        run.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        atomic_json(run, out / 'run.json')
    print(json.dumps({'status': 'complete', 'n': len(items), 'conditions': list(run['results'])}))


if __name__ == '__main__':
    main()

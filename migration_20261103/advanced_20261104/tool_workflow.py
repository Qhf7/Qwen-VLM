"""Bounded model-selected RGB crop tool on the official native Qwen3-VL path.

The native visual path is a separate foundation-model experiment; compressed
bridge LoRA weights are not applied to a different visual representation.
Import and --self-test never load a model or open a dataset.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('ANOMALYGPT_REPO', '/root/autodl-tmp/AnomalyGPT'))
for folder in (HERE.parent / 'code', REPO, REPO / 'code/code/model'):
    sys.path.insert(0, str(folder))
PIPELINE_VERSION = 'qwen3vl-native-bounded-crop-tool-v1'
CONDITIONS = ('no_tool', 'duplicate_original', 'center_crop', 'random_crop', 'model_tool')
IMAGE_SIZE = 448
ROI_AREA = 1.0 / 9.0
PLAN_PROMPT = (
    'Inspect this product image for a visible anomaly. You may inspect at most one close-up. '
    'Choose whether another view is useful. Output exactly one JSON object, with no prose: '
    '{"action":"crop","center":[x,y]} or {"action":"answer"}. '
    'x and y are finite normalized coordinates between 0 and 1 in the original image; '
    'x increases rightward, y downward. The crop has fixed width and height of one third '
    'of the original image. Use only visible image content. Do not guess an unseen defect.'
)
FINAL_PROMPT = (
    'Now give the final decision for the original product. If a close-up image is supplied, '
    'it is sampled from the original at the recorded bounds. An unchanged view is explicitly '
    'marked as a duplicate. Is there any anomaly in the original product? '
    'Answer with Yes or No first, followed by one brief sentence about visible evidence. '
    'Do not request further tools.'
)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    value = {}
    for key, entry in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = entry
    return value


def parse_request(text):
    """Strict schema, no prose/fences/extra keys, bools, NaN, or coercion."""
    if not isinstance(text, str):
        return None, 'not_text'
    try:
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (ValueError, TypeError):
        return None, 'invalid_json'
    if not isinstance(value, dict) or type(value.get('action')) is not str:
        return None, 'invalid_object'
    if value['action'] == 'answer':
        return (value, None) if set(value) == {'action'} else (None, 'unexpected_keys')
    if value['action'] != 'crop' or set(value) != {'action', 'center'}:
        return None, 'invalid_action_or_keys'
    center = value['center']
    if not isinstance(center, list) or len(center) != 2:
        return None, 'invalid_center_shape'
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in center):
        return None, 'center_out_of_bounds_or_nonfinite'
    return {'action': 'crop', 'center': [float(x) for x in center]}, None


def crop_bounds(center):
    if len(center) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in center):
        raise ValueError('Crop center must be finite and normalized')
    side = math.sqrt(ROI_AREA)
    x = min(max(float(center[0]) - side / 2, 0), 1 - side)
    y = min(max(float(center[1]) - side / 2, 0), 1 - side)
    return [x, y, x + side, y + side]


def stable_center(path, seed):
    import random
    value = int.from_bytes(hashlib.sha256((str(seed) + '\0' + path).encode()).digest()[:8], 'big')
    rng = random.Random(value)
    side = math.sqrt(ROI_AREA)
    return [rng.uniform(side / 2, 1 - side / 2) for _ in range(2)]


def request_for_condition(condition, plan, error, path, seed):
    if condition in ('no_tool', 'duplicate_original'):
        return {'action': 'answer'}, None
    if condition == 'center_crop':
        return {'action': 'crop', 'center': [0.5, 0.5]}, None
    if condition == 'random_crop':
        return {'action': 'crop', 'center': stable_center(path, seed)}, None
    if condition == 'model_tool':
        return ({'action': 'answer'} if error else plan), error
    raise ValueError(condition)


def build_final_inputs(raw, condition, request, planner_error=None):
    """Only RGB and model/control request enter inference. No GT or maps."""
    from PIL import Image
    original = raw.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    bounds = crop_bounds(request['center']) if request['action'] == 'crop' else None
    views = [original]
    if bounds is not None:
        w, h = raw.size
        crop = raw.transform((IMAGE_SIZE, IMAGE_SIZE), Image.Transform.EXTENT,
                             (bounds[0] * w, bounds[1] * h, bounds[2] * w, bounds[3] * h),
                             resample=Image.Resampling.BICUBIC)
        views.append(crop)
        result = 'Tool crop result; normalized xyxy bounds: ' + json.dumps(bounds, separators=(',', ':')) + '.'
    elif condition != 'no_tool':
        views.append(original.copy())
        result = 'No crop executed. Second image is an unchanged duplicate of the first.'
        if planner_error:
            result += ' The tool request was invalid and was rejected.'
    else:
        result = 'No crop executed; only the original image is supplied.'
    # The final decision is a fresh prompt with actual returned visual evidence.
    # Arbitrary planner prose is not injected into the final classifier.
    content = [{'type': 'image'} for _ in views] + [{'type': 'text', 'text': result + '\n' + FINAL_PROMPT}]
    messages = [{'role': 'user', 'content': content}]
    return views, messages, {'request_executed': request, 'tool_calls': int(bounds is not None),
                            'bbox_xyxy_relative': bounds, 'tool_result': result,
                            'view_roles': ['original'] + ([] if condition == 'no_tool' else
                                ['crop' if bounds is not None else 'duplicate_original'])}


def prepare(processor, model, views, messages, device):
    import torch
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[rendered], images=views, return_tensors='pt', images_kwargs={'do_resize': False})
    grids = encoded['image_grid_thw'].tolist()
    patch, merge = int(processor.image_processor.patch_size), int(processor.image_processor.merge_size)
    tokens = [t * h * w // (merge * merge) for t, h, w in grids]
    if len(grids) != len(views) or any(t != 1 or h * patch != IMAGE_SIZE or w * patch != IMAGE_SIZE
                                       for t, h, w in grids):
        raise AssertionError('Official processor changed fixed448 image geometry')
    if any(token != 196 for token in tokens) or int((encoded['input_ids'] == model.config.image_token_id).sum()) != sum(tokens):
        raise AssertionError('Unexpected visual token budget')
    if not bool((encoded['attention_mask'] == 1).all()):
        raise AssertionError('Only one unpadded prompt is supported')
    batch = {k: v.to(device=device, dtype=torch.bfloat16 if k == 'pixel_values' else v.dtype)
             if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    return batch, {'n_images': len(views), 'image_grid_thw': grids,
                   'image_tokens_total': sum(tokens), 'prompt_tokens_total': int(encoded['input_ids'].shape[1])}


def planner(model, processor, raw, cfg, device):
    import torch
    from PIL import Image
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    view = raw.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PLAN_PROMPT}]}]
    batch, token_info = prepare(processor, model, [view], messages, device)
    model.model.rope_deltas = None
    torch.cuda.synchronize(device)
    began = time.perf_counter()
    generated = model.generate(**batch, generation_config=cfg, use_model_defaults=False)
    suffix = generated[0, batch['input_ids'].shape[1]:].detach().cpu().tolist()
    text = processor.tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    request, error = parse_request(text)
    del generated, batch
    return {'raw_text': text, 'token_ids': suffix, 'request': request, 'error': error,
            'seconds': time.perf_counter() - started, 'generate_seconds': elapsed,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device), **token_info}


def run_case(model, processor, image_path, conditions, seed, planner_cfg, final_cfg, device):
    """Inference API accepts no labels, categories, masks, or cache rows."""
    import torch
    from PIL import Image
    from migration_roi_eval import score_and_generate, single_token_id
    load_started = time.perf_counter()
    with Image.open(image_path) as source:
        raw = source.convert('RGB')
    digest = sha256_file(image_path)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    started = time.perf_counter()
    plan = planner(model, processor, raw, planner_cfg, device)
    yes, no = (single_token_id(processor.tokenizer, x) for x in ('Yes', 'No'))
    outputs = []
    for condition in conditions:
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        request, rejected = request_for_condition(condition, plan['request'], plan['error'], image_path, seed)
        views, messages, tool_info = build_final_inputs(raw, condition, request, rejected)
        batch, budget = prepare(processor, model, views, messages, device)
        prediction = score_and_generate(model, processor, batch, yes, no, final_cfg, device)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - began
        output = {'condition': condition, 'image_sha256': digest, 'planner': plan,
                  **tool_info, **budget, **prediction,
                  'final_seconds': seconds, 'image_load_hash_seconds': load_seconds,
                  'standalone_seconds_without_planner': load_seconds + seconds,
                  'standalone_seconds_including_planner': load_seconds + seconds + plan['seconds'],
                  'deployment_seconds': load_seconds + seconds + (plan['seconds'] if condition == 'model_tool' else 0),
                  'planner_invalid': bool(plan['error']), 'request_rejected': bool(rejected),
                  'peak_gpu_allocated_bytes_final': torch.cuda.max_memory_allocated(device),
                  'peak_gpu_reserved_bytes_final': torch.cuda.max_memory_reserved(device),
                  'peak_gpu_allocated_bytes_end_to_end': max(torch.cuda.max_memory_allocated(device), plan['peak_gpu_allocated_bytes']),
                  'score_definition': 'unprocessed final first-answer-token logit(Yes)-logit(No)'}
        outputs.append(output)
        del batch, views
    if sha256_file(image_path) != digest:
        raise AssertionError('Source RGB changed during inference')
    return outputs, time.perf_counter() - started


def self_test():
    accepted = ['{"action":"answer"}', '{"action":"crop","center":[0,1]}',
                '{"action":"crop","center":[0.25,0.75]}']
    rejected = ['{}', '{"action":"crop","center":[true,0]}',
                '{"action":"crop","center":[-0.1,1]}', '{"action":"crop","center":[NaN,0]}',
                '{"action":"crop","center":[0,0],"label":1}',
                chr(96)*3+'json\n{"action":"answer"}\n'+chr(96)*3,
                '{"action":"answer","center":[0,0]}',
                '{"action":"crop","action":"answer"}']
    for value in accepted:
        assert parse_request(value)[1] is None, value
    for value in rejected:
        assert parse_request(value)[1] is not None, value
    for center in ([0, 0], [1, 1], [0.5, 0.5]):
        x1, y1, x2, y2 = crop_bounds(center)
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        assert abs((x2 - x1) * (y2 - y1) - ROI_AREA) < 1e-12
    assert stable_center('x', 42) == stable_center('x', 42)
    from PIL import Image
    raw = Image.new('RGB', (800, 300), color=(20, 40, 60))
    for condition in CONDITIONS:
        request, error = request_for_condition(condition, {'action': 'crop', 'center': [1., 0.]}, None, 'x', 42)
        views, messages, info = build_final_inputs(raw, condition, request, error)
        assert len(views) == (1 if condition == 'no_tool' else 2)
        assert all(image.size == (448, 448) for image in views)
        assert info['tool_calls'] <= 1
    print(json.dumps({'passed': True, 'accepted_schema_cases': len(accepted),
                      'rejected_schema_cases': len(rejected), 'conditions_checked': list(CONDITIONS),
                      'gpu_model_loaded': False, 'dataset_opened': False}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct')
    ap.add_argument('--cache', default='runs/migration_20261103/teacher_cache.pt')
    ap.add_argument('--base-manifest', default='runs/migration_20261103/lora_20261104/base_manifest.json')
    ap.add_argument('--out', type=Path, default=Path('runs/migration_20261103/advanced_20261104/tool_val'))
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--limit', type=int, default=48, help='Fixed validation pilot; 0=all380')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--condition', action='append', choices=CONDITIONS)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--planner-max-new-tokens', type=int, default=64)
    ap.add_argument('--max-new-tokens', type=int, default=32)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.limit < 0 or min(args.max_new_tokens, args.planner_max_new_tokens) <= 0:
        raise ValueError('Invalid size or token budgets')
    from migration_roi_eval import load_shared_cache, select_validation_indices
    from migration_metrics import evaluate_predictions
    cache, cache_audit = load_shared_cache(args.cache)
    items = cache['items']
    indices = select_validation_indices(items, args.limit, args.seed)
    selected = [items[i] for i in indices]
    conditions = list(dict.fromkeys(args.condition or CONDITIONS))
    del cache
    gc.collect()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'run.json').exists() or (args.out / 'predictions.jsonl').exists():
        raise FileExistsError('Preserve existing results; choose a new output directory')
    plan = {'pipeline_version': PIPELINE_VERSION, 'config': {**vars(args), 'out': str(args.out)},
            'conditions': conditions, 'n': len(selected), 'cache_audit': cache_audit,
            'selected_cache_indices': indices, 'selected_paths': [x['path'] for x in selected],
            'selected_manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
            'planner_prompt': PLAN_PROMPT, 'final_prompt': FINAL_PROMPT,
            'inference_inputs': 'RGB only; cache labels/categories used for subset selection and metrics only',
            'native_model_adapter': None, 'language_bridge_adapter_used': False,
            'max_tool_calls': 1, 'image_size': IMAGE_SIZE, 'roi_area': ROI_AREA,
            'validation_only': True, 'external_queries': 0, 'test_queries': 0,
            'source_sha256': sha256_file(__file__), 'status': 'planned', 'gpu_model_loaded': False}
    atomic_json(plan, args.out / 'plan.json')
    if not args.run:
        print(json.dumps({'status': 'planned', 'gpu_model_loaded': False, 'n': len(selected)})); return
    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    from migration_native_parity import audit_dtypes, source_audit
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU evaluation requires BF16 CUDA')
    source = source_audit(args.model)
    manifest = json.loads(Path(args.base_manifest).read_text())
    if Path(manifest['model_path']).resolve() != Path(args.model).resolve():
        raise ValueError('Base manifest model path differs')
    for name, entry in manifest['files'].items():
        file = Path(args.model) / name
        if file.stat().st_size != entry['bytes'] or sha256_file(file) != entry['sha256']:
            raise ValueError('Base model file identity changed: ' + name)
    if not source['passed']:
        raise RuntimeError('Installed native source failed audited SHA/version gate')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats(args.device)
    began = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={'': args.device}, attn_implementation='sdpa', low_cpu_mem_usage=True, local_files_only=True).eval()
    model.requires_grad_(False)
    dtype_audit = audit_dtypes(model)
    if not dtype_audit['passed']:
        raise RuntimeError('Complete native model must be BF16')
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    common = dict(do_sample=False, num_beams=1, use_cache=True,
                  bos_token_id=model.generation_config.bos_token_id,
                  eos_token_id=model.generation_config.eos_token_id,
                  pad_token_id=model.generation_config.pad_token_id)
    planner_cfg = GenerationConfig(max_new_tokens=args.planner_max_new_tokens, **common)
    final_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, **common)
    run = {**plan, 'status': 'running', 'gpu_model_loaded': True, 'native_source_audit': source,
           'dtype_audit': dtype_audit, 'base_manifest_sha256': sha256_file(args.base_manifest),
           'load_seconds': time.perf_counter() - began, 'results': {}}
    atomic_json(run, args.out / 'run.json')
    records = {condition: [] for condition in conditions}
    started = time.perf_counter()
    try:
        with open(args.out / 'predictions.jsonl', 'x') as handle, torch.inference_mode():
            for position, (index, item) in enumerate(zip(indices, selected)):
                outputs, elapsed = run_case(model, processor, str(item['path']), conditions, args.seed,
                                           planner_cfg, final_cfg, args.device)
                for output in outputs:
                    record = {**item, 'cache_index': index, **output}
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    records[output['condition']].append(record)
                handle.flush()
                atomic_json({'done': position + 1, 'n': len(selected), 'elapsed_seconds': time.perf_counter() - started},
                            args.out / 'progress.json')
                if (position + 1) % 8 == 0 or position + 1 == len(selected):
                    print(json.dumps({'done': position + 1, 'n': len(selected), 'elapsed_seconds': time.perf_counter() - started}), flush=True)
        for condition, rows in records.items():
            values = lambda key: [row[key] for row in rows]
            summary = {'metrics': evaluate_predictions(selected, values('score_yes_minus_no'), values('generated_text')),
                       'n': len(rows), 'actual_tool_call_rate': sum(values('tool_calls')) / len(rows),
                       'invalid_planner_rate': sum(values('planner_invalid')) / len(rows),
                       'rejected_request_rate': sum(values('request_rejected')) / len(rows),
                       'mean_final_seconds': sum(values('final_seconds')) / len(rows),
                       'mean_standalone_seconds_without_planner': sum(values('standalone_seconds_without_planner')) / len(rows),
                       'mean_deployment_seconds': sum(values('deployment_seconds')) / len(rows),
                       'mean_standalone_seconds_including_planner': sum(values('standalone_seconds_including_planner')) / len(rows),
                       'peak_gpu_allocated_bytes_final': max(values('peak_gpu_allocated_bytes_final')),
                       'peak_gpu_reserved_bytes_final': max(values('peak_gpu_reserved_bytes_final')),
                       'peak_gpu_allocated_bytes_end_to_end': max(values('peak_gpu_allocated_bytes_end_to_end')),
                       'final_image_tokens_min_max': [min(values('image_tokens_total')), max(values('image_tokens_total'))],
                       'final_prompt_tokens_min_max': [min(values('prompt_tokens_total')), max(values('prompt_tokens_total'))],
                       'planner_reused_across_controls': True}
            atomic_json(summary, args.out / (condition + '_metrics.json'))
            run['results'][condition] = summary
        run.update(status='complete', elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        run.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        atomic_json(run, args.out / 'run.json')
    print(json.dumps({'status': run['status'], 'n': len(selected), 'output': str(args.out)}))


if __name__ == '__main__':
    main()

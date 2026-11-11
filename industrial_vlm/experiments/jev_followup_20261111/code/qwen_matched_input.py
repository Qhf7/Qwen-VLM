#!/usr/bin/env python3
"""Server-only native Qwen base/DD scoring on stage08's exact 448 PNGs.

prepare --root ROOT
preflight --root ROOT
run --root ROOT [--arm base|decision_decision] [--repair-tail]
evaluate --root ROOT
self-test

Formal design: 120 approved development queries x two frozen Qwen arms.
The existing native score._score_image performs all causal/token scoring.
This harness adds provenance, exact image-grid observation and durable resume.
No training, API, GT mask or confirmation access. Run every command on AutoDL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

SCHEMA = 'qwen-stage08-matched-png-v1'
HOSTNAME = 'autodl-container-132444827f-9c594038'
DATA_ROOT = Path('/root/autodl-tmp/industrial_vlm')
PYTHON = '/root/autodl-tmp/vlm/.venv/bin/python'
MODEL = DATA_ROOT / 'models/Qwen3-VL-8B-Instruct'
ADAPTER = DATA_ROOT / 'runs/multitask_v2/decision_decision/adapter'
NATIVE_ROOT = DATA_ROOT / 'src'
ARMS = ('base', 'decision_decision')
SIZE, PIXELS, MAX_LENGTH = 448, 200704, 2048
EXPECTED_GRID = [[1, 28, 28]]
MERGED_TOKENS = 196
PROMPT = 'Inspect this industrial image for defects. Return JSON only: {"is_anomalous": true or false}.'
EXPECTED_VERSIONS = dict(python='3.12.3', torch='2.5.1', transformers='5.14.1',
                         peft='0.21.2', bitsandbytes='0.48.2', accelerate='1.14.0')
EXPECTED_CODE = {
    'score.py': 'b2dd6dc5330804f884560c1cd35c238f253d2b8b30b73a9e14458b9e878928e0',
    'modeling.py': '7f86efc21f98e42eae4c615a22189c33f562ecbf88a507dc3ea8318a9d6be370',
    'data.py': '958d6f88e7c9cd96fead8e3feaad2760fab1edb7b7695407e1e616acadc1f6ea',
    'evaluate.py': '36c92459ff9f0489d58b5261def6915e2201013860c56ea0b2287d326bc0c58c',
    'metrics.py': '4a9c7904499c1e12f915810bafdf283ca1129b67bc85706ce413ab90e7e094cc',
}
LIMITATIONS = [
    'Previously viewed development validation; no blind confirmation or deployment claim.',
    'Exact same stage08 query PNG bytes, one image and no references in both Qwen arms.',
    'Qwen JSON boolean prompt/scoring and Jev typed Noul protocol differ.',
    'Jev internal visual tokens and compute are unknown; matching pixels/images is not matching FLOPs or tokens.',
    'Qwen two-candidate normalized score is not calibrated confidence.',
    'Historical Qwen original-resolution scores and historical Jev renderer remain separate evidence.',
]


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Expected safe stage-relative path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escaped experiment root')
    return path


def publish(path, value):
    """Immutable, fsynced artifact. Identical reruns do not replace it."""
    path = Path(path)
    if path.exists():
        require(canonical(read(path)) == canonical(value), 'Frozen artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.publish.tmp')
    with temporary.open('xb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_status(stage, value):
    temporary = stage / 'status.json.tmp'
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, stage / 'status.json')


@contextmanager
def locked(stage):
    stage.mkdir(parents=True, exist_ok=True)
    with (stage / '.run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def host_gate():
    require(socket.gethostname() == HOSTNAME, 'Execute only on the designated AutoDL instance')
    require(sys.executable == PYTHON, 'Use the native Qwen interpreter: ' + PYTHON)


def versions():
    values = dict(python=platform.python_version())
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'Pillow', 'numpy', 'tokenizers'):
        values[name] = importlib.metadata.version(name)
    require(all(values[k] == v for k, v in EXPECTED_VERSIONS.items()), 'Native Qwen environment version gate failed')
    return values


def native_modules():
    for name, expected in EXPECTED_CODE.items():
        require(sha(NATIVE_ROOT / 'industrial_vlm' / name) == expected, 'Native source changed: ' + name)
    sys.path.insert(0, str(NATIVE_ROOT))
    score = importlib.import_module('industrial_vlm.score')
    modeling = importlib.import_module('industrial_vlm.modeling')
    data = importlib.import_module('industrial_vlm.data')
    for module in (score, modeling, data):
        require(Path(module.__file__).resolve().parent == (NATIVE_ROOT / 'industrial_vlm').resolve(),
                'Wrong native module imported')
    require(data.DECISION_PROMPT == PROMPT and score.SCHEMA_VERSION == 'industrial-vlm-boolean-score-v1',
            'Native boolean question/schema changed')
    return score, modeling


def model_identity(score):
    result = dict(model=str(MODEL), adapter=str(ADAPTER), model_files=score._model_files(MODEL),
                  adapter_files=score._model_files(ADAPTER))
    config = read(MODEL / 'config.json')
    require(config['model_type'] == 'qwen3_vl', 'Expected native Qwen3-VL architecture')
    vision = config['vision_config']
    require((vision['patch_size'], vision['spatial_merge_size'], vision['temporal_patch_size']) == (16, 2, 2),
            'Unexpected Qwen vision geometry')
    adapter = read(ADAPTER / 'adapter_config.json')
    require(adapter['r'] == 16 and adapter['lora_alpha'] == 32 and adapter['lora_dropout'] == .05,
            'DD LoRA hyperparameters differ')
    require(set(adapter['target_modules']) == {'q_proj', 'k_proj', 'v_proj', 'o_proj'} and
            adapter['peft_type'] == 'LORA' and adapter['task_type'] == 'CAUSAL_LM', 'DD LoRA targets differ')
    require(adapter['base_model_name_or_path'] == str(MODEL), 'DD base model identity differs')
    result['lora'] = {k: adapter[k] for k in ('r', 'lora_alpha', 'lora_dropout', 'target_modules', 'peft_type', 'task_type')}
    return result


def verify_asset(source_stage, query):
    from PIL import Image
    metadata = query['asset']
    require(metadata['size'] == [SIZE, SIZE], 'Shared PNG size differs')
    path = inside(source_stage, metadata['path'])
    require(path.parent == source_stage / 'assets', 'Expected stage08 assets directory')
    require(path.stat().st_size == metadata['png_bytes'] and sha(path) == metadata['png_sha256'], 'Shared PNG bytes differ')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (SIZE, SIZE), 'Expected exact RGB448 PNG')
        image.verify()
    return path


def source_queries(root, source_stage, limit):
    path = source_stage / 'contract.json'
    source = read(path)
    require(source['schema'] == 'jev-independent-support-robustness-v1' and source['size'] == SIZE and
            source['model'] == 'jev-1.14.0' and source['mode'] == 'formal_independent_support_robustness' and
            source['limit_queries'] is None, 'Need the formal stage08 query contract')
    queries = source['queries']
    require(len(queries) == 120 and len({q['source_id'] for q in queries}) == 120, 'Expected120 unique source queries')
    require(queries == sorted(queries, key=lambda q: q['source_id']), 'Query order must remain source-sorted')
    require(len(Counter(q['category'] for q in queries)) == 12 and set(Counter(q['category'] for q in queries).values()) == {10},
            'Expected12 categories x10 approved queries')
    frozen_plan = root / 'source_baseline/frozen/plan.json'
    frozen_config = root / 'source_baseline/frozen/config.json'
    for name, path_ in [('query_plan', frozen_plan), ('historical_config', frozen_config)]:
        require(Path(source['source_files'][name]['path']).resolve() == path_.resolve() and
                sha(path_) == source['source_files'][name]['sha256'], 'Frozen source identity changed')
    approved = {q['source_id']: q for q in read(frozen_plan)['queries']}
    for query in queries:
        sid = query['source_id']; previous = approved[sid]
        require(query['original_sha256'] == previous['sha256'] and query['category'] == previous['category'] and
                query['original_bytes'] == previous['source_bytes'], 'Shared query original identity differs')
    if limit is not None:
        require(1 <= limit <= 120, 'Invalid separate preflight size')
        queries = queries[:limit]
    result = []
    for query in queries:
        path_ = verify_asset(source_stage, query)
        result.append(dict(source_id=query['source_id'], category=query['category'],
                           original_sha256=query['original_sha256'], png_sha256=query['asset']['png_sha256'],
                           png_bytes=query['asset']['png_bytes'], image=str(path_)))
    return result, dict(path=str(path), sha256=sha(path), query_plan_sha256=sha(frozen_plan),
                        historical_config_sha256=sha(frozen_config))


def protocol():
    return dict(prompt=PROMPT, source_stage_arm='shared0', arms=list(ARMS), image_size=[SIZE, SIZE],
                min_pixels=PIXELS, max_pixels=PIXELS, max_length=MAX_LENGTH, shots=0, images_per_query=1,
                expected_image_grid_thw=EXPECTED_GRID, expected_merged_visual_tokens=MERGED_TOKENS,
                quantization='Native language-only NF4; double quantization; BF16 compute; visual BF16; SDPA',
                scorer='Unmodified industrial_vlm.score._score_image; both candidates; positions-1 causal shift',
                threshold=dict(field='anomaly_log_odds', comparison='strict >', value=0.),
                no_calibration_fitting=True, no_labels_in_inference=True, confirmation_opened=False,
                latency='Includes PNG preprocessing, native tensor audit/hash and two full forwards; excludes model loading/file hashes; first image after each load is cold',
                image_decisions=240, candidate_forward_passes=480, extra_warmup_forwards=0, limitations=LIMITATIONS)


def prepare(args):
    host_gate(); deps = versions(); score, _ = native_modules()
    root = args.root.resolve(); stage = inside(root, args.stage); source_stage = inside(root, args.source_stage)
    require(stage != source_stage, 'Output must differ from stage08')
    require(args.limit_queries is None or args.stage != 'stage09_qwen_matched', 'Preflight subset requires a separate stage directory')
    with locked(stage):
        queries, source = source_queries(root, source_stage, args.limit_queries)
        value = dict(schema=SCHEMA, mode='formal240' if args.limit_queries is None else 'separate_preflight_not_formal',
                     root=str(root), stage=args.stage, source_stage=args.source_stage, source_stage_contract=source,
                     limit_queries=args.limit_queries, runner_sha256=sha(__file__), native_code_sha256=EXPECTED_CODE,
                     dependencies=deps, protocol=protocol(), models=model_identity(score), queries=queries,
                     schedule=[dict(arm=arm, source_id=q['source_id']) for arm in ARMS for q in queries])
        publish(stage / 'contract.json', value)
        if not (stage / 'status.json').exists():
            write_status(stage, dict(schema=SCHEMA, status='prepared', expected=len(value['schedule']), completed=0, timestamp_utc=now()))
        print(json.dumps(dict(status='prepared', stage=str(stage), expected=len(value['schedule']), contract_sha256=digest(canonical(value)))), flush=True)


def verify_contract(args):
    host_gate(); root = args.root.resolve(); stage = inside(root, args.stage); value = read(stage / 'contract.json')
    require(value['schema'] == SCHEMA and value['root'] == str(root) and value['stage'] == args.stage, 'Stage09 contract identity differs')
    require(value['runner_sha256'] == sha(__file__) and value['native_code_sha256'] == EXPECTED_CODE and
            value['dependencies'] == versions() and value['protocol'] == protocol(), 'Stage09 code/environment/protocol changed')
    source_stage = inside(root, value['source_stage'])
    queries, source = source_queries(root, source_stage, value['limit_queries'])
    require(queries == value['queries'] and source == value['source_stage_contract'], 'Shared source contract or PNGs changed')
    score, modeling = native_modules()
    require(model_identity(score) == value['models'], 'Frozen model/adapter bytes differ')
    expected = [dict(arm=arm, source_id=q['source_id']) for arm in ARMS for q in queries]
    require(value['schedule'] == expected, 'Frozen stage09 schedule changed')
    return stage, value, score, modeling


def fixed_processor(processor):
    ip = processor.image_processor
    require((int(ip.patch_size), int(ip.merge_size), int(ip.temporal_patch_size)) == (16, 2, 2), 'Native processor geometry differs')
    for key in ('min_pixels', 'max_pixels'):
        if hasattr(ip, key):
            setattr(ip, key, PIXELS)
    # Transformers5.14 uses its native mutable SizeDict dataclass, not dict.
    # Set known area fields in place so preprocess retains its native type.
    require('shortest_edge' in ip.size and 'longest_edge' in ip.size, 'Unknown native processor area schema')
    ip.size['shortest_edge'] = PIXELS
    ip.size['longest_edge'] = PIXELS
    processor._industrial_max_pixels = PIXELS
    return dict(size=dict(ip.size), patch_size=int(ip.patch_size), merge_size=int(ip.merge_size),
                temporal_patch_size=int(ip.temporal_patch_size))


def audit_inputs(inputs, processor):
    import torch
    grid = inputs['image_grid_thw'].tolist()
    require(grid == EXPECTED_GRID, 'Actual native image grid differs from448 square')
    merged = sum(t * h * w for t, h, w in grid) // int(processor.image_processor.merge_size) ** 2
    image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    require(image_token is not None and image_token != processor.tokenizer.unk_token_id, 'Missing native image token')
    count = int((inputs['input_ids'] == image_token).sum().item())
    require(merged == MERGED_TOKENS and count == MERGED_TOKENS, 'Native merged/image token count differs')
    pixels = inputs['pixel_values'].detach().cpu().contiguous()
    require(tuple(pixels.shape)[:1] == (784,), 'Unexpected native patch count')
    return dict(image_grid_thw=grid, merged_visual_tokens=merged, image_pad_tokens=count,
                pixel_values_shape=list(pixels.shape), pixel_values_dtype=str(pixels.dtype),
                pixel_values_sha256=digest(pixels.view(torch.uint8).numpy().tobytes()))


@contextmanager
def observe_native_inputs(modeling, processor):
    original = modeling.make_inputs; observations = []
    def observed(*args, **kwargs):
        output = original(*args, **kwargs)
        observations.append(audit_inputs(output, processor))
        return output
    modeling.make_inputs = observed
    try:
        yield observations
    finally:
        modeling.make_inputs = original


def gpu_gate():
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), 'Need one native BF16-capable CUDA GPU')
    return dict(gpu=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda,
                device_capability=list(torch.cuda.get_device_capability(0)), hostname=socket.gethostname())


def loading_audit(model, modeling, arm):
    import torch
    from bitsandbytes.nn import Linear4bit
    require(modeling.input_device(model).type == 'cuda', 'CPU/offloaded inference is forbidden')
    require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Model must be frozen eval')
    quantized = [n for n, m in model.named_modules() if isinstance(m, Linear4bit)]
    require(len(quantized) == 252 and all(modeling._is_language(n) for n in quantized), 'Expected252 language-only NF4 linears')
    for n, p in model.named_parameters():
        if modeling._is_visual(n) and p.is_floating_point():
            require(p.dtype == torch.bfloat16, 'Visual parameter lost BF16')
    qc = model.config.quantization_config
    qc = qc if isinstance(qc, dict) else qc.to_dict()
    require(qc['load_in_4bit'] and qc['bnb_4bit_quant_type'] == 'nf4' and qc['bnb_4bit_use_double_quant'] and
            str(qc['bnb_4bit_compute_dtype']) in ('bfloat16', 'torch.bfloat16'), 'NF4 double-quant BF16 gate failed')
    require(model.config._attn_implementation == 'sdpa', 'Expected native SDPA implementation')
    lora = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and len(module.lora_A):
            require(modeling.LANGUAGE_ATTENTION.search(name) is not None and not modeling._is_visual(name), 'LoRA outside language attention')
            require(set(module.lora_A) == {'default'} and module.r['default'] == 16 and module.lora_alpha['default'] == 32 and
                    module.lora_dropout['default'].p == .05, 'Loaded DD LoRA differs')
            lora.append(name)
    require(len(lora) == (144 if arm == 'decision_decision' else 0), 'Loaded adapter module count differs')
    return dict(quantized_modules=quantized, loaded_lora_modules=lora, visual_dtype='torch.bfloat16',
                quantization=qc, attention_implementation=model.config._attn_implementation, all_parameters_frozen=True)


def preflight(args):
    stage, contract, _, modeling = verify_contract(args)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, min_pixels=PIXELS, max_pixels=PIXELS)
    fixed = fixed_processor(processor); checks = []
    positions = sorted({0, len(contract['queries']) // 2, len(contract['queries']) - 1})
    for i in positions:
        q = contract['queries'][i]
        inputs = modeling.make_inputs(processor, q['image'], PROMPT, max_length=MAX_LENGTH)
        checks.append(dict(source_id=q['source_id'], **audit_inputs(inputs, processor)))
    value = dict(schema=SCHEMA, contract_sha256=digest(canonical(contract)), status='passed',
                 dependencies=versions(), hardware=gpu_gate(), processor=fixed, checks=checks, model_forward_calls=0)
    with locked(stage):
        publish(stage / 'preflight.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def validate_record(row, contract, score):
    queries = {q['source_id']: q for q in contract['queries']}
    arm, sid = row.get('arm'), row.get('source_id')
    require(arm in ARMS and sid in queries, 'Unknown score target')
    q = queries[sid]
    require(not {'label', 'is_anomalous', 'gt_boxes', 'boxes', 'answer'}.intersection(row), 'Inference score contains ground truth fields')
    require(row['schema_version'] == score.SCHEMA_VERSION and row['harness_schema'] == SCHEMA and
            row['contract_sha256'] == digest(canonical(contract)) and row['category'] == q['category'] and
            row['image_sha256'] == q['png_sha256'] and row['original_sha256'] == q['original_sha256'] and
            row['image'] == q['image'] and row['prompt'] == PROMPT and row['id'] == sid + '__' + arm, 'Score provenance changed')
    expected = score.normalized_score(row['log_likelihood_true'], row['log_likelihood_false'])
    require(all(row.get(k) == v for k, v in expected.items()), 'Score normalization differs from raw likelihoods')
    require(row['image_grid_thw'] == EXPECTED_GRID and row['merged_visual_tokens'] == MERGED_TOKENS and
            row['input_audit']['image_grid_thw'] == EXPECTED_GRID and
            row['input_audit']['merged_visual_tokens'] == MERGED_TOKENS and
            row['input_audit']['image_pad_tokens'] == MERGED_TOKENS and row['native_input_calls'] == 3 and
            isinstance(row['input_audit']['pixel_values_sha256'], str) and
            len(row['input_audit']['pixel_values_sha256']) == 64, 'Saved native image audit differs')
    require(row['token_boundary_audit'] == 'native full token IDs == prompt IDs + offset-verified continuation IDs' and
            row['n_value_tokens'] == {'true': 1, 'false': 1} and
            row['value_token_ids'] == {'true': [1866], 'false': [3849]}, 'Saved boolean token audit differs')
    require(all(type(row[k]) in (float, int) and math.isfinite(row[k]) and row[k] >= 0
                for k in ('latency_seconds', 'peak_memory_allocated_bytes')), 'Invalid saved resource observations')
    return arm, sid


def read_scores(stage, contract, score, repair_tail=False):
    path = stage / 'scores.jsonl'
    if not path.exists():
        return []
    rows = []; seen = set()
    with path.open('r+b' if repair_tail else 'rb') as stream:
        while True:
            offset = stream.tell(); line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                require(not stream.read().strip(), 'Corrupt score before final line')
                require(repair_tail and not (stage / 'prediction_freeze.json').exists(), 'Incomplete final score; rerun with --repair-tail before freeze')
                stream.truncate(offset); stream.flush(); os.fsync(stream.fileno()); break
            pair = validate_record(row, contract, score)
            require(pair not in seen, 'Duplicate score target')
            seen.add(pair); rows.append(row)
            if not line.endswith(b'\n'):
                require(repair_tail and not (stage / 'prediction_freeze.json').exists(), 'Missing final newline; use --repair-tail before freeze')
                stream.seek(0, os.SEEK_END); stream.write(b'\n'); stream.flush(); os.fsync(stream.fileno())
    return rows


def run(args):
    stage, contract, score, modeling = verify_contract(args)
    import torch
    with locked(stage):
        rows = read_scores(stage, contract, score, args.repair_tail)
        completed = {(r['arm'], r['source_id']) for r in rows}
        hardware = gpu_gate(); queries = {q['source_id']: q for q in contract['queries']}
        def status(state, **extra):
            value = dict(schema=SCHEMA, status=state, expected=len(contract['schedule']), completed=len(completed),
                         per_arm=dict(Counter(a for a, _ in completed)), contract_sha256=digest(canonical(contract)),
                         timestamp_utc=now(), confirmation_opened=False, **extra)
            write_status(stage, value)
            return value
        if (stage / 'prediction_freeze.json').exists():
            freeze = read(stage / 'prediction_freeze.json')
            require(freeze['contract_sha256'] == digest(canonical(contract)) and
                    freeze['scores_sha256'] == sha(stage / 'scores.jsonl') and
                    len(completed) == len(contract['schedule']), 'Frozen predictions changed')
            print(json.dumps(status('complete')), flush=True); return
        try:
            for arm in ARMS:
                if args.arm is not None and args.arm != arm:
                    continue
                pending = [q for q in contract['queries'] if (arm, q['source_id']) not in completed]
                if not pending:
                    continue
                status('loading', active_arm=arm)
                model, processor = modeling.load_model(str(MODEL), adapter=str(ADAPTER) if arm == 'decision_decision' else None,
                                                       training=False, max_pixels=PIXELS, local_files_only=True)
                fixed = fixed_processor(processor)
                audit = dict(schema=SCHEMA, contract_sha256=digest(canonical(contract)), arm=arm, hardware=hardware,
                             processor=fixed, model_loading_audit=loading_audit(model, modeling, arm))
                publish(stage / (arm + '_loading_audit.json'), audit)
                device = modeling.input_device(model); status('running', active_arm=arm)
                with (stage / 'scores.jsonl').open('a', encoding='utf-8') as output:
                    for pending_index, q in enumerate(pending):
                        require(sha(q['image']) == q['png_sha256'], 'Shared PNG changed before inference')
                        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                        started = time.perf_counter()
                        with observe_native_inputs(modeling, processor) as observations:
                            value = score._score_image(model, processor, Path(q['image']), PROMPT, MAX_LENGTH)
                        torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - started
                        require(len(observations) == 3 and all(a == observations[0] for a in observations), 'Native pixel/grid inputs differ across candidates')
                        row = dict(schema_version=score.SCHEMA_VERSION, harness_schema=SCHEMA,
                                   contract_sha256=digest(canonical(contract)), arm=arm, id=q['source_id'] + '__' + arm,
                                   source_id=q['source_id'], category=q['category'], image=q['image'], image_sha256=q['png_sha256'],
                                   original_sha256=q['original_sha256'], prompt=PROMPT, **value, image_grid_thw=EXPECTED_GRID,
                                   merged_visual_tokens=MERGED_TOKENS, input_audit=observations[0], native_input_calls=len(observations),
                                   first_image_after_model_load=pending_index == 0, latency_seconds=elapsed,
                                   peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device), timestamp_utc=now())
                        validate_record(row, contract, score)
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        output.flush(); os.fsync(output.fileno())
                        rows.append(row); completed.add((arm, q['source_id']))
                        print(json.dumps(dict(event='scored', arm=arm, completed=len(completed), expected=len(contract['schedule']),
                                              seconds=elapsed, merged_visual_tokens=MERGED_TOKENS)), flush=True)
                        status('running', active_arm=arm)
                del model, processor; gc.collect(); torch.cuda.empty_cache()
            if len(completed) == len(contract['schedule']):
                freeze = dict(schema=SCHEMA, contract_sha256=digest(canonical(contract)),
                              scores_sha256=sha(stage / 'scores.jsonl'), accepted=len(rows),
                              labels_opened_before_prediction_freeze=False, confirmation_opened=False)
                publish(stage / 'prediction_freeze.json', freeze)
                status('complete')
            else:
                status('partial')
        except Exception as error:
            status('failed', error_type=type(error).__name__, error=str(error))
            raise


def evaluate(args):
    stage, contract, score, _ = verify_contract(args)
    with locked(stage):
        require((stage / 'prediction_freeze.json').exists(), 'Finish and freeze all predictions before opening labels')
        freeze = read(stage / 'prediction_freeze.json')
        require(freeze['contract_sha256'] == digest(canonical(contract)) and freeze['scores_sha256'] == sha(stage / 'scores.jsonl'), 'Prediction freeze changed')
        rows = read_scores(stage, contract, score)
        require(len(rows) == len(contract['schedule']) == freeze['accepted'], 'Incomplete prediction freeze')
        # Only approved120 boolean labels are read, after immutable prediction freeze.
        root = args.root.resolve(); config = read(root / 'source_baseline/frozen/config.json')
        labels_path = root / 'source_baseline/frozen/labels.json'; labels = read(labels_path)
        require(digest(canonical(labels)) == config['labels_sha256'], 'Frozen label identity differs')
        require(all(type(labels[q['source_id']]) in (bool, int) and labels[q['source_id']] in (0, 1) for q in contract['queries']), 'Invalid frozen boolean labels')
        by_arm = {}
        for arm in ARMS:
            records = [dict(r, label=bool(labels[r['source_id']])) for r in rows if r['arm'] == arm]
            categories = defaultdict(list)
            for row in records:
                categories[row['category']].append(row)
            by_arm[arm] = dict(overall=score.ranking_metrics(records),
                               default_equal_candidate_threshold=score.threshold_metrics(records, 0.),
                               by_category={c:score.ranking_metrics(v) for c,v in sorted(categories.items())},
                               mean_latency_seconds=sum(r['latency_seconds'] for r in records)/len(records),
                               peak_memory_allocated_bytes=max(r['peak_memory_allocated_bytes'] for r in records))
        result = dict(schema=SCHEMA, status='complete', mode=contract['mode'], prediction_freeze=freeze,
                      labels_file_sha256=sha(labels_path), arms=by_arm, limitations=LIMITATIONS,
                      matched_jev_contract=contract['source_stage_contract'], image_size=[SIZE,SIZE], merged_visual_tokens=MERGED_TOKENS,
                      fitted_thresholds=False, confirmation_opened=False)
        publish(stage / 'metrics.json', result)
        lines = ['# Qwen共同448输入比较', '',
                 'Qwen量化底座与DD LoRA采用stage08新Jev shared0逐字节相同的120张PNG，每次只含一张query、无参考图。',
                 '复用原生score._score_image：只对JSON布尔值token计分，保留共同前后缀核验与causal shift。', '',
                 '| 模型 | AUROC | AP | Recall(logodds>0) | FPR(logodds>0) |',
                 '|---|---:|---:|---:|---:|']
        for arm in ARMS:
            m=by_arm[arm]; t=m['default_equal_candidate_threshold']
            lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in (m['overall']['auroc'],m['overall']['average_precision'],t['recall'],t['fpr']))+' |')
        lines += ['', '实际native image_grid_thw=[1,28,28]，merged视觉token=196/图；240图像判断对应480次候选forward，无额外GPU预热。', '',
                  '## 解释边界', ''] + ['- '+x for x in LIMITATIONS] + ['']
        text='\n'.join(lines); path=stage/'REPORT.md'
        if path.exists():
            require(path.read_text(encoding='utf-8') == text, 'Frozen report differs')
        else:
            with path.open('x',encoding='utf-8') as stream:
                stream.write(text); stream.flush(); os.fsync(stream.fileno())
        require(sha(stage/'scores.jsonl') == freeze['scores_sha256'], 'Predictions changed during evaluation')
        print(json.dumps(dict(status='complete', accepted=len(rows), metrics=str(stage/'metrics.json'), arms=by_arm), ensure_ascii=False), flush=True)


def self_test(args):
    host_gate(); versions(); score, _ = native_modules()
    require(PIXELS == SIZE*SIZE and 28*28//4 == MERGED_TOKENS, 'Geometry constant mismatch')
    for a,b in [(-1.,-3.),(-10001.,-1.),(-1.,-10001.),(-5.,-5.)]:
        result=score.normalized_score(a,b)
        require(result['anomaly_log_odds'] == a-b and 0 <= result['anomaly_score'] <= 1, 'Native score sanity failed')
    require(canonical({'b':2,'a':1}) == canonical({'a':1,'b':2}), 'Canonical hashing unstable')
    require(inside(Path('/tmp'), 'matched_fixture').parent == Path('/tmp').resolve(), 'Stage path sanity failed')
    for unsafe in ('../escape','/tmp/escape'):
        try:
            inside(Path('/tmp'),unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe path accepted')
    print(json.dumps(dict(status='passed', schema=SCHEMA, no_model_load=True, no_model_forward=True)), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode', required=True)
    for name in ('prepare','preflight','run','evaluate'):
        p=modes.add_parser(name); p.add_argument('--root',type=Path,required=True)
        p.add_argument('--stage',default='stage09_qwen_matched')
        if name=='prepare':
            p.add_argument('--source-stage',default='stage08_support');p.add_argument('--limit-queries',type=int)
        if name=='run':
            p.add_argument('--arm',choices=ARMS);p.add_argument('--repair-tail',action='store_true')
    modes.add_parser('self-test')
    args=parser.parse_args()
    return {'prepare':prepare,'preflight':preflight,'run':run,'evaluate':evaluate,'self-test':self_test}[args.mode](args)


if __name__ == '__main__':
    main()

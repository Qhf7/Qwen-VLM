"""Read-only diagnostics for fully committed, frozen external structured runs.

No model, RGB, mask, tensor cache, or server is accessed. Completion metadata
and declared hashes are checked before prediction bytes are opened; those bytes
are then hashed before parsing. Incomplete outputs are skipped without opening
their prediction file. Diagnostics do not alter the official parser or metric.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'code'))
from migration_metrics import parse_generated
from structured_geometry import LOCATIONS, SCHEMA, location_for_box, parse_structured
from summarize_external import (EXPECTED_MANIFEST, EXPECTED_N, exact_metrics,
                                load_result, run_paths, validate_saved_policy)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_jsonl(path, expected_sha):
    """Hash the same immutable bytes that are subsequently parsed."""
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha,
            'Prediction bytes differ from committed SHA')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


class DuplicateKey(ValueError):
    pass


def unique_object(pairs):
    if len({key for key, _ in pairs}) != len(pairs):
        raise DuplicateKey('Duplicate JSON key')
    return dict(pairs)


def diagnose_text(text):
    """Mutually exclusive first failure in the frozen parser's check order."""
    payload = None
    first_line = None
    reason = 'valid'
    found = re.fullmatch(r'\s*(Yes|No)\s*\n\s*(\{.*\})\s*', text, flags=re.DOTALL)
    if found is None:
        reason = 'outer_yesno_newline_json_contract'
    else:
        first_line = int(found.group(1) == 'Yes')
        try:
            payload = json.loads(found.group(2), object_pairs_hook=unique_object)
        except DuplicateKey:
            reason = 'duplicate_json_keys'
        except (ValueError, TypeError):
            reason = 'json_syntax'
        if reason == 'valid':
            if not isinstance(payload, dict) or set(payload) != {'bbox', 'location'}:
                reason = 'json_schema_keys'
            else:
                box, location = payload['bbox'], payload['location']
                if first_line == 0:
                    if box is not None or location != 'none':
                        reason = 'no_with_box_or_non_none_location'
                elif box is None:
                    reason = 'yes_with_null_bbox'
                elif not isinstance(box, list) or len(box) != 4:
                    reason = 'bbox_shape'
                elif any(type(value) is not int for value in box):
                    reason = 'bbox_coordinate_type'
                elif any(value < 0 or value > 1000 for value in box):
                    reason = 'bbox_out_of_range'
                elif box[0] >= box[2] or box[1] >= box[3]:
                    reason = 'bbox_degenerate_or_unordered'
                elif location not in LOCATIONS:
                    reason = 'location_enum_or_none_for_yes'
                elif location != location_for_box(box):
                    reason = 'location_bbox_center_inconsistency'
    strict = parse_structured(text)
    require((reason == 'valid') == (strict is not None),
            'Diagnostic validity differs from frozen strict parser')
    box = payload.get('bbox') if isinstance(payload, dict) else None
    location = payload.get('location') if isinstance(payload, dict) else None
    geometry = (isinstance(box, list) and len(box) == 4
                and all(type(value) is int for value in box)
                and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
    return dict(reason=reason, strict_valid=strict is not None,
                decision=parse_generated(text), outer_decision=first_line,
                payload=payload, bbox=box, location=location,
                valid_bbox_geometry=geometry, strict=strict)


def describe(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, median=None, p25=None, p75=None, min=None, max=None)
    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p25=quantile(.25), p75=quantile(.75), min=values[0], max=values[-1])


def count(values):
    return dict(sorted(Counter(values).items()))


def location_label(value):
    if value in LOCATIONS or value == 'none':
        return value
    return 'invalid:' + json.dumps(value, ensure_ascii=False, sort_keys=True)


def summarize_records(records, tokenizer=None):
    rows = [diagnose_text(record['generated']) for record in records]
    def group(indices):
        selected = [rows[index] for index in indices]
        labels = [records[index]['label'] for index in indices]
        geometry = [row['bbox'] for row in selected if row['valid_bbox_geometry']]
        yes = [row for row in selected if row['decision'] == 1]
        return dict(n=len(indices), n_normal=labels.count(0), n_anomaly=labels.count(1),
                    n_strict_valid=sum(row['strict_valid'] for row in selected),
                    format_valid_rate=sum(row['strict_valid'] for row in selected) / len(indices) if indices else None,
                    strict_first_failure=count(row['reason'] for row in selected),
                    yesno=count('invalid' if row['decision'] is None else 'Yes' if row['decision'] else 'No'
                                for row in selected),
                    n_json_object=sum(isinstance(row['payload'], dict) for row in selected),
                    bbox_kind=count('unparsed' if not isinstance(row['payload'], dict) or 'bbox' not in row['payload']
                                    else 'null' if row['bbox'] is None else 'valid_geometry' if row['valid_bbox_geometry']
                                    else 'invalid_geometry' for row in selected),
                    n_valid_bbox_geometry=len(geometry),
                    n_strict_valid_yes=sum(row['strict_valid'] and row['decision'] == 1 for row in selected),
                    n_yes_without_valid_bbox_geometry=sum(not row['valid_bbox_geometry'] for row in yes),
                    n_yes_location_none=sum(row['location'] == 'none' for row in yes),
                    n_no_with_non_null_bbox=sum(row['decision'] == 0 and row['bbox'] is not None for row in selected),
                    locations=count(location_label(row['location']) for row in selected),
                    bbox_center_cells=count(location_for_box(box) for box in geometry),
                    bbox_coordinates={name: describe(box[axis] for box in geometry)
                                      for axis, name in enumerate(('x1', 'y1', 'x2', 'y2'))},
                    bbox_area_fraction=describe((box[2]-box[0])*(box[3]-box[1])/1_000_000 for box in geometry),
                    bbox_center_x=describe((box[0]+box[2])/2000 for box in geometry),
                    bbox_center_y=describe((box[1]+box[3])/2000 for box in geometry))
    result = group(list(range(len(records))))
    result['by_true_label'] = {str(label): group([index for index, record in enumerate(records) if record['label'] == label])
                               for label in (0, 1)}
    result['per_category'] = {category: group([index for index, record in enumerate(records)
                                             if record['category'] == category])
                              for category in sorted({record['category'] for record in records})}
    texts = [record['generated'] for record in records]
    result['generation_lengths'] = dict(
        characters=describe(len(text) for text in texts),
        lines=describe(len(text.splitlines()) for text in texts),
        number_lexemes=describe(len(re.findall(r'[-+]?\d+(?:\.\d+)?', text)) for text in texts),
        reencoded_token_count=describe(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
                                 if tokenizer is not None else None,
        token_count_status='local_tokenizer_reencoding_of_saved_decoded_text_excludes_EOS' if tokenizer is not None
                           else 'unavailable_no_local_tokenizer_requested',
        note='Decoded saved text has no EOS. Reencoding does not establish the original generation token sequence or stop reason.')
    return result


def validate_manifest(path):
    require(sha256(path) == EXPECTED_MANIFEST, 'External manifest is not the frozen residual1575')
    manifest = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(len(manifest) == EXPECTED_N and len({row['source_id'] for row in manifest}) == EXPECTED_N,
            'Frozen manifest size/identity differs')
    require(sum(row['label'] == 0 for row in manifest) == 392
            and sum(row['label'] == 1 for row in manifest) == 1183
            and len({row['category'] for row in manifest}) == 15,
            'Frozen manifest class/category counts differ')
    return manifest


def completion_gate(advanced, historical, model, seed, manifest_sha):
    """No predictions or posthoc target files are read in this function."""
    run, output = run_paths(advanced, historical, model, seed)
    status = dict(model=model, seed=seed, output=str(output))
    required = ('summary.json', 'inference_complete.json', 'frozen_config.json', 'predictions.sha256')
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        return None, dict(**status, state='skipped_incomplete', reason='Missing completion metadata: ' + ', '.join(missing))
    summary, inference, frozen = (read(output / name) for name in required[:3])
    if summary.get('state') != 'complete' or inference.get('state') != 'inference_complete':
        return None, dict(**status, state='skipped_incomplete', reason='Completion states are not final')
    require(summary.get('seed') == seed and summary.get('n') == EXPECTED_N, 'Summary seed/count differs')
    require(summary.get('manifest_sha256') == manifest_sha == EXPECTED_MANIFEST, 'Summary manifest differs')
    require(summary.get('condition') == ('initial_C_structured_prompt' if model == 'Cstructured' else 'S_best'),
            'Summary structured condition differs')
    require(summary.get('schema') == SCHEMA and summary.get('max_new_tokens') == 80,
            'Summary schema/generation budget differs')
    require(summary.get('external_queries') == EXPECTED_N and summary.get('historical_visa_test_queries') == 0,
            'Summary query denominator/history differs')
    require(summary.get('mask_reads_during_inference') == 0 and summary.get('posthoc_mask_reads') == 1183,
            'Summary mask-read contract differs')
    require(inference.get('processed') == EXPECTED_N and inference.get('mask_reads') == 0,
            'Inference-before-GT completion marker differs')
    checksum = (output / 'predictions.sha256').read_text().split()
    digest = checksum[0] if checksum else ''
    require(re.fullmatch(r'[0-9a-f]{64}', digest) is not None
            and summary.get('predictions_sha256') == inference.get('predictions_sha256') == digest,
            'Declared prediction hashes differ')
    require(sha256(output / 'frozen_config.json') == inference.get('frozen_config_sha256'),
            'Frozen config SHA differs from inference marker')
    require(all(key in summary and summary[key] == value for key, value in frozen.items()),
            'Summary differs from frozen config')
    freeze = read(advanced / 'external_selection_freeze.json')
    selected = freeze['models'][model + '_s%d' % seed]
    require(sha256(run / 'run.json') == selected['run_sha256']
            and sha256(run / 'complete.json') == selected['complete_sha256'],
            'Source training metadata differs from pre-external freeze')
    require(summary['structured_run_sha256'] == selected['run_sha256']
            and summary['base_manifest_sha256'] == selected['base_manifest_sha256'],
            'Summary source identity differs from pre-external freeze')
    for filename in ('structured_model.py', 'structured_geometry.py', 'structured_external.py', 'advanced_generation.py'):
        require(sha256(advanced / filename) == freeze['source_sha256'][filename], 'Frozen source changed: ' + filename)
    require(summary.get('model_source_sha256') == freeze['source_sha256']['structured_model.py'],
            'Saved model source differs')
    config = read(run / 'run.json')['config']
    validate_saved_policy(advanced, run, output, model, seed, config, summary)
    return dict(run=run, output=output, summary=summary, prediction_sha256=digest), None


def load_completed(advanced, historical, model, seed, manifest):
    gate, skipped = completion_gate(advanced, historical, model, seed, EXPECTED_MANIFEST)
    if skipped is not None:
        return None, skipped
    rows = checked_jsonl(gate['output'] / 'predictions.jsonl', gate['prediction_sha256'])
    require(len(rows) == EXPECTED_N and len({row['source_id'] for row in rows}) == EXPECTED_N,
            'Prediction count/unique identities differ')
    for row, expected in zip(rows, manifest):
        require(all(row[key] == expected[key] for key in ('source_id', 'path', 'category', 'label'))
                and row['split'] == 'external', 'Prediction order/metadata differs from frozen manifest')
        require(type(row['score']) in (int, float) and math.isfinite(row['score'])
                and isinstance(row['generated'], str), 'Invalid score/text')
    # The existing full audit recomputes all official metrics and saved targets.
    # This second read occurs only after the stricter pre-read completion gate.
    audit, skipped = load_result(advanced, historical, model, seed, manifest, EXPECTED_MANIFEST)
    require(skipped is None and audit is not None, 'Completed result disappeared during audit')
    require(audit['predictions_sha256'] == gate['prediction_sha256'], 'Prediction changed during audit')
    gate.update(records=rows, audit=audit)
    return gate, None


def visa_comparison(advanced, model, seed, tokenizer):
    run = advanced / ('arm_S_s%d' % seed)
    complete = read(run / 'complete.json')
    require(complete.get('state') == 'complete' and complete.get('steps') == 500
            and complete.get('preflight') is False, 'VisA source training is incomplete')
    mode = 'initial' if model == 'Cstructured' else 'best'
    if mode == 'best':
        fresh = read(run / 'fresh_best_reload.json')
        require(fresh.get('state') == 'complete' and fresh.get('generated_exact_match') is True
                and fresh.get('fresh_reload_max_abs_score_error') == 0, 'Fresh VisA reconstruction gate failed')
    name = 'initial_val_predictions.json' if mode == 'initial' else 'best_val_generated_predictions.json'
    path = run / name
    records = read(path)
    require(len(records) == 380 and sum(row['label'] == 1 for row in records) == 208
            and len({row['category'] for row in records}) == 12
            and len({row['path'] for row in records}) == 380,
            'VisA comparison denominator/category/identity differs')
    from summarize_structured import recompute, recursive_compare
    metrics = recompute(records)
    recursive_compare(read(run / (mode + '_val_metrics.json')), metrics)
    return dict(domain='VisA_existing_development_validation', mode=mode,
                predictions_sha256=sha256(path), n=380, n_normal=172, n_anomaly=208,
                metrics=metrics, diagnostics=summarize_records(records, tokenizer),
                note='Different images, categories and class proportions from external MVTec; this is a descriptive domain comparison.')


def prompt_diagnostics(advanced, tokenizer):
    """Recover the frozen prompt expression without importing a model module."""
    model_path = advanced / 'structured_model.py'
    language_path = advanced.parent / 'code/migration_language.py'
    module = ast.parse(language_path.read_text())
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('FEATURE_MARKER', 'DESCRIPTIONS'):
                constants[key] = ast.literal_eval(node.value)
    tree = ast.parse(model_path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_template_ids')
    assignment = next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'user' for target in node.targets))
    expression = compile(ast.Expression(assignment.value), str(model_path), 'eval')
    prompts = {}
    for category in constants['DESCRIPTIONS']:
        user = eval(expression, {'__builtins__': {}}, dict(constants, LOCATIONS=LOCATIONS, category=category))
        wrapper = None
        if tokenizer is not None:
            rendered = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': user}],
                tokenize=False, add_generation_prompt=True)
            before, after = rendered.split(constants['FEATURE_MARKER'])
            wrapper = dict(before_feature_slot=len(tokenizer.encode(before, add_special_tokens=False)),
                           after_feature_slot=len(tokenizer.encode(after, add_special_tokens=False)),
                           note='Counts text positions around the visual embedding slot; excludes inserted visual embeddings.')
        prompts[category] = dict(description=constants['DESCRIPTIONS'][category], user_text=user,
                                 user_characters=len(user), chat_wrapper_token_counts=wrapper)
    return dict(source=str(model_path), source_sha256=sha256(model_path),
                method_line=method.lineno, prompt_assignment_line=assignment.lineno,
                language_source_sha256=sha256(language_path),
                external_category='industrial_component',
                difference='External uses an industrial component; VisA uses its category description. Schema instructions are the same.',
                schema=SCHEMA, max_new_tokens=80, prompts=prompts)


def gate_token_diagnostics(run, model):
    """Saved original token traces cover only the fixed VisA engineering gate."""
    name = 'initial_generation_policy.json' if model == 'Cstructured' else 'best_generation_policy.json'
    policy = read(run / name)
    selected = policy['selected']
    key = 'batched' if selected == 'batched' else 'scalar'
    tokens, stops = policy[key + '_token_ids'], policy[key + '_stop_ids']
    require(len(tokens) == len(stops) == 8, 'Saved gate token denominator differs')
    return dict(scope='Fixed VisA validation8 engineering gate only; not external generated-token distribution',
                n=8, selected=selected, original_token_count_excluding_stop=describe(map(len, tokens)),
                stop_ids=count('None' if value is None else str(value) for value in stops),
                n_without_stop=sum(value is None for value in stops),
                source_sha256=sha256(run / name))


def markdown(report):
    lines = ['# External structured output diagnostics', '',
             'Read-only posthoc diagnostics; frozen prompts, parser, checkpoints and thresholds are unchanged.', '',
             '| Model | Seed | n (normal/anomaly) | Strict valid | Yes / No / invalid | Main strict failure |',
             '|---|---:|---:|---:|---:|---|']
    for result in report['results']:
        diag = result['diagnostics']; reasons = {k:v for k,v in diag['strict_first_failure'].items() if k != 'valid'}
        main = max(reasons, key=reasons.get) if reasons else 'none'
        yn = diag['yesno']
        lines.append('| %s | %d | %d (%d/%d) | %d/%d | %d / %d / %d | %s (%d) |' % (
            result['model'], result['seed'], diag['n'], diag['n_normal'], diag['n_anomaly'],
            diag['n_strict_valid'], diag['n'], yn.get('Yes',0), yn.get('No',0), yn.get('invalid',0),
            main, reasons.get(main,0)))
    lines += ['', 'Strict failure counts are mutually exclusive first failures; excluding the `valid` entry, they sum to all invalid outputs. '
              'The JSON includes each category, true-label denominator, bbox geometry, location and decoded-text length distributions.', '']
    for result in report['results']:
        if 'visa' in result:
            external = result['diagnostics']; visa = result['visa']['diagnostics']
            lines.append('- %s seed %d: external format %.4f (%d/%d); VisA format %.4f (%d/%d). Different class/category composition prevents treating this as a paired image comparison.' % (
                result['model'], result['seed'], external['format_valid_rate'], external['n_strict_valid'], external['n'],
                visa['format_valid_rate'], visa['n_strict_valid'], visa['n']))
    if report['skipped']:
        lines += ['', 'Skipped incomplete outputs (raw predictions were not opened):']
        lines += ['- %s seed %d: %s' % (row['model'], row['seed'], row['reason']) for row in report['skipped']]
    if report['errors']:
        lines += ['', 'Failed integrity checks:']
        lines += ['- %s seed %d: %s' % (row['model'], row['seed'], row['error']) for row in report['errors']]
    lines += ['', 'Coverage: %d/%d requested outputs passed; %d/6 of the full structured suite were included in this invocation.' % (
                  len(report['results']), report['coverage']['requested_outputs'], len(report['results'])),
              'Prompt source: `structured_model.py::_template_ids`; external uses the generic noun `an industrial component`, '
              'while VisA uses category descriptions. Both retain the same schema instructions and 80-token budget.',
              'Token counts are available only when a local tokenizer is explicitly supplied; saved decoded text excludes EOS, '
              'so reencoded length is a diagnostic and does not establish the original generation token trace.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--advanced', type=Path, default=HERE)
    parser.add_argument('--historical', type=Path, default=HERE.parent / 'lora_20261104')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--models', nargs='+', choices=('Cstructured', 'S'), default=['Cstructured', 'S'])
    parser.add_argument('--seeds', type=int, nargs='+', choices=(42,43,44), default=[42,43,44])
    parser.add_argument('--include-visa', action='store_true')
    parser.add_argument('--tokenizer', type=Path, help='Optional local tokenizer directory; no network/model weights are loaded')
    parser.add_argument('--out', type=Path, required=True, help='New diagnostic JSON path outside frozen run directories')
    args = parser.parse_args()
    args.advanced = args.advanced.resolve(); args.historical = args.historical.resolve(); args.out = args.out.resolve()
    manifest_path = args.manifest or args.advanced / 'external_mvtec/manifest.jsonl'
    require(not any(re.fullmatch(r'arm_S_s\d+', part) for part in args.out.parts),
            'Diagnostic output must be outside frozen run directories')
    require(not args.out.exists() and not args.out.with_suffix('.md').exists(), 'Diagnostic output already exists')
    manifest = validate_manifest(manifest_path)
    tokenizer = None
    if args.tokenizer:
        require(args.tokenizer.is_dir(), 'Tokenizer must be an existing local directory')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    report = dict(state='complete', scope='Frozen external MVTec residual1575 posthoc structured diagnostics',
                  manifest_sha256=EXPECTED_MANIFEST, diagnostic_source_sha256=sha256(__file__),
                  official_parser_source_sha256=sha256(args.advanced / 'structured_geometry.py'),
                  tokenizer_directory=str(args.tokenizer.resolve()) if args.tokenizer else None,
                  prompts=prompt_diagnostics(args.advanced, tokenizer), results=[], skipped=[], errors=[])
    for model in args.models:
        for seed in args.seeds:
            try:
                loaded, skipped = load_completed(args.advanced, args.historical, model, seed, manifest)
                if skipped is not None:
                    report['skipped'].append(skipped); continue
                output = loaded['output']
                result = dict(model=model, seed=seed, state='complete_and_recomputed',
                              input_sha256={name:sha256(output/name) for name in (
                                  'summary.json','inference_complete.json','frozen_config.json',
                                  'predictions.jsonl','predictions.sha256','posthoc_targets.json','metrics.json')},
                              generation_policy=loaded['audit']['generation_policy'],
                              saved_val8_token_trace=gate_token_diagnostics(loaded['run'], model),
                              metrics=loaded['audit']['metrics'],
                              diagnostics=summarize_records(loaded['records'], tokenizer))
                exact_metrics(result['diagnostics']['format_valid_rate'],
                              result['metrics']['localization']['format_valid_rate'])
                if args.include_visa:
                    result['visa'] = visa_comparison(args.advanced, model, seed, tokenizer)
                report['results'].append(result)
            except (ValueError, AssertionError, KeyError, TypeError, FileNotFoundError, json.JSONDecodeError) as error:
                report['errors'].append(dict(model=model, seed=seed, error=str(error)))
    report['coverage'] = dict(requested_outputs=len(args.models)*len(args.seeds),
                              requested_models=args.models, requested_seeds=args.seeds,
                              complete_outputs=len(report['results']), full_structured_suite_outputs=6,
                              full_structured_suite_complete={(row['model'],row['seed']) for row in report['results']}
                                 == {(model,seed) for model in ('Cstructured','S') for seed in (42,43,44)})
    report['state'] = ('integrity_error' if report['errors'] else 'partial' if report['skipped'] else
                       'complete' if report['coverage']['full_structured_suite_complete'] else 'requested_subset_complete')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    args.out.with_suffix('.md').write_text(markdown(report))
    print(json.dumps(dict(state=report['state'], n_results=len(report['results']), n_skipped=len(report['skipped']),
                          n_errors=len(report['errors']), output=str(args.out)), ensure_ascii=False))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

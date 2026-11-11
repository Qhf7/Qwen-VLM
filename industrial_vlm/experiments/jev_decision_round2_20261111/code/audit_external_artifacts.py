"""Read saved external/validation artifacts only; never load a model or RGB/mask."""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

BASE = Path('/root/autodl-tmp/AnomalyGPT')
ADV = BASE / 'runs/migration_20261103/advanced_20261104'
NATIVE = Path('/root/autodl-tmp/industrial_vlm')
OUT = NATIVE / 'experiments/jev_decision_round2_20261111'
sys.path.insert(0, str(BASE))
from migration_metrics import evaluate_predictions

def read(p):
    return json.loads(Path(p).read_text())

def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def schema(row):
    return {k: type(v).__name__ for k, v in row.items()}

def equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b

def run():
    assert socket.gethostname() == 'autodl-container-132444827f-9c594038'
    frozen = read(ADV / 'external_selection_freeze.json')
    protocol = read(ADV / 'external_protocol.json')
    summary = read(ADV / 'external_summary.json')
    manifest = rows(ADV / 'external_manifest.jsonl')
    source_ids = [x['source_id'] for x in manifest]
    hashes = [x['image_sha256'] for x in manifest]
    assert len(manifest) == len(set(source_ids)) == len(set(hashes)) == 1575
    assert all(x['source_id'] == 'mvtec_' + x['image_sha256'] for x in manifest)
    external = []
    for item in summary['rows']:
        path = Path(item['output']) / 'predictions.jsonl'
        pred = rows(path)
        aligned = len(pred) == len(manifest) and all(
            all(p[k] == m[k] for k in ('source_id', 'path', 'category', 'label'))
            and p['split'] == 'external' for p, m in zip(pred, manifest))
        assert aligned and all(type(x['score']) in (int, float) and math.isfinite(x['score']) for x in pred)
        recomputed = evaluate_predictions(pred, [p['score'] for p in pred], [p['generated'] for p in pred])
        saved_metrics = read(path.parent / 'metrics.json')
        assert equal(recomputed, {k:saved_metrics[k] for k in recomputed})
        assert sha(path) == item['predictions_sha256']
        external.append(dict(
            model=item['model'], seed=item['seed'], path=str(path), sha256=sha(path),
            schema=schema(pred[0]), n=len(pred), unique_sources=len(set(x['source_id'] for x in pred)),
            same_order_source_label_and_path_as_manifest=aligned, source_id_is_image_sha256=True,
            score_n_unique=len(set(x['score'] for x in pred)), score_min=min(x['score'] for x in pred),
            score_max=max(x['score'] for x in pred), metrics_recomputed_equal=True,
            score_metrics=recomputed['overall']['scores'], generated_metrics=recomputed['overall']['generated'],
            category_auroc=recomputed['category_macro']['scores']['image_auroc'],
            checkpoint_sha256=item['checkpoint_sha256'], checkpoint_step=item['checkpoint_step'],
            new_trainable_parameters=item['newly_trained_parameters_for_this_condition'],
            exposures=item['training_exposures_for_this_condition']))
    assert len(external) == 21
    model_roles = {
        'A': 'validation-selected language migration bridge baseline; no additional training for this condition',
        'C': 'frozen bridge + language q_proj/v_proj LoRA; CE',
        'D': 'joint bridge + language LoRA; CE',
        'Cpair': 'frozen bridge + LoRA; paired normal/anomaly sampling, CE',
        'R': 'frozen bridge + LoRA; paired CE + ranking',
        'Cstructured': 'same selected C checkpoint; structured instruction/output control, no extra training',
        'S': 'structured supervision continuation on language LoRA'}
    aggregate = {}
    for arm, v in summary['aggregate'].items():
        keys = ['overall.scores.image_auroc', 'overall.scores.image_ap', 'overall.scores.fpr',
                'overall.generated.fpr', 'overall.generated.recall', 'category_macro.scores.image_auroc']
        aggregate[arm] = {k: v['metrics'][k] for k in keys}
    pilot = []
    for name in ['external_base_region', 'external_quality_region']:
        folder = NATIVE / 'evaluations' / name
        pred = rows(folder / 'predictions.jsonl')
        meta = read(folder / 'meta.json')
        pilot.append(dict(path=str(folder), n=len(pred), schema=schema(pred[0]),
                          has_continuous_score=any('score' in k or 'logit' in k for k in pred[0]),
                          sha256=sha(folder / 'predictions.jsonl'),
                          data=meta['config']['data'], adapter=meta['config']['adapter'],
                          overlap_external_source_ids=len(set(p['source_id'] for p in pred) & set(source_ids))))
    thresholds = []
    for p in sorted((NATIVE / 'evaluations/multitask_v2_scores').glob('*_threshold.json')):
        d = read(p)
        thresholds.append(dict(path=str(p), sha256=sha(p), **{k:d[k] for k in [
            'kind', 'source_scores', 'source_scores_sha256', 'threshold_log_odds', 'comparison',
            'target_empirical_fpr', 'n_calibration_normals', 'actual_calibration_false_positives',
            'empirical_calibration_fpr', 'confidence_calibrated']}))
    tool = read(ADV / 'tool_summary.json')
    tool_rows = rows(ADV / 'tool_val/predictions.jsonl')
    tool_run = read(ADV / 'tool_val/run.json')
    by_condition = {}
    for condition in tool_run['conditions']:
        pred = [x for x in tool_rows if x['condition'] == condition]
        key = [(x['path'], x['image_sha256'], x['category'], x['label']) for x in pred]
        assert len(pred) == len(set(x['image_sha256'] for x in pred)) == 380
        assert all(x['split'] == 'val' and math.isfinite(x['score_yes_minus_no'])
                   and x['score_yes_minus_no'] == x['yes_logit'] - x['no_logit'] for x in pred)
        metrics = evaluate_predictions(pred, [x['score_yes_minus_no'] for x in pred], [x['generated_text'] for x in pred])
        arm = tool['validation']['arms'][condition]
        assert equal(metrics, arm['metrics'])
        by_condition[condition] = dict(n=len(pred), unique_images=380, source_keys=key,
            score_metrics=metrics['overall']['scores'], generated_metrics=metrics['overall']['generated'],
            tool_calls=sum(x['tool_calls'] for x in pred),
            planner_action_counts=dict(collections.Counter(x['planner']['request']['action'] for x in pred)),
            actual_action_counts=dict(collections.Counter(x['request_executed']['action'] for x in pred)),
            n_images=sorted(set(x['n_images'] for x in pred)),
            visual_tokens=sorted(set(x['image_tokens_total'] for x in pred)),
            deployment_mean_seconds=arm['costs']['deployment_seconds']['mean'])
    common_keys = by_condition['no_tool']['source_keys']
    assert all(v.pop('source_keys') == common_keys for v in by_condition.values())
    duplicate = {x['image_sha256']:x for x in tool_rows if x['condition']=='duplicate_original'}
    model_tool = [x for x in tool_rows if x['condition']=='model_tool']
    same_model_tool_duplicate = all(
        x['score_yes_minus_no'] == duplicate[x['image_sha256']]['score_yes_minus_no']
        and x['generated_text'] == duplicate[x['image_sha256']]['generated_text'] for x in model_tool)
    assert same_model_tool_duplicate
    code_audit = {name:dict(path=str(BASE/name), frozen_sha256=digest,
                          current_sha256=sha(BASE/name), matches_frozen=sha(BASE/name)==digest)
                  for name,digest in frozen['source_sha256'].items()}
    audit = dict(schema='round2-external-artifact-audit-v1', created_at=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), mode='saved-artifacts-only; server CPU; no model/API/RGB/mask',
        source_summary_sha256=sha(ADV/'external_summary.json'), source_protocol_sha256=sha(ADV/'external_protocol.json'),
        manifest=dict(path=str(ADV/'external_manifest.jsonl'),sha256=sha(ADV/'external_manifest.jsonl'),
                      n=1575, normal=392, anomaly=1183, categories=15,
                      unique_source_ids=1575, unique_recorded_image_sha256=1575,
                      recorded_hashes_checked_without_reopening_RGB=True),
        external_prediction_rows=21*1575, independent_external_images=1575,
        all_21_runs_same_source_order=True, three_seeds_are_training_replicates_not_4725_independent_images=True,
        external_score_definition='unprocessed first-answer-token logit(Yes)-logit(No)',
        external_score_threshold='score > 0; score == 0 predicts No; no normal-only external calibration',
        model_roles=model_roles, runs=external, aggregate=aggregate, external_pilot=pilot,
        native_validation_thresholds=thresholds, source_code=code_audit,
        tool=dict(path=str(ADV/'tool_val'), predictions_sha256=sha(ADV/'tool_val/predictions.jsonl'),
                  run_sha256=sha(ADV/'tool_val/run.json'), summary_sha256=sha(ADV/'tool_summary.json'),
                  n_rows=1900, independent_images=380, seed=42, n_normal=172,n_anomaly=208,
                  image_size=448, native_adapter=None, bridge_adapter_used=False,
                  split='validation development; not external or independent heldout', schema=schema(tool_rows[0]),
                  all_arms_same_source_order_and_image_sha256=True, arms=by_condition,
                  model_tool_exactly_equal_duplicate_original=same_model_tool_duplicate,
                  masks_used_only_posthoc_geometry=True),
        reuse=dict(external_ranking=True, fixed_zero_threshold_binary=True,
                   calibrated_external_three_state=False, matched_native_DD_Jev_external_comparison=False,
                   native_5arm_fixed_controls=True, Jev_tool_head_comparison=False),
        missing=['A calibration-only frozen two-threshold policy for the SAME external score contract.',
                 'Matched native DD/Jev external scores on the same approved source set, resolution and references.',
                 'Jev planner actions / tool choices and downstream fixed final classifier outcomes.',
                 'Independent evaluation sources for a new tool-policy claim; current 380 are development.'],
        cost=dict(this_audit_API_requests=0,this_audit_GPU_forwards=0,models_loaded=0,
                  budget_rule='New experimental cost requires a frozen sample/condition plan; no monetary/time estimate inferred.'),
        access_boundary=dict(
            current_audit_reads='External MVTec saved predictions, VisA validation tool predictions, validation threshold metadata, static code only.',
            earlier_misclassified_path='evaluations/multitask_v2_scores/base_regression/meta.json',
            incident='Earlier recovery reconnaissance read native-test base_regression meta/metrics and threshold metadata before its role was recognized. No native-test scores.jsonl, generated predictions, or RGB were read. Stopped immediately; excluded from round2 analysis.',
            sealed_native_test_rows_reopened=False,stage13_labels_read=False,
            stages01_13_modified=False,external_selection_or_threshold_fitting=False),
        script_sha256=sha(__file__))
    lines=['# 外部连续分数与五臂工具实验可复用审计','',
        '本审计于服务器对已保存产物执行 CPU 统计；未加载模型、未发 API、未打开 RGB 或 mask、未重新训练或选择阈值。','',
        '## 结论','',
        '- 已找到真正的连续分数：migration 外部 `predictions.jsonl` 包含 float `score`；不能仅凭文件名判断它只是生成结果。',
        '- 共 7 个条件 × 3 个训练 seed × 1,575 图 = 33,075 行；独立外部图仍只有 1,575 张（392 正常、1,183 异常、15 类）。21 次模型评价使用完全相同的 source/path/label 顺序与记录的 image SHA。',
        '- 已重算全部 21 份外部指标、5 份工具指标并与冻结结果核对。所有 prediction SHA 与历史汇总一致。',
        '- 足够复用外部 AUROC/AP 和固定零阈值二分类诊断；尚不足以声称已完成外部校准三态或原生 DD／Jev 的同条件对照。',
        '- 现有五臂为原生 Qwen3-VL-8B、448、seed42、val380；model_tool 的 380 次 planner 全为 answer，0 次 crop，其最终分数与生成文本逐图等于 duplicate_original。不能把它写成自主裁剪或 Jev 工具头已提升。','',
        '## 三种评分口径','',
        '| 产物 | 连续分数 | 决策规则／阈值来源 | 可复用范围 |','|---|---|---|---|',
        '| migration MVTec residual1575 | 首答 token logit(Yes) − logit(No) | 严格 >0；等于0判No | 本分支外部排序与固定二分类 |',
        '| native 5-arm val380 | 最终独立 prompt 首答 token logit(Yes) − logit(No) | 严格 >0 | 原生开发工具对照 |',
        '| native boolean validation120 | JSON true/false 完整候选序列 log likelihood 差值 | validation 的60张正常图，normal-only order statistic，strict >，校准经验FPR5% | 仅对应的base/adapter/prompt/分辨率contract |',
        '| native MVTec pilot150 | 仅 raw_text＋parsed；无连续score/logit字段 | 生成 JSON 决策 | 不能由解析布尔值补成连续分数 |','',
        '生成 FPR 使用首个 Yes/No 的解析结果，invalid按既定规则计错；score FPR 使用连续 score 的严格阈值。两列即使数值相同，也属于不同决策链。AUROC/AP由连续score计算，不依赖生成解析。native normal-only阈值只控制校准样本经验FPR，不保证未来或跨域FPR；不能将其直接套在migration logit差值或Jev概率上。','',
        '## 外部结果（三 seed 均值±样本标准差）','',
        '| 条件 | AUROC | AP | score FPR(>0) | generated FPR | generated Recall | 类别macro AUROC |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,v in aggregate.items():
        def fmt(k):
            z=v[k];return f"{100*z['mean']:.2f}±{100*z['sample_sd']:.2f}%"
        lines.append('| '+arm+' | '+' | '.join(fmt(k) for k in [
            'overall.scores.image_auroc','overall.scores.image_ap','overall.scores.fpr',
            'overall.generated.fpr','overall.generated.recall','category_macro.scores.image_auroc'])+' |')
    lines += ['', 'seed 的差异表示训练复现波动，不是三批独立测试图；不得把三个seed行拼接成4,725个独立样本来缩小置信区间。以上外部集已被历史研究查看，不能重新标注为盲测集。','',
              '条件定义：'+ '；'.join(k+'：'+v for k,v in model_roles.items())+'。','',
              '## 五臂开发结果（val380）','',
              '| 条件 | AUROC | Recall(>0) | score FPR | 实际crop数 | 图像数／视觉token | 平均部署秒 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,v in by_condition.items():
        m=v['score_metrics'];lines.append(f"| {arm} | {100*m['image_auroc']:.2f}% | {100*m['recall']:.2f}% | {100*m['fpr']:.2f}% | {v['tool_calls']} | {v['n_images']}／{v['visual_tokens']} | {v['deployment_mean_seconds']:.3f} |")
    lines += ['', 'center/random crop 的位置是固定控制，不使用GT；GT geometry仅在推理后审计。原图固定448；crop宽高各1/3，面积1/9，最多一次调用。除no_tool外均给两张图（无crop时第二张为原图重复），视觉token匹配，文本token与时延未完全匹配。planner跨控制复用；固定控制部署成本不含不必要planner开销。','',
              '## 下一步及预算门禁','',
              '1. 保留现有21份冻结外部分数，仅生成可追溯的分析表；本轮不重新挑阈值、checkpoint或seed。',
              '2. 外部三态需先在合适的独立开发／校准来源固定两个阈值及拒答损失，记录calibration身份与score contract，再对预注册外部集一次评价；同一外部标签不能同时调阈值和报告泛化。',
              '3. Jev工具头需冻结动作空间、统一图像预算、固定下游分类器，再与no_tool／duplicate／center／random／Qwen planner逐图配对；现有Jev异常判定分数不能冒充工具动作日志。',
              '4. 如复用val380原始图做新的开发工具对照，完整五臂至少需要380×5=1,900次最终判定（若全重跑），外加planner调用；如只补Jev planner，调用数按新contract单独列出。此为计算次数示意，非批准预算，也不是新独立盲测。',
              '5. 当前审计新增API=0、GPU forward=0；没有真实调用方案与费率前不捏造费用或时长。','',
              '## 准确路径与访问边界','',
              f'- 外部汇总：`{ADV}/external_summary.json`；manifest：`{ADV}/external_manifest.jsonl`。',
              '- A/C路径：`migration_20261103/lora_20261104/arm_C_s{42,43,44}/external_mvtec_residual_v1{_A,}/predictions.jsonl`。',
              '- D/Cpair/R路径：`advanced_20261104/{arm_D,Cpair,R}_s{42,43,44}/external_mvtec_residual_v1/predictions.jsonl`；Cstructured为`arm_S_s*/external_mvtec_residual_v1_initial_c`；S为同根`external_mvtec_residual_v1`。',
              f'- 五臂：`{ADV}/tool_val/{{run.json,predictions.jsonl,audit.json}}`；汇总`{ADV}/tool_summary.json`。',
              '- scorer：`/root/autodl-tmp/AnomalyGPT/migration_language.py:297-301`；阈值与指标：`migration_metrics.py:59-74`；native正常阈值：`industrial_vlm/src/industrial_vlm/score.py:165-187`。',
              '- 恢复初期曾误将base_regression当external，读取其meta/metrics和阈值元数据后识别为native test并停止；未读native test scores/predictions/RGB。本报告不将这些误读元数据纳入新的评价。',
              '- 本轮不读取stage13确认标签、不改01—13、不重开封存native test逐样本结果；已查看过的历史confirmation不能作为新round2调参集。',
              '- 全部路径、schema、SHA、21 run与五臂细项见同名JSON。']
    OUT.mkdir(parents=True,exist_ok=True)
    for name,value in [('EXTERNAL_ARTIFACT_AUDIT.json',json.dumps(audit,ensure_ascii=False,indent=2)+'\n'),
                       ('EXTERNAL_ARTIFACT_AUDIT.md','\n'.join(lines)+'\n')]:
        p=OUT/name
        if p.exists():raise FileExistsError(p)
        p.write_text(value)
    print(json.dumps(dict(status='complete',external_runs=21,external_images=1575,tool_rows=1900,
        code_matches_frozen=all(x['matches_frozen'] for x in code_audit.values()),
        outputs={n:sha(OUT/n) for n in ['EXTERNAL_ARTIFACT_AUDIT.json','EXTERNAL_ARTIFACT_AUDIT.md']}),ensure_ascii=False))

if __name__ == '__main__':
    run()

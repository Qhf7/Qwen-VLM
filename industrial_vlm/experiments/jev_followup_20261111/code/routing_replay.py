#!/usr/bin/env python3
"""Server-only frozen offline routing replay; never calls a model/API.

prepare --root ROOT --run runs/modelgo_roi freezes policy before stage05.
evaluate --root ROOT --run runs/modelgo_roi requires complete stage05.
self-test uses synthetic CPU-only data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys

import roi_api as api
import evaluate_roi as evaluation

SCHEMA = 'jev-offline-sequential-routing-v1'
HOSTNAME = 'autodl-container-132444827f-9c594038'
LOW, HIGH = .25, .75
AREA = .5
SEEDS = (42, 43, 44)
THRESHOLD = .5
RANDOM_PREFIX = 'jev-routing-matched-random-v1'
METRICS = evaluation.METRICS


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return api.file_sha(path)


def publish(path, value):
    evaluation.publish(path, value)


def protocol_body(root, run):
    """Read contract/candidates only; no probabilities, targets or metrics."""
    contract_path = run / 'contract.json'
    contract = read(contract_path)
    require(contract.get('schema') == api.SCHEMA and contract.get('mode') == 'formal_development_candidate_review'
            and contract.get('limit_queries') is None, 'Routing requires the formal paired contract')
    candidates_path = api.inside(root, contract['source_candidates'])
    candidates = read(candidates_path)
    require(candidates.get('status') == 'complete' and len(candidates.get('records', [])) == 120, 'Candidate stage incomplete')
    require(sha(candidates_path) == contract['source_sha256']['candidates'], 'Candidate identity changed')
    require(sha(api.__file__) == contract['runner_sha256'], 'Frozen API source changed')
    records = contract['records']
    require(len(records) == len({r['source_id'] for r in records}) == 120, 'Expected120 unique query identities')
    require(len(contract['schedule']) == 240 and {(r['source_id'],r['arm']) for r in contract['schedule']} ==
            {(r['source_id'],a) for r in records for a in ('roi','duplicate')}, 'Formal paired schedule differs')
    source = {r['source_id']:r for r in candidates['records']}
    for row in records:
        original = source[row['source_id']]
        require(row['bbox_original_norm'] == original['bbox_original_norm'] and
                row['query_crop_bytes_identical'] == (original['assets']['query']['png_sha256'] ==
                                                       original['assets']['crop']['png_sha256']), 'Candidate routing geometry differs')
    return dict(schema=SCHEMA, run=run.relative_to(root).as_posix(),
        formal_contract_sha256=sha(contract_path), formal_contract_canonical_sha256=digest(canonical(contract)),
        candidates_sha256=sha(candidates_path), routing_source_sha256=sha(__file__),
        evaluation_source_sha256=sha(evaluation.__file__), api_source_sha256=sha(api.__file__),
        dependencies=dict(python=platform.python_version(), numpy=importlib.metadata.version('numpy')),
        n_queries=120, threshold=THRESHOLD, decision='p>=0.5 retains the fixed candidate',
        policies=dict(duplicate='One duplicate arm response per query; one logical request',
            roi='One ROI arm response per query; one logical request',
            sequential_uncertain=dict(first='duplicate', lower=LOW, upper=HIGH, interval='closed_both',
                require_different_crop=True, action='Use ROI response iff duplicate probability in [.25,.75] and crop bytes differ; else duplicate',
                calls='one duplicate for every query plus one ROI for every triggered query'),
            matched_random=dict(seeds=list(SEEDS), eligible='crop bytes differ',
                count='Same number as sequential_uncertain over the full120; never label dependent',
                selection='Sort sha256(jev-routing-matched-random-v1|seed|source_id) ascending, then source_id; first K',
                first='duplicate', action='Use ROI for selected K and duplicate otherwise',
                interpretation='Three descriptive matched random selections; not trained seeds or independent datasets'),
            geometry_small=dict(max_area=AREA, comparison='area<=0.5', require_different_crop=True,
                first='duplicate', action='Use ROI iff original-normalized candidate area<=0.5 and crop bytes differ')),
        partitions=['all120','teacher_unseen','teacher_training_exposed','query_crop_bytes_identical',
                    'query_crop_bytes_different','full_image_candidate','nonfull_image_candidate'],
        teacher_unseen_definition='teacher_split != train; fixed development subset',
        group_routing='Compute routing and matched K once on all120; subgroups only evaluate that frozen route',
        metrics=list(METRICS), bootstrap=dict(replicates=2000, seed=42, strata='category x image_label',
            paired=True, route_frozen_during_resampling=True, policy_reselection_uncertainty=False),
        input_budget='Logical successful model requests and observed usage of selected saved calls; no monetary estimate',
        collection_budget='The full240 table was collected; replay logical budgets do not undo actual calls/retries',
        no_gt_or_metric_policy_tuning=True, no_oracle_arm_selection=True, no_api_calls=True,
        offline_replay_only=True, online_agent_benefit_measured=False, confirmation_opened=False,
        freeze_gate='prepare reads only formal contract and fixed candidates; evaluate requires stage05 complete and identical frozen predictions')


def prepare(args):
    require(socket.gethostname() == HOSTNAME, 'Routing preparation must execute on designated server')
    root = args.root.resolve()
    run = api.inside(root,args.run)
    target = root/'stages/06_routing_protocol.json'
    with api.output_lock(root/'stages/routing_lock'):
        body = protocol_body(root,run)
        if target.exists():
            old=read(target)
            require(old.get('protocol') == body, 'Frozen routing protocol differs')
            document=old
        else:
            document=dict(schema=SCHEMA,status='frozen',timestamp_utc=now(),hostname=socket.gethostname(),
                          ground_truth_or_metrics_read=False,protocol=body,protocol_sha256=digest(canonical(body)))
            publish(target,document)
        print(json.dumps(dict(status='frozen',protocol=str(target),protocol_sha256=document['protocol_sha256'],
                              ground_truth_or_metrics_read=False),ensure_ascii=False),flush=True)
    return 0


def matched_random_indices(rows,count,seed):
    eligible=[i for i,r in enumerate(rows) if not r['query_crop_bytes_identical']]
    require(type(count) is int and 0<=count<=len(eligible),'Matched random K exceeds eligible population')
    def order(index):
        sid=rows[index]['source_id']
        return digest((RANDOM_PREFIX+'|'+str(seed)+'|'+sid).encode()),sid
    return set(sorted(eligible,key=order)[:count])


def route_specs(rows,completed):
    uncertain=set()
    geometric=set()
    for i,row in enumerate(rows):
        p=completed[row['source_id'],'duplicate']['probability']
        if not row['query_crop_bytes_identical']:
            if LOW<=p<=HIGH:
                uncertain.add(i)
            box=row['candidate_bbox_original_norm']
            area=(box[2]-box[0])*(box[3]-box[1])
            if area<=AREA:
                geometric.add(i)
    specs={'duplicate':dict(upgrades=set(),single_arm='duplicate'),
           'roi':dict(upgrades=set(),single_arm='roi'),
           'sequential_uncertain':dict(upgrades=uncertain,single_arm=None),
           'geometry_small':dict(upgrades=geometric,single_arm=None)}
    for seed in SEEDS:
        specs['matched_random_s'+str(seed)]=dict(upgrades=matched_random_indices(rows,len(uncertain),seed),single_arm=None)
    return specs


def usage_budget(rows,completed,spec,indices):
    selected=[]
    upgrades=spec['upgrades']
    for i in indices:
        sid=rows[i]['source_id']
        if spec['single_arm']:
            selected.append(completed[sid,spec['single_arm']])
        else:
            selected.append(completed[sid,'duplicate'])
            if i in upgrades:
                selected.append(completed[sid,'roi'])
    fields=sorted({key for value in completed.values() for key in value['usage']})
    usage={}
    for field in fields:
        observed=[row['usage'][field] for row in selected if field in row['usage']]
        usage[field]=dict(observed_sum=float(sum(observed)),observed_calls=len(observed),
                          missing_calls=len(selected)-len(observed),complete=len(observed)==len(selected))
    return dict(n_queries=len(indices),logical_successful_requests=len(selected),
        average_requests_per_query=len(selected)/len(indices) if indices else None,
        extra_roi_calls=0 if spec['single_arm'] else sum(i in upgrades for i in indices),
        observed_usage=usage,calls_without_any_usage=sum(not row['usage'] for row in selected),
        saved_call_seconds_sum=float(sum(row['seconds'] for row in selected)),
        latency_note='Sum of separately observed call durations, not measured online sequential wall latency',
        billing_note='Replay budget only; all240 source calls and their retries were actually collected; missing usage is unknown')


def partitions(rows):
    return dict(all120=list(range(len(rows))),
        teacher_unseen=[i for i,r in enumerate(rows) if r['teacher_split']!='train'],
        teacher_training_exposed=[i for i,r in enumerate(rows) if r['teacher_split']=='train'],
        query_crop_bytes_identical=[i for i,r in enumerate(rows) if r['query_crop_bytes_identical']],
        query_crop_bytes_different=[i for i,r in enumerate(rows) if not r['query_crop_bytes_identical']],
        full_image_candidate=[i for i,r in enumerate(rows) if r['full_image_candidate']],
        nonfull_image_candidate=[i for i,r in enumerate(rows) if not r['full_image_candidate']])


def fmt(value):
    return 'NA' if value is None else f'{100*value:.2f}%'


def report(result):
    group=result['groups']['all120']
    lines=['# Jev 离线顺序路由回放', '',
        '本阶段复用已采集的240个配对响应，不新增模型请求。策略在读取GT和正式指标前冻结。',
        '路由仅使用先到的duplicate概率、候选几何与中性source ID哈希，不按GT或较优答案选臂。', '',
        '| 策略 | 逻辑请求数 | ROI升级数 | 全异常保留召回 | 正常出框率 | 全异常平均IoU | IoU≥0.5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,method in group['methods'].items():
        m,b=method['performance']['pooled'],method['budget']
        lines.append(f"| {name} | {b['logical_successful_requests']} | {b['extra_roi_calls']} | {fmt(m['anomaly_retained_recall'])} | {fmt(m['normal_positive_box_rate'])} | {m['anomaly_bbox_iou_mean_all']:.6f} | {fmt(m['anomaly_iou_at_05_all'])} |")
    delta=group['uncertain_minus_duplicate']['pooled']
    lines+=['','## 固定策略与结果分析','',
        '不确定性路由：duplicate概率落在闭区间[0.25,0.75]且裁片字节不同，才读取ROI响应；否则保留duplicate。',
        '几何策略：候选原图面积比例≤0.5且裁片字节不同才升级。随机控制从非重复裁片中按种子哈希排序，选择与不确定性路由相同数量。',
        '所有策略均p≥0.5保留候选框，未拟合阈值。全120上的升级集合先确定，再原样用于分组评价。', '',
        '| 不确定性路由−duplicate | 差值 | 配对bootstrap95区间 |','|---|---:|---|']
    for name,item in delta.items():
        interval=item['ci95']
        ci='NA' if interval is None else f'[{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] pp'
        point='NA' if item['delta'] is None else f"{100*item['delta']:+.2f} pp"
        lines.append(f'| {name} | {point} | {ci} |')
    conclusions=result['analysis']
    lines+=['',*['- '+line for line in conclusions], '', '## 分组与限制','',
        'JSON另存teacher未见、teacher训练暴露、query/crop字节相同或不同、整图框与非整图框的完整分母指标、成本及配对区间。',
        '三seed随机结果是描述性控制，不是三个独立数据集；小子集FPR不稳定。Bootstrap固定已产生的路由集合，不包含策略重选、服务随机性或未来分布变化。',
        '这是全表已采集后的离线路由回放。逻辑请求数与usage重组不代表节省了已发生的240目标采集成本，也不证明在线Agent净收益。',
        '两臂均含两幅448图；同字节裁片仍保留来源响应，差异可能来自服务随机性和中性UUID。区域概率未当作整图校准概率。',
        '本阶段没有新API、训练、GT调参或confirmation访问；不估货币费用。', '',
        f"协议SHA256：`{result['routing_protocol_sha256']}`",
        f"正式预测SHA256：`{result['formal_predictions_sha256']}`",
        f"stage05 SHA256：`{result['stage05_sha256']}`",'']
    return '\n'.join(lines)


def evaluate(args):
    require(socket.gethostname()==HOSTNAME,'Routing evaluation must execute on designated server')
    root=args.root.resolve()
    run=api.inside(root,args.run)
    with api.output_lock(root/'stages/routing_lock'):
        protocol_path=root/'stages/06_routing_protocol.json'
        frozen=read(protocol_path)
        require(frozen.get('status')=='frozen' and frozen.get('ground_truth_or_metrics_read') is False,'Pre-GT routing freeze required')
        body=protocol_body(root,run)
        require(frozen['protocol']==body and frozen['protocol_sha256']==digest(canonical(body)),'Routing protocol/source changed')
        status,contract,completed,starts,attempts=evaluation.validate_formal(root,run)
        prediction_sha=sha(run/'predictions.jsonl')
        prediction_freeze=read(root/'stages/04_prediction_freeze.json')
        require(prediction_freeze['files_sha256']['formal_predictions']==prediction_sha,'Prediction freeze mismatch')
        stage05_path=root/'stages/05_evaluation.json'
        stage05=read(stage05_path)
        require(stage05.get('status')=='complete' and stage05.get('prediction_freeze')==prediction_freeze
                and stage05.get('confirmation_opened') is False,'Complete same-prediction stage05 required')
        require(stage05.get('script_sha256')==sha(evaluation.__file__),'Stage05 evaluator identity differs')
        require(frozen['timestamp_utc']<stage05['timestamp_utc'],'Routing protocol was not frozen before stage05')
        rows=stage05['records']
        require(len(rows)==len({r['source_id'] for r in rows})==120 and
                [r['source_id'] for r in rows]==sorted(r['source_id'] for r in rows),'Stage05 row identities/order differ')
        expected={r['source_id']:r for r in contract['records']}
        require(set(expected)=={r['source_id'] for r in rows},'Stage05/formal population differs')
        targets=[dict(source_id=r['source_id'],label=r['label'],bbox=r['gt_bbox_original_norm'],ground_truth=r['ground_truth']) for r in rows]
        require(digest(canonical(targets))==stage05['gt_targets_sha256'],'Saved target identity differs')
        for row in rows:
            sid=row['source_id']
            require(row['teacher_split']==expected[sid]['teacher_split'] and
                    row['candidate_bbox_original_norm']==expected[sid]['bbox_original_norm'] and
                    row['query_crop_bytes_identical']==expected[sid]['query_crop_bytes_identical'],'Stage05 route metadata differs')
            require(type(row['label']) is int and row['label'] in (0,1),'Invalid saved binary label')
            for arm in ('roi','duplicate'):
                require(row['probabilities'][arm]==completed[sid,arm]['probability'],'Stage05/saved probability differs')
        specs=route_specs(rows,completed)
        gates={}
        chosen={}
        for name,spec in specs.items():
            arms=[spec['single_arm'] or ('roi' if i in spec['upgrades'] else 'duplicate') for i in range(len(rows))]
            chosen[name]=arms
            gates[name]=[completed[row['source_id'],arm]['probability']>=THRESHOLD for row,arm in zip(rows,arms)]
        groups={}
        for name,indices in partitions(rows).items():
            methods={policy:dict(performance=evaluation.package(rows,gates[policy],indices),
                budget=usage_budget(rows,completed,spec,indices)) for policy,spec in specs.items()}
            comparisons={policy:evaluation.bootstrap_difference(rows,gates[policy],gates['duplicate'],indices)
                         for policy in specs if policy!='duplicate'}
            groups[name]=dict(n=len(indices),methods=methods,vs_duplicate=comparisons,
                uncertain_minus_duplicate=comparisons['sequential_uncertain'],
                uncertain_minus_roi=evaluation.bootstrap_difference(rows,gates['sequential_uncertain'],gates['roi'],indices))
        all_methods=groups['all120']['methods']
        base=all_methods['duplicate']['performance']['pooled']
        route=all_methods['sequential_uncertain']['performance']['pooled']
        extra=all_methods['sequential_uncertain']['budget']['extra_roi_calls']
        diou=route['anomaly_bbox_iou_mean_all']-base['anomaly_bbox_iou_mean_all']
        drec=route['anomaly_retained_recall']-base['anomaly_retained_recall']
        dfpr=route['normal_positive_box_rate']-base['normal_positive_box_rate']
        analysis=[f'不确定性路由升级{extra}/120张图，逻辑请求数{120+extra}，比duplicate多{extra}次；每次ROI升级仍是一条双图请求。',
            f'相对duplicate：完整异常平均IoU变化{diou:+.6f}，保留召回变化{100*drec:+.2f}pp，正常出框率变化{100*dfpr:+.2f}pp。',
            '只报告已冻结规则的收益与代价，不据此自动追加付费调用或调阈值。']
        if diou<=0 and drec<=0 and dfpr>=0:
            analysis.append('当前点估计没有显示该顺序路由的质量收益，额外调用不能据此认定有价值。')
        else:
            analysis.append('指标存在不同方向或非零变化，需结合完整配对区间、误报和成本评估；本轮不宣称部署或在线Agent收益。')
        result=dict(schema=SCHEMA,status='complete',timestamp_utc=now(),hostname=socket.gethostname(),
            script_sha256=sha(__file__),routing_protocol_sha256=sha(protocol_path),
            protocol_canonical_sha256=frozen['protocol_sha256'],formal_predictions_sha256=prediction_sha,
            stage05_sha256=sha(stage05_path),gt_targets_sha256=stage05['gt_targets_sha256'],
            full_table_collection_operations=stage05['operations'],groups=groups,analysis=analysis,
            routing_records=[dict(source_id=row['source_id'],teacher_split=row['teacher_split'],
                query_crop_bytes_identical=row['query_crop_bytes_identical'],
                chosen_arm={p:chosen[p][i] for p in specs},
                upgraded={p:(i in specs[p]['upgrades']) for p in specs if not specs[p]['single_arm']}) for i,row in enumerate(rows)],
            no_api_calls=True,no_gt_tuning=True,no_oracle_arm_selection=True,offline_replay_only=True,
            online_agent_benefit_measured=False,confirmation_opened=False)
        path=root/'stages/06_routing_results.json'
        if path.exists():
            previous=read(path)
            require(all(previous[k]==result[k] for k in result if k!='timestamp_utc'),'Existing routing result differs')
            result=previous
        else:
            publish(path,result)
        document=report(result)
        report_path=root/'ROUTING_REPORT.md'
        if report_path.exists():
            require(report_path.read_text(encoding='utf-8')==document,'Existing routing report differs')
        else:
            with report_path.open('x',encoding='utf-8') as stream:
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
        require(sha(run/'predictions.jsonl')==prediction_sha and sha(stage05_path)==result['stage05_sha256'],'Frozen sources changed during replay')
        print(json.dumps(dict(status='complete',n=120,routing_results=str(path),report=str(report_path),
                              extra_roi_calls=extra,no_api_calls=True),ensure_ascii=False),flush=True)
    return 0


def self_test():
    rows=[]
    probabilities=[.25,.75,.5,.249,.751,.5]
    completed={}
    for i,p in enumerate(probabilities):
        sid='synthetic_'+str(i)
        identical=i==2
        rows.append(dict(source_id=sid,category='fixture',label=i%2,teacher_split='val',
            query_crop_bytes_identical=identical,full_image_candidate=identical,
            candidate_bbox_original_norm=[0.,0.,1.,1.] if identical else [0.,0.,.5,1.],candidate_iou=.5))
        for arm in ('roi','duplicate'):
            completed[sid,arm]=dict(probability=p if arm=='duplicate' else 1-p,usage={'input_tokens':10},seconds=2.)
    specs=route_specs(rows,completed)
    require(specs['sequential_uncertain']['upgrades']=={0,1,5},'Inclusive uncertain interval/duplicate exclusion failed')
    require(specs['geometry_small']['upgrades']=={0,1,3,4,5},'Area inclusive gate failed')
    for seed in SEEDS:
        picked=specs['matched_random_s'+str(seed)]['upgrades']
        require(len(picked)==3 and 2 not in picked and picked==matched_random_indices(rows,3,seed),'Matched random selection failed')
    budget=usage_budget(rows,completed,specs['sequential_uncertain'],list(range(6)))
    require(budget['logical_successful_requests']==9 and budget['observed_usage']['input_tokens']['observed_sum']==90,'Sequential request/token accounting failed')
    require(usage_budget(rows,completed,specs['roi'],list(range(6)))['logical_successful_requests']==6,'Single ROI baseline budget failed')
    hidden=[{**r,'label':1-r['label']} for r in rows]
    require(route_specs(hidden,completed)==specs,'Route depends on GT label')
    print(json.dumps(dict(status='passed',synthetic_only=True,network_calls=0,checks=['inclusive_boundary','duplicate_exclusion',
        'geometry_gate','matched_random_count','deterministic_random','no_label_routing','sequential_cost','single_arm_cost'])))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name,func in [('prepare',prepare),('evaluate',evaluate)]:
        item=sub.add_parser(name)
        item.add_argument('--root',type=Path,required=True)
        item.add_argument('--run',default='runs/modelgo_roi')
        item.set_defaults(func=func)
    item=sub.add_parser('self-test')
    item.set_defaults(func=lambda args:self_test())
    args=parser.parse_args()
    try:
        return args.func(args)
    except Exception as error:
        print(json.dumps(dict(status='blocked',exception_class=type(error).__name__,
            reason=str(error) if isinstance(error,ValueError) else 'Inspect frozen inputs and runtime; arbitrary exception text suppressed')),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Publish static review materials on the approved server; never infer or label."""
import hashlib
import json
import os
from pathlib import Path
import socket
import zipfile

ROOT = Path('/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111')
OLD = ROOT.parent / 'jev_followup_20261111'
OUT = ROOT / 'human_semantic_review'
CONTRACT_SHA = '16941598c4589bdc4eedbbe2990d1e75bee0dde3db41be889bc3e23fac2473c1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def dump(path, data):
    write(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def text(path, value):
    write(path, value.encode())


def main():
    assert socket.gethostname() == 'autodl-container-132444827f-9c594038'
    assert not (OUT / 'review_packet').exists(), 'Publish once; preserve existing annotation work'
    contract_raw = (ROOT / 'online_route_pilot/contract.json').read_bytes()
    assert sha(contract_raw) == CONTRACT_SHA
    c = json.loads(contract_raw)
    assert len(c['records']) == 24 and len({r['source_id'] for r in c['records']}) == 24
    assert (ROOT / 'online_route_pilot/prediction_freeze.json').is_file()
    packet = OUT / 'review_packet'
    packet.mkdir(parents=True)
    public, private, forms = [], [], []
    stage_a = ['# 第一步：独立查看整图', '', '先完成整图字段，再打开第二步。只依据本图可见证据；分辨率不足或无法排除正常变化时填 uncertain。', '']
    stage_b = ['# 第二步：复核候选区域', '', '应已完成整图判断。框来自固定候选，提出候选不代表已经判定异常；裁片可能与原图相同。记录可见证据，不猜测原因。', '']
    ordered = sorted(c['records'], key=lambda r: sha(('human-review-order-v1|' + r['source_id']).encode()))
    for i, row in enumerate(ordered, 1):
        item = f'item_{i:02d}'
        assets = {}
        for role in ('query', 'crop'):
            asset = row['assets'][role]
            src = (OLD / asset['path']).resolve()
            assert src.is_relative_to(OLD.resolve())
            raw = src.read_bytes()
            assert sha(raw) == asset['png_sha256'] and len(raw) == asset['png_bytes']
            rel = f'images/{item}_{role}.png'
            write(packet / rel, raw)
            assets[role] = {'path': rel, 'sha256': sha(raw), 'bytes': len(raw)}
        public.append({'item_id': item, 'assets': assets, 'candidate_bbox_in_query_xyxy_01': row['bbox_448_norm']})
        private.append({'item_id': item, 'source_id': row['source_id'], 'query_sha256': assets['query']['sha256'], 'crop_sha256': assets['crop']['sha256']})
        forms.append({'schema': 'human-visible-evidence-review-v1', 'item_id': item, 'status': 'pending',
                      'reviewer_id': None, 'completed_utc': None,
                      'whole_image': {'visible_anomaly': None, 'evidence': None, 'defect_type_open_text': None, 'uncertainty_reason': None},
                      'candidate': {'visible_anomaly': None, 'evidence': None, 'relationship_to_whole_image': None, 'uncertainty_reason': None},
                      'causal_claim_made': False})
        stage_a += [f'## {item}', '', f'![{item}整图]({assets["query"]["path"]})', '']
        bounds = ', '.join(f'{v:.6f}' for v in row['bbox_448_norm'])
        stage_b += [f'## {item}', '', f'原图画布中的候选范围：[{bounds}]，顺序 left,top,right,bottom；0–1归一化，含原448画布白边。', '',
                    f'![{item}整图]({assets["query"]["path"]})', '', f'![{item}裁片]({assets["crop"]["path"]})', '']
    dump(packet / 'items.json', public)
    for reviewer in ('A', 'B'):
        text(packet / f'annotator_{reviewer}.jsonl', ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in forms))
    text(packet / '01_whole_images.md', '\n'.join(stage_a))
    text(packet / '02_candidates.md', '\n'.join(stage_b))
    instructions = '''# 24图人工视觉事实复核材料

当前状态：等待两位复核者独立填写，尚无人工金标。

1. 复核者A、B分别复制自己的annotator_A/B.jsonl，独立填写；先只打开01_whole_images.md，完成whole_image部分。
2. 再打开02_candidates.md，填写candidate部分；不要事先查看模型分数、原标签、mask或另一位的表单。
3. visible_anomaly填 yes / no / uncertain。yes须写可见证据；no不等于保证没有不可见缺陷；uncertain须写原因，不能强行二选一。
4. evidence记可见形态和位置；defect_type_open_text允许自由描述或unknown，不猜原因，不据类别名称推断答案。
5. relationship_to_whole_image填 consistent / inconsistent / uncertain，比较候选区域的证据是否支持已记下的整图描述。
6. 完成单项后填reviewer_id、completed_utc并将status从pending改为completed。未经人工填写的null必须保留，不得当成no。
7. 保留两个原始表单。分歧由第三人或双方逐项裁决，保存独立adjudication表及理由；未裁决的条目保持不确定。

图像是冻结的448输入，不提供更高清原图，判断仅对应本输入条件。阶段2提供的候选来自已有方法；候选图不构成答案。
本包隐藏了预测、标签和source ID，来源映射另存于包外。所有样本都来自已看开发数据，不能称为新的盲测。
包内manifest提供图像SHA用于核验，不包含缺陷标签。这里的盲标指复核者看不到旧结果，不保证未曾见过图。

后续验收：24项×两人，所有状态/必填证据/时间完整、分歧逐项保留；以“图像来源”为统计单位，不能将两位复核者当48个独立来源。
这些人工记录完成后才可成为视觉事实参考。缺陷类型/解释支持评测还需固定待验证命题和独立确认来源；本包不自行证明模型准确率。
'''
    text(packet / 'README.md', instructions)
    dump(OUT / 'source_mapping_private.json', {'schema': 'human-review-source-mapping-v1', 'online_contract_sha256': CONTRACT_SHA, 'records': private})
    forbidden = ['source_id', 'historical_duplicate_probability', 'teacher_split', 'is_anomalous', 'mask_path', 'visa_']
    for path in packet.iterdir():
        if path.suffix in ('.jsonl', '.json', '.md'):
            value = path.read_text()
            assert all(term not in value for term in forbidden), path
    files = {str(p.relative_to(packet)): sha(p.read_bytes()) for p in packet.rglob('*') if p.is_file()}
    dump(packet / 'MANIFEST.json', {'schema': 'human-review-static-packet-v1', 'items': 24, 'annotators': 2, 'human_completed': 0, 'files': files})
    archive = OUT / 'review_packet.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(packet.rglob('*')):
            if p.is_file():
                z.write(p, str(Path('review_packet') / p.relative_to(packet)))
    dump(OUT / 'preparation_certificate.json', {'schema': 'human-review-material-preparation-v1', 'status': 'prepared_waiting_for_human_annotation',
         'items': 24, 'source_images': 24, 'image_assets': 48, 'blank_forms': 48, 'human_completed': 0,
         'online_contract_sha256': CONTRACT_SHA, 'builder_sha256': sha(Path(__file__).read_bytes()),
         'archive_sha256': sha(archive.read_bytes()), 'no_labels_or_masks_read': True, 'API_calls': 0, 'model_forwards': 0,
         'private_mapping_outside_packet': True, 'stage13_read': False})
    print(json.dumps({'status': 'prepared_waiting_for_human_annotation', 'items': 24, 'blank_forms': 48, 'API_calls': 0}))


if __name__ == '__main__':
    main()

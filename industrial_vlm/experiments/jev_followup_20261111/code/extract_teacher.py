"""Extract only approved Jev query maps from the existing server cache."""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

from audit_sources import digest, publish

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def processed(image):
    resized = ImageOps.contain(image, (448, 448), method=Image.Resampling.LANCZOS)
    offset = [(448-resized.width)//2, (448-resized.height)//2]
    canvas = Image.new('RGB', (448,448), (255,255,255))
    canvas.paste(resized, tuple(offset))
    buffer=io.BytesIO()
    canvas.save(buffer, format='PNG', compress_level=6)
    return buffer.getvalue(), list(resized.size), offset

def exclusive_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, 'Existing asset differs'
    else:
        with path.open('xb') as stream:
            stream.write(data)
    return dict(path=str(path), png_sha256=hashlib.sha256(data).hexdigest(),
                png_bytes=len(data), size=[448,448])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    args=parser.parse_args()
    assert socket.gethostname()=='autodl-container-132444827f-9c594038'
    root=args.root.resolve()
    first=json.loads((root/'stages/01_source_audit.json').read_text())
    assert first['status']=='complete' and first['passed']==168
    plan=json.loads((root/'source_baseline/frozen/plan.json').read_text())
    sidecar=json.loads(args.cache.with_suffix('.json').read_text())
    cache_sha=digest(args.cache)
    assert cache_sha==sidecar['cache_sha256']
    cache=torch.load(args.cache, map_location='cpu', mmap=True, weights_only=False)
    assert cache['pipeline_version']=='visa-square-binary-sklearn-v1'
    assert cache['geometry']=='square224'
    manifest_sha=hashlib.sha256(json.dumps(cache['items'],sort_keys=True).encode()).hexdigest()
    assert manifest_sha==cache['manifest_sha256']==sidecar['manifest_sha256']
    assert cache['decoder_sha256']==sidecar['decoder_sha256']==digest(cache['decoder'])
    items={item['path']:(i,item) for i,item in enumerate(cache['items'])}
    assert len(items)==len(cache['items'])
    image_rows=plan['queries']+[r for rows in plan['supports_by_category'].values() for r in rows]
    exposure=[]
    for row in image_rows:
        path=str((args.images/row['relative_image']).resolve())
        assert path in items
        index,item=items[path]
        assert item['category']==row['category']
        exposure.append(dict(source_id=row['source_id'], category=row['category'],
            role='query' if row in plan['queries'] else 'support', cache_index=index,
            teacher_split=item['split'], teacher_training_exposed=item['split']=='train',
            original_sha256=row['sha256']))
    stage2=dict(schema='jev-teacher-source-alignment-v1', status='complete',
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(), cache_sha256=cache_sha, manifest_sha256=manifest_sha,
        decoder_sha256=cache['decoder_sha256'], script_sha256=digest(__file__),
        counts_by_query_split=dict(Counter(x['teacher_split'] for x in exposure if x['role']=='query')),
        counts_by_support_split=dict(Counter(x['teacher_split'] for x in exposure if x['role']=='support')),
        unseen_definition='teacher_split != train; development diagnostic, not new blind test',
        tensor_content_access='Only approved 120 query maps indexed; no other cache map/CLS content evaluated',
        cache_integrity='Whole-file SHA and item manifest metadata verified',
        confirmation_opened=False, ground_truth_masks_opened=False, records=exposure)
    publish(root/'stages/02_teacher_alignment.json', stage2)
    versions={k:importlib.metadata.version(k) for k in ['Pillow','numpy','torch','requests']}
    protocol=dict(schema='jev-candidate-roi-v1', model='jev-1.14.0',
        query_plan_sha256=digest(root/'source_baseline/frozen/plan.json'),
        candidate_rule='map >= numpy.quantile(map, 0.95, method=linear); union bbox; never tune on GT',
        teacher_geometry='RGB direct resize 224; raw orientation; audited EXIF required to equal 1',
        asset_geometry='448x448; EXIF transpose; RGB; LANCZOS contain; centered white pad; PNG compress_level6',
        candidate_crop='floor xmin/ymin, ceil xmax/ymax in raw original pixels; clamp; no extra margin',
        roi_arms=['original_plus_candidate_crop','original_plus_duplicate_original'],
        normal_references=0, assets_per_request=2, keep_threshold=0.5, expected_targets=240,
        endpoint='https://api.modelgo.ai/v1/systemone', workers=3, connect_timeout=60, read_timeout=120,
        automatic_http_retries=0, max_started_per_target=3, explicit_retry_required=True,
        evaluation='All 60 anomaly denominators and all60 normal denominators; also teacher-unseen subset',
        exposure_caveat='Teacher-supervised training overlap is disclosed; all120 not unseen evaluation',
        ground_truth_gate='No GT masks until candidate and formal ROI prediction hashes frozen',
        confirmation_opened=False, versions=versions)
    publish(root/'stages/03_protocol.json', protocol)
    candidate_rows=[]
    exif_counts=Counter()
    for row in plan['queries']:
        path=(args.images/row['relative_image']).resolve()
        index,item=items[str(path)]
        probability=cache['maps'][index].detach().cpu().numpy()
        assert probability.shape==(224,224) and np.isfinite(probability).all()
        assert probability.min()>=0 and probability.max()<=1
        threshold=float(np.quantile(probability,.95,method='linear'))
        ys,xs=np.nonzero(probability>=threshold)
        norm=[float(xs.min()/224),float(ys.min()/224),float((xs.max()+1)/224),float((ys.max()+1)/224)]
        with Image.open(path) as raw:
            orientation=int(raw.getexif().get(274,1))
            exif_counts[orientation]+=1
            assert orientation==1, 'Nontrivial EXIF needs explicit coordinate transform'
            original=ImageOps.exif_transpose(raw).convert('RGB')
            width,height=original.size
            png,resized,pad=processed(original)
            bounds=[max(0,math.floor(norm[0]*width)),max(0,math.floor(norm[1]*height)),
                    min(width,math.ceil(norm[2]*width)),min(height,math.ceil(norm[3]*height))]
            assert bounds[2]>bounds[0] and bounds[3]>bounds[1]
            crop_png,crop_resized,crop_pad=processed(original.crop(tuple(bounds)))
        query_asset=exclusive_bytes(root/'assets/query'/f"{row['source_id']}.png",png)
        crop_asset=exclusive_bytes(root/'assets/crop'/f"{row['source_id']}.png",crop_png)
        for asset in [query_asset,crop_asset]:
            asset['path']=str(Path(asset['path']).relative_to(root))
        normalized_448=[(norm[0]*resized[0]+pad[0])/448,(norm[1]*resized[1]+pad[1])/448,
                        (norm[2]*resized[0]+pad[0])/448,(norm[3]*resized[1]+pad[1])/448]
        candidate_rows.append(dict(source_id=row['source_id'], category=row['category'],
            teacher_split=item['split'], cache_index=index, original_sha256=row['sha256'],
            map_sha256=hashlib.sha256(probability.tobytes()).hexdigest(), quantile_threshold=threshold,
            bbox_original_norm=norm, bbox_448_norm=normalized_448, crop_bounds_original=bounds,
            original_size=[width,height], resized_size=resized,padding=pad,
            crop_resized_size=crop_resized,crop_padding=crop_pad,exif_orientation=orientation,
            bbox_area_fraction=(norm[2]-norm[0])*(norm[3]-norm[1]),
            map_range=[float(probability.min()),float(probability.max())],
            assets=dict(query=query_asset,crop=crop_asset)))
    stage3=dict(schema='jev-fixed-candidates-v1', status='complete',
        protocol_sha256=digest(root/'stages/03_protocol.json'),
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        versions=versions,n_queries=len(candidate_rows), exif_orientation_counts=dict(exif_counts),
        full_image_boxes=sum(r['bbox_area_fraction']==1 for r in candidate_rows),
        confirmation_opened=False,ground_truth_masks_opened=False, records=candidate_rows)
    publish(root/'stages/03_candidates.json', stage3)
    print(json.dumps({k:v for k,v in stage2.items() if k!='records'}),flush=True)
    print(json.dumps({k:v for k,v in stage3.items() if k!='records'}),flush=True)

if __name__=='__main__':
    main()

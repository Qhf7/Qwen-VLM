"""Image-only ImageBind/BigDec cache for the frozen MVTec audit manifest."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(os.environ.get('ANOMALYGPT_REPO', '/root/autodl-tmp/AnomalyGPT'))
sys.path.insert(0, str(REPO / 'code/code/model'))
sys.path.insert(0, str(REPO))
from dec_big import BigDec, patch_features, atomic_json, atomic_save
from visa_pipeline import PIPELINE_VERSION, load_image_tensor


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--decoder', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    if args.out.exists() or args.out.with_suffix('.json').exists():
        raise FileExistsError('External cache output already exists')
    manifest_raw = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    records = [json.loads(line) for line in manifest_raw.splitlines() if line]
    if not records or len({r['source_id'] for r in records}) != len(records):
        raise ValueError('Invalid external image manifest')
    root = args.data_root.resolve()
    items = []
    for record in records:
        path = root / record['path']
        if sha256(path) != record['image_sha256']:
            raise ValueError('External image identity changed: ' + str(path))
        items.append(dict(path=str(path), category='industrial_component',
                          label=int(record['label']), split='external',
                          source_id=record['source_id'], category_original=record['category']))
    torch.set_num_threads(8)
    from ImageBind import imagebind_model
    from ImageBind.models.imagebind_model import ModalityType
    encoder, _ = imagebind_model.imagebind_huge({'layers': [7, 15, 23, 31]})
    encoder.load_state_dict(torch.load(REPO/'pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth', map_location='cpu'), strict=True)
    encoder.requires_grad_(False).half().cuda().eval()
    source = torch.load(args.decoder, map_location='cpu')
    if source['pipeline_version'] != PIPELINE_VERSION:
        raise ValueError('Teacher decoder pipeline mismatch')
    decoder = BigDec().float().cuda().eval()
    decoder.load_state_dict(source['dec'], strict=True)
    decoder.requires_grad_(False)
    cls = torch.empty(len(items), 1024, dtype=torch.float32)
    maps = torch.empty(len(items), 224, 224, dtype=torch.float32)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            end = min(start + args.batch_size, len(items))
            pixels = torch.cat([load_image_tensor(items[i]['path']) for i in range(start, end)]).half()
            encoded = encoder({ModalityType.VISION: pixels})['vision']
            feature = patch_features(encoded[1], end-start)
            probability = decoder(feature).sigmoid().squeeze(1)
            cls[start:end] = encoded[0].float().cpu()
            maps[start:end] = probability.float().cpu()
            if end % 128 < args.batch_size or end == len(items):
                print(json.dumps(dict(cached=end, total=len(items), elapsed=time.time()-started)), flush=True)
        singleton = []
        for index in (0, len(items)//2, len(items)-1):
            encoded = encoder({ModalityType.VISION: load_image_tensor(items[index]['path']).half()})['vision']
            probability = decoder(patch_features(encoded[1], 1)).sigmoid()[0, 0]
            map_error = float((probability.cpu()-maps[index]).abs().max())
            cls_error = float((encoded[0].float().cpu()[0]-cls[index]).abs().max())
            if map_error >= 0.02 or cls_error >= 0.01:
                raise AssertionError(('Batch/singleton mismatch', index, map_error, cls_error))
            singleton.append(dict(index=index, map_max_abs=map_error, cls_max_abs=cls_error))
    metadata = dict(schema='imagebind-bigdec-external-cache-v1', pipeline_version=PIPELINE_VERSION,
                    manifest_sha256=manifest_sha, decoder_sha256=sha256(args.decoder),
                    decoder_seed=source.get('teacher_seed'), geometry='direct-square224',
                    images=len(items), singleton_checks=singleton,
                    gt_masks_opened=False, images_for_training=False,
                    generation_or_qwen_queries=0, elapsed_seconds=time.time()-started)
    atomic_save(dict(**metadata, items=items, cls=cls, maps=maps), str(args.out))
    metadata['cache_sha256'] = sha256(args.out)
    atomic_json(metadata, str(args.out.with_suffix('.json')))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()

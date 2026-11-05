"""Cache one fixed repaired localization teacher for every migration arm."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = '/root/autodl-tmp/AnomalyGPT'
sys.path.insert(0, os.path.join(REPO, 'code/code/model'))
sys.path.insert(0, REPO)
from dec_big import BigDec, patch_features, atomic_json, atomic_save
from split_visa import splits
from visa_pipeline import PIPELINE_VERSION, load_image_tensor


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', default=REPO + '/runs/dec_big_fixed_s42/best.pt')
    ap.add_argument('--out', default=REPO + '/runs/migration_20261103/teacher_cache.pt')
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Existing teacher cache is preserved: ' + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    from ImageBind import imagebind_model
    from ImageBind.models.imagebind_model import ModalityType
    encoder, _ = imagebind_model.imagebind_huge({'layers': [7, 15, 23, 31]})
    encoder.load_state_dict(torch.load(REPO + '/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth', map_location='cpu'), strict=True)
    encoder.requires_grad_(False).half().cuda().eval()
    checkpoint = torch.load(args.decoder, map_location='cpu')
    assert checkpoint['pipeline_version'] == PIPELINE_VERSION
    decoder = BigDec().float().cuda().eval()
    decoder.load_state_dict(checkpoint['dec'], strict=True)
    decoder.requires_grad_(False)
    items = [dict(path=p, category=c, label=int(y), split=s)
             for s, rows in zip(('train', 'val', 'test'), splits()) for p, c, y in rows]
    n = len(items)
    maps = torch.empty(n, 224, 224, dtype=torch.float32)
    cls = torch.empty(n, 1024, dtype=torch.float32)
    begin = time.time()
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            chunk = items[start:start+args.batch_size]
            x = torch.cat([load_image_tensor(v['path']) for v in chunk]).half()
            encoded = encoder({ModalityType.VISION: x})['vision']
            features = patch_features(encoded[1], len(chunk))
            m = decoder(features).sigmoid().squeeze(1)
            c = encoded[0].float()
            assert m.shape == (len(chunk), 224, 224) and c.shape == (len(chunk), 1024)
            assert torch.isfinite(m).all() and torch.isfinite(c).all()
            maps[start:start+len(chunk)] = m.cpu()
            cls[start:start+len(chunk)] = c.cpu()
            if start % (args.batch_size * 50) == 0:
                print('[cache] %d/%d %.1fs' % (start+len(chunk), n, time.time()-begin), flush=True)
        # Recompute samples independently to catch batching/layout changes.
        checks = []
        for j in (0, n//2, n-1):
            encoded = encoder({ModalityType.VISION: load_image_tensor(items[j]['path']).half()})['vision']
            m = decoder(patch_features(encoded[1], 1)).sigmoid()[0, 0]
            err = float((m.cpu() - maps[j]).abs().max())
            cls_err = float((encoded[0].float().cpu()[0] - cls[j]).abs().max())
            checks.append(dict(index=j, map_max_abs=err, cls_max_abs=cls_err))
            # FP16 encoder kernels can differ slightly between batch sizes.
            assert err < 0.02 and cls_err < 0.01, checks
    manifest = dict(pipeline_version=PIPELINE_VERSION,
                    training_protocol='supervised_custom_60_20_20',
                    decoder=str(args.decoder), decoder_sha256=sha256(args.decoder),
                    teacher_seed=42, dtype='float32', geometry='square224',
                    cache_batch_size=args.batch_size, singleton_checks=checks,
                    counts={s: sum(v['split'] == s for v in items) for s in ('train','val','test')},
                    manifest_sha256=hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
                    elapsed_seconds=time.time()-begin)
    atomic_save(dict(**manifest, items=items, cls=cls, maps=maps), str(out))
    manifest['cache_sha256'] = sha256(str(out))
    atomic_json(manifest, str(out.with_suffix('.json')))
    print('[done]', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()

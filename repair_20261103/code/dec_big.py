"""ImageBind patch decoder with aligned VisA labels and validation-selected weights."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = os.environ.get('ANOMALYGPT_REPO', '/root/autodl-tmp/AnomalyGPT')
sys.path.insert(0, os.path.join(REPO, 'code/code/model'))
sys.path.insert(0, REPO)
from visa_pipeline import (PIPELINE_VERSION, load_image_tensor, load_mask_tensor,
                           mask_path_for_image, pixel_metrics)
from split_visa import splits


def patch_features(raw, batch_size):
    """ImageBind exposes [CLS+patch, batch, width], not batch first."""
    out = []
    for f in raw:
        if f.ndim != 3 or f.shape[0] != 257 or f.shape[1] != batch_size or f.shape[2] != 1280:
            raise ValueError('Unexpected ImageBind feature layout: %s, batch=%d' % (tuple(f.shape), batch_size))
        out.append(f.transpose(0, 1)[:, 1:, :].contiguous())
    if len(out) != 4:
        raise ValueError('Expected four ImageBind feature levels')
    return out


class BigDec(nn.Module):
    def __init__(self, in_dim=1280, d=512):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(in_dim, d) for _ in range(4)])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, feats):
        maps = []
        for layer, f in zip(self.proj, feats):
            if f.ndim != 3 or f.shape[1:] != (256, 1280):
                raise ValueError('Decoder requires [batch,256,1280], got %s' % (tuple(f.shape),))
            h = layer(f.float())
            maps.append(h.transpose(1, 2).reshape(h.shape[0], h.shape[2], 16, 16))
        if len(maps) != 4:
            raise ValueError('Decoder requires four feature levels')
        y = self.up(torch.stack(maps).mean(0))
        return F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)


def loss_fn(logit, gt):
    # Stable BCE and unclamped logits avoid a dead gradient at sigmoid saturation.
    bce = F.binary_cross_entropy_with_logits(logit, gt, reduction='none')
    p = logit.sigmoid()
    pt = p * gt + (1-p) * (1-gt)
    focal = ((1-pt).square() * bce).mean()
    dims = (1, 2, 3)
    dice = 1 - (2*(p*gt).sum(dims)+1)/(p.sum(dims)+gt.sum(dims)+1)
    return focal + dice.mean()


def atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def atomic_save(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def audit_data():
    summary = {'pipeline_version': PIPELINE_VERSION, 'protocol': 'supervised_custom_60_20_20'}
    groups = splits()
    sets = [set(x[0] for x in g) for g in groups]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for name, group in zip(('train','val','test'), groups):
        positives, empty = 0, 0
        cats = {}
        for p, c, lab in group:
            if lab:
                gt = load_mask_tensor(mask_path_for_image(p, lab), device='cpu')
                positives += 1
                empty += int(gt.sum() == 0)
            cats[c] = cats.get(c, 0)+1
        summary[name] = {'n':len(group), 'n_anomaly':positives,
                         'empty_anomaly_masks_after_resize':empty, 'categories':cats}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


class Features:
    def __init__(self, encoder, cache_gb=2.0):
        self.encoder = encoder
        self.limit = int(cache_gb * 1024**3)
        self.cache = OrderedDict()
        self.nbytes = 0
        self.layout_reported = False

    @torch.no_grad()
    def __call__(self, items):
        fresh = {p: self.cache[p] for p, _, _ in items if p in self.cache}
        missing = list(dict.fromkeys(p for p, _, _ in items if p not in fresh))
        if missing:
            from ImageBind.models.imagebind_model import ModalityType
            x = torch.cat([load_image_tensor(p) for p in missing]).half()
            raw = self.encoder({ModalityType.VISION:x})['vision'][1]
            feats = patch_features(raw, len(missing))
            if not self.layout_reported:
                print('[layout] raw=%s canonical=%s' % ([list(f.shape) for f in raw], [list(f.shape) for f in feats]), flush=True)
                self.layout_reported = True
            for j, p in enumerate(missing):
                fs = [f[j:j+1].cpu() for f in feats]
                if not all(torch.isfinite(f).all() for f in fs):
                    raise ValueError('Nonfinite encoder features')
                fresh[p] = fs
                size = sum(f.nelement()*f.element_size() for f in fs)
                if self.limit >= size:
                    while self.cache and self.nbytes+size > self.limit:
                        _, old = self.cache.popitem(last=False)
                        self.nbytes -= sum(f.nelement()*f.element_size() for f in old)
                    self.cache[p] = fs
                    self.nbytes += size
        result = []
        for p, _, _ in items:
            fs = fresh[p]
            if p in self.cache:
                self.cache.move_to_end(p)
            result.append(fs)
        return [torch.cat([fs[k] for fs in result]).cuda() for k in range(4)]


def targets(items):
    return torch.stack([load_mask_tensor(mask_path_for_image(p, lab)) for p, _, lab in items])


@torch.no_grad()
def evaluate(dec, feats, items, batch_size):
    dec.eval()
    records = []
    total_loss = 0.0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start+batch_size]
        gt = targets(chunk)
        logits = dec(feats(chunk))
        assert logits.shape == gt.shape == (len(chunk), 1, 224, 224)
        total_loss += float(loss_fn(logits, gt))*len(chunk)
        pred = logits.sigmoid().cpu().numpy()
        truths = gt.cpu().numpy()
        for (p,c,lab), g, s in zip(chunk, truths, pred):
            records.append((c, lab, g[0], s[0]))
    metrics = pixel_metrics(records)
    metrics['loss'] = total_loss/len(items)
    dec.train()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-every', type=int, default=200)
    ap.add_argument('--cache-gb', type=float, default=2.0)
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--split', choices=['val','test'], default='test')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--audit-data', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default='runs/dec_big_fixed_s42')
    a = ap.parse_args()
    out = os.path.join(REPO, a.out)
    os.makedirs(out, exist_ok=True)
    if a.audit_data:
        atomic_json(audit_data(), os.path.join(out, 'data_audit.json'))
        return
    if a.steps < 1 or a.bs < 1 or a.val_every < 1 or a.cache_gb < 0:
        raise ValueError('Positive steps, batch size, validation interval; nonnegative cache required')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from ImageBind import imagebind_model
    from ImageBind.models.imagebind_model import ModalityType
    enc, _ = imagebind_model.imagebind_huge({'layers':[7,15,23,31]})
    enc.load_state_dict(torch.load(os.path.join(REPO, 'pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth'), map_location='cpu', weights_only=True), strict=True)
    enc.half().cuda().eval()
    for p in enc.parameters():
        p.requires_grad = False
    dec = BigDec().cuda()
    feats = Features(enc, a.cache_gb)
    tr, va, te = splits()
    meta = {'pipeline_version':PIPELINE_VERSION, 'protocol':'supervised_custom_60_20_20',
            'seed':a.seed, 'config':vars(a), 'parameters':sum(p.numel() for p in dec.parameters()),
            'feature_layout':'ImageBind [257,B,1280] -> [B,256,1280], CLS removed',
            'selection':'validation macro pixel AUROC',
            'train_n':len(tr), 'val_n':len(va), 'test_n':len(te)}
    print('[config]', json.dumps(meta), flush=True)
    if a.eval:
        ckpt = a.ckpt or os.path.join(out,'best.pt')
        ck = torch.load(ckpt, map_location='cpu', weights_only=True)
        if ck.get('pipeline_version') != PIPELINE_VERSION:
            raise ValueError('Checkpoint is from a different or unversioned preprocessing pipeline')
        dec.load_state_dict(ck['dec'], strict=True)
        result = evaluate(dec, feats, va if a.split=='val' else te, a.bs)
        result.update({'split':a.split, 'checkpoint':ckpt, 'step':ck['step'], 'protocol':meta['protocol']})
        atomic_json(result, os.path.join(out,'metrics.json'))
        print('[metrics]',json.dumps(result),flush=True)
        return
    if os.path.exists(os.path.join(out,'final.pt')) or os.path.exists(os.path.join(out,'best.pt')):
        raise FileExistsError('Choose a fresh output directory; existing checkpoints are preserved')
    atomic_json(meta, os.path.join(out,'run.json'))
    tr_n=[x for x in tr if not x[2]]; tr_a=[x for x in tr if x[2]]
    if a.smoke:
        # Fixed real training samples only, two classes, normal/abnormal balanced.
        fixed=[]
        for c in ('candle','capsules'):
            for lab in (0,1):
                pool=[x for x in tr if x[1]==c and x[2]==lab]
                fixed.append(next(x for x in pool if not lab or targets([x]).sum()>0))
        smoke_fs=feats(fixed); smoke_gt=targets(fixed)
        before=float(loss_fn(dec(smoke_fs),smoke_gt))
        opt=torch.optim.AdamW(dec.parameters(),lr=min(a.lr,3e-4),weight_decay=1e-4)
        for step in range(a.steps):
            opt.zero_grad(set_to_none=True)
            logits=dec(smoke_fs); loss=loss_fn(logits,smoke_gt)
            assert logits.shape==(4,1,224,224)
            if not torch.isfinite(loss): raise ValueError('Nonfinite smoke loss')
            loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(dec.parameters(),1.0))
            if not np.isfinite(grad) or grad<=0: raise ValueError('Missing/nonfinite gradients')
            opt.step()
            if (step+1)%10==0: print('[smoke %d/%d] loss %.6f grad %.6f'%(step+1,a.steps,float(loss),grad),flush=True)
        with torch.no_grad(): after=float(loss_fn(dec(smoke_fs),smoke_gt))
        smoke_metrics=evaluate(dec,feats,fixed,4)
        result={'pipeline_version':PIPELINE_VERSION,'initial_loss':before,'final_loss':after,
                'reduction':before-after, 'passed':(after<before-0.03 and
                smoke_metrics['pixel_auroc_macro'] >= 0.9 and smoke_metrics['pixel_ap_macro'] >= 0.1),
                'metrics':smoke_metrics}
        atomic_json(result,os.path.join(out,'smoke.json'))
        print('[smoke_result]',json.dumps(result),flush=True)
        if not result['passed']: raise RuntimeError('Small-set learning check failed')
        return
    opt=torch.optim.AdamW(dec.parameters(),lr=a.lr,weight_decay=1e-4)
    # Cosine schedule works for short smoke runs too; no zero-length OneCycle warmup.
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.steps,eta_min=a.lr*1e-3)
    best=-1.0; t0=time.time()
    for step in range(1,a.steps+1):
        dec.train(); opt.zero_grad(set_to_none=True)
        batch=[random.choice(tr_a if random.random()<0.5 else tr_n) for _ in range(a.bs)]
        gt=targets(batch); logits=dec(feats(batch))
        assert logits.shape==gt.shape==(len(batch),1,224,224)
        loss=loss_fn(logits,gt)
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(dec.parameters(),1.0))
        if not np.isfinite(grad): raise ValueError('Nonfinite gradient')
        opt.step(); sched.step()
        if step==1 or step%20==0:
            print('[step %d/%d] loss %.6f lr %.3g grad %.4f elapsed %.1fs'%(step,a.steps,float(loss),sched.get_last_lr()[0],grad,time.time()-t0),flush=True)
        if step%a.val_every==0 or step==a.steps:
            val=evaluate(dec,feats,va,a.bs)
            print('[val %d] %s'%(step,json.dumps(val)),flush=True)
            record={'step':step,**val}
            with open(os.path.join(out,'validation.jsonl'),'a') as f: f.write(json.dumps(record)+'\n')
            ck={'dec':dec.state_dict(),'step':step,'pipeline_version':PIPELINE_VERSION,
                'seed':a.seed,'val_metrics':val,'config':vars(a),'protocol':meta['protocol']}
            if val['pixel_auroc_macro']>best:
                best=val['pixel_auroc_macro']; atomic_save(ck,os.path.join(out,'best.pt'))
                print('[save] best validation AUROC %.6f at %d'%(best,step),flush=True)
            if step==a.steps: atomic_save(ck,os.path.join(out,'final.pt'))
    atomic_json({'steps':a.steps,'best_val_pixel_auroc':best,'elapsed_seconds':time.time()-t0,
                 'pipeline_version':PIPELINE_VERSION},os.path.join(out,'train_complete.json'))
    print('[done]',flush=True)


if __name__=='__main__':
    main()

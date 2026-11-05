#!/bin/bash
set -Eeuo pipefail
cd /root/autodl-tmp/AnomalyGPT
PY=/root/autodl-tmp/conda-envs/anomalygpt/bin/python
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
stage=startup
trap 'rc=$?; echo "FAILED stage=$stage rc=$rc at=$(date -Is)"; exit "$rc"' ERR
echo "START $(date -Is)"
stage=integration_smoke
$PY -u train_vicuna_clean.py --decoder-kind upsample --decoder-init runs/dec_big_fixed_s42/best.pt --freeze-decoder --steps 30 --accum 2 --val-every 30 --val-limit 24 --out runs/vic_fixed_smoke_20261103
stage=smoke_gate
$PY - <<'PY'
import torch
ck=torch.load('runs/vic_fixed_smoke_20261103/final.pt',map_location='cpu')
original=torch.load('runs/dec_big_fixed_s42/best.pt',map_location='cpu')
assert ck['step']==30 and ck['decoder_kind']=='upsample' and ck['freeze_decoder']
assert all(torch.equal(v,original['dec'][k]) for k,v in ck['dec'].items())
assert all(torch.isfinite(v).all() for name in ['prompt_learner','llama_proj'] for v in ck[name].values())
assert ck['val_auroc'] if 'val_auroc' in ck else True
print('INTEGRATION_SMOKE_PASSED_FROZEN_DECODER_IDENTICAL')
PY
stage=full_training
$PY -u train_vicuna_clean.py --decoder-kind upsample --decoder-init runs/dec_big_fixed_s42/best.pt --freeze-decoder --seed 42 --steps 1500 --accum 4 --val-every 300 --val-limit 300 --out runs/vic_fixed_s42
stage=full_test
$PY -u train_vicuna_clean.py --eval --ckpt runs/vic_fixed_s42/best.pt --out runs/eval_vic_fixed_s42
stage=verify_delivery
$PY - <<'PY'
import torch,json,datetime
best=torch.load('runs/vic_fixed_s42/best.pt',map_location='cpu')
original=torch.load('runs/dec_big_fixed_s42/best.pt',map_location='cpu')
assert all(torch.equal(v,original['dec'][k]) for k,v in best['dec'].items())
pixel=json.load(open('runs/dec_big_fixed_s42/metrics.json'))
image=json.load(open('runs/eval_vic_fixed_s42/metrics.json'))
result={'time':datetime.datetime.now().isoformat(),'pipeline_version':best['pipeline_version'],
        'training_protocol':best['training_protocol'],'seed':best['seed'],'steps_completed':1500,
        'selected_step':best['step'],'selected_val_auroc':best['val_auroc'],
        'frozen_decoder_verified_identical':True,'pixel_metrics':pixel,'image_metrics':image}
with open('runs/vic_fixed_delivery_20261103.json','w') as f:json.dump(result,f,indent=2)
print('VERIFIED_DELIVERY',json.dumps({k:v for k,v in result.items() if k not in ['pixel_metrics','image_metrics']}))
print('IMAGE_METRICS',json.dumps(image))
PY
echo "DONE $(date -Is)"

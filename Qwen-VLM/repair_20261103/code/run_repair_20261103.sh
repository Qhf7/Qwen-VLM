#!/bin/bash
set -Eeuo pipefail
cd /root/autodl-tmp/AnomalyGPT
PY=/root/autodl-tmp/conda-envs/qwen3vl2/bin/python
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
stage=startup
trap 'rc=$?; echo "FAILED stage=$stage rc=$rc at=$(date -Is)"; exit "$rc"' ERR
echo "START $(date -Is)"
stage=preflight
$PY -m py_compile visa_pipeline.py dec_big.py train_native_seg.py train_vicuna_clean.py qwen3vl_native_bridge.py eval_native.py eval_pixel.py
$PY - <<'PY'
import json
r=json.load(open('runs/dec_fixed_smoke_20261103/smoke.json'))
assert r['passed'] and r['metrics']['pixel_auroc_macro']>=0.9 and r['metrics']['pixel_ap_macro']>=0.1, r
print('SMOKE_GATE_PASSED',r['initial_loss'],r['final_loss'])
PY
stage=decoder_training
$PY -u dec_big.py --steps 1200 --bs 4 --seed 42 --val-every 200 --cache-gb 2 --out runs/dec_big_fixed_s42
stage=decoder_test
$PY -u dec_big.py --eval --split test --bs 4 --out runs/dec_big_fixed_s42
stage=old_decoder_recheck
$PY -u eval_pixel.py --ckpt runs/native_clean/best.pt --split test --out runs/eval_pixel_old_corrected_20261103
stage=summary
$PY - <<'PY'
import json,datetime
new=json.load(open('runs/dec_big_fixed_s42/metrics.json'))
old=json.load(open('runs/eval_pixel_old_corrected_20261103/metrics.json'))
keys=['pixel_auroc_macro','pixel_ap_macro','pixel_auroc_weighted','image_auroc_from_map99']
r={'time':datetime.datetime.now().isoformat(),'pipeline_version':new['pipeline_version'],
   'protocol':'supervised_custom_60_20_20','old_checkpoint_corrected_evaluation':old,
   'retrained_corrected_decoder':new,'differences':{k:new[k]-old[k] for k in keys}}
with open('runs/repair_20261103_summary.json','w') as f:json.dump(r,f,indent=2)
print('CORRECTED_COMPARISON',json.dumps({k:{'old':old[k],'new':new[k],'delta':new[k]-old[k]} for k in keys}))
PY
echo "DONE $(date -Is)"

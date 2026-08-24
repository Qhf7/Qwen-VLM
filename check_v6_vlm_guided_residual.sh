#!/usr/bin/env bash
set -euo pipefail
ssh -i "${HOME}/.ssh/id_ed25519" -o IdentitiesOnly=yes -p 21091 root@connect.westc.seetacloud.com 'bash -s' <<'REMOTE'
set -u
STATE=/root/autodl-tmp/vlm_guided_residual_v6_state.txt
echo TIME; date
[[ -s "$STATE" ]] || { echo STATE_MISSING; exit 0; }
cat "$STATE"; . "$STATE"
echo GPU; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo PROCESSES; pgrep -af '[t]rain_v6.py|[p]roxy_v6.py|[i]nfer_v6.py|[r]un_v6_pipeline.sh' || true
echo SCREEN; screen -list | grep -F "$SCREEN_NAME" || true
echo STAGE
if grep -q '^VALID:' "$REMOTE_RUN/validate.log" 2>/dev/null; then
  echo 'complete 100%'
elif pgrep -af '[i]nfer_v6.py' | grep -F "$REMOTE_RUN" >/dev/null; then echo inference
elif pgrep -af '[p]roxy_v6.py' | grep -F "$REMOTE_RUN" >/dev/null; then echo proxy
elif pgrep -af '[t]rain_v6.py' | grep -F "$REMOTE_RUN/final" >/dev/null; then echo final_training
elif pgrep -af '[t]rain_v6.py' | grep -F "$REMOTE_RUN/fold0" >/dev/null; then echo fold_training
else echo inactive_or_between_stages; fi
echo TRAIN_PROGRESS
for path in "$REMOTE_RUN/fold0/checkpoints/train_progress.json" "$REMOTE_RUN/final/checkpoints/train_progress.json"; do
  [[ ! -s "$path" ]] || { echo "--- $path"; cat "$path"; }
done
echo NOTE
echo 'The current legacy progress JSON may expose total_steps as total until the next v6 process starts; V6_PROGRESS log lines are authoritative for the running process.'
echo LOG_TAIL; tail -60 "$REMOTE_RUN/screen.log" 2>/dev/null || true
echo PROXY_RESULTS
for path in "$REMOTE_RUN"/fold0/proxy_epoch[0-9].json; do
  [[ ! -s "$path" ]] || { echo "--- $path"; cat "$path"; }
done
echo SELECTION; cat "$REMOTE_RUN/fold0/selection.json" 2>/dev/null || true
echo INFER; tail -30 "$REMOTE_RUN/infer.log" 2>/dev/null || true
echo VALIDATE; cat "$REMOTE_RUN/validate.log" 2>/dev/null || true
echo OUTPUTS; find "$REMOTE_RUN" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -50
REMOTE

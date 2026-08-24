#!/usr/bin/env bash
set -u

HOST="root@connect.westc.seetacloud.com"
PORT="21091"
KEY="${HOME}/.ssh/id_ed25519"
INTERVAL="${1:-10}"
ONCE=0
if [[ "$INTERVAL" == "--once" ]]; then
  ONCE=1
  INTERVAL=10
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 2 )); then
  echo "Usage: bash watch_v6_progress.sh [refresh_seconds|--once]" >&2
  exit 2
fi

cleanup() {
  printf '\nLocal monitor stopped. The remote training process is still running.\n'
}
trap cleanup INT TERM

while true; do
  snapshot="$({
    ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=8 -p "$PORT" "$HOST" 'bash -s' <<'REMOTE'
set -u
STATE=/root/autodl-tmp/vlm_guided_residual_v6_state.txt
if [[ ! -s "$STATE" ]]; then
  echo "STATUS=state_missing"
  exit 0
fi
. "$STATE"
now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
gpu="$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"
stage=inactive
active_log="$REMOTE_RUN/screen.log"
if grep -q '^VALID:' "$REMOTE_RUN/validate.log" 2>/dev/null; then
  stage=complete
elif pgrep -af '[i]nfer_v6.py' | grep -F "$REMOTE_RUN" >/dev/null; then
  stage=final_reference_calibration_or_inference
  active_log="$REMOTE_RUN/infer.log"
elif pgrep -af '[p]roxy_v6.py' | grep -F "$REMOTE_RUN" >/dev/null; then
  stage=heldout_proxy
  active_log="$(find "$REMOTE_RUN/fold0" -maxdepth 1 -type f -name 'proxy_epoch*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
elif pgrep -af '[t]rain_v6.py' | grep -F "$REMOTE_RUN/final" >/dev/null; then
  stage=final_training
  active_log="$REMOTE_RUN/final/train.log"
elif pgrep -af '[t]rain_v6.py' | grep -F "$REMOTE_RUN/fold0" >/dev/null; then
  stage=fold_training
  active_log="$REMOTE_RUN/fold0/train.log"
fi

overall="$(grep -h '^PROGRESS ' "$active_log" 2>/dev/null | tail -1)"
stage_bar="$(grep -h '^V6_PROGRESS ' "$active_log" 2>/dev/null | tail -1)"
process="$(pgrep -af '[t]rain_v6.py|[p]roxy_v6.py|[i]nfer_v6.py' | grep -F "$REMOTE_RUN" | head -1)"
pid="$(printf '%s\n' "$process" | awk '{print $1}')"
process_detail="none"
if [[ -n "$pid" ]]; then
  process_detail="$(ps -o stat=,etime=,pcpu=,pmem=,rss= -p "$pid" 2>/dev/null | xargs)"
fi
proxy_done="$(find "$REMOTE_RUN/fold0" -maxdepth 1 -type f -name 'proxy_epoch[0-9].json' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
fold_epoch1=no; fold_epoch2=no; final_model=no; zip=no; valid=no
[[ -s "$REMOTE_RUN/fold0/checkpoints/epoch1.pt" ]] && fold_epoch1=yes
[[ -s "$REMOTE_RUN/fold0/checkpoints/epoch2.pt" ]] && fold_epoch2=yes
[[ -s "$REMOTE_RUN/final/checkpoints/vlm_guided_residual_v6.pt" ]] && final_model=yes
[[ -s "$REMOTE_RUN/submission_vlm_guided_residual_v6.zip" ]] && zip=yes
grep -q '^VALID:' "$REMOTE_RUN/validate.log" 2>/dev/null && valid=yes
selection="$(python3 - "$REMOTE_RUN/fold0/selection.json" 2>/dev/null <<'PY' || true
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if p.exists():
    x = json.loads(p.read_text())
    print(f"best_epoch={x.get('best_epoch')} selection={x.get('best_selection')}")
PY
)"
mtime="unknown"
age="unknown"
if [[ -s "$active_log" ]]; then
  mtime="$(date -r "$active_log" '+%Y-%m-%d %H:%M:%S')"
  age="$(( $(date +%s) - $(stat -c %Y "$active_log") ))s"
fi

echo "STATUS=ok"
echo "TIME=$now"
echo "STAGE=$stage"
echo "GPU=$gpu"
echo "PROCESS=$process_detail"
echo "OVERALL=$overall"
echo "STAGE_BAR=$stage_bar"
echo "LOG_UPDATED=$mtime age=$age"
echo "CHECKPOINTS=fold_epoch1:$fold_epoch1 fold_epoch2:$fold_epoch2 proxies:$proxy_done/3 final_model:$final_model zip:$zip valid:$valid"
echo "SELECTION=${selection:-pending}"
echo "RUN=$REMOTE_RUN"
if [[ "$stage" == "complete" ]]; then echo "DONE=1"; else echo "DONE=0"; fi
REMOTE
  } 2>&1)"
  rc=$?

  if [[ -t 1 ]]; then
    printf '\033[2J\033[H'
  fi
  printf 'VLM-Guided Residual v6 - live remote monitor\n'
  printf 'Refresh: every %ss | Stop monitor only: Ctrl-C\n' "$INTERVAL"
  printf '%s\n' '--------------------------------------------------------------------------'
  if (( rc != 0 )); then
    printf 'SSH check failed (exit %d):\n%s\n' "$rc" "$snapshot"
  else
    printf '%s\n' "$snapshot" | sed \
      -e 's/^TIME=/Time:        /' \
      -e 's/^STAGE=/Stage:       /' \
      -e 's/^GPU=/GPU:         /' \
      -e 's/^PROCESS=/Process:     /' \
      -e 's/^OVERALL=/Overall:     /' \
      -e 's/^STAGE_BAR=/Stage bar:   /' \
      -e 's/^LOG_UPDATED=/Log update:  /' \
      -e 's/^CHECKPOINTS=/Pipeline:    /' \
      -e 's/^SELECTION=/Selection:   /' \
      -e 's/^RUN=/Remote run:  /' \
      -e '/^STATUS=/d' \
      -e '/^DONE=/d'
    printf '%s\n' '--------------------------------------------------------------------------'
    printf '%s\n' 'Remaining-time guide (historical same-server measurements):'
    printf '%s\n' '  current fold training              use the live Overall ETA shown above'
    printf '%s\n' '  each held-out proxy                about 3.5-4.2 h, three total'
    printf '%s\n' '  final training after selection     about 0-3.5 h (best epoch 0/1/2)'
    printf '%s\n' '  final bank + calibration + infer   about 9-11 h'
    printf '%s\n' '  estimated complete pipeline        about 20-26 h remaining'
  fi

  (( ONCE )) && exit "$rc"
  printf '%s\n' "$snapshot" | grep -q '^DONE=1$' && exit 0
  sleep "$INTERVAL"
done

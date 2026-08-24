#!/usr/bin/env bash
set -euo pipefail
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST="root@connect.westc.seetacloud.com"; PORT=21091; KEY="${HOME}/.ssh/id_ed25519"
STATE_VALUES="$(ssh -i "$KEY" -o IdentitiesOnly=yes -p "$PORT" "$HOST" 'cat /root/autodl-tmp/vlm_guided_residual_v6_state.txt')"
REMOTE_RUN="$(printf '%s\n' "$STATE_VALUES" | sed -n 's/^REMOTE_RUN=//p')"
[[ -n "$REMOTE_RUN" ]] || { echo REMOTE_RUN_MISSING >&2; exit 2; }
ssh -i "$KEY" -o IdentitiesOnly=yes -p "$PORT" "$HOST" \
  "grep -q '^VALID:' '$REMOTE_RUN/validate.log' && test -s '$REMOTE_RUN/submission_vlm_guided_residual_v6.zip'"
NAME="v6_vlm_guided_residual_results_$(basename "$REMOTE_RUN")"
TARGET="$LOCAL_ROOT/$NAME"
[[ ! -e "$TARGET" ]] || { echo "refusing existing local target: $TARGET" >&2; exit 3; }
mkdir "$TARGET"
scp -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" \
  "$HOST:$REMOTE_RUN/submission_vlm_guided_residual_v6.zip" "$TARGET/"
for file in infer.log validate.log screen.log; do
  scp -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" "$HOST:$REMOTE_RUN/$file" "$TARGET/" 2>/dev/null || true
done
scp -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" \
  "$HOST:$REMOTE_RUN/submission_vlm_guided_residual_v6/run_meta.json" "$TARGET/" 2>/dev/null || true
shasum -a 256 "$TARGET/submission_vlm_guided_residual_v6.zip"
echo "DOWNLOADED=$TARGET"

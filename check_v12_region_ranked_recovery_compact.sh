#!/usr/bin/env bash
set -euo pipefail
HOST=${HOST:-root@connect.westc.seetacloud.com}; PORT=${PORT:-21091}; RUN=${REMOTE_RUN:?set REMOTE_RUN}
ssh -p "$PORT" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "$HOST" bash -s -- "$RUN" <<'REMOTE'
set -euo pipefail
RUN=$1; echo "TIME=$(date '+%F %T %Z')"; echo "RUN=$RUN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
ps -eo pid,etime,pcpu,pmem,rss,state,args | grep -E 'v12_region_ranked_recovery_compact|lowmem_reference_v12|infer_stream_memory' | grep -v grep || true
grep -E 'V12_RECOVERY|memory [0-9]+/|calibrate [0-9]+/|test [0-9]+/|Killed|COMPLETE|ERROR' "$RUN/pipeline.log" 2>/dev/null | tail -35 || true
find "$RUN" -maxdepth 4 -type f \( -name '*.pt' -o -name '*.zip' -o -name '*.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -25 || true
REMOTE

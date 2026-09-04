#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/vlm
source .venv/bin/activate

passes_gate() {
  local out_dir="$1"
  python - "$out_dir" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
for path in sorted(out_dir.glob("step*_metrics.json")):
    metrics = json.loads(path.read_text())
    if (
        metrics["macro_aupr"] >= 0.7795
        and metrics["macro_f1max"] >= 0.7380
        and metrics["small_defect_recall"] >= 0.988095
        and metrics["predicted_area"]["ok_view_mean"] <= 0.10
        and metrics["predicted_area"]["view_p95"] <= 0.25
    ):
        print(f"[gate] pass: {path}")
        raise SystemExit(0)
raise SystemExit(1)
PY
}

while pgrep -f "e1.train_fold1.*e1_fold1_ablation_c_layer15" >/dev/null; do
  sleep 30
done

if passes_gate artifacts/e1_fold1_ablation_c_layer15; then
  echo "[sweep] layer15 passed; stop remaining layer experiments"
  exit 0
fi

for layer in 23 31; do
  out_dir="artifacts/e1_fold1_ablation_c_layer${layer}"
  mkdir -p "$out_dir"
  python -m e1.train_fold1 \
    --model-id models/Qwen2.5-VL-3B-Instruct \
    --cache-root Train448 \
    --dev-cache-root Dataset448 \
    --gt-root Dataset \
    --out-dir "$out_dir" \
    --empty-mask-dice-only-nonempty \
    --lambda-seg 0.1 \
    --feature-layers "$layer" \
    > "$out_dir/run.log" 2>&1

  if passes_gate "$out_dir"; then
    echo "[sweep] layer${layer} passed; stop remaining layer experiments"
    exit 0
  fi
done

echo "[sweep] layer15/layer23/layer31 all completed without a fold1 gate pass"

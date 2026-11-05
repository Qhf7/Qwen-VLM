"""Wait for GPU stages, then verify and render real results; no final archive overwrite."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/multitask_v2"
PYTHON = "/root/autodl-tmp/vlm/.venv/bin/python"
STATE = EXP / "report_state.json"


def state(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_unix": time.time(), **value}, indent=2) + "\n")
    temporary.replace(STATE)


def main():
    deadline = time.monotonic() + 2 * 60 * 60
    state({"status": "waiting", "stage": "waiting_for_gpu_stages"})
    while True:
        states = [json.loads((EXP / name).read_text()) if (EXP / name).exists()
                  else {"status": "not_started"} for name in ("state.json", "scoring_state.json")]
        if any(item["status"] == "failed" for item in states):
            raise RuntimeError("A GPU pipeline failed; results not finalized")
        if all(item["status"] == "complete" for item in states):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("GPU completion not observed within two hours")
        time.sleep(15)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"}
    for script in ("verify_multitask_v2.py", "analyze_multitask_v2.py",
                   "plot_multitask_v2.py", "analyze_scores_v2.py"):
        state({"status": "running", "stage": script})
        with (EXP / "logs" / (script + ".log")).open("a") as output:
            subprocess.run([PYTHON, "-u", str(ROOT / "ops" / script)], env=env,
                           cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    state({"status": "complete", "stage": "verification_and_result_reports",
           "confirmation_evaluated": False, "archive_pending_final_interpretation": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        old = json.loads(STATE.read_text()) if STATE.exists() else {}
        state({**old, "status": "failed", "error": str(exc)})
        raise

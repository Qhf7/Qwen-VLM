"""Bounded preparation wait followed by validation and the fixed pilot."""
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for attempt in range(300):
    if (root / "prepared/visa/audit_report.json").exists():
        break
    time.sleep(2)
else:
    raise RuntimeError("VisA preparation did not complete within ten minutes; inspect preparation log")
subprocess.run([sys.executable, str(root / "ops/validate_prepared.py")], check=True)
subprocess.run([sys.executable, "-u", str(root / "ops/run_pipeline.py")], check=True)

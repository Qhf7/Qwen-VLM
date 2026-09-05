#!/usr/bin/env python3
"""One-shot Codex repair after switching accounts or model providers.

The command runs both safeguards used by this project:

1. Synchronize historical rollout/SQLite ``model_provider`` values with the
   provider selected in ``config.toml``.
2. Remove replay-only ``rs_*`` reasoning items that have no encrypted content
   and therefore cannot be sent to a stateless Responses backend.

It is a dry-run unless ``--apply`` is supplied. Close Codex first so it does
not keep an in-memory copy of a session while the files are being repaired.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
PROVIDER_SCRIPT = TOOLS_DIR / "repair_codex_session_provider.py"
REASONING_SCRIPT = TOOLS_DIR / "repair_codex_unpersisted_reasoning.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Codex provider mappings and stale rs_* reasoning items."
    )
    parser.add_argument("--codex-home", type=Path, help="Codex data directory")
    parser.add_argument("--provider", help="Override target provider id")
    parser.add_argument("--from-provider", action="append", default=[], help="Only remap this provider (repeatable)")
    parser.add_argument("--provider-state-db", type=Path, help="Override state_*.sqlite used for provider mappings")
    parser.add_argument("--history-db", type=Path, help="Override thread_history_*.sqlite used for stale-item cleanup")
    parser.add_argument("--session-id", help="Limit stale-reasoning cleanup to one session")
    parser.add_argument("--include-locked", action="store_true", help="Allow repair of an inactive locked session")
    parser.add_argument("--apply", action="store_true", help="Apply changes; otherwise only report planned changes")
    return parser.parse_args()


def run(command: list[str]) -> int:
    print("$ " + " ".join(map(str, command)))
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parse_args()
    common: list[str] = []
    if args.codex_home:
        common += ["--codex-home", str(args.codex_home)]
    if args.provider_state_db:
        common += ["--state-db", str(args.provider_state_db)]
    if args.apply:
        common.append("--apply")

    provider = [sys.executable, str(PROVIDER_SCRIPT), *common]
    if args.provider:
        provider += ["--provider", args.provider]
    for source in args.from_provider:
        provider += ["--from-provider", source]
    code = run(provider)
    if code:
        return code

    reasoning = [sys.executable, str(REASONING_SCRIPT)]
    if args.codex_home:
        reasoning += ["--codex-home", str(args.codex_home)]
    if args.history_db:
        reasoning += ["--state-db", str(args.history_db)]
    if args.session_id:
        reasoning += ["--session-id", args.session_id]
    if args.include_locked:
        reasoning.append("--include-locked")
    if args.apply:
        reasoning.append("--apply")
    return run(reasoning)


if __name__ == "__main__":
    raise SystemExit(main())

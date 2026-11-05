"""Run project commands over an already authenticated SSH control connection.

Authentication stays in SSH; no passwords are stored in this project.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("INDUSTRIAL_VLM_SSH_HOST"))
    parser.add_argument("--port", default=os.environ.get("INDUSTRIAL_VLM_SSH_PORT", "22"))
    parser.add_argument("--control", default=os.environ.get(
        "INDUSTRIAL_VLM_SSH_CONTROL", "/tmp/industrial_vlm_ssh_%C"))
    parser.add_argument("--script", type=Path)
    parser.add_argument("--upload", nargs=2, metavar=("LOCAL", "REMOTE"))
    parser.add_argument("--download", nargs=2, metavar=("REMOTE", "LOCAL"))
    parser.add_argument("command", nargs="?")
    args = parser.parse_args()
    if not args.host:
        parser.error("provide --host or set INDUSTRIAL_VLM_SSH_HOST")
    ssh = ["ssh", "-o", f"ControlPath={args.control}", "-p", args.port, args.host]
    if args.upload or args.download:
        src, dst = args.upload or args.download
        remote_shell = shlex.join(ssh[:-1])
        if args.upload:
            dst = args.host + ":" + dst
        else:
            src = args.host + ":" + src
        subprocess.run(["rsync", "-a", "--partial", "-e", remote_shell, src, dst], check=True)
    elif args.script:
        subprocess.run(ssh + ["bash -s"], input=args.script.read_bytes(), check=True)
    elif args.command:
        subprocess.run(ssh + [args.command], check=True)
    else:
        parser.error("provide command, --script, --upload, or --download")


if __name__ == "__main__":
    main()

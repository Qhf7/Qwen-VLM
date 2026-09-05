#!/usr/bin/env python3
"""Repair Codex history after switching accounts or model providers.

The Codex desktop app stores a session's model provider in two places:

1. The first ``session_meta`` record in each rollout JSONL file.
2. ``threads.model_provider`` in the newest ``state_*.sqlite`` database.

This utility makes those values agree with the provider selected by
``$CODEX_HOME/config.toml``. It is dry-run by default. Every applied repair
creates a rollback manifest plus a consistent SQLite backup under
``$CODEX_HOME/provider-repair-backups``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPT_VERSION = 1
BUILTIN_PROVIDERS = {"openai", "ollama", "lmstudio", "amazon-bedrock"}
PROVIDER_FIELD_RE = re.compile(
    rb'("model_provider"\s*:\s*)("(?:\\.|[^"\\])*")'
)


class RepairError(RuntimeError):
    """A user-actionable repair failure."""


@dataclass(frozen=True)
class RolloutRecord:
    path: Path
    relative_path: str
    session_id: str
    provider: str
    head: bytes
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ThreadRecord:
    thread_id: str
    provider: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize historical Codex session providers with the provider "
            "currently selected in config.toml. Runs as a dry-run unless "
            "--apply is supplied."
        )
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        help="Codex data directory (default: $CODEX_HOME, otherwise ~/.codex)",
    )
    parser.add_argument(
        "--provider",
        help="Target provider id (default: model_provider from config.toml)",
    )
    parser.add_argument(
        "--from-provider",
        action="append",
        default=[],
        help=(
            "Only replace this old provider. Repeat for multiple ids. "
            "By default, replace every provider different from the target."
        ),
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        help="State database path (default: highest-numbered state_*.sqlite)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned repair. Without this flag, only report changes.",
    )
    parser.add_argument(
        "--include-locked",
        action="store_true",
        help="Allow modifying a session with a writer lock (unsafe unless Codex is closed)",
    )
    parser.add_argument(
        "--restore",
        type=Path,
        metavar="BACKUP_DIR",
        help="Restore providers from a backup created by this script",
    )
    return parser.parse_args()


def get_codex_home(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def read_config(codex_home: Path) -> dict:
    path = codex_home / "config.toml"
    if not path.is_file():
        raise RepairError(f"Config file not found: {path}")
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RepairError(f"Cannot read {path}: {exc}") from exc


def resolve_target_provider(config: dict, override: str | None) -> str:
    target = override or config.get("model_provider") or "openai"
    if not isinstance(target, str) or not target.strip():
        raise RepairError("The target provider id must be a non-empty string")
    target = target.strip()
    configured = config.get("model_providers", {})
    if target not in BUILTIN_PROVIDERS and target not in configured:
        choices = ", ".join(sorted(BUILTIN_PROVIDERS | set(configured)))
        raise RepairError(
            f"Target provider {target!r} is not defined in config.toml. "
            f"Known providers: {choices}"
        )
    return target


def find_state_db(codex_home: Path, override: Path | None) -> Path | None:
    if override is not None:
        path = override.expanduser().resolve()
        if not path.is_file():
            raise RepairError(f"State database not found: {path}")
        return path

    candidates: list[tuple[int, Path]] = []
    for path in codex_home.glob("state_*.sqlite"):
        match = re.fullmatch(r"state_(\d+)\.sqlite", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def iter_rollout_paths(codex_home: Path) -> Iterable[Path]:
    for directory_name in ("sessions", "archived_sessions"):
        root = codex_home / directory_name
        if root.is_dir():
            yield from sorted(root.rglob("*.jsonl"))


def read_rollout(path: Path, codex_home: Path) -> RolloutRecord | None:
    try:
        with path.open("rb") as handle:
            head = handle.readline()
        item = json.loads(head)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"Cannot parse the first record of {path}: {exc}") from exc

    if item.get("type") != "session_meta":
        return None
    payload = item.get("payload")
    if not isinstance(payload, dict):
        return None
    provider = payload.get("model_provider")
    session_id = payload.get("id")
    if not isinstance(provider, str) or not isinstance(session_id, str):
        return None
    metadata = path.stat()
    return RolloutRecord(
        path=path,
        relative_path=str(path.relative_to(codex_home)),
        session_id=session_id,
        provider=provider,
        head=head,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def scan_rollouts(codex_home: Path) -> tuple[list[RolloutRecord], list[Path]]:
    records: list[RolloutRecord] = []
    skipped: list[Path] = []
    for path in iter_rollout_paths(codex_home):
        record = read_rollout(path, codex_home)
        if record is None:
            skipped.append(path)
        else:
            records.append(record)
    return records, skipped


def open_state_db(path: Path, *, readonly: bool) -> sqlite3.Connection:
    try:
        if readonly:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        else:
            connection = sqlite3.connect(path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(threads)").fetchall()
        }
        if not {"id", "model_provider"}.issubset(columns):
            connection.close()
            raise RepairError(f"{path} has no compatible threads table")
        return connection
    except sqlite3.Error as exc:
        raise RepairError(f"Cannot open state database {path}: {exc}") from exc


def scan_threads(state_db: Path | None) -> list[ThreadRecord]:
    if state_db is None:
        return []
    connection = open_state_db(state_db, readonly=True)
    try:
        rows = connection.execute(
            "SELECT id, model_provider FROM threads ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    return [ThreadRecord(str(row[0]), str(row[1])) for row in rows]


def provider_selected(provider: str, target: str, sources: set[str]) -> bool:
    return provider != target and (not sources or provider in sources)


def locked_session_ids(codex_home: Path) -> set[str]:
    lock_dir = codex_home / "thread-writer-locks"
    if not lock_dir.is_dir():
        return set()
    return {
        path.stem
        for path in lock_dir.glob("*.lock")
        if path.name != ".coordination.lock"
    }


def replace_provider_in_head(head: bytes, expected: str, target: str) -> bytes:
    match = PROVIDER_FIELD_RE.search(head)
    if match is None:
        raise RepairError("session_meta does not contain a model_provider field")
    try:
        observed = json.loads(match.group(2).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("model_provider is not a valid JSON string") from exc
    if observed != expected:
        raise RepairError(
            f"session provider changed concurrently: expected {expected!r}, got {observed!r}"
        )
    replacement = json.dumps(target, ensure_ascii=False).encode("utf-8")
    updated = head[: match.start(2)] + replacement + head[match.end(2) :]
    parsed = json.loads(updated)
    if parsed.get("payload", {}).get("model_provider") != target:
        raise RepairError("failed to verify rewritten session_meta")
    return updated


def copy_xattrs(source: Path, destination: Path) -> None:
    if not hasattr(os, "listxattr"):
        return
    try:
        for name in os.listxattr(source):
            os.setxattr(destination, name, os.getxattr(source, name))
    except OSError:
        # Extended attributes are not required by Codex; preserve them when possible.
        pass


def rewrite_head(path: Path, old_head: bytes, new_head: bytes) -> None:
    metadata = path.stat()
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.provider-repair-", dir=path.parent
        )
        with os.fdopen(descriptor, "wb") as destination, path.open("rb") as source:
            observed = source.readline()
            if observed != old_head:
                raise RepairError(f"Session changed while being repaired: {path}")
            destination.write(new_head)
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        temporary = Path(temporary_name)
        os.chmod(temporary, stat.S_IMODE(metadata.st_mode))
        copy_xattrs(path, temporary)
        os.replace(temporary, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def create_backup(
    codex_home: Path,
    target: str,
    rollout_records: list[RolloutRecord],
    thread_records: list[ThreadRecord],
    state_db: Path | None,
) -> Path:
    backup_root = codex_home / "provider-repair-backups"
    backup_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = backup_root / timestamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = backup_root / f"{timestamp}-{suffix}"
        suffix += 1
    backup_dir.mkdir(mode=0o700)

    config_path = codex_home / "config.toml"
    shutil.copy2(config_path, backup_dir / "config.toml")

    session_manifest = backup_dir / "session_heads.jsonl"
    with session_manifest.open("w", encoding="utf-8") as handle:
        for record in rollout_records:
            entry = {
                "relative_path": record.relative_path,
                "session_id": record.session_id,
                "old_provider": record.provider,
                "new_provider": target,
                "original_head_b64": base64.b64encode(record.head).decode("ascii"),
                "original_size": record.size,
                "original_mtime_ns": record.mtime_ns,
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    thread_manifest = backup_dir / "thread_providers.jsonl"
    with thread_manifest.open("w", encoding="utf-8") as handle:
        for record in thread_records:
            entry = {
                "thread_id": record.thread_id,
                "old_provider": record.provider,
                "new_provider": target,
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if state_db is not None:
        source = open_state_db(state_db, readonly=True)
        backup_path = backup_dir / state_db.name
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        os.chmod(backup_path, 0o600)

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "codex_home": str(codex_home),
        "target_provider": target,
        "rollout_count": len(rollout_records),
        "thread_count": len(thread_records),
        "state_db": state_db.name if state_db is not None else None,
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in backup_dir.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    return backup_dir


def update_thread_providers(
    state_db: Path | None, records: list[ThreadRecord], target: str
) -> None:
    if state_db is None or not records:
        return
    connection = open_state_db(state_db, readonly=False)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for record in records:
            cursor = connection.execute(
                "UPDATE threads SET model_provider = ? "
                "WHERE id = ? AND model_provider = ?",
                (target, record.thread_id, record.provider),
            )
            if cursor.rowcount != 1:
                current = connection.execute(
                    "SELECT model_provider FROM threads WHERE id = ?", (record.thread_id,)
                ).fetchone()
                if current is None or current[0] != target:
                    raise RepairError(
                        f"Thread changed concurrently: {record.thread_id}"
                    )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def restore_changed_rollouts(records: list[RolloutRecord]) -> None:
    for record in reversed(records):
        with record.path.open("rb") as handle:
            current_head = handle.readline()
        rewrite_head(record.path, current_head, record.head)


def verify(
    codex_home: Path,
    state_db: Path | None,
    target: str,
    sources: set[str],
) -> tuple[Counter, Counter, int]:
    rollout_records, _ = scan_rollouts(codex_home)
    thread_records = scan_threads(state_db)
    rollout_counts = Counter(record.provider for record in rollout_records)
    thread_counts = Counter(record.provider for record in thread_records)

    remaining_rollouts = [
        record
        for record in rollout_records
        if provider_selected(record.provider, target, sources)
    ]
    remaining_threads = [
        record
        for record in thread_records
        if provider_selected(record.provider, target, sources)
    ]
    if remaining_rollouts or remaining_threads:
        raise RepairError(
            "Verification found unrepaired providers: "
            f"{len(remaining_rollouts)} rollout(s), {len(remaining_threads)} DB row(s)"
        )

    rollout_by_id = {record.session_id: record.provider for record in rollout_records}
    mismatches = sum(
        1
        for record in thread_records
        if record.thread_id in rollout_by_id
        and rollout_by_id[record.thread_id] != record.provider
    )
    if mismatches:
        raise RepairError(
            f"Verification found {mismatches} provider mismatch(es) between rollout and DB"
        )
    return rollout_counts, thread_counts, mismatches


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise RepairError(f"Backup manifest not found: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RepairError(f"Invalid {path} line {line_number}: {exc}") from exc
    return records


def restore_backup(
    codex_home: Path,
    state_db: Path | None,
    backup_dir: Path,
    *,
    apply: bool,
) -> None:
    backup_dir = backup_dir.expanduser().resolve()
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RepairError(f"Not a provider-repair backup: {backup_dir}")
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(summary.get("script_version", 0)) > SCRIPT_VERSION:
        raise RepairError("Backup was created by a newer script version")

    session_entries = load_jsonl(backup_dir / "session_heads.jsonl")
    thread_entries = load_jsonl(backup_dir / "thread_providers.jsonl")
    locked = locked_session_ids(codex_home)
    affected_ids = {
        str(entry["session_id"]) for entry in session_entries
    } | {str(entry["thread_id"]) for entry in thread_entries}
    conflicts = sorted(locked & affected_ids)
    if conflicts:
        raise RepairError(
            "Cannot restore sessions with writer locks. Close Codex first: "
            + ", ".join(conflicts)
        )

    print(f"Restore backup: {backup_dir}")
    print(f"Rollout heads: {len(session_entries)}")
    print(f"Database rows: {len(thread_entries)}")
    print("Run again with --apply to restore.")
    if not apply:
        return

    planned_heads: list[tuple[Path, bytes, bytes]] = []
    for entry in session_entries:
        path = codex_home / str(entry["relative_path"])
        if not path.is_file():
            raise RepairError(f"Cannot restore missing rollout: {path}")
        with path.open("rb") as handle:
            current_head = handle.readline()
        old_head = base64.b64decode(entry["original_head_b64"], validate=True)
        current = json.loads(current_head)
        if current.get("payload", {}).get("id") != entry["session_id"]:
            raise RepairError(f"Session id mismatch while restoring {path}")
        planned_heads.append((path, current_head, old_head))

    restored_heads: list[tuple[Path, bytes, bytes]] = []
    try:
        for path, current_head, old_head in planned_heads:
            rewrite_head(path, current_head, old_head)
            restored_heads.append((path, old_head, current_head))

        if state_db is not None and thread_entries:
            connection = open_state_db(state_db, readonly=False)
            try:
                connection.execute("BEGIN IMMEDIATE")
                for entry in thread_entries:
                    cursor = connection.execute(
                        "UPDATE threads SET model_provider = ? WHERE id = ?",
                        (entry["old_provider"], entry["thread_id"]),
                    )
                    if cursor.rowcount != 1:
                        raise RepairError(
                            f"Cannot restore missing DB thread: {entry['thread_id']}"
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
    except Exception:
        for path, old_head, current_head in reversed(restored_heads):
            rewrite_head(path, old_head, current_head)
        raise
    print(f"Restored {len(restored_heads)} rollout(s) and {len(thread_entries)} DB row(s).")


def print_counts(label: str, counts: Counter) -> None:
    rendered = ", ".join(f"{provider}={count}" for provider, count in sorted(counts.items()))
    print(f"{label}: {rendered or '(none)'}")


def run(args: argparse.Namespace) -> int:
    codex_home = get_codex_home(args.codex_home)
    if not codex_home.is_dir():
        raise RepairError(f"Codex data directory not found: {codex_home}")
    state_db = find_state_db(codex_home, args.state_db)

    if args.restore is not None:
        restore_backup(codex_home, state_db, args.restore, apply=args.apply)
        return 0

    config = read_config(codex_home)
    target = resolve_target_provider(config, args.provider)
    sources = {provider.strip() for provider in args.from_provider if provider.strip()}
    if target in sources:
        raise RepairError("--from-provider must not include the target provider")

    rollout_records, skipped = scan_rollouts(codex_home)
    thread_records = scan_threads(state_db)
    rollout_candidates = [
        record
        for record in rollout_records
        if provider_selected(record.provider, target, sources)
    ]
    thread_candidates = [
        record
        for record in thread_records
        if provider_selected(record.provider, target, sources)
    ]

    rollout_counts = Counter(record.provider for record in rollout_records)
    thread_counts = Counter(record.provider for record in thread_records)
    print(f"Codex home: {codex_home}")
    print(f"Target provider: {target}")
    print(f"State database: {state_db or '(not found)'}")
    print_counts("Rollout providers", rollout_counts)
    print_counts("Database providers", thread_counts)
    print(
        f"Planned changes: {len(rollout_candidates)} rollout(s), "
        f"{len(thread_candidates)} DB row(s)"
    )
    if skipped:
        print(f"Skipped non-session JSONL files: {len(skipped)}")

    locked = locked_session_ids(codex_home)
    affected_ids = {record.session_id for record in rollout_candidates} | {
        record.thread_id for record in thread_candidates
    }
    conflicts = sorted(locked & affected_ids)
    if conflicts and not args.include_locked:
        raise RepairError(
            "Refusing to modify sessions with writer locks. Close those Codex tasks "
            "or close Codex, then retry. Locked ids: " + ", ".join(conflicts)
        )

    if not rollout_candidates and not thread_candidates:
        print("No repair is needed.")
        return 0
    if not args.apply:
        print("Dry-run only. Re-run with --apply to make these changes.")
        return 0

    backup_dir = create_backup(
        codex_home, target, rollout_candidates, thread_candidates, state_db
    )
    print(f"Backup: {backup_dir}")
    changed: list[RolloutRecord] = []
    try:
        for record in rollout_candidates:
            new_head = replace_provider_in_head(record.head, record.provider, target)
            rewrite_head(record.path, record.head, new_head)
            changed.append(record)
        update_thread_providers(state_db, thread_candidates, target)
    except Exception:
        if changed:
            restore_changed_rollouts(changed)
        raise

    verified_rollouts, verified_threads, _ = verify(
        codex_home, state_db, target, sources
    )
    print_counts("Verified rollout providers", verified_rollouts)
    print_counts("Verified database providers", verified_threads)
    print(
        f"Repair complete: {len(rollout_candidates)} rollout(s) and "
        f"{len(thread_candidates)} DB row(s) now use {target!r}."
    )
    print(
        "Rollback command: "
        f"{Path(__file__).resolve()} --codex-home {json.dumps(str(codex_home))} "
        f"--restore {json.dumps(str(backup_dir))} --apply"
    )
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (RepairError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

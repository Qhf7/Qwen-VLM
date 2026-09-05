#!/usr/bin/env python3
"""Remove stale Responses reasoning items that were recorded without content.

When a Responses request is stateless (``store=false``), a reasoning item can
only be replayed if its encrypted content is available.  A small number of
old Codex rollouts contain ``response_item`` entries such as ``{"id":"rs_1"}``
with neither ``encrypted_content`` nor content.  Replaying those entries makes
the current backend return ``Item ... not found``.  This utility replaces only
those item/event lines with harmless token-count events (keeping byte offsets
stable) and removes the matching projection rows from ``thread_history``.

It is a dry-run by default.  Applied changes are backed up as exact original
lines and database rows, so later turns appended to a session are preserved by
the restore operation.
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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPT_VERSION = 1


class RepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class BadItem:
    session_id: str
    item_id: str
    turn_id: str | None
    response_line: int
    event_line: int | None
    path: Path
    relative_path: str
    response_bytes: bytes
    event_bytes: bytes | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and remove Codex reasoning response items with an rs_* id "
            "but no encrypted content. Dry-run unless --apply is supplied."
        )
    )
    parser.add_argument("--codex-home", type=Path, help="Codex data directory")
    parser.add_argument("--session-id", help="Limit repair to one session id")
    parser.add_argument("--state-db", type=Path, help="thread_history SQLite path")
    parser.add_argument("--apply", action="store_true", help="Apply the repair")
    parser.add_argument(
        "--include-locked",
        action="store_true",
        help="Allow a locked but inactive session (close Codex if unsure)",
    )
    parser.add_argument(
        "--restore", type=Path, metavar="BACKUP_DIR", help="Restore a repair backup"
    )
    return parser.parse_args()


def codex_home(value: Path | None) -> Path:
    configured = os.environ.get("CODEX_HOME")
    path = value or (Path(configured).expanduser() if configured else Path.home() / ".codex")
    return path.expanduser().resolve()


def iter_rollouts(home: Path) -> Iterable[Path]:
    for name in ("sessions", "archived_sessions"):
        root = home / name
        if root.is_dir():
            yield from sorted(root.rglob("*.jsonl"))


def parse_line(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"Invalid JSONL record: {exc}") from exc
    return value if isinstance(value, dict) else {}


def scan_file(path: Path, home: Path, session_filter: str | None) -> list[BadItem]:
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        return []
    metadata = parse_line(lines[0]).get("payload", {})
    session_id = metadata.get("id") if isinstance(metadata, dict) else None
    if not isinstance(session_id, str) or (session_filter and session_id != session_filter):
        return []

    malformed: dict[str, tuple[int, str | None, bytes]] = {}
    completed_events: dict[str, tuple[int, bytes]] = {}
    for line_number, raw in enumerate(lines, 1):
        record = parse_line(raw)
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "response_item" and payload.get("type") == "reasoning":
            item_id = payload.get("id")
            if (
                isinstance(item_id, str)
                and item_id.startswith("rs_")
                and not payload.get("encrypted_content")
            ):
                malformed[item_id] = (line_number, payload.get("internal_chat_message_metadata_passthrough", {}).get("turn_id"), raw)
        if record.get("type") == "event_msg" and payload.get("type") == "item_completed":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "Reasoning":
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id.startswith("rs_"):
                    completed_events[item_id] = (line_number, raw)

    output: list[BadItem] = []
    for item_id, (response_line, turn_id, response_bytes) in malformed.items():
        event = completed_events.get(item_id)
        output.append(
            BadItem(
                session_id=session_id,
                item_id=item_id,
                turn_id=turn_id,
                response_line=response_line,
                event_line=event[0] if event else None,
                path=path,
                relative_path=str(path.relative_to(home)),
                response_bytes=response_bytes,
                event_bytes=event[1] if event else None,
            )
        )
    return output


def scan_rollouts(home: Path, session_filter: str | None) -> list[BadItem]:
    result: list[BadItem] = []
    for path in iter_rollouts(home):
        result.extend(scan_file(path, home, session_filter))
    return result


def find_history_db(home: Path, override: Path | None) -> Path | None:
    if override:
        path = override.expanduser().resolve()
        if not path.is_file():
            raise RepairError(f"History database not found: {path}")
        return path
    candidates: list[tuple[int, Path]] = []
    for path in home.glob("thread_history_*.sqlite"):
        match = re.fullmatch(r"thread_history_(\d+)\.sqlite", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda pair: pair[0])[1]


def open_history(path: Path, readonly: bool) -> sqlite3.Connection:
    try:
        if readonly:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        else:
            db = sqlite3.connect(path, timeout=10)
        db.execute("PRAGMA busy_timeout = 10000")
        columns = {row[1] for row in db.execute("PRAGMA table_info(thread_items)")}
        required = {"thread_id", "turn_id", "item_id", "rollout_ordinal", "created_at_ms", "item_json", "item_type", "updated_at_ordinal"}
        if not required.issubset(columns):
            db.close()
            raise RepairError(f"{path} has no compatible thread_items table")
        return db
    except sqlite3.Error as exc:
        raise RepairError(f"Cannot open {path}: {exc}") from exc


def locked_ids(home: Path) -> set[str]:
    root = home / "thread-writer-locks"
    if not root.is_dir():
        return set()
    return {path.stem for path in root.glob("*.lock") if path.name != ".coordination.lock"}


def active_ids(history_db: Path | None) -> set[str]:
    if history_db is None:
        return set()
    db = open_history(history_db, readonly=True)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(thread_turns)")}
        if not {"thread_id", "status"}.issubset(columns):
            return set()
        if "rollout_ordinal" in columns:
            rows = db.execute(
                "SELECT thread_id FROM thread_turns AS current "
                "WHERE lower(current.status) IN ('inprogress','in_progress','running') "
                "AND current.rollout_ordinal = ("
                "  SELECT MAX(previous.rollout_ordinal) FROM thread_turns AS previous "
                "  WHERE previous.thread_id = current.thread_id"
                ")"
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT DISTINCT thread_id FROM thread_turns "
                "WHERE lower(status) IN ('inprogress','in_progress','running')"
            ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        db.close()


def no_op_line(raw: bytes) -> bytes:
    newline = b"\n" if raw.endswith(b"\n") else b""
    length = len(raw) - len(newline)
    compact = b'{"type":"event_msg","payload":{"type":"token_count"}}'
    if len(compact) > length:
        compact = b"{}"
    return compact + (b" " * (length - len(compact))) + newline


def atomic_rewrite_lines(path: Path, replacements: dict[int, bytes]) -> None:
    original = path.read_bytes().splitlines(keepends=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.repair-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for line_number, raw in enumerate(original, 1):
                output.write(replacements.get(line_number, raw))
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def create_backup(home: Path, history_db: Path | None, bad_items: list[BadItem]) -> Path:
    root = home / "provider-repair-backups"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = root / f"{stamp}-unpersisted-reasoning"
    suffix = 1
    while backup.exists():
        backup = root / f"{stamp}-unpersisted-reasoning-{suffix}"
        suffix += 1
    backup.mkdir(mode=0o700)

    with (backup / "rollout_lines.jsonl").open("w", encoding="utf-8") as handle:
        for item in bad_items:
            handle.write(
                json.dumps(
                    {
                        "relative_path": item.relative_path,
                        "session_id": item.session_id,
                        "item_id": item.item_id,
                        "turn_id": item.turn_id,
                        "response_line": item.response_line,
                        "response_b64": base64.b64encode(item.response_bytes).decode(),
                        "event_line": item.event_line,
                        "event_b64": base64.b64encode(item.event_bytes).decode() if item.event_bytes else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    db_rows: list[dict] = []
    if history_db is not None:
        db = open_history(history_db, readonly=True)
        try:
            for item in bad_items:
                rows = db.execute(
                    "SELECT thread_id,turn_id,item_id,rollout_ordinal,created_at_ms,item_json,item_type,updated_at_ordinal "
                    "FROM thread_items WHERE thread_id=? AND item_id=?",
                    (item.session_id, item.item_id),
                ).fetchall()
                for row in rows:
                    db_rows.append(dict(zip(("thread_id","turn_id","item_id","rollout_ordinal","created_at_ms","item_json","item_type","updated_at_ordinal"), row)))
        finally:
            db.close()
    (backup / "thread_items.json").write_text(json.dumps(db_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if history_db is not None:
        source = open_history(history_db, readonly=True)
        destination = sqlite3.connect(backup / history_db.name)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "target_sessions": sorted({item.session_id for item in bad_items}),
        "bad_item_count": len(bad_items),
        "history_db": history_db.name if history_db else None,
    }
    (backup / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in backup.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    return backup


def update_history(history_db: Path | None, bad_items: list[BadItem]) -> list[dict]:
    if history_db is None or not bad_items:
        return []
    db = open_history(history_db, readonly=False)
    deleted: list[dict] = []
    try:
        db.execute("BEGIN IMMEDIATE")
        for item in bad_items:
            rows = db.execute(
                "SELECT thread_id,turn_id,item_id,rollout_ordinal,created_at_ms,item_json,item_type,updated_at_ordinal "
                "FROM thread_items WHERE thread_id=? AND item_id=?",
                (item.session_id, item.item_id),
            ).fetchall()
            for row in rows:
                deleted.append(dict(zip(("thread_id","turn_id","item_id","rollout_ordinal","created_at_ms","item_json","item_type","updated_at_ordinal"), row)))
            db.execute(
                "DELETE FROM thread_items WHERE thread_id=? AND item_id=?",
                (item.session_id, item.item_id),
            )
        db.commit()
        return deleted
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def restore_lines(home: Path, entries: list[dict]) -> None:
    by_path: dict[Path, dict[int, bytes]] = {}
    for entry in entries:
        path = home / entry["relative_path"]
        if not path.is_file():
            raise RepairError(f"Missing rollout while restoring: {path}")
        by_path.setdefault(path, {})[int(entry["response_line"])] = base64.b64decode(entry["response_b64"])
        if entry.get("event_line") and entry.get("event_b64"):
            by_path[path][int(entry["event_line"])] = base64.b64decode(entry["event_b64"])
    for path, replacements in by_path.items():
        atomic_rewrite_lines(path, replacements)


def restore_db(history_db: Path | None, rows: list[dict]) -> None:
    if history_db is None or not rows:
        return
    db = open_history(history_db, readonly=False)
    try:
        db.execute("BEGIN IMMEDIATE")
        db.executemany(
            "INSERT OR REPLACE INTO thread_items "
            "(thread_id,turn_id,item_id,rollout_ordinal,created_at_ms,item_json,item_type,updated_at_ordinal) "
            "VALUES (:thread_id,:turn_id,:item_id,:rollout_ordinal,:created_at_ms,:item_json,:item_type,:updated_at_ordinal)",
            rows,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def verify(
    home: Path,
    history_db: Path | None,
    session_filter: str | None,
    bad_items: list[BadItem],
) -> int:
    remaining = scan_rollouts(home, session_filter)
    if remaining:
        raise RepairError(f"Verification found {len(remaining)} malformed reasoning item(s)")
    if history_db is not None and bad_items:
        db = open_history(history_db, readonly=True)
        try:
            for item in bad_items:
                count = db.execute(
                    "SELECT COUNT(*) FROM thread_items WHERE thread_id=? AND item_id=?",
                    (item.session_id, item.item_id),
                ).fetchone()[0]
                if count:
                    raise RepairError(
                        f"Verification found projection row for removed item {item.item_id}"
                    )
        finally:
            db.close()
    return 0


def restore_backup(
    home: Path,
    history_db: Path | None,
    backup: Path,
    apply: bool,
    include_locked: bool,
) -> int:
    backup = backup.expanduser().resolve()
    manifest_path = backup / "manifest.json"
    if not manifest_path.is_file():
        raise RepairError(f"Not a repair backup: {backup}")
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(summary.get("script_version", 0)) > SCRIPT_VERSION:
        raise RepairError("Backup was created by a newer script version")
    entries = [json.loads(line) for line in (backup / "rollout_lines.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = json.loads((backup / "thread_items.json").read_text(encoding="utf-8"))
    ids = {entry["session_id"] for entry in entries}
    conflicts = sorted(ids & locked_ids(home))
    if conflicts and not include_locked:
        raise RepairError("Writer locks present; close Codex before restore: " + ", ".join(conflicts))
    print(f"Restore backup: {backup}")
    print(f"Rollout item(s): {len(entries)}; projection row(s): {len(rows)}")
    if not apply:
        print("Dry-run only. Re-run with --apply to restore.")
        return 0
    restore_lines(home, entries)
    restore_db(history_db, rows)
    print("Restore complete.")
    return 0


def run(args: argparse.Namespace) -> int:
    home = codex_home(args.codex_home)
    if not home.is_dir():
        raise RepairError(f"Codex home not found: {home}")
    history_db = find_history_db(home, args.state_db)
    if args.restore:
        return restore_backup(
            home, history_db, args.restore, args.apply, args.include_locked
        )

    bad_items = scan_rollouts(home, args.session_id)
    counts = Counter(item.session_id for item in bad_items)
    print(f"Codex home: {home}")
    print(f"History database: {history_db or '(not found)'}")
    print(f"Malformed unpersisted reasoning item(s): {len(bad_items)}")
    for session_id, count in sorted(counts.items()):
        ids = sorted({item.item_id for item in bad_items if item.session_id == session_id})
        print(f"  {session_id}: {count} ({', '.join(ids)})")
    if not bad_items:
        print("No repair is needed.")
        return 0

    affected = {item.session_id for item in bad_items}
    active = active_ids(history_db)
    running = sorted(affected & active)
    if running:
        raise RepairError("Refusing to modify active session(s): " + ", ".join(running))
    locked = sorted(affected & locked_ids(home))
    if locked and not args.include_locked:
        raise RepairError(
            "Writer locks present; close Codex or pass --include-locked after confirming "
            "the session is inactive: " + ", ".join(locked)
        )
    if not args.apply:
        print("Dry-run only. Re-run with --apply to make these changes.")
        return 0

    backup = create_backup(home, history_db, bad_items)
    print(f"Backup: {backup}")
    replacements: dict[Path, dict[int, bytes]] = {}
    for item in bad_items:
        replacements.setdefault(item.path, {})[item.response_line] = no_op_line(item.response_bytes)
        if item.event_line and item.event_bytes:
            replacements[item.path][item.event_line] = no_op_line(item.event_bytes)

    changed: list[Path] = []
    deleted: list[dict] = []
    try:
        for path, lines in replacements.items():
            atomic_rewrite_lines(path, lines)
            changed.append(path)
        deleted = update_history(history_db, bad_items)
        verify(home, history_db, args.session_id, bad_items)
    except Exception:
        if changed:
            entries = [json.loads(line) for line in (backup / "rollout_lines.jsonl").read_text(encoding="utf-8").splitlines()]
            restore_lines(home, entries)
        if deleted:
            restore_db(history_db, deleted)
        raise

    print(f"Repair complete: {len(bad_items)} malformed reasoning item(s) removed from replay.")
    print(f"Projection rows deleted: {len(deleted)}")
    print(
        "Rollback command: "
        f"{Path(__file__).resolve()} --codex-home {json.dumps(str(home))} "
        f"--restore {json.dumps(str(backup))} --apply --include-locked"
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

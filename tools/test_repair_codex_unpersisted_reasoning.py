#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("repair_codex_unpersisted_reasoning.py")


class RepairScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / ".codex"
        rollout = self.home / "sessions/2026/01/01/rollout-demo.jsonl"
        rollout.parent.mkdir(parents=True)
        self.rollout = rollout
        records = [
            {"type": "session_meta", "payload": {"id": "demo", "model_provider": "custom"}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "Reasoning", "id": "rs_1"}}},
            {"type": "response_item", "payload": {"type": "reasoning", "id": "rs_1", "encrypted_content": None, "content": None}},
            {"type": "response_item", "payload": {"type": "reasoning", "id": "rs_good", "encrypted_content": "abc", "content": None}},
            {"type": "response_item", "payload": {"type": "message", "id": "msg_1", "role": "user", "content": []}},
        ]
        rollout.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        self.original = rollout.read_bytes()
        self.db_path = self.home / "thread_history_1.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path)
        db.executescript(
            "CREATE TABLE thread_items (thread_id TEXT, turn_id TEXT, item_id TEXT, rollout_ordinal INTEGER, "
            "created_at_ms INTEGER, item_json TEXT, item_type TEXT, updated_at_ordinal INTEGER, "
            "PRIMARY KEY(thread_id,turn_id,item_id));"
            "CREATE TABLE thread_turns (thread_id TEXT, status TEXT);"
        )
        db.execute(
            "INSERT INTO thread_items VALUES (?,?,?,?,?,?,?,?)",
            ("demo", "turn", "rs_1", 1, 1, '{"type":"reasoning","id":"rs_1","content":[]}', "reasoning", 1),
        )
        db.execute(
            "INSERT INTO thread_items VALUES (?,?,?,?,?,?,?,?)",
            ("demo", "turn", "item_legacy", 2, 1, '{"type":"reasoning","id":"item_legacy","content":[]}', "reasoning", 1),
        )
        db.commit()
        db.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), "--codex-home", str(self.home), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_apply_and_restore(self) -> None:
        dry = self.run_script()
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("Malformed unpersisted reasoning item(s): 1", dry.stdout)
        self.assertEqual(self.rollout.read_bytes(), self.original)

        applied = self.run_script("--apply")
        self.assertEqual(applied.returncode, 0, applied.stderr)
        self.assertEqual(len(self.rollout.read_bytes()), len(self.original))
        self.assertNotIn(b'"id": "rs_1"', self.rollout.read_bytes())
        self.assertIn(b'"id": "rs_good"', self.rollout.read_bytes())
        db = sqlite3.connect(self.db_path)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM thread_items WHERE item_id='rs_1'").fetchone()[0], 0)
        db.close()

        backups = sorted((self.home / "provider-repair-backups").iterdir())
        restored = self.run_script("--restore", str(backups[0]), "--apply")
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertEqual(self.rollout.read_bytes(), self.original)
        db = sqlite3.connect(self.db_path)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM thread_items WHERE item_id='rs_1'").fetchone()[0], 1)
        db.close()


if __name__ == "__main__":
    unittest.main()

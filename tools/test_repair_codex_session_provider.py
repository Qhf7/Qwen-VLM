#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("repair_codex_session_provider.py")


def write_rollout(path: Path, session_id: str, provider: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "model_provider": provider,
                "cwd": "/tmp/example",
            },
        },
        {"timestamp": "2026-01-01T00:00:01Z", "type": "event_msg"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def provider_from_rollout(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())["payload"]["model_provider"]


class RepairScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.codex_home = Path(self.temporary.name) / ".codex"
        self.codex_home.mkdir()
        (self.codex_home / "config.toml").write_text(
            'model_provider = "custom"\n\n'
            '[model_providers.custom]\n'
            'name = "Current relay"\n'
            'wire_api = "responses"\n',
            encoding="utf-8",
        )
        self.rollouts = {
            "old-openai": self.codex_home
            / "sessions/2026/01/01/rollout-old-openai.jsonl",
            "old-relay": self.codex_home
            / "sessions/2026/01/02/rollout-old-relay.jsonl",
            "current": self.codex_home
            / "sessions/2026/01/03/rollout-current.jsonl",
            "archived": self.codex_home
            / "archived_sessions/rollout-archived.jsonl",
        }
        providers = {
            "old-openai": "openai",
            "old-relay": "api111",
            "current": "custom",
            "archived": "openai",
        }
        for session_id, path in self.rollouts.items():
            write_rollout(path, session_id, providers[session_id])

        self.state_db = self.codex_home / "state_5.sqlite"
        connection = sqlite3.connect(self.state_db)
        connection.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO threads (id, model_provider) VALUES (?, ?)", providers.items()
        )
        connection.commit()
        connection.close()
        self.original = providers

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), "--codex-home", str(self.codex_home), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def database_providers(self) -> dict[str, str]:
        connection = sqlite3.connect(self.state_db)
        try:
            return dict(connection.execute("SELECT id, model_provider FROM threads"))
        finally:
            connection.close()

    def test_dry_run_apply_and_restore(self) -> None:
        dry_run = self.run_script()
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("Planned changes: 3 rollout(s), 3 DB row(s)", dry_run.stdout)
        self.assertEqual(self.database_providers(), self.original)

        applied = self.run_script("--apply")
        self.assertEqual(applied.returncode, 0, applied.stderr)
        self.assertEqual(
            self.database_providers(), {session_id: "custom" for session_id in self.original}
        )
        self.assertTrue(
            all(provider_from_rollout(path) == "custom" for path in self.rollouts.values())
        )

        backups = sorted((self.codex_home / "provider-repair-backups").iterdir())
        self.assertEqual(len(backups), 1)
        restored = self.run_script("--restore", str(backups[0]), "--apply")
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertEqual(self.database_providers(), self.original)
        self.assertEqual(
            {
                session_id: provider_from_rollout(path)
                for session_id, path in self.rollouts.items()
            },
            self.original,
        )

    def test_from_provider_limits_scope(self) -> None:
        applied = self.run_script("--from-provider", "api111", "--apply")
        self.assertEqual(applied.returncode, 0, applied.stderr)
        expected = dict(self.original)
        expected["old-relay"] = "custom"
        self.assertEqual(self.database_providers(), expected)


if __name__ == "__main__":
    unittest.main()

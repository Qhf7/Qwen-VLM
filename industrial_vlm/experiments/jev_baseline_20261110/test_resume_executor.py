"""Offline-only amendment tests. Temporary files and mock HTTP; no real API."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import runner as r
import resume_executor as e
from test_runner import Fixture

class Response:
    headers = {'x-request-id': 'offline-response'}
    def __init__(self, status=200): self.status_code = status
    def json(self): return {'model': r.MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': .6}}, 'usage': {'input_tokens': 4}}

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.f = Fixture(self.root); self.f.prepare(); self.f.materialize()
            self.output = self.root / 'run'; args = self.f.run_args(self.output, limit=None); args.workers = 3; args.dry_run = True
            self.assertEqual(r.run(args), 0)
        self.config, self.plan, self.labels = r.load_frozen(self.f.plan_dir)
        self.doc, self.images = r.verified_images(self.f.plan_dir, self.config, self.plan)
        self.contract_bytes = (self.output / 'contract.json').read_bytes()
        self.contract_hash = r.sha(r.canonical(r.read_json(self.output / 'contract.json')))
        self.args = argparse.Namespace(plan_dir=self.f.plan_dir, output=self.output, retry_failed=True,
            dry_run=False, key_env='JEV_OFFLINE_FIXTURE_KEY', stop_file=self.output / 'STOP')
    def tearDown(self): self.tmp.cleanup()
    def payload(self, q, shots):
        return r.request_payload(q, self.plan['supports_by_category'][q['category']], shots, self.images,
            self.contract_hash + '|' + q['source_id'] + '|' + str(shots))
    def add_original(self, q, shots, event='started'):
        body = self.payload(q, shots)
        row = {'source_id': q['source_id'], 'shots': shots, 'contract_sha256': self.contract_hash,
            'request_sha256': r.sha(r.canonical(body)), 'attempt': 1, 'timestamp_utc': r.now(), 'event': event}
        r.append_json(self.output / 'attempts.jsonl', row)
        return row
    def add_success(self, q, shots):
        attempt = self.add_original(q, shots)
        row = {k: v for k, v in attempt.items() if k not in ('attempt', 'event')}
        row.update(category=q['category'], response_model=r.MODEL, probability=.6, seconds=1, usage={}, request_id='original')
        r.append_json(self.output / 'predictions.jsonl', row)
    def test_full_history_bounded_retry_and_contract_unchanged(self):
        q = self.plan['queries'][0]; self.add_success(q, 0)
        started = self.add_original(q, 1)
        r.append_json(self.output / 'attempts.jsonl', {**started, 'event': 'failed', 'error': 'http_502', 'http_status': 502, 'seconds': 1})
        self.add_original(q, 4)  # Interrupted old-worker target.
        old_prediction = (self.output / 'predictions.jsonl').read_bytes()
        calls = {}; guard = threading.Lock()
        def post(url, **kwargs):
            body = kwargs['json']; ident = r.sha(r.canonical(body))
            with guard:
                calls[ident] = calls.get(ident, 0) + 1; n = calls[ident]
            if ident == r.sha(r.canonical(self.payload(q, 0))): raise AssertionError('Completed target sent again')
            if ident == r.sha(r.canonical(self.payload(q, 1))) and n == 1: return Response(502)
            if ident == r.sha(r.canonical(self.payload(q, 4))) and n == 1: raise r.requests.Timeout()
            return Response()
        with patch.dict(os.environ, {'JEV_OFFLINE_FIXTURE_KEY': 'SYNTHETIC_SECRET'}), patch.object(r.requests, 'post', side_effect=post), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.execute(self.args), 2)
            self.assertEqual(len(r.jsonl(self.output / 'predictions.jsonl')), 359)
            self.assertEqual(e.execute(self.args), 0)
            before = sum(calls.values()); self.assertEqual(e.execute(self.args), 0); self.assertEqual(sum(calls.values()), before)
        self.assertEqual((self.output / 'contract.json').read_bytes(), self.contract_bytes)
        self.assertTrue((self.output / 'predictions.jsonl').read_bytes().startswith(old_prediction))
        amendment = r.read_json(self.output / 'operational_amendment.json')
        self.assertEqual(amendment['initial_completed_count'], 1); self.assertEqual(len(amendment['interrupted_unknown_targets']), 1)
        self.assertEqual(amendment['actual_workers'], 6)
        starts = [x for x in r.jsonl(self.output / 'attempts.jsonl') if x['event'] == 'started']
        counts = r.Counter(r.result_key(x) for x in starts)
        self.assertEqual(counts[q['source_id'], 1], 3); self.assertEqual(counts[q['source_id'], 4], 3)
        self.assertLessEqual(max(counts.values()), 3)
        rows = r.jsonl(self.output / 'predictions.jsonl'); self.assertEqual(len(rows), 360)
        self.assertTrue(all(x['amendment_sha256'] == r.sha(r.canonical(amendment)) for x in rows[1:]))
        logs = ''.join(x.read_text() for x in self.output.iterdir() if x.is_file())
        self.assertNotIn('SYNTHETIC_SECRET', logs); self.assertNotIn('data_base64', logs)
        # Original runner accepts appended operational provenance and validates exact old request identity.
        eval_args = argparse.Namespace(plan_dir=self.f.plan_dir, output=self.output)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(r.np, 'percentile', wraps=r.np.percentile):
            self.assertEqual(r.evaluate(eval_args), 0)
    def test_stop_file_drains_and_requires_retry_authorization(self):
        q = self.plan['queries'][0]; self.add_original(q, 0)
        self.args.retry_failed = False
        with self.assertRaises(ValueError): e.execute(self.args)
        self.args.retry_failed = True
        def post(url, **kwargs):
            self.args.stop_file.touch(); return Response()
        with patch.dict(os.environ, {'JEV_OFFLINE_FIXTURE_KEY': 'SYNTHETIC_SECRET'}), patch.object(r.requests, 'post', side_effect=post) as mocked, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.execute(self.args), 2)
            self.assertLessEqual(mocked.call_count, 6)
        sessions = r.jsonl(self.output / 'operational_sessions.jsonl'); self.assertTrue(sessions[-1]['stop_file_observed'])
        self.assertEqual((self.output / 'contract.json').read_bytes(), self.contract_bytes)
    def test_original_prefix_tamper_rejected(self):
        q = self.plan['queries'][0]; self.add_success(q, 0); self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(e.execute(self.args), 0)
        row = r.jsonl(self.output / 'predictions.jsonl')[0]; row['probability'] = .7
        (self.output / 'predictions.jsonl').write_text(r.canonical(row).decode() + '\n')
        with self.assertRaises(ValueError): e.execute(self.args)

if __name__ == '__main__': unittest.main(verbosity=2)

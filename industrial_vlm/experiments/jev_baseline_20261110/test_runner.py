"""Offline contract tests only; all images/probabilities here are synthetic fixtures."""
import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import runner as r

class Fixture:
    def __init__(self, root):
        self.root = root; self.plan_dir = root / 'frozen'; self.images = root / 'VisA'
        query = []; train = []; audit = []; labels = {}
        for c in range(12):
            cat = 'fixture' + str(c)
            for split, count in [('query', 10), ('train', 64)]:
                for i in range(count):
                    abnormal = i >= count // 2
                    im = Image.new('RGB', (32 + c, 40 + i), (c * 13, i * 3 % 256, 140))
                    b = io.BytesIO(); im.save(b, format='PNG'); raw = b.getvalue(); digest = r.sha(raw)
                    relative = cat + '/Data/Images/' + ('Anomaly' if abnormal else 'Normal') + '/' + split + str(i) + '.PNG'
                    source = self.images / relative; source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(raw)
                    sid = 'visa_' + digest; image = str(r.REMOTE_ROOT / relative)
                    row = {'source_id': sid, 'category': cat, 'is_anomalous': abnormal, 'image': image,
                           'answer': 'SECRET_GT_ANSWER', 'boxes': [[1, 2, 3, 4]], 'anomaly_area_fraction': 0.123}
                    audit.append({'source_id': sid, 'image': image, 'sha256': digest, 'bytes': len(raw)})
                    if split == 'query': query.append(row); labels[sid] = int(abnormal)
                    else: train.extend([row, dict(row)])
        # Query/train pixel bytes must differ; dimensions encode disjoint split identity.
        # The train loop currently overlaps first 10 images; separate raw train bytes below.
        for row in train[::2]:
            original = row['source_id']; source = self.images / str(Path(row['image']).relative_to(str(r.REMOTE_ROOT)))
            im = Image.open(source).convert('RGB'); im.putpixel((0, 0), (1, 2, 3)); b = io.BytesIO(); im.save(b, format='PNG'); raw = b.getvalue(); source.write_bytes(raw)
            digest = r.sha(raw); row['source_id'] = 'visa_' + digest
            # propagate paired duplicate and historical identity.
            idx = train.index(row); train[idx + 1]['source_id'] = row['source_id']
            for item in audit:
                if item['source_id'] == original and item['image'] == row['image']:
                    item.update(source_id=row['source_id'], sha256=digest, bytes=len(raw))
        self.queries = root / 'query.jsonl'; self.train = root / 'train.jsonl'; self.audit = root / 'audit.json'
        self.queries.write_text('\n'.join(json.dumps(x) for x in query) + '\n')
        self.train.write_text('\n'.join(json.dumps(x) for x in train) + '\n')
        r.write_json(self.audit, {'images': audit})
    def prepare(self):
        r.prepare(argparse.Namespace(queries=self.queries, train=self.train, audit=self.audit, output=self.plan_dir, image_root=self.images))
    def materialize(self):
        return r.materialize(argparse.Namespace(plan_dir=self.plan_dir, image_root=self.images))
    def run_args(self, output, limit=1):
        return argparse.Namespace(plan_dir=self.plan_dir, output=output, endpoint='https://fixture.example/v1/systemone',
            limit_queries=limit, workers=1, key_env='JEV_OFFLINE_FIXTURE_KEY', retry_failed=False, dry_run=False, timeout=1, max_consecutive_errors=1)

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_metrics_ties(self):
        m = r.metrics([0, 1, 0, 1], [.5] * 4)
        self.assertEqual(m['auroc'], .5); self.assertEqual(m['ap'], .5)
        self.assertEqual(m['recall'], 1); self.assertEqual(m['fpr'], 1)
        self.assertEqual(m['brier'], .25); self.assertEqual(m['ece_10_equal_width'], 0)
        self.assertEqual(r.metrics([0, 1], [0, 1])['macro_f1'], 1)
    def test_probability_contract(self):
        for p in [True, '0.7', float('nan'), -1, 1.1]:
            with self.assertRaises(ValueError): r.response_probability({'model': r.MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': p}}})
        self.assertEqual(r.response_probability({'model': r.MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': .7}}}), .7)
        with self.assertRaises(ValueError): r.response_probability({'model': 'jev-latest', 'answers': {}})
    def test_endpoint_and_usage(self):
        for endpoint in ['http://fixture/v1/systemone', 'https://fixture/v1/content_generation/tasks', 'https://key@fixture/v1/systemone', 'https://fixture/v1/systemone?key=x']:
            with self.assertRaises(ValueError): r.validate_endpoint(endpoint)
        self.assertEqual(r.safe_usage({'usage': {'input_tokens': 3, 'secret': 'NO', 'output_tokens': True}}), {'input_tokens': 3})
    def test_preprocessing_and_label_free_payload(self):
        f = Fixture(self.root); f.prepare(); self.assertEqual(f.materialize(), 0)
        config, plan, labels = r.load_frozen(f.plan_dir); doc, images = r.verified_images(f.plan_dir, config, plan)
        self.assertEqual(len(doc['records']), 168); q = plan['queries'][0]; supports = plan['supports_by_category'][q['category']]
        qbytes = []
        for shots in (0, 1, 4):
            payload = r.request_payload(q, supports, shots, images, 'fixture|' + str(shots))
            wire = json.dumps(payload)
            for forbidden in ['SECRET_GT_ANSWER', 'source_id', 'relative_image', 'is_anomalous', 'boxes', 'anomaly_area_fraction', '/Normal/', '/Anomaly/']:
                self.assertNotIn(forbidden, wire)
            self.assertEqual(len(payload['assets']), 1 + shots)
            qbytes.append(base64.b64decode(payload['assets'][payload['state']['query']]['data_base64']))
            self.assertEqual(Image.open(io.BytesIO(qbytes[-1])).size, (448, 448))
            self.assertEqual(len(payload['state']['normal_references']), shots)
        self.assertEqual(qbytes[0], qbytes[1]); self.assertEqual(qbytes[0], qbytes[2])
        plan['queries'][0]['category'] = 'tampered'; r.write_json(f.plan_dir / 'plan.json', plan)
        with self.assertRaises(ValueError): r.load_frozen(f.plan_dir)
    def test_original_and_processed_hash_rejection(self):
        f = Fixture(self.root); f.prepare(); plan = r.read_json(f.plan_dir / 'plan.json')
        path = f.images / plan['queries'][0]['relative_image']; original = path.read_bytes(); path.write_bytes(b'bad')
        self.assertEqual(f.materialize(), 2)
        self.assertEqual(r.read_json(f.plan_dir / 'materialized.json')['failures'][0]['error'], 'original_bytes_mismatch')
        path.write_bytes(original); self.assertEqual(f.materialize(), 0)
        config, plan, _ = r.load_frozen(f.plan_dir); target = f.plan_dir / r.read_json(f.plan_dir / 'materialized.json')['records'][0]['image']; target.write_bytes(b'bad')
        with self.assertRaises(ValueError): r.verified_images(f.plan_dir, config, plan)
    def test_resume_rejects_duplicates_and_configuration_change(self):
        f = Fixture(self.root); f.prepare(); f.materialize(); output = self.root / 'run'; args = f.run_args(output)
        class Response:
            status_code = 200; headers = {'x-request-id': 'fixture-id'}
            def json(self): return {'model': r.MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': .6}}, 'usage': {'input_tokens': 9}}
        with patch.dict(os.environ, {'JEV_OFFLINE_FIXTURE_KEY': 'SYNTHETIC_SECRET'}), patch.object(r.requests, 'post', return_value=Response()) as post:
            self.assertEqual(r.run(args), 0); self.assertEqual(post.call_count, 3)
            self.assertEqual(r.run(args), 0); self.assertEqual(post.call_count, 3)
        text = ''.join(x.read_text() for x in output.iterdir() if x.is_file())
        self.assertNotIn('SYNTHETIC_SECRET', text); self.assertNotIn('data_base64', text)
        self.assertEqual(r.evaluate(argparse.Namespace(plan_dir=f.plan_dir, output=output)), 2)
        self.assertFalse((output / 'metrics.json').exists())
        args.endpoint = 'https://different.example/v1/systemone'
        with self.assertRaises(ValueError): r.run(args)
        args.endpoint = 'https://fixture.example/v1/systemone'
        line = r.jsonl(output / 'predictions.jsonl')[0]; r.append_json(output / 'predictions.jsonl', line)
        with self.assertRaises(ValueError): r.run(args)
    def test_timeout_requires_explicit_retry(self):
        f = Fixture(self.root); f.prepare(); f.materialize(); args = f.run_args(self.root / 'timeout')
        with patch.dict(os.environ, {'JEV_OFFLINE_FIXTURE_KEY': 'SYNTHETIC_SECRET'}), patch.object(r.requests, 'post', side_effect=r.requests.Timeout()) as post:
            self.assertEqual(r.run(args), 2); self.assertEqual(post.call_count, 1)
            with self.assertRaises(ValueError): r.run(args)
            self.assertEqual(post.call_count, 1)
    def test_persisted_probability_rejection(self):
        row = {'source_id': 'x', 'shots': 0, 'contract_sha256': 'contract', 'response_model': r.MODEL,
               'probability': 2, 'request_sha256': '0' * 64, 'seconds': 1, 'usage': {}}
        with self.assertRaises(ValueError): r.validate_records([row], {('x', 0)}, 'contract')

if __name__ == '__main__':
    unittest.main(verbosity=2)

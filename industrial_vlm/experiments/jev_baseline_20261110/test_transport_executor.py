"""Offline fixture-only transport amendment tests; never call a real API."""
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
import resume_executor as old
import transport_executor as new
from test_runner import Fixture

class Response:
    headers = {'x-request-id': 'fixture'}
    def __init__(self, status=200): self.status_code = status
    def json(self): return {'model': r.MODEL, 'answers': {'anomaly': {'type': 'noul', 'noul': .6}}, 'usage': {'input_tokens': 3}}

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.f = Fixture(self.root); self.f.prepare(); self.f.materialize()
            self.output = self.root / 'run'; args = self.f.run_args(self.output, limit=None); args.workers = 3; args.dry_run = True
            r.run(args)
        self.contract = r.read_json(self.output / 'contract.json'); self.contract_bytes = (self.output / 'contract.json').read_bytes()
        self.ch = r.sha(r.canonical(self.contract)); self.config, self.plan, _ = r.load_frozen(self.f.plan_dir)
        _, self.images = r.verified_images(self.f.plan_dir, self.config, self.plan)
        self.args = argparse.Namespace(plan_dir=self.f.plan_dir, output=self.output, retry_failed=True,
            dry_run=True, key_env='OFFLINE_TRANSPORT_KEY', stop_file=self.output / 'STOP_V2')
        oldargs = argparse.Namespace(**vars(self.args));oldargs.stop_file=self.output/'STOP'
        with contextlib.redirect_stdout(io.StringIO()): old.execute(oldargs)
        self.old_amendment = r.read_json(self.output / 'operational_amendment.json')
        self.old_hash = r.sha(r.canonical(self.old_amendment))
        self.old_bytes = (self.output / 'operational_amendment.json').read_bytes()
    def tearDown(self): self.tmp.cleanup()
    def request(self,q,s): return r.request_payload(q,self.plan['supports_by_category'][q['category']],s,self.images,self.ch+'|'+q['source_id']+'|'+str(s))
    def attempt(self,q,s,n):
        return {'source_id':q['source_id'],'shots':s,'contract_sha256':self.ch,'request_sha256':r.sha(r.canonical(self.request(q,s))),
            'timestamp_utc':r.now(),'attempt':n,'event':'started','operational_phase':'rolling6','amendment_sha256':self.old_hash}
    def add_success(self,q,s):
        a=self.attempt(q,s,1);r.append_json(self.output/'attempts.jsonl',a)
        row={**a,'category':q['category'],'response_model':r.MODEL,'probability':.6,'seconds':1,'usage':{},'request_id':'old'}
        row.pop('event');row.pop('attempt');r.append_json(self.output/'predictions.jsonl',row)
    def test_old_three_cap_increases_to_five_and_payloads_immutable(self):
        schedule=r.schedule_for(self.plan,None);last=schedule[-1]
        for q,s in schedule[:-1]:self.add_success(q,s)
        q,s=last
        for n in (1,2,3):
            a=self.attempt(q,s,n);r.append_json(self.output/'attempts.jsonl',a)
            r.append_json(self.output/'attempts.jsonl',{**a,'event':'failed','http_status':502,'error':'http_502','seconds':15.2})
        old_predictions=(self.output/'predictions.jsonl').read_bytes()
        expected=r.canonical(self.request(q,s));calls=[]
        class Session:
            trust_env=True
            def post(self,url,**kwargs):
                self_outer.assertEqual(r.canonical(kwargs['json']),expected)
                self_outer.assertEqual(kwargs['timeout'],(60.,120.));self_outer.assertFalse(kwargs['allow_redirects'])
                calls.append(kwargs);return Response(502 if len(calls)==1 else 200)
        self_outer=self;self.args.dry_run=False
        with patch.dict(os.environ,{'OFFLINE_TRANSPORT_KEY':'SYNTHETIC_KEY'}),patch.object(r.requests,'Session',return_value=Session()) as factory,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(new.execute(self.args),0)
            self.assertEqual(len(calls),2);self.assertEqual(factory.call_count,1)
        self.assertEqual((self.output/'contract.json').read_bytes(),self.contract_bytes)
        self.assertEqual((self.output/'operational_amendment.json').read_bytes(),self.old_bytes)
        self.assertTrue((self.output/'predictions.jsonl').read_bytes().startswith(old_predictions))
        rows=r.jsonl(self.output/'predictions.jsonl');self.assertEqual(len(rows),360)
        self.assertEqual(rows[-1]['operational_phase'],'transport_rolling6')
        starts=[x for x in r.jsonl(self.output/'attempts.jsonl') if x['event']=='started' and r.result_key(x)==(q['source_id'],s)]
        self.assertEqual([x['attempt'] for x in starts],[1,2,3,4,5])
        with patch.object(new.getpass,'getpass',side_effect=AssertionError('must not ask key')),patch.object(r.requests,'Session',side_effect=AssertionError('must not send')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(new.execute(self.args),0)
        logs=''.join(x.read_text() for x in self.output.iterdir() if x.is_file())
        self.assertNotIn('SYNTHETIC_KEY',logs);self.assertNotIn('data_base64',logs)
        summary=r.read_json(self.output/'phase_engineering_summary_v2.json')
        self.assertEqual(summary['phases']['transport_rolling6']['successful_targets'],1)
    def test_sanitized_exception_classes_and_session_reuse(self):
        new.SESSION_LOCAL=threading.local()
        class Session:
            trust_env=True
            def post(self,*args,**kwargs):
                try:raise TimeoutError('SECRET_REQUEST_CONTENT')
                except TimeoutError as cause:raise r.requests.ConnectionError('SECRET_AUTH') from cause
        with patch.object(r.requests,'Session',return_value=Session()) as factory:
            a=new.successful_response('https://fixture/v1/systemone','SYNTHETIC_KEY',{})
            b=new.successful_response('https://fixture/v1/systemone','SYNTHETIC_KEY',{})
        self.assertEqual(factory.call_count,1)
        self.assertEqual(a[-1]['exception_class'],'ConnectionError');self.assertIn('TimeoutError',a[-1]['cause_classes'])
        self.assertNotIn('SECRET',str(a));self.assertEqual(a[-1]['stage'],'transport_unknown')
    def test_prefix_tamper_and_stop_v2(self):
        self.args.dry_run=False;self.args.stop_file.touch()
        with patch.dict(os.environ,{'OFFLINE_TRANSPORT_KEY':'SYNTHETIC_KEY'}),patch.object(r.requests,'Session',side_effect=AssertionError('STOP must not submit')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(new.execute(self.args),2)
        self.assertTrue(r.jsonl(self.output/'operational_sessions_v2.jsonl')[-1]['stop_file_observed'])
        # frozen original logs are protected by both amendment chains.
        a=r.read_json(self.output/'operational_amendment_v2.json');self.assertEqual(a['actual_read_timeout_seconds'],120)
        self.assertEqual(a['max_total_started_per_target'],5)

if __name__=='__main__':unittest.main(verbosity=2)

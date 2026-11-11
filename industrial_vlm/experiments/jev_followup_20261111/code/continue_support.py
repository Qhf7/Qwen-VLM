"""Run the already-frozen server support experiment with bounded recovery."""
import argparse
import datetime
import fcntl
import getpass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    assert socket.gethostname()=='autodl-container-132444827f-9c594038'
    preflight=json.loads((root/'stage08_preflight/status.json').read_text())
    assert preflight['status']=='complete' and preflight['completed']==5
    code=root/'code';stage=root/'stage08_support'
    def event(state,**extra):
        value=dict(stage=8,status=state,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        with (root/'pipeline_events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n');stream.flush();os.fsync(stream.fileno())
        status=dict(schema='jev-continuous-research-v1',hostname=socket.gethostname(),
            current_stage=8,confirmation_opened=False,**value)
        tmp=root/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(root/'status.json')
        print(json.dumps(value),flush=True)
    with (root/'.support_coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key=os.environ.get('MODELGO_API_KEY') or getpass.getpass('ModelGo API key (hidden, memory only): ')
        assert key and key.strip()==key
        env=os.environ.copy();env['MODELGO_API_KEY']=key;key=None
        for invocation in range(12):
            if (root/'STOP_PIPELINE').exists() or (stage/'STOP').exists():
                event('stopped_by_user');return 2
            status=json.loads((stage/'status.json').read_text())
            if status['status']=='complete':break
            if status.get('exhausted_unaccepted',0):
                event('bounded_attempts_exhausted',run_status=status);return 2
            command=[sys.executable,str(code/'support_robustness.py'),'run','--root',str(root)]
            if status.get('started_attempts',0):command.append('--retry-failed')
            event('running_support' if invocation==0 else 'explicit_bounded_recovery',invocation=invocation+1,run_status=status)
            with (root/'stage08_pipeline_commands.log').open('a') as output:
                result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT)
            status=json.loads((stage/'status.json').read_text())
            if status['status']=='complete':break
            if status.get('exhausted_unaccepted',0):
                event('bounded_attempts_exhausted',run_status=status);return 2
            if result.returncode not in (0,2):
                event('runtime_requires_audit',returncode=result.returncode,run_status=status);return result.returncode
        else:
            event('invocation_cap_requires_audit',run_status=status);return 2
        env.pop('MODELGO_API_KEY',None)
        event('predictions_complete',run_status=status)
        with (root/'stage08_pipeline_commands.log').open('a') as output:
            result=subprocess.run([sys.executable,str(code/'support_robustness.py'),'evaluate','--root',str(root)],
                stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:
            event('evaluation_requires_audit',returncode=result.returncode);return result.returncode
        event('complete',result='stage08_support/metrics.json',next='merge matched Qwen stage09 and prepare isolated confirmation protocol')
        return 0

if __name__=='__main__':
    sys.exit(main())

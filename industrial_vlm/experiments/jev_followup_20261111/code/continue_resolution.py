"""Wait for support completion, run bounded resolution calls, then analyze."""
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
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    assert socket.gethostname()=='autodl-container-132444827f-9c594038'
    pre=json.loads((root/'stage10_preflight/status.json').read_text())
    assert pre['status']=='complete' and pre['completed']==1
    def event(status,**extra):
        value=dict(stage=10,status=status,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        with (root/'pipeline_events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value),flush=True)
    def execute(command,env=None):
        with (root/'stage10_pipeline_commands.log').open('a') as output:
            return subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
    with (root/'.resolution_coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key=os.environ.get('MODELGO_API_KEY') or getpass.getpass('ModelGo API key (hidden, memory only): ')
        assert key and key.strip()==key
        env=os.environ.copy();env['MODELGO_API_KEY']=key;key=None
        event('waiting_support_completion')
        while True:
            if (root/'STOP_PIPELINE').exists():event('stopped_by_user');return 2
            support=json.loads((root/'stage08_support/status.json').read_text())
            if support['status']=='complete' and (root/'stage08_support/metrics.json').exists():break
            time.sleep(5)
        stage=root/'stage10_resolution';code=root/'code'
        for invocation in range(8):
            state=json.loads((stage/'status.json').read_text())
            if state['status']=='complete':break
            if (root/'STOP_PIPELINE').exists() or (stage/'STOP').exists():event('stopped_by_user');return 2
            if state.get('exhausted_unaccepted',0):event('bounded_attempts_exhausted',run_status=state);return 2
            command=[sys.executable,str(code/'resolution_ablation.py'),'run','--root',str(root)]
            if state['started_attempts']:command.append('--retry-failed')
            event('running_resolution',invocation=invocation+1,run_status=state)
            result=execute(command,env)
            state=json.loads((stage/'status.json').read_text())
            if state['status']=='complete':break
            if result not in (0,2):event('runtime_requires_audit',returncode=result);return result
        else:event('invocation_cap_requires_audit');return 2
        env.pop('MODELGO_API_KEY',None)
        event('predictions_complete',run_status=state)
        result=execute([sys.executable,str(code/'resolution_ablation.py'),'evaluate','--root',str(root)])
        if result:event('evaluation_requires_audit',returncode=result);return result
        event('complete',result='stage10_resolution/metrics.json',next='joint frozen followup analysis')
        return 0

if __name__=='__main__':sys.exit(main())

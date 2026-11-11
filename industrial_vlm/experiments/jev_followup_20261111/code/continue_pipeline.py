"""Server stage coordinator. Secrets remain in process memory only."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    assert socket.gethostname()=='autodl-container-132444827f-9c594038'
    root=args.root.resolve()
    py=sys.executable
    code=root/'code'
    log=root/'pipeline_events.jsonl'
    def event(stage,status,**extra):
        item=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  stage=stage,status=status,**extra)
        with log.open('a') as stream:
            stream.write(json.dumps(item,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
        state=dict(schema='jev-continuous-research-v1',hostname=socket.gethostname(),
                   current_stage=stage,status=status,updated_at=item['timestamp_utc'],
                   confirmation_opened=False,**extra)
        temp=root/'status.tmp'
        temp.write_text(json.dumps(state,indent=2));temp.replace(root/'status.json')
        print(json.dumps(item),flush=True)
    def execute(script,arguments,env=None):
        with (root/'pipeline_commands.log').open('a') as stream:
            result=subprocess.run([py,str(code/script),*arguments],env=env,
                                  stdout=stream,stderr=subprocess.STDOUT)
        return result.returncode
    with (root/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy()
        key=env.get('MODELGO_API_KEY') or getpass.getpass('ModelGo API key (hidden, memory only): ')
        assert key and key.strip()==key
        env['MODELGO_API_KEY']=key
        key=None
        event(4,'monitoring_formal_roi',auto_transition=True)
        output=root/'runs/modelgo_roi'
        for invocation in range(4):
            while True:
                if (root/'STOP_PIPELINE').exists():
                    event(4,'stopped_by_user');return 2
                with (output/'.run.lock').open('a') as run_lock:
                    try:
                        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        busy=True
                    else:
                        busy=False
                        fcntl.flock(run_lock,fcntl.LOCK_UN)
                if not busy:break
                time.sleep(5)
            status=json.loads((output/'status.json').read_text())
            if status['status']=='complete':break
            if (output/'STOP').exists():
                event(4,'stopped_by_run_control');return 2
            if status.get('exhausted_unaccepted_targets',0):
                event(4,'bounded_retry_exhausted',run_status=status);return 2
            if invocation==3:
                event(4,'requires_transport_audit',run_status=status);return 2
            event(4,'explicit_bounded_retry',invocation=invocation+1,
                  note='Unknown server completion can duplicate billing; max3 starts/target frozen',run_status=status)
            execute('roi_api.py',['run','--root',str(root),'--output','runs/modelgo_roi','--retry-failed'],env)
        else:
            event(4,'incomplete');return 2
        env.pop('MODELGO_API_KEY',None)
        event(4,'complete',run_status=status)
        event(5,'evaluating_frozen_predictions')
        result=execute('evaluate_roi.py',['--root',str(root),'--run','runs/modelgo_roi'])
        if result:
            event(5,'evaluation_requires_audit',returncode=result);return result
        event(5,'complete',result='stages/05_evaluation.json')
        event(6,'running_budget_routing_replay')
        result=execute('routing_replay.py',['evaluate','--root',str(root)])
        if result:
            event(6,'routing_requires_audit',returncode=result);return result
        event(6,'complete',result='stages/06_routing_results.json',
              next='calibration and multi-support protocols require explicit isolation audit')
        return 0

if __name__=='__main__':
    sys.exit(main())

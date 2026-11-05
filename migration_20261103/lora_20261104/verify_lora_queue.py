"""Run fresh-process checkpoint checks after the serial training queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def save(value, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve()
    status=dict(state='waiting_for_training',checks=[])
    save(status,root/'verify_status.json')
    while True:
        training=json.load(open(root/'status.json'))
        if training['state']=='failed':
            status.update(state='blocked_by_training_failure',error=training.get('error'))
            save(status,root/'verify_status.json')
            return
        if training['state']=='complete':break
        time.sleep(15)
    checks=[('C',42,True),('B',42,False)]+[('C',s,False) for s in (42,43,44)]
    for arm,seed,preflight in checks:
        status.update(state='running',arm=arm,seed=seed,preflight=preflight)
        save(status,root/'verify_status.json')
        cmd=[sys.executable,'-u','verify_lora_checkpoint.py','--root',str(root),
             '--arm',arm,'--seed',str(seed),'--mode','final']
        if preflight:cmd+=['--preflight']
        rc=subprocess.call(cmd)
        if rc:
            status.update(state='failed',exit_code=rc)
            save(status,root/'verify_status.json')
            raise RuntimeError('Fresh checkpoint audit failed')
        status['checks'].append(dict(arm=arm,seed=seed,preflight=preflight,state='passed'))
    status.update(state='complete')
    save(status,root/'verify_status.json')


if __name__=='__main__':main()

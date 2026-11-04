"""Wait for D/pairwise training, then reload all nine formal checkpoints."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/advanced_20261104')
    status_path = root / 'status.json'
    verification = dict(state='waiting_for_training', checks=[])
    def save():
        temp = root / 'verification_status.tmp'
        temp.write_text(json.dumps(verification, indent=2) + '\n')
        temp.replace(root / 'verification_status.json')
    save()
    while True:
        status = json.loads(status_path.read_text())
        if status['state'] == 'failed':
            verification.update(state='blocked_by_training_failure', training_error=status.get('error'))
            save()
            return
        if status['state'] == 'complete':
            break
        time.sleep(15)
    root_arg = '/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/advanced_20261104'
    runs = [root / ('arm_D_s%d' % seed) for seed in (42, 43, 44)]
    runs += [root / ('Cpair_s%d' % seed) for seed in (42, 43, 44)]
    runs += [root / ('R_s%d' % seed) for seed in (42, 43, 44)]
    for run in runs:
        verification.update(state='running', run=run.name)
        save()
        cmd = [sys.executable, '-u', 'verify_advanced_checkpoint.py', '--root', root_arg,
               '--run', str(run), '--mode', 'final']
        code = subprocess.call(cmd)
        if code:
            verification.update(state='failed', run=run.name, exit_code=code)
            save()
            raise RuntimeError('Fresh reload failed for ' + run.name)
        result = json.loads((run/'fresh_final_reload.json').read_text())
        verification['checks'].append(dict(run=run.name, state='passed', n=result['n']))
        save()
    verification.update(state='complete')
    save()


if __name__ == '__main__':
    main()

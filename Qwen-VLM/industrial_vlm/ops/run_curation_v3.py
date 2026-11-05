"""Sequential bounded v3 execution; archived rounds and confirmation stay untouched."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/curation_v3'
PY='/root/autodl-tmp/vlm/.venv/bin/python'
ARMS=('normal_random','normal_hard','region_highres')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(path.read_text())


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+chr(10))
    temporary.replace(path)


def freeze_inputs(name,paths):
    observed={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    ledger=EXP/(name+'.json')
    if ledger.exists():
        if read(ledger)!=observed:raise RuntimeError(f'Frozen inputs differ: {name}')
    else:write(ledger,observed)


def verify_frozen():
    for ledger in (EXP/'initial_input_hashes.json',EXP/'selected_input_hashes.json'):
        if not ledger.exists():continue
        for relative,expected in read(ledger).items():
            if sha(ROOT/relative)!=expected:raise RuntimeError(f'Frozen source/input changed: {relative}')


def stage(name,arguments,marker=None,kind=None):
    verify_frozen()
    marker=ROOT/marker if marker else None
    if marker is not None and marker.exists():
        data=read(marker)
        if data.get('status') in ('complete','passed'):
            if kind=='train':
                stored=read(marker.parent/'run_config.json')
                supplied=read(Path(arguments[arguments.index('--config')+1]))
                if any(stored.get(k)!=v for k,v in supplied.items()):
                    raise RuntimeError(f'Completed train config differs: {name}')
                if (not data.get('reload_check',{}).get('passed')
                    or sha(marker.parent/'exposures.jsonl')!=data['exposure_trace_sha256']
                    or sha(Path(stored['train_file']))!=stored['train_sha256']):
                    raise RuntimeError(f'Completed training failed identity checks: {name}')
            if kind in ('score','generation') and data['n_completed']!=data['config']['n_expected']:
                raise RuntimeError(f'Completed eval has incomplete rows: {name}')
            print(json.dumps({'event':'already_complete','stage':name}),flush=True)
            return
        if kind=='train' and any(marker.parent.iterdir()):
            raise RuntimeError(f'Incomplete training preserved: {marker.parent}')
    command=[PY,'-u',*arguments]
    write(EXP/'state.json',{'status':'running','stage':name,'command':command,'confirmation_evaluated':False})
    print(json.dumps({'event':'start','stage':name,'unix':time.time()}),flush=True)
    env={**os.environ,'PYTHONPATH':str(EXP/'runtime_src'),'OMP_NUM_THREADS':'8',
        'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    started=time.perf_counter()
    with (EXP/'logs'/(name+'.log')).open('a') as log:
        code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    duration=time.perf_counter()-started
    with (EXP/'stage_timings.jsonl').open('a') as stream:
        stream.write(json.dumps({'stage':name,'wall_seconds_including_process_hash_load':duration,
                                 'exit_code':code,'finished_unix':time.time()})+chr(10))
    if code:raise RuntimeError(f'{name} exited {code}; inspect its stage log')
    if marker and (not marker.exists() or read(marker).get('status') not in ('complete','passed')):
        raise RuntimeError(f'Missing completion marker: {name}')
    print(json.dumps({'event':'finished','stage':name,'seconds':duration}),flush=True)


def main():
    EXP.mkdir(parents=True,exist_ok=True)
    (EXP/'logs').mkdir(exist_ok=True)
    runtime=EXP/'runtime_src'
    if not runtime.exists():
        shutil.copytree(ROOT/'src',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    initial=(list(runtime.rglob('*.py'))+[ROOT/'ops'/name for name in
        ('run_curation_v3.py','prepare_curation_v3.py','score_train_mining_v3.py','preflight_curation_v3.py')]
        +[ROOT/'prepared/curation_v3'/name for name in
        ('plan.json','mining_normal_candidates.jsonl')]+[ROOT/'reports/curation_v3/实验协议.md'])
    freeze_inputs('initial_input_hashes',initial)
    stage('preserve_v2',[str(ROOT/'ops/verify_delivery_multitask_v2.py')])
    mining=ROOT/'evaluations/curation_v3/mining_base_train_normal'
    mine_args=[str(ROOT/'ops/score_train_mining_v3.py')]
    if (mining/'meta.json').exists():mine_args+=['--resume']
    stage('mining_base_train_normal',mine_args,
          'evaluations/curation_v3/mining_base_train_normal/meta.json','score')
    if not (ROOT/'prepared/curation_v3/selection_audit.json').exists():
        stage('select_train_normals',[str(ROOT/'ops/prepare_curation_v3.py'),'--phase','select'])
    if not all((ROOT/'configs/curation_v3'/(arm+'.json')).exists() for arm in ARMS):
        raise RuntimeError('Selected configurations are missing')
    paths=[p for p in (ROOT/'prepared/curation_v3').glob('*.json*') if p.name!='highres_smoke.jsonl']
    paths += [p for p in (ROOT/'configs/curation_v3').glob('*.json') if p.name!='highres_smoke.json']
    freeze_inputs('selected_input_hashes',paths)
    stage('native_highres_preflight',[str(ROOT/'ops/preflight_curation_v3.py')],
          'experiments/curation_v3/native_input_preflight.json')
    stage('highres_smoke',['-m','industrial_vlm.train','--config',str(ROOT/'configs/curation_v3/highres_smoke.json')],
          'runs/curation_v3/highres_smoke/summary.json','train')
    for arm in ARMS:
        stage('train_'+arm,['-m','industrial_vlm.train','--config',str(ROOT/'configs/curation_v3'/(arm+'.json'))],
              f'runs/curation_v3/{arm}/summary.json','train')
    refs={'region_low_low_region':'evaluations/multitask_v2/region_region_region'}
    matrix=[('normal_random_decision','normal_random','decision',524288),
            ('normal_hard_decision','normal_hard','decision',524288),
            ('region_low_high_region','v2','region',1048576),
            ('region_high_low_region','region_highres','region',524288),
            ('region_high_high_region','region_highres','region',1048576)]
    for name,arm,task,pixels in matrix:
        adapter=(ROOT/'runs/multitask_v2/region_region/adapter' if arm=='v2'
                 else ROOT/'runs/curation_v3'/arm/'adapter')
        relative='evaluations/curation_v3/'+name
        output=ROOT/relative
        arguments=['-m','industrial_vlm.evaluate','--model',str(ROOT/'models/Qwen3-VL-8B-Instruct'),
            '--adapter',str(adapter),'--data',str(ROOT/'prepared/curation_v3'/f'val_{task}_120.jsonl'),
            '--image-root',str(ROOT/'datasets'),'--output',str(output),'--run-name',name,
            '--branch','curation_v3_validation','--max-pixels',str(pixels),'--max-length','2048',
            '--max-new-tokens','128','--bootstrap-samples','200']
        if (output/'meta.json').exists():arguments+=['--resume']
        stage('val_'+name,arguments,relative+'/meta.json','generation')
        refs[name]=relative
    write(EXP/'evaluation_refs.json',refs)
    for arm in ('normal_random','normal_hard'):
        output=ROOT/'evaluations/curation_v3'/(arm+'_validation')
        arguments=['-m','industrial_vlm.score','score','--model',str(ROOT/'models/Qwen3-VL-8B-Instruct'),
            '--adapter',str(ROOT/'runs/curation_v3'/arm/'adapter'),
            '--data',str(ROOT/'prepared/curation_v3/val_decision_120.jsonl'),
            '--image-root',str(ROOT/'datasets'),'--output',str(output),'--data-role','validation',
            '--max-pixels','524288','--max-length','2048','--run-name',arm+'_validation']
        if (output/'meta.json').exists():arguments+=['--resume']
        stage('score_'+arm,arguments,str(output.relative_to(ROOT))+'/meta.json','score')
    verify_frozen()
    stage('preserve_v2_final',[str(ROOT/'ops/verify_delivery_multitask_v2.py')])
    write(EXP/'state.json',{'status':'complete','stage':'all_training_validation_complete',
        'confirmation_evaluated':False,'final_report_pending':True,'finished_unix':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if EXP.exists():
            prior=read(EXP/'state.json') if (EXP/'state.json').exists() else {}
            write(EXP/'state.json',{**prior,'status':'failed','error':str(exc)})
        raise

"""Frozen S/C-structured-prompt MVTec audit: infer first, open masks afterward.

The caller owns the shared GPU lock. No external result is used to change a
prompt, checkpoint, generation budget, or target transformation in this file.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from structured_model import StructuredLanguageBridge, load_fixed_c
from structured_geometry import (SCHEMA, sha256_file, mask_to_target,
                                 localization_metrics, parse_structured)
import numpy as np
import torch
from peft import set_peft_model_state_dict
from migration_advanced_train import adapter_state, bridge_digest
from migration_metrics import evaluate_predictions
from visa_pipeline import PIPELINE_VERSION
from advanced_generation import force_generation_policy

EXPECTED_MANIFEST_SHA256 = '4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a'
EXPECTED_COUNTS = {'n': 1575, 'normal': 392, 'anomaly': 1183, 'categories': 15}


def atomic_json(value, path):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def relative_under(root, relative):
    candidate=Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('Manifest data path must be relative without parent traversal')
    result=(Path(root)/candidate).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Manifest data path escapes data root')
    return result


def validate_manifest(records):
    """Metadata-only; intentionally does not resolve or open any mask."""
    assert len(records)==EXPECTED_COUNTS['n']
    assert len({r['source_id'] for r in records})==len(records)
    assert len({r['path'] for r in records})==len(records)
    labels=[int(r['label']) for r in records]
    assert set(labels)=={0,1} and sum(labels)==EXPECTED_COUNTS['anomaly']
    assert len(labels)-sum(labels)==EXPECTED_COUNTS['normal']
    assert len({r['category'] for r in records})==EXPECTED_COUNTS['categories']
    for r in records:
        assert r['source_id']=='mvtec_'+r['image_sha256']
        assert r['prompt_category'] in ('generic_industrial_component','industrial_component')
        assert int(r['width'])>0 and int(r['height'])>0
        if r['label']:
            assert r['mask_path'] and r['mask_sha256']
        else:
            assert r['mask_path'] is None and r['mask_sha256'] is None


def validate_external_cache(cache, sidecar, records, manifest_sha, decoder_sha):
    assert sidecar['schema']==cache['schema']=='imagebind-bigdec-external-cache-v1'
    for metadata in (cache,sidecar):
        assert metadata['manifest_sha256']==manifest_sha
        assert metadata['decoder_sha256']==decoder_sha
        assert metadata['pipeline_version']==PIPELINE_VERSION
        assert metadata['geometry']=='direct-square224'
        assert metadata['images']==len(records)
        assert metadata['gt_masks_opened'] is False
        assert metadata['images_for_training'] is False
        assert metadata['generation_or_qwen_queries']==0
    assert len(cache['items'])==len(records)
    for item,record in zip(cache['items'],records):
        assert item['source_id']==record['source_id']
        assert item['label']==record['label'] and item['category_original']==record['category']
        assert item['category']=='industrial_component' and item['split']=='external'
        assert Path(item['path']).as_posix().endswith('/'+record['path'])
    for key,shape in (('cls',(len(records),1024)),('maps',(len(records),224,224))):
        assert tuple(cache[key].shape)==shape and cache[key].dtype==torch.float32
        assert torch.isfinite(cache[key]).all()
    assert cache['maps'].min()>=0 and cache['maps'].max()<=1


def posthoc_targets(records, data_root, inference_marker, prediction_path):
    """Only legal after every immutable raw model output has been written."""
    marker=json.loads(Path(inference_marker).read_text())
    assert marker['state']=='inference_complete' and marker['processed']==len(records)
    assert marker['mask_reads']==0 and marker['predictions_sha256']==sha256_file(prediction_path)
    from PIL import Image
    targets=[];mask_reads=0
    for record in records:
        if int(record['label'])==0:
            target=mask_to_target(None,0)
        else:
            path=relative_under(data_root,record['mask_path'])
            if sha256_file(path)!=record['mask_sha256']:
                raise ValueError('Frozen external mask identity changed: '+record['mask_path'])
            with Image.open(path) as mask:
                if mask.size!=(record['width'],record['height']):
                    raise ValueError('Mask/image geometry mismatch: '+record['path'])
                target=mask_to_target(np.array(mask),1)
            mask_reads+=1
        targets.append(dict(source_id=record['source_id'],category=record['category'],**target))
    return targets,mask_reads


def load_and_validate_sources(run_dir, initial_c):
    spec=json.loads((run_dir/'run.json').read_text());identity=spec['identity'];cfg=spec['config']
    assert identity['schema']==SCHEMA and not cfg['preflight']
    for filename,expected_sha in spec['source_sha256'].items():
        assert sha256_file(Path(__file__).parent/filename)==expected_sha,('Training source changed',filename)
    complete=json.loads((run_dir/'complete.json').read_text())
    assert complete['state']=='complete' and not complete['preflight'] and complete['steps']==500
    assert complete['seed']==identity['seed'] and complete['test_queries']==complete['external_queries']==0
    assert cfg['max_new_tokens']==80 and (cfg['microbatch'],cfg['accum'],cfg['lr'])==(1,4,1e-4)
    source_c=Path(cfg['source_c'])
    assert str(source_c.resolve())==identity['source_c']
    for path,key in ((source_c/'run.json','source_c_run_sha256'),(source_c/'best.pt','source_c_best_sha256'),
                     (Path(cfg['cache']),'cache_sha256'),(Path(cfg['targets']),'targets_sha256'),
                     (Path(cfg['base_manifest']),'base_manifest_sha256')):
        assert sha256_file(path)==identity[key],('Source identity changed',str(path))
    source=json.loads((source_c/'run.json').read_text());cid=source['identity'];ccfg=source['config']
    assert cid['arm']=='C' and cid['seed']==identity['seed']
    assert not ccfg.get('preflight',False) and not ccfg.get('paired_sampling',False)
    assert ccfg.get('objective','ce')=='ce' and (ccfg['rank'],ccfg['alpha'])==(8,16)
    assert cid['cache_sha256']==identity['cache_sha256']
    assert cid['base_manifest_sha256']==identity['base_manifest_sha256']
    assert cid['pipeline_version']==PIPELINE_VERSION
    assert sha256_file(cid['bridge_path'])==cid['bridge_sha256']
    source_bridge=torch.load(cid['bridge_path'],map_location='cpu')
    for key in ('pipeline_version','decoder_sha256','cache_sha256','cache_manifest_sha256'):
        assert source_bridge[key]==cid[key],('Original bridge metadata mismatch',key)
    assert source_bridge['backend']=='qwen' and source_bridge['config']['seed']==identity['seed']
    train_sidecar=json.loads(Path(cfg['cache']).with_suffix('.json').read_text())
    assert train_sidecar['cache_sha256']==identity['cache_sha256']
    assert train_sidecar['decoder_sha256']==cid['decoder_sha256']
    assert train_sidecar['manifest_sha256']==cid['cache_manifest_sha256']
    target_sidecar=json.loads(Path(cfg['targets']).with_suffix('.json').read_text())
    assert target_sidecar['cache_sha256']==identity['cache_sha256']
    assert target_sidecar['manifest_sha256']==identity['targets_sha256'] and target_sidecar['test_mask_reads']==0
    base=json.loads(Path(cfg['base_manifest']).read_text())
    for name,entry in base['files'].items():
        path=Path(base['model_path'])/name
        assert path.stat().st_size==entry['bytes'] and sha256_file(path)==entry['sha256']
    checkpoint=source_c/'best.pt' if initial_c else run_dir/'best.pt'
    state=torch.load(checkpoint,map_location='cpu')
    assert state['identity']==(cid if initial_c else identity)
    assert state['step']==(identity['source_c_step'] if initial_c else complete['best_step'])
    return spec,source,base,state,checkpoint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--external-cache',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True,help='Opened only after all model predictions are finalized')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--initial-c',action='store_true',help='Original C best with the identical structured prompt; no S continuation')
    args=ap.parse_args();run=args.run.resolve()
    out=args.out or run/('external_mvtec_structured_initial_C_v1' if args.initial_c else 'external_mvtec_structured_S_v1')
    if out.exists():raise FileExistsError('Existing outputs are preserved: '+str(out))
    manifest_sha=sha256_file(args.manifest)
    assert manifest_sha==EXPECTED_MANIFEST_SHA256
    protocol=json.loads(args.protocol.read_text())
    assert protocol['protocol']=='visa_to_mvtec_residual_official_test_v1'
    assert protocol['manifest_sha256']==manifest_sha and protocol['retained_n']==1575
    assert protocol['n_normal']==392 and protocol['n_anomaly']==1183
    records=[json.loads(x) for x in args.manifest.read_text().splitlines() if x]
    validate_manifest(records)
    spec,source,base,state,checkpoint=load_and_validate_sources(run,args.initial_c)
    cfg,identity=spec['config'],spec['identity'];decoder_sha=source['identity']['decoder_sha256']
    sidecar=json.loads(args.external_cache.with_suffix('.json').read_text())
    external_cache_sha=sha256_file(args.external_cache)
    assert external_cache_sha==sidecar['cache_sha256']
    cache=torch.load(args.external_cache,map_location='cpu')
    validate_external_cache(cache,sidecar,records,manifest_sha,decoder_sha)
    torch.set_num_threads(8);torch.manual_seed(identity['seed'])
    model=StructuredLanguageBridge('qwen',device='cuda',init_seed=identity['seed'])
    load_fixed_c(model,cfg['source_c'])
    assert str(model.model_path)==base['model_path']
    assert bridge_digest(model)==identity['source_bridge_sha256']
    if not args.initial_c:
        result=set_peft_model_state_dict(model.language_model,state['adapter']);assert not result.unexpected_keys
    current=adapter_state(model)
    assert current.keys()==state['adapter'].keys() and all(torch.equal(current[k],v) for k,v in state['adapter'].items())
    assert any(torch.count_nonzero(v).item() for k,v in current.items() if 'lora_B' in k)
    generation_path=run/('initial_generation_policy.json' if args.initial_c else 'best_generation_policy.json')
    if generation_path.exists():
        policy=json.loads(generation_path.read_text());force_generation_policy(model,policy)
        assert policy['max_new_tokens']==80 and policy['batch_size']==cfg['eval_batch_size']
    else:
        policy=dict(selected='historical_scalar',note='Pre-generation-optimization run without policy file')
    out.mkdir(parents=True,exist_ok=False)
    frozen=dict(run=str(run),seed=identity['seed'],condition='initial_C_structured_prompt' if args.initial_c else 'S_best',
                checkpoint_step=state['step'],checkpoint_sha256=sha256_file(checkpoint),
                source_c_best_sha256=identity['source_c_best_sha256'],source_c_run_sha256=identity['source_c_run_sha256'],
                structured_run_sha256=sha256_file(run/'run.json'),train_cache_sha256=identity['cache_sha256'],
                train_targets_sha256=identity['targets_sha256'],base_manifest_sha256=identity['base_manifest_sha256'],
                decoder_sha256=decoder_sha,manifest_sha256=manifest_sha,protocol_sha256=sha256_file(args.protocol),
                external_cache_sha256=external_cache_sha,prompt_category='industrial_component',
                max_new_tokens=80,batch_size=cfg['eval_batch_size'],schema=SCHEMA,
                generation_policy=policy['selected'],generation_policy_sha256=sha256_file(generation_path) if generation_path.exists() else None,
                evaluator_sha256=sha256_file(__file__),model_source_sha256=sha256_file(Path(__file__).parent/'structured_model.py'),
                mask_reads_at_start=0,selection='Frozen before external model queries; no external selection')
    atomic_json(frozen,out/'frozen_config.json')
    torch.cuda.reset_peak_memory_stats();model.eval();started=time.time();output=[]
    prediction_path=out/'predictions.jsonl';temporary=prediction_path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for start in range(0,len(records),cfg['eval_batch_size']):
            end=min(start+cfg['eval_batch_size'],len(records))
            # Model receives neither labels nor masks or target boxes. A single
            # generic prompt is used for every true external object category.
            cls=cache['cls'][start:end].cuda();maps=cache['maps'][start:end].cuda()
            packed=model.pack(cls,maps,['industrial_component']*(end-start))
            torch.cuda.synchronize();tick=time.time()
            with torch.no_grad():
                scores=model.score(packed).float().cpu().tolist()
                texts=model.generate(packed,max_new_tokens=80)
            torch.cuda.synchronize();latency=(time.time()-tick)/(end-start)
            for record,score,text in zip(records[start:end],scores,texts):
                item=dict(source_id=record['source_id'],path=record['path'],category=record['category'],
                          label=int(record['label']),split='external',score=float(score),generated=text,
                          parsed=parse_structured(text),latency_seconds=latency)
                stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');output.append(item)
            stream.flush();del packed,cls,maps
            if end%64<cfg['eval_batch_size'] or end==len(records):
                progress=dict(state='inferring',processed=end,total=len(records),elapsed_seconds=time.time()-started,mask_reads=0)
                atomic_json(progress,out/'progress.json');print(json.dumps(progress),flush=True)
    assert len(output)==len(records)
    temporary.replace(prediction_path);prediction_sha=sha256_file(prediction_path)
    (out/'predictions.sha256').write_text(prediction_sha+'  predictions.jsonl\n')
    inference=dict(state='inference_complete',processed=len(output),mask_reads=0,predictions_sha256=prediction_sha,
                   inference_seconds=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
                   frozen_config_sha256=sha256_file(out/'frozen_config.json'))
    atomic_json(inference,out/'inference_complete.json')
    # Ground-truth mask access starts here, after every raw score/generation
    # is durably committed and hashed. It cannot change any model prediction.
    targets,mask_reads=posthoc_targets(records,args.data_root,out/'inference_complete.json',prediction_path)
    scores=[x['score'] for x in output];texts=[x['generated'] for x in output]
    metrics=evaluate_predictions(output,scores,texts)
    metrics['localization']=localization_metrics(targets,texts)
    metrics['localization_per_category']={c:localization_metrics([t for t in targets if t['category']==c],
        [r['generated'] for r in output if r['category']==c]) for c in sorted({r['category'] for r in records})}
    atomic_json(metrics,out/'metrics.json')
    atomic_json(targets,out/'posthoc_targets.json')
    assert sha256_file(prediction_path)==prediction_sha,'Posthoc evaluation changed predictions'
    summary=dict(state='complete',**frozen,n=len(output),counts=dict(Counter(str(r['label']) for r in output)),
                 metrics=metrics,predictions_sha256=prediction_sha,posthoc_targets_sha256=sha256_file(out/'posthoc_targets.json'),
                 inference_seconds=inference['inference_seconds'],total_elapsed_seconds=time.time()-started,
                 mean_latency_seconds=sum(r['latency_seconds'] for r in output)/len(output),
                 peak_memory_bytes=inference['peak_memory_bytes'],mask_reads_during_inference=0,
                 posthoc_mask_reads=mask_reads,historical_visa_test_queries=0,external_queries=len(output),
                 limitation='Public cross-dataset transfer audit; base pretraining exposure unknown. S adds mask supervision and answer tokens; no causal LoRA-only attribution.')
    atomic_json(summary,out/'summary.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

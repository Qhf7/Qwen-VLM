"""Fresh-process exact reload audit for structured language LoRA experiments."""
import argparse
import json
from pathlib import Path

from structured_model import StructuredLanguageBridge, load_fixed_c
from structured_geometry import SCHEMA, sha256_file
import torch
from peft import set_peft_model_state_dict
from migration_advanced_train import adapter_state, bridge_digest
from dec_big import atomic_json
from advanced_generation import force_generation_policy


def choose_batches(records, batch_size, n):
    """Keep original evaluation batches for bit-exact BF16 score comparison."""
    chunks=[records[j:j+batch_size] for j in range(0,len(records),batch_size)]
    priority=[]
    for category in sorted({r['category'] for r in records}):
        for label in (0,1):
            found=next((j for j,chunk in enumerate(chunks) if any(r['category']==category and r['label']==label for r in chunk)),None)
            if found is not None and found not in priority:
                priority.append(found)
    for j in (0,len(chunks)//2,len(chunks)-1):
        if j not in priority:priority.append(j)
    selected=[chunks[j] for j in sorted(priority)]
    if sum(map(len,selected))<n:
        for j,chunk in enumerate(chunks):
            if j not in priority:
                selected.append(chunk)
                if sum(map(len,selected))>=n:break
    return selected


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--mode',choices=['best','final'],default='final');ap.add_argument('--n',type=int,default=32)
    args=ap.parse_args();run=args.run.resolve()
    assert (run/'complete.json').exists()
    spec=json.loads((run/'run.json').read_text());cfg=spec['config'];identity=spec['identity']
    assert identity['schema']==SCHEMA
    for path,key in ((cfg['cache'],'cache_sha256'),(cfg['targets'],'targets_sha256'),(cfg['base_manifest'],'base_manifest_sha256')):
        assert sha256_file(path)==identity[key]
    assert sha256_file(Path(cfg['source_c'])/'best.pt')==identity['source_c_best_sha256']
    assert sha256_file(Path(cfg['source_c'])/'run.json')==identity['source_c_run_sha256']
    torch.set_num_threads(8);torch.manual_seed(identity['seed'])
    model=StructuredLanguageBridge('qwen',device='cuda',init_seed=identity['seed'])
    load_fixed_c(model,cfg['source_c'])
    source_hash=bridge_digest(model);assert source_hash==identity['source_bridge_sha256']
    state=torch.load(run/(args.mode+'.pt'),map_location='cpu');assert state['identity']==identity
    result=set_peft_model_state_dict(model.language_model,state['adapter']);assert not result.unexpected_keys
    actual=adapter_state(model);assert actual.keys()==state['adapter'].keys()
    assert all(torch.equal(actual[k],v) for k,v in state['adapter'].items())
    assert any(torch.count_nonzero(v).item() for k,v in actual.items() if 'lora_B' in k)
    cache=torch.load(cfg['cache'],map_location='cpu');items=cache['items']
    reference_name='final_val_predictions.json' if args.mode=='final' else 'best_val_generated_predictions.json'
    references=json.loads((run/reference_name).read_text())
    chunks=choose_batches(references,cfg['eval_batch_size'],args.n)
    errors=[];n=0;labels=set();categories=set()
    generation_path=run/(args.mode+'_generation_policy.json')
    if generation_path.exists():
        policy=json.loads(generation_path.read_text());force_generation_policy(model,policy)
    else:
        policy=dict(selected='historical_scalar',note='Pre-generation-optimization run without policy file')
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            ids=[r['index'] for r in chunk]
            assert all(items[i]['split']=='val' for i in ids)
            packed=model.pack(cache['cls'][ids].cuda(),cache['maps'][ids].cuda(),[items[i]['category'] for i in ids])
            scores=model.score(packed).float().cpu().tolist()
            texts=model.generate(packed,max_new_tokens=cfg['max_new_tokens'])
            errors.extend(abs(r['score']-s) for r,s in zip(chunk,scores))
            assert texts==[r['generated'] for r in chunk], 'Fresh process generation mismatch'
            n+=len(chunk);labels.update(r['label'] for r in chunk);categories.update(r['category'] for r in chunk)
    assert max(errors)==0.,('Fresh process score mismatch',max(errors))
    assert bridge_digest(model)==source_hash
    result=dict(state='complete',mode=args.mode,step=state['step'],n=n,categories=sorted(categories),labels=sorted(labels),
                original_batch_composition_preserved=True,fresh_reload_max_abs_score_error=max(errors),
                generated_exact_match=True,adapter_tensor_exact_match=True,adapter_B_nonzero=True,
                bridge_unchanged=True,source_sha256=sha256_file(__file__),test_queries=0,external_queries=0,
                generation_policy=policy['selected'],generation_policy_sha256=sha256_file(generation_path) if generation_path.exists() else None)
    atomic_json(result,str(run/('fresh_'+args.mode+'_reload.json')));print(json.dumps(result))


if __name__=='__main__':main()

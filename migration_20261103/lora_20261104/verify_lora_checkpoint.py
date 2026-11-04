"""Fresh-process trained checkpoint audit on previously evaluated val images."""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from migration_language import FrozenLanguageBridge
from migration_lora_train import (adapter_state, configure_arm, load_bridge,
                                  load_delta, sha256_file)
from dec_big import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--arm', choices=['B', 'C'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--mode', choices=['best', 'final'], default='final')
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    with open(root.parent/'gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        d = root / (('preflight_' if args.preflight else 'arm_') + args.arm + '_s' + str(args.seed))
        assert (d/'complete.json').exists()
        run = json.load(open(d/'run.json'))
        cfg, identity = run['config'], run['identity']
        assert sha256_file(identity['bridge_path']) == identity['bridge_sha256']
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        model = FrozenLanguageBridge('qwen', device='cuda', init_seed=args.seed)
        load_bridge(model, torch.load(identity['bridge_path'], map_location='cpu'))
        configure_arm(model, args.arm, cfg['rank'], cfg['alpha'], cfg['lr_bridge'], cfg['lr_lora'])
        state = load_delta(model, d/(args.mode+'.pt'), identity)
        cache = torch.load(cfg['cache'], map_location='cpu')
        reference_name = 'final_val_predictions.json' if args.mode == 'final' else 'best_val_generated_predictions.json'
        reference = json.load(open(d/reference_name))[:8]
        paths = {x['path']:i for i,x in enumerate(cache['items'])}
        ids = [paths[x['path']] for x in reference]
        assert all(cache['items'][i]['split']=='val' for i in ids)
        model.eval()
        with torch.no_grad():
            packed = model.pack(cache['cls'][ids].cuda(), cache['maps'][ids].cuda(),
                                [cache['items'][i]['category'] for i in ids])
            scores = model.score(packed).float().cpu().tolist()
            texts = model.generate(packed, max_new_tokens=32)
        error = max(abs(x['score']-s) for x,s in zip(reference,scores))
        assert error == 0., ('Fresh model score mismatch', error)
        assert texts == [x['generated'] for x in reference], 'Fresh model generation mismatch'
        trained_lora_nonzero = None
        if args.arm == 'C':
            delta = adapter_state(model)
            trained_lora_nonzero = any(torch.count_nonzero(v).item()>0 for k,v in delta.items() if 'lora_B' in k)
            if args.mode == 'final':
                assert trained_lora_nonzero, 'Final trained adapter is still zero'
            with model.language_model.disable_adapter():
                disabled = model.score(packed).float().cpu().tolist()
            initial = {x['path']:x for x in json.load(open(d/'initial_val_predictions.json'))}
            disabled_error = max(abs(initial[x['path']]['score']-s) for x,s in zip(reference,disabled))
            assert disabled_error == 0., ('Disabled adapter did not recover A', disabled_error)
        result = dict(arm=args.arm,seed=args.seed,checkpoint=args.mode,step=state['step'],
                      preflight=args.preflight,n=len(ids),split='val',test_queries=0,
                      fresh_process_max_abs_score_error=error,generated_exact_match=True,
                      trained_lora_B_nonzero=trained_lora_nonzero,
                      disabled_adapter_max_abs_score_error=disabled_error if args.arm=='C' else None)
        atomic_json(result,str(d/('fresh_'+args.mode+'_reload.json')))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

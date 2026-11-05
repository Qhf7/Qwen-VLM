"""Optional exact-gated batch greedy decoding, independent of historical model.

Use only VisA validation samples for the engineering gate. A chosen policy is
serialized before metric evaluation. Batched GEMMs may change BF16 argmax;
exact token/EOS agreement is required, otherwise the whole condition remains
scalar. No generation formatting, stopping, prompt, weights, or token budget
is changed to gain agreement. Checkpoint verifiers force the saved policy.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import types

import torch

GENERATION_VERSION='exact-gated-leftpad-greedy-v1'


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def validate_budget(max_new_tokens):
    if isinstance(max_new_tokens,bool) or not isinstance(max_new_tokens,int) or max_new_tokens<1:
        raise ValueError('max_new_tokens must be a positive integer')


def decoded(model, token_rows):
    return [model.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False) for ids in token_rows]


@torch.no_grad()
def scalar_token_trace(model,pack,max_new_tokens=80):
    """Exact copy of historical scalar computation, returning tokens and EOS."""
    validate_budget(max_new_tokens)
    rows=[];stops=[];embedding=model.language_model.get_input_embeddings()
    for index,length in enumerate(pack['prefix_lengths'].tolist()):
        embeds=pack['inputs_embeds'][index:index+1,:length]
        mask=torch.ones((1,length),dtype=torch.long,device=model.device)
        positions=torch.arange(length,device=model.device).unsqueeze(0)
        out=model.language_model(inputs_embeds=embeds,attention_mask=mask,
                                 position_ids=positions,use_cache=True,return_dict=True)
        cache=out.past_key_values;ids=[];stop=None
        logits=model.lm_head(out.last_hidden_state[:,-1])
        for step in range(max_new_tokens):
            token_id=int(logits.argmax(dim=-1).item())
            if token_id in model.stop_ids:
                stop=token_id;break
            ids.append(token_id)
            if step+1==max_new_tokens:break
            if cache is None:raise RuntimeError('Missing scalar autoregressive cache')
            token=torch.tensor([[token_id]],dtype=torch.long,device=model.device)
            mask=torch.cat([mask,torch.ones((1,1),dtype=torch.long,device=model.device)],dim=1)
            positions=torch.tensor([[length+step]],dtype=torch.long,device=model.device)
            out=model.language_model(inputs_embeds=embedding(token),attention_mask=mask,
                                     position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
            cache=out.past_key_values;logits=model.lm_head(out.last_hidden_state[:,-1])
        rows.append(ids);stops.append(stop)
        del cache,out,logits
    return dict(tokens=rows,stop_ids=stops,texts=decoded(model,rows))


@torch.no_grad()
def batched_token_trace(model,pack,max_new_tokens=80):
    """Batch cached greedy decoding with left-padded prompts and masked EOS rows.

    Pass 2D monotonic per-row text positions, as the historical text-only Qwen
    route does. The Qwen text model expands these to its three rotary axes.
    Cache positions are physical left-padded indices managed by the text model.
    """
    validate_budget(max_new_tokens)
    lengths=pack['prefix_lengths'].to(device=model.device,dtype=torch.long)
    batch_size=int(lengths.numel())
    if batch_size<1:raise ValueError('No generation rows')
    maximum=int(lengths.max().item());width=pack['inputs_embeds'].shape[-1]
    embeds=pack['inputs_embeds'].new_zeros((batch_size,maximum,width))
    mask=torch.zeros((batch_size,maximum),device=model.device,dtype=torch.long)
    for index,length in enumerate(lengths.tolist()):
        embeds[index,maximum-length:]=pack['inputs_embeds'][index,:length]
        mask[index,maximum-length:]=1
    positions=(mask.cumsum(dim=1)-1).clamp(min=0)
    embedding=model.language_model.get_input_embeddings()
    out=model.language_model(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,
                             use_cache=True,return_dict=True)
    cache=out.past_key_values;logits=model.lm_head(out.last_hidden_state[:,-1])
    active=torch.ones(batch_size,dtype=torch.bool,device=model.device)
    ids=[[] for _ in range(batch_size)];stops=[None]*batch_size
    for step in range(max_new_tokens):
        choices=logits.argmax(dim=-1)
        for index,token_id in enumerate(choices.tolist()):
            if not bool(active[index]):continue
            if token_id in model.stop_ids:
                stops[index]=int(token_id);active[index]=False
            else:ids[index].append(int(token_id))
        if not bool(active.any()) or step+1==max_new_tokens:break
        if cache is None:raise RuntimeError('Missing batched autoregressive cache')
        # Finished rows never re-enter decoding and newly appended EOS/pad slots
        # are invisible as keys. A live row always feeds its last predicted ID.
        choices=torch.where(active,choices,torch.full_like(choices,int(model.pad_id)))
        mask=torch.cat([mask,active.to(dtype=torch.long).unsqueeze(1)],dim=1)
        positions=(lengths+step).unsqueeze(1)
        out=model.language_model(inputs_embeds=embedding(choices.unsqueeze(1)),attention_mask=mask,
                                 position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
        cache=out.past_key_values;logits=model.lm_head(out.last_hidden_state[:,-1])
    del cache,out,logits
    return dict(tokens=ids,stop_ids=stops,texts=decoded(model,ids))


class BatchedGenerationMixin:
    """Unconfigured models use scalar generation; no implicit batch opt-in."""
    def scalar_generate(self,pack,max_new_tokens=80):
        return scalar_token_trace(self,pack,max_new_tokens)['texts']

    def generate(self,pack,max_new_tokens=80):
        policy=getattr(self,'generation_policy',None)
        if policy is None:
            return self.scalar_generate(pack,max_new_tokens)
        if policy['version']!=GENERATION_VERSION or max_new_tokens!=policy['max_new_tokens']:
            raise ValueError('Generation request differs from the frozen policy')
        if policy['selected']=='scalar':
            return self.scalar_generate(pack,max_new_tokens)
        if int(pack['prefix_lengths'].numel())>policy['batch_size']:
            raise ValueError('Generation batch exceeds the exact-gated batch size')
        return batched_token_trace(self,pack,max_new_tokens)['texts']


def attach_generation(model):
    """Opt in without modifying shared FrozenLanguageBridge class or source."""
    if not hasattr(model,'historical_scalar_generate'):
        model.historical_scalar_generate=model.generate
    model.scalar_generate=types.MethodType(BatchedGenerationMixin.scalar_generate,model)
    model.generate=types.MethodType(BatchedGenerationMixin.generate,model)
    return model


def fixed_visa_gate_ids(items,n=8):
    if n!=8:raise ValueError('Frozen engineering gate contains exactly eight VisA validation images')
    # First four available categories with both labels, one example per label;
    # every gate has both decisions and mixed category prompt lengths.
    categories=sorted({x['category'] for x in items if x['split']=='val'})
    ids=[]
    for category in categories:
        pair=[next((i for i,x in enumerate(items) if x['split']=='val' and x['category']==category and int(x['label'])==label),None) for label in (0,1)]
        if None not in pair:ids.extend(pair)
        if len(ids)==n:break
    if len(ids)!=n:raise ValueError('Need four validation categories containing both labels for the fixed gate')
    return ids


def synchronize(model):
    if torch.device(model.device).type=='cuda':torch.cuda.synchronize()


def freeze_generation_policy(model,cache,max_new_tokens=80,batch_size=4,context=None):
    """Use only fixed val8 to select engineering implementation, then freeze."""
    if batch_size<2:raise ValueError('Batch candidate must have at least two rows')
    validate_budget(max_new_tokens);attach_generation(model)
    inherited=model.historical_scalar_generate
    if (getattr(inherited,'__func__',None) is BatchedGenerationMixin.generate):
        raise RuntimeError('Generation attachment lost original scalar implementation')
    ids=fixed_visa_gate_ids(cache['items'])
    records=[cache['items'][i] for i in ids]
    assert all(x['split']=='val' for x in records)
    was_training=model.training;model.eval()
    cuda=torch.device(model.device).type=='cuda'
    memory_before=torch.cuda.memory_allocated() if cuda else None
    scalar_rows=[];batch_rows=[];scalar_stops=[];batch_stops=[];scalar_texts=[];batch_texts=[]
    scalar_seconds=0.;batch_seconds=0.;parent_exact=True;failure=None;used_lengths=[]
    try:
        for offset in range(0,len(ids),batch_size):
            chunk=ids[offset:offset+batch_size]
            packed=model.pack(cache['cls'][chunk].to(model.device),cache['maps'][chunk].to(model.device),[cache['items'][i]['category'] for i in chunk])
            used_lengths.extend(packed['prefix_lengths'].tolist())
            synchronize(model);tick=time.perf_counter();scalar=scalar_token_trace(model,packed,max_new_tokens);synchronize(model)
            scalar_seconds+=time.perf_counter()-tick
            # On the first chunk verify the retained scalar copy against the
            # historical inherited generator, independently of batch agreement.
            if offset==0 and hasattr(model,'historical_scalar_generate'):
                parent_exact=scalar['texts']==model.historical_scalar_generate(packed,max_new_tokens=max_new_tokens)
                if not parent_exact:raise AssertionError('Scalar token trace diverges from historical generator')
            scalar_rows.extend(scalar['tokens']);scalar_stops.extend(scalar['stop_ids']);scalar_texts.extend(scalar['texts'])
            synchronize(model);tick=time.perf_counter();candidate=batched_token_trace(model,packed,max_new_tokens);synchronize(model)
            batch_seconds+=time.perf_counter()-tick
            batch_rows.extend(candidate['tokens']);batch_stops.extend(candidate['stop_ids']);batch_texts.extend(candidate['texts'])
            del packed
    except Exception as error:
        if not parent_exact:raise
        failure=type(error).__name__+': '+str(error)[:500]
        gc.collect()
        if torch.device(model.device).type=='cuda':torch.cuda.empty_cache()
    finally:
        model.train(was_training)
    token_exact=(failure is None and scalar_rows==batch_rows and scalar_stops==batch_stops and scalar_texts==batch_texts)
    differences=[dict(index=i,scalar_token_ids=s,batched_token_ids=b,scalar_stop_id=se,batched_stop_id=be)
                 for i,s,b,se,be in zip(ids,scalar_rows,batch_rows,scalar_stops,batch_stops) if s!=b or se!=be]
    policy=dict(version=GENERATION_VERSION,selected='batched' if token_exact else 'scalar',gate_split='VisA_existing_val_only',
                max_new_tokens=max_new_tokens,batch_size=batch_size,gate_indices=ids,gate_paths=[x['path'] for x in records],
                gate_labels=[int(x['label']) for x in records],gate_categories=[x['category'] for x in records],
                gate_prefix_lengths=used_lengths,
                exact_token_ids_and_stop_ids=token_exact,scalar_parent_exact=parent_exact,
                gate_scalar_seconds=scalar_seconds,gate_batch_seconds=batch_seconds,
                measured_gate_speedup=scalar_seconds/batch_seconds if batch_seconds>0 else None,
                scalar_token_ids=scalar_rows,scalar_stop_ids=scalar_stops,batched_token_ids=batch_rows,batched_stop_ids=batch_stops,
                failure=failure,differences=differences,context=context,
                memory_allocated_before_gate=memory_before,peak_memory_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                source_sha256=file_sha(__file__),selection_rule='Select batch only on val8 exact tokens+EOS; failure fixes whole condition to scalar; no external metrics',
                padding='left prompt padding; masked pad/EOS rows; per-row monotonic text positions; common physical cache growth')
    model.generation_policy=policy
    return policy


def force_generation_policy(model,policy):
    """Verifier/evaluator uses stored decision; it must not choose on test data."""
    if policy['version']!=GENERATION_VERSION or policy['selected'] not in ('scalar','batched'):
        raise ValueError('Unknown generation policy')
    if policy['source_sha256']!=file_sha(__file__):raise ValueError('Generation implementation differs from frozen policy')
    if policy['selected']=='batched' and not policy['exact_token_ids_and_stop_ids']:
        raise ValueError('Batched policy lacks exact parity gate')
    attach_generation(model);model.generation_policy=dict(policy)
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--source-c',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--max-new-tokens',type=int,default=80)
    ap.add_argument('--batch-size',type=int,default=4)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    from structured_model import StructuredLanguageBridge,load_fixed_c
    torch.set_num_threads(8);torch.manual_seed(args.seed)
    cache=torch.load(args.cache,map_location='cpu')
    model=StructuredLanguageBridge('qwen',device='cuda',init_seed=args.seed)
    load_fixed_c(model,args.source_c)
    policy=freeze_generation_policy(model,cache,args.max_new_tokens,args.batch_size,
                                    context=dict(seed=args.seed,source_c=str(args.source_c),source_c_best_sha256=file_sha(args.source_c/'best.pt')))
    atomic_json(policy,args.out);print(json.dumps(policy),flush=True)


if __name__=='__main__':main()

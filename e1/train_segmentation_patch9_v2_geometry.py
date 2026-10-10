"""Frozen Qwen3 Patch9 training on real normal and anomalous views."""
import argparse
import hashlib
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from b0.single_view import candidate_ids
from e1.model import QwenE1
from e1.patch9_training_data import check_splits, audit_frame, make_schedule, example, KINDS
from e1.patch9_validation import evaluate, write_json
from e1.dataset_config import BASE_DEV, BASE_TRAIN, assert_original_paths, assert_original_split

EVAL_STEPS = (250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
AMP_DEVICE = 'cuda' if DEVICE.type == 'cuda' else 'cpu'
AMP_DTYPE = torch.bfloat16 if DEVICE.type == 'cuda' else torch.float16

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode(proc, image):
    batch=proc.image_processor(images=Image.fromarray(image),return_tensors='pt')
    inputs={k:v.to(DEVICE) for k,v in batch.items()}
    if inputs['image_grid_thw'].tolist()!=[[1,28,28]]:
        raise ValueError('Expected Qwen3 28x28 grid')
    return inputs

def build_model(model_id):
    proc=AutoProcessor.from_pretrained(model_id,min_pixels=448**2,max_pixels=448**2)
    base=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,dtype=AMP_DTYPE,attn_implementation='eager').to(DEVICE).eval()
    layers=tuple(base.config.vision_config.deepstack_visual_indexes)
    if layers!=(5,11,17):
        raise ValueError(f'Unexpected visual layers {layers}')
    model=QwenE1(base,candidate_ids(proc.tokenizer,'Yes'),candidate_ids(proc.tokenizer,'No'),
                 feature_layers=layers,decoder_channels=288).to(DEVICE)
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.decoder.region_head.requires_grad_(False)
    return model,proc

def parameters(model):
    return [p for p in model.parameters() if p.requires_grad]

def gradient_check(model):
    names=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            if not name.startswith('decoder.') or p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(f'Gradient check failed: {name}')
            names.append(name)
        elif p.grad is not None:
            raise RuntimeError(f'Frozen parameter has gradient: {name}')
    if not names:
        raise RuntimeError('No trainable parameters')
    return names

def optimize_batch(model,proc,train,batch,args,opt):
    if len(batch)!=8 or sorted(batch.kind)!=sorted(KINDS):
        raise ValueError('Expected exact eight-example mixture')
    model.train()
    model.base_model.eval()
    opt.zero_grad(set_to_none=True)
    details=[]
    for selection in batch.itertuples(index=False):
        image,mask=example(train,selection,args.root,args.gt_root)
        target=torch.from_numpy(mask.astype(np.float32)[None,None]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model.segmentation_only(encode(proc,image),target,
                                           allow_empty=True,dice_nonempty_only=True)
        if result.mask_logits.shape!=(1,1,448,448) or not torch.isfinite(result.loss):
            raise RuntimeError('Invalid output/loss')
        if not mask.any() and float(result.dice_loss.detach())!=0:
            raise RuntimeError('Empty target must contribute zero Dice loss')
        (result.loss/8).backward()
        probability=torch.sigmoid(result.mask_logits.float()).detach()
        details.append(dict(kind=selection.kind,loss=float(result.loss.detach()),
                            focal=float(result.focal_loss.detach()),dice=float(result.dice_loss.detach()),
                            probability=float(probability.mean()),area=float((probability>=.5).float().mean())))
    names=gradient_check(model)
    norm=float(torch.nn.utils.clip_grad_norm_(parameters(model),1.))
    if not math.isfinite(norm) or norm==0:
        raise RuntimeError('Invalid gradient norm')
    opt.step()
    return details,norm,names

def predictor(model,proc):
    def predict(image):
        with torch.inference_mode(),torch.autocast(AMP_DEVICE,dtype=AMP_DTYPE,enabled=DEVICE.type in ('cuda','mps')):
            inputs=encode(proc,image)
            features,_,_=model._vision_features(inputs['pixel_values'],inputs['image_grid_thw'])
            logits,_,_=model.decoder(features)
            logits=F.interpolate(logits,size=(448,448),mode='bilinear',align_corners=False)
            return logits.float().sigmoid()[0,0].cpu().numpy()
    return predict

def save(path,state):
    temporary=path.with_suffix('.pt.tmp')
    torch.save(state,temporary)
    temporary.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-id',default='models/Qwen3-VL-2B-Instruct')
    p.add_argument('--train',type=Path,default=BASE_TRAIN)
    p.add_argument('--dev',type=Path,default=BASE_DEV)
    p.add_argument('--root',type=Path,default=Path('Dataset'))
    p.add_argument('--gt-root',type=Path,default=Path('Dataset'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.steps<1 or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Positive steps and CUDA BF16 required')
    args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.out/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.out/'training_history.csv').exists() and not args.resume:
        raise FileExistsError('Choose a fresh output directory')
    seed=20260901
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train=pd.read_parquet(args.train).reset_index(drop=True)
    dev=pd.read_parquet(args.dev).reset_index(drop=True)
    assert_original_split(train, "train")
    assert_original_split(dev, "dev")
    assert_original_paths(train, "train")
    assert_original_paths(dev, "dev")
    check_splits(train,dev)
    records=audit_frame(train,args.root,args.gt_root)
    audit_frame(dev,args.root,args.gt_root)
    report={name:dict(images=len(df),normal=int(df.view_label.eq(0).sum()),
                     anomaly=int(df.view_label.eq(1).sum()),classes=int(df.class_name.nunique()))
            for name,df in [('train',train),('dev',dev)]}
    report['passed']=True
    schedule=make_schedule(train,records,args.steps,seed)
    schedule.to_parquet(args.out/'sampling_schedule.parquet',index=False)
    write_json(args.out/'data_validation.json',report)
    eval_steps=sorted({s for s in EVAL_STEPS if s<=args.steps}|{args.steps})
    protocol=dict(model=args.model_id,steps=args.steps,seed=seed,feature_layers=[5,11,17],decoder_channels=288,
        train_sha256=digest(args.train),dev_sha256=digest(args.dev),schedule_sha256=digest(args.out/'sampling_schedule.parquet'),
        mixture=list(KINDS),logical_batch=8,micro_batch=1,learning_rate=1e-4,weight_decay=.01,
        warmup=50,gradient_clip=1.,scheduler='cosine',precision='BF16; FP32 loss and trainable weights',
        loss='Focal(gamma=2,alpha_positive=.25)+Dice(nonempty only); empty=Focal',
        geometry_sha256=digest(Path(__file__).with_name('patch9_v2_geometry.py')),
        model_code_sha256=digest(Path(__file__).with_name('model.py')),
        trainer_sha256=digest(__file__),patch_fraction=.4,patch_overlap=.25,
        evaluation_steps=eval_steps,evaluation_variants=['full','mean_w0.5'],
        selection='anomaly category-macro Dice: mean_w0.5',data=report)
    model,proc=build_model(args.model_id)
    opt=torch.optim.AdamW(parameters(model),lr=1e-4,weight_decay=.01)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/50 if s<50 else
        .5*(1+math.cos(math.pi*(s-50)/max(1,args.steps-50))))
    protocol['trainable_parameters']=sum(p.numel() for p in parameters(model))
    write_json(args.out/'frozen_training_protocol.json',protocol)
    if args.validate_only:
        before=next(model.decoder.parameters()).detach().clone()
        values,norm,names=optimize_batch(model,proc,train,schedule[schedule.step.eq(1)],args,opt)
        changed=not torch.equal(before,next(model.decoder.parameters()).detach())
        if not changed:
            raise RuntimeError('Optimizer did not update decoder')
        model.eval()
        smoke=pd.concat([dev[dev.view_label.eq(label)].head(2) for label in (0,1)])
        result=evaluate(predictor(model,proc),smoke,args.root,args.gt_root,args.out/'smoke_eval',1)
        write_json(args.out/'forward_validation.json',dict(passed=True,optimizer_updated=changed,
            examples=values,gradient_norm=norm,gradient_parameters=names,mask_shape=[1,1,448,448],
            base_has_gradient=any(v.grad is not None for v in model.base_model.parameters()),
            trainable_parameters=protocol['trainable_parameters'],evaluation_n=len(smoke)))
        return
    start=0; best=-1.; best_step=0; history=[]
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['protocol']!=protocol:
            raise ValueError('Resume protocol mismatch')
        model.decoder.load_state_dict(state['decoder'])
        opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        start=state['step']; best=state['best']; best_step=state['best_step']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        if (args.out/'training_history.csv').exists():
            history=pd.read_csv(args.out/'training_history.csv').query('step<=@start').to_dict('records')
    begun=time.perf_counter()
    for step,batch in schedule[schedule.step.gt(start)].groupby('step',sort=True):
        t=time.perf_counter()
        values,norm,_=optimize_batch(model,proc,train,batch,args,opt)
        sched.step()
        record=dict(step=int(step),loss=float(np.mean([v['loss'] for v in values])),grad_norm=norm,
                    lr=opt.param_groups[0]['lr'],step_s=time.perf_counter()-t)
        for kind in sorted(set(KINDS)):
            subset=[v for v in values if v['kind']==kind]
            for key in ('loss','focal','dice','probability','area'):
                record[f'{kind}_{key}']=float(np.mean([v[key] for v in subset]))
        history.append(record)
        tmp=args.out/'history.tmp'
        pd.DataFrame(history).to_csv(tmp,index=False); tmp.replace(args.out/'training_history.csv')
        def state():
            return dict(step=int(step),decoder=model.decoder.state_dict(),optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(),best=best,best_step=best_step,protocol=protocol,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if step%10==0 or step==1:
            print(f'[train] {step}/{args.steps} loss={record["loss"]:.5f}',flush=True)
        if step%50==0:
            save(args.out/'latest.pt',state())
        if step in eval_steps:
            save(args.out/f'step{step:05d}.pt',state())
            model.eval()
            result=evaluate(predictor(model,proc),dev,args.root,args.gt_root,args.out,int(step))
            score=result['variants']['mean_w0.5']['macro_dice']
            if score>best:
                best=score; best_step=int(step)
                save(args.out/'best.pt',state())
            save(args.out/'latest.pt',state())
    write_json(args.out/'training_summary.json',dict(completed_steps=args.steps,best_step=best_step,
               best_macro_dice=best,elapsed_this_session_s=time.perf_counter()-begun))

if __name__=='__main__':
    main()

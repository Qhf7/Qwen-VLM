"""Check native-forward parity and real-image classification gradients on CUDA."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from e1.evaluate_classification_fold1 import build_classification_model, encode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-id',default='models/Qwen3-VL-2B-Instruct')
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,default=Path('Train448'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.manual_seed(20260901)
    device=torch.device('cuda')
    model,processor=build_classification_model(args.model_id,None,device,
        dtype=torch.bfloat16,feature_layers=(5,11,17,23))
    pool=pd.read_parquet(args.pool)
    results=[]
    for label in (0,1):
        row=pool[pool.view_label.eq(label)].iloc[0]
        with Image.open(args.cache_root/row.image_path) as im:
            assert im.size==(448,448)
            inputs=encode(processor,im.convert('RGB'),device)
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            native=model.base_model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            features,embeds,deepstack=model._vision_features(inputs['pixel_values'],inputs['image_grid_thw'])
            assert len(deepstack)==3
            assert all(f.shape[-2:]==(28,28) for f in features)
            empty=embeds.new_empty((1,0,model.base_model.config.text_config.hidden_size))
            manual=model._language_logits(inputs,embeds,empty,deepstack)
            torch.testing.assert_close(manual,native,rtol=0.005,atol=0.05)
            parity=float((manual-native).abs().max())
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(inputs,sample_labels=torch.tensor([label],device=device))
        assert torch.isfinite(output.loss)
        output.loss.backward()
        gradient_stats={}
        for name,module in [('decoder',model.decoder),('prompt',model.prompt_learner)]:
            grads=[v.grad for v in module.parameters() if v.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            norm=sum(float(g.float().square().sum()) for g in grads)**0.5
            assert norm>0
            gradient_stats[name]=norm
        assert all(v.grad is None for v in model.base_model.parameters())
        assert all(v.grad is None for v in model.decoder.mask_head.parameters())
        assert all(v.grad is None for v in model.decoder.region_head.parameters())
        results.append(dict(view_label=label,image_path=row.image_path,
            loss=float(output.loss.detach()),p_yes=float(output.anomaly_logits.sigmoid().detach()),
            native_max_abs_difference=parity,gradient_norm=gradient_stats))
    report=dict(passed=True,precision='bfloat16',examples=results)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

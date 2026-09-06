"""Evaluate native Qwen image-classification scores on individual images."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from b0.single_view import candidate_ids
from e1.classification_model import ClassificationOutput
from e1.evaluate_classification_fold1 import evaluate_classification, sha256_file


class NativeClassifier(torch.nn.Module):
    def __init__(self, base, tokenizer):
        super().__init__()
        self.base=base
        self.yes_ids=candidate_ids(tokenizer,'Yes')
        self.no_ids=candidate_ids(tokenizer,'No')
        self.autocast_dtype=torch.bfloat16

    def forward(self,inputs):
        logits=self.base(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        score=torch.logsumexp(logits[:,self.yes_ids],dim=1)-torch.logsumexp(logits[:,self.no_ids],dim=1)
        return ClassificationOutput(anomaly_logits=score)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-id',default='models/Qwen3-VL-2B-Instruct')
    p.add_argument('--dev',type=Path,default=Path('artifacts/splits/fold1/dev.parquet'))
    p.add_argument('--cache-root',type=Path,default=Path('Dataset448'))
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    device=torch.device('cuda')
    dev=pd.read_parquet(args.dev)
    assert len(dev)==3600 and dev.image_path.nunique()==3600
    assert all((args.cache_root/x).is_file() for x in dev.image_path)
    processor=AutoProcessor.from_pretrained(args.model_id,min_pixels=448**2,max_pixels=448**2)
    base=AutoModelForImageTextToText.from_pretrained(args.model_id,dtype=torch.bfloat16,
        attn_implementation='eager').eval().to(device)
    model=NativeClassifier(base,processor.tokenizer)
    results=evaluate_classification(model,processor,dev,args.cache_root,device,args.out_dir,'baseline')
    results.update(model_id=args.model_id,dev_sha256=sha256_file(args.dev),dtype='bfloat16')
    (args.out_dir/'baseline_metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()

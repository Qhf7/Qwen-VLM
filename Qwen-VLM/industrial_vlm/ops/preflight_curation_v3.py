"""Check native high-resolution encodings; no model weights or confirmation images."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/'experiments/curation_v3/native_input_preflight.json'
    if output.exists():
        raise FileExistsError(output)
    from transformers import AutoProcessor
    from industrial_vlm.modeling import make_inputs
    from industrial_vlm.evaluate import _atomic_json
    import torch
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(root/'models/Qwen3-VL-8B-Instruct',
                                            max_pixels=1048576, local_files_only=True)
    image_processor = processor.image_processor
    if hasattr(image_processor, 'max_pixels'):
        image_processor.max_pixels = 1048576
    if isinstance(getattr(image_processor,'size',None),dict) and 'longest_edge' in image_processor.size:
        image_processor.size = {**image_processor.size,'longest_edge':1048576}
    records, longest = [], None
    start = time.perf_counter()
    for role, filename in [('train','train_region_highres.jsonl'),('validation','val_region_120.jsonl')]:
        rows = [json.loads(s) for s in (root/'prepared/curation_v3'/filename).read_text().splitlines() if s.strip()]
        unique = {r['source_id']:r for r in rows}
        for row in unique.values():
            data = make_inputs(processor,row['image'],row['prompt'],row['answer'],max_length=2048)
            length = int(data['input_ids'].shape[-1])
            target = int((data['labels'][:,1:] != -100).sum())
            visual = int((data['input_ids'] == 151655).sum())
            grid = data['image_grid_thw'][0].tolist()
            if visual != grid[0]*grid[1]*grid[2]//4 or length-target+128 > 2048:
                raise ValueError('Native grid/token or generation reserve contract failed')
            records.append({'source_id':row['source_id'],'role':role,'input_tokens':length,
                            'visual_tokens':visual,'target_tokens':target,'grid_thw':grid})
            if role == 'train' and (longest is None or length > longest[0]):
                longest = (length,row)
            del data
            if len(records)%100==0:
                print(f'Native encodings checked: {len(records)}/888',flush=True)
    if len(records)!=888 or len({r['source_id'] for r in records})!=888:
        raise ValueError('High-resolution preflight source count differs')
    smoke_data=root/'prepared/curation_v3/highres_smoke.jsonl'
    smoke_config=root/'configs/curation_v3/highres_smoke.json'
    if smoke_data.exists() or smoke_config.exists():
        raise FileExistsError('Preflight smoke inputs already exist')
    row={**longest[1], 'id':longest[1]['source_id']+'__v3_highres_smoke'}
    smoke_data.write_text(json.dumps(row,ensure_ascii=False)+chr(10))
    config=json.loads((root/'configs/curation_v3/region_highres.json').read_text())
    config.update(train_file=str(smoke_data), val_file=None, output=str(root/'runs/curation_v3/highres_smoke'),
                  max_steps=1,save_steps=0,eval_steps=0)
    _atomic_json(smoke_config,config)
    _atomic_json(output,{'status':'passed','max_pixels':1048576,'max_length':2048,
        'n_unique':888,'seconds':time.perf_counter()-start,'confirmation_images_opened':False,
        'max_input_tokens':max(r['input_tokens'] for r in records),
        'max_visual_tokens':max(r['visual_tokens'] for r in records),
        'smoke_source_id':row['source_id'],'records':records})
    print(json.dumps({'status':'passed','n_unique':888,'max_input_tokens':longest[0]}),flush=True)


if __name__=='__main__':
    main()

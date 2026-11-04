"""Convert hash-verified legacy Vicuna weights to tensor-only safetensors.

Uses tmpfs to keep original model and experiment disks intact. The two source
digests are independently recorded in the October conversion audit.
"""
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file, load_file

SOURCE = Path('/root/autodl-tmp/AnomalyGPT/pretrained_ckpt/vicuna_ckpt/vicuna-7b-v1.1')
DEST = Path('/dev/shm/migration_vicuna_bf16')
EXPECTED = [
    '5ed572be140240b212049a3e791271eed8a04c40bc732c91a4da4b1469db23b1',
    'd9382da358f0ec38c4fca3bcf1e3e65274ae4c78090e0775b4bb4dea6a518e08',
]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(exist_ok=False)
    for name in ('config.json', 'generation_config.json', 'tokenizer.model',
                 'tokenizer_config.json', 'special_tokens_map.json'):
        shutil.copyfile(SOURCE/name, DEST/name)
    index = json.loads((SOURCE/'pytorch_model.bin.index.json').read_text())
    records = []
    for j, source_name in enumerate(sorted(set(index['weight_map'].values()))):
        source = SOURCE/source_name
        actual = digest(source)
        assert actual == EXPECTED[j], 'Legacy model source differs from recorded verified checkpoint'
        state = torch.load(source, map_location='cpu', weights_only=True)
        assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
        target_name = source_name.replace('pytorch_model-', 'model-').replace('.bin', '.safetensors')
        save_file({k:v.contiguous() for k,v in state.items()}, str(DEST/target_name), metadata={'format':'pt'})
        reloaded = load_file(str(DEST/target_name), device='cpu')
        assert set(state) == set(reloaded) and all(torch.equal(v, reloaded[k]) for k,v in state.items())
        records.append(dict(source=source_name, source_sha256=actual, target=target_name,
                            target_sha256=digest(DEST/target_name), all_tensors_equal=True))
        print('[converted]', source_name, len(state), flush=True)
        del state, reloaded
    index['weight_map'] = {k:v.replace('pytorch_model-', 'model-').replace('.bin','.safetensors')
                           for k,v in index['weight_map'].items()}
    (DEST/'model.safetensors.index.json').write_text(json.dumps(index, indent=2))
    report = dict(source=str(SOURCE), destination=str(DEST), torch=torch.__version__,
                  weights_only=True, records=records, all_tensors_equal=True)
    p = Path('/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/vicuna_conversion.json')
    p.write_text(json.dumps(report, indent=2))
    print('[done] tensor-for-tensor equality verified', flush=True)


if __name__ == '__main__':
    main()

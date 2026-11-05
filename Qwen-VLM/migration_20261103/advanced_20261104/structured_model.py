"""Structured Qwen text adapter; inherited visual pack never receives masks."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('ANOMALYGPT_REPO', '/root/autodl-tmp/AnomalyGPT'))
for path in (REPO, REPO / 'code/code/model', HERE, HERE.parent / 'code', HERE.parent.parent / 'repair_20261103/code'):
    if path.is_dir():
        sys.path.insert(0, str(path))

from migration_language import FrozenLanguageBridge, FEATURE_MARKER, DESCRIPTIONS
from structured_geometry import SCHEMA, LOCATIONS, parse_structured


class StructuredLanguageBridge(FrozenLanguageBridge):
    """Fixed schema instructions plus unchanged first-answer Yes-minus-No score."""
    def __init__(self, *args, **kwargs):
        # Parent checks its historical canonical answers during construction.
        self.structured_ready = False
        super().__init__(*args, **kwargs)
        self.structured_ready = True

    def _check_answer(self, answer):
        if self.structured_ready and parse_structured(answer) is None:
            raise ValueError('Invalid structured training target')
        return super()._check_answer(answer)

    def _template_ids(self, category):
        if category in self._prompt_ids:
            return self._prompt_ids[category]
        if self.backend != 'qwen' or category not in DESCRIPTIONS:
            raise ValueError('Structured extension requires a supported Qwen category')
        user = ('Image features: ' + FEATURE_MARKER + '\nThis is a photo of ' + DESCRIPTIONS[category]
                + ' for anomaly detection. Is there any anomaly in the image?\n'
                + 'Reply with exactly Yes or No on the first line. On the second line output one JSON object '
                + 'with only bbox and location. For No use {"bbox":null,"location":"none"}. '
                + 'For Yes, bbox is [x1,y1,x2,y2], integer coordinates from 0 to 1000 relative to image width '
                + 'and height; enclose all abnormal regions in one box. x1<x2 and y1<y2. '
                + 'location is the grid cell containing the box center, chosen from '
                + ', '.join(LOCATIONS) + '. Do not output any other text.')
        rendered = self.tok.apply_chat_template(
            [{'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
        if rendered.count(FEATURE_MARKER) != 1:
            raise AssertionError('Feature marker must appear once')
        before, after = rendered.split(FEATURE_MARKER)
        ids = (self.tok.encode(before, add_special_tokens=False),
               self.tok.encode(after, add_special_tokens=False))
        self._prompt_ids[category] = ids
        return ids


def load_fixed_c(model, source_run):
    """Restore pre-existing C best for the same seed, independent of D/R scores."""
    import json
    import torch
    from migration_advanced_train import configure_arm, load_bridge, load_delta, sha256_file
    source_run = Path(source_run)
    if not (source_run / 'complete.json').is_file():
        raise ValueError('Original C run is not complete')
    run = json.loads((source_run / 'run.json').read_text())
    identity = run['identity']
    if identity['arm'] != 'C' or identity['seed'] != model.init_seed:
        raise ValueError('Source must be original C of the matching seed')
    if sha256_file(identity['bridge_path']) != identity['bridge_sha256']:
        raise ValueError('Source bridge changed')
    load_bridge(model, torch.load(identity['bridge_path'], map_location='cpu'))
    cfg = run['config']
    if cfg.get('preflight', False) or cfg.get('paired_sampling', False) or cfg.get('objective', 'ce') != 'ce':
        raise ValueError('Use original unpaired CE C, not the later paired control')
    named, targets, groups = configure_arm(model, 'C', cfg['rank'], cfg['alpha'], 0., cfg['lr_lora'])
    state = load_delta(model, source_run / 'best.pt', identity)
    if (cfg['rank'], cfg['alpha']) != (8, 16):
        raise ValueError('Frozen structured protocol uses r8 alpha16')
    return named, targets, groups, run, state

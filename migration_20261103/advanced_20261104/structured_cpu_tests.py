"""No-checkpoint CPU checks for target isolation and structured geometry."""
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from structured_geometry import build_targets, self_test, target_answer
from structured_model import StructuredLanguageBridge


def test_mask_manifest():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        path=root/'candle/Data/Masks/Anomaly/000.png';path.parent.mkdir(parents=True)
        mask=np.zeros((8,16),dtype=np.uint8);mask[2:6,4:12]=3;Image.fromarray(mask).save(path)
        items=[dict(path='/remote/VisA/candle/Data/Images/Anomaly/000.JPG',label=1,split='train',category='candle'),
               dict(path='/remote/VisA/candle/Data/Images/Normal/001.JPG',label=0,split='val',category='candle'),
               dict(path='/unavailable/test/Data/Images/Anomaly/never_opened.JPG',label=1,split='test',category='candle')]
        targets=build_targets(items,root)
        assert len(targets)==2 and targets[0]['bbox']==[250,250,750,750]
        assert targets[0]['foreground_pixels']==32 and targets[1]['mask_path'] is None
        return dict(temp_mask_manifest=True,historical_test_mask_not_read=True)


class MockTokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        return ''.join(m['content'] for m in messages)+'ASSISTANT:'
    def encode(self,text,add_special_tokens=False):
        if text.startswith('Yes'):return [1]+[3]*(len(text)-3)
        if text.startswith('No'):return [2]+[3]*(len(text)-2)
        return [4]*len(text)


class MockLanguage(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(8,4096)
        self.config=type('Cfg',(),dict(max_position_embeddings=10000))()
    def get_input_embeddings(self):return self.embedding


class MockPrompt(nn.Module):
    def forward(self,maps):return torch.ones((len(maps),18,4096))


def test_pack_isolation():
    model=StructuredLanguageBridge.__new__(StructuredLanguageBridge);nn.Module.__init__(model)
    model.backend='qwen';model.device=torch.device('cpu');model.dtype=torch.float32
    model.structured_ready=True;model._prompt_ids={};model.tok=MockTokenizer()
    model.yes_id=1;model.no_id=2;model.eos_id=7;model.language_model=MockLanguage()
    model.prompt_learner=MockPrompt();model.cls_proj=nn.Linear(1024,4096)
    answers=[target_answer(dict(label=1,bbox=[0,0,200,200],location='top_left')),
             target_answer(dict(label=0,bbox=None,location='none'))]
    inputs=(torch.ones((2,1024)),torch.ones((2,224,224)),['candle','pcb1'])
    inference=model.pack(*inputs)
    training=model.pack(*inputs,answers)
    for index,length in enumerate(training['prefix_lengths']):
        assert (training['labels'][index,:length]==-100).all()
        assert torch.equal(inference['inputs_embeds'][index,:length],training['inputs_embeds'][index,:length])
    assert [int(training['labels'][i,length]) for i,length in enumerate(training['prefix_lengths'])]==[1,2]
    assert training['labels'][0,-1]==7
    try:model._check_answer('Yes, binary only')
    except ValueError:pass
    else:raise AssertionError('Unstructured training target should fail')
    return dict(teacher_forcing_prefix_masked=True,inference_prefix_independent_of_target=True,
                first_yesno_token_preserved=True,eos_in_labels=True)


if __name__=='__main__':
    print(json.dumps(dict(geometry=self_test(),manifest=test_mask_manifest(),pack=test_pack_isolation())))

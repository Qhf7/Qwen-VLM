"""Offline CPU regression of advanced trainable, pairing and delta contracts.

Uses a randomly initialized tiny LlamaModel and the actual functions extracted
from migration_advanced_train.py. It downloads no weights, reads no dataset,
and does not import remote-only dec_big/visa runtime dependencies. This verifies
mechanics, not Qwen checkpoint compatibility or experiment efficacy.
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import tempfile

import torch
from torch import nn
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from transformers import LlamaConfig, LlamaModel


def source_functions():
    source = Path(__file__).with_name('migration_advanced_train.py')
    names = {'configure_arm', 'paired_schedule', 'bridge_state', 'load_bridge',
             'adapter_state', 'tensor_digest', 'bridge_digest', 'load_delta'}
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    scope = dict(torch=torch, random=random, hashlib=hashlib, json=json, Path=Path,
                 LoraConfig=LoraConfig, get_peft_model=get_peft_model,
                 get_peft_model_state_dict=get_peft_model_state_dict,
                 set_peft_model_state_dict=set_peft_model_state_dict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope, hashlib.sha256(source.read_bytes()).hexdigest()


class TinyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=32,
                             attention_dropout=0.0)
        self.language_model = LlamaModel(config)
        self.prompt_learner = nn.Linear(6, 16)
        self.cls_proj = nn.Linear(5, 16)
        self.lm_head = nn.Linear(16, 32, bias=False)
        self.register_buffer('tokens', torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
        self.register_buffer('visual', torch.arange(12).reshape(2, 6).float() / 10)
        self.register_buffer('cls', torch.arange(10).reshape(2, 5).float() / 10)

    def forward(self):
        visual = self.prompt_learner(self.visual) + self.cls_proj(self.cls)
        embedded = self.language_model.get_input_embeddings()(self.tokens) + visual[:, None, :]
        hidden = self.language_model(inputs_embeds=embedded, use_cache=False,
                                     return_dict=True).last_hidden_state
        return self.lm_head(hidden)


def new_model(functions, arm):
    torch.manual_seed(137)
    model = TinyBridge()
    named, targets, groups = functions['configure_arm'](model, arm, 2, 4, 1e-5, 1e-4)
    model.eval()
    return model, named, targets, groups


def test_arm(functions, arm):
    model, named, targets, groups = new_model(functions, arm)
    expected = {}
    for name, parameter in model.named_parameters():
        allowed = ((arm in ('B', 'D') and name.startswith(('prompt_learner.', 'cls_proj.')))
                   or (arm in ('C', 'D', 'R') and name.startswith('language_model.')
                       and ('.lora_A.' in name or '.lora_B.' in name)))
        assert parameter.requires_grad is allowed, (arm, name)
        if allowed:
            assert parameter.dtype == torch.float32
            expected[name] = parameter
    assert set(expected) == {name for name, _ in named}
    ids = [id(parameter) for group in groups for parameter in group['params']]
    assert len(ids) == len(set(ids)) == len(named)
    assert set(ids) == {id(parameter) for _, parameter in named}
    assert len(targets) == (4 if arm in ('C', 'D', 'R') else 0)
    if arm == 'D':
        assert [(group['name'], group['lr']) for group in groups] == [
            ('bridge', 1e-5), ('language_lora', 1e-4)]
        for group in groups:
            subset = {name for name, parameter in named if id(parameter) in {id(p) for p in group['params']}}
            assert all(name.startswith(('prompt_learner.', 'cls_proj.')) for name in subset) if group['name'] == 'bridge' else all('.lora_' in name for name in subset)
    else:
        assert len(groups) == 1
        assert groups[0]['lr'] == (1e-5 if arm == 'B' else 1e-4)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    audits = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        logits = model()
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 32),
                                                (model.tokens + 1).reshape(-1))
        score = logits[:, 1, 1] - logits[:, 1, 2]
        rank = torch.nn.functional.softplus(-(score[1] - score[0]))
        loss = ce + (rank if arm == 'R' else 0)
        loss.backward()
        grad = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                group = ('lora_A' if '.lora_A.' in name else 'lora_B' if '.lora_B.' in name else name.split('.')[0])
                grad[group] = grad.get(group, 0.0) + float(parameter.grad.norm())
            else:
                assert parameter.grad is None, name
        assert all(value > 0 for key, value in grad.items() if key != 'lora_A' or step == 2), grad
        optimizer.step()
        audits.append(grad)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, initial[name]), name
    bridge_changed = any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()
                         if name.startswith(('prompt_learner.', 'cls_proj.')))
    assert bridge_changed is (arm in ('B', 'D'))
    return model, named, dict(arm=arm, trainable_tensors=len(named), groups=len(groups),
                             target_modules=len(targets), bridge_changed=bridge_changed,
                             gradient_norms=audits)


def test_d_roundtrip(functions, model, named):
    identity = {key: 'toy_' + key for key in ('bridge_sha256', 'cache_sha256', 'base_manifest_sha256',
                 'sample_schedule_sha256', 'pipeline_version')}
    identity.update(arm='D', seed=137)
    checkpoint = dict(identity=identity, step=2, bridge=functions['bridge_state'](model),
                      adapter=functions['adapter_state'](model))
    original = {name: parameter.detach().clone() for name, parameter in named}
    with torch.no_grad():
        reference = model().detach().clone()
        for _, parameter in named:
            parameter.add_(0.1)
    assert any(not torch.equal(parameter, original[name]) for name, parameter in named)
    with tempfile.TemporaryDirectory(prefix='advanced_contract_') as temporary:
        path = Path(temporary) / 'delta.pt'
        torch.save(checkpoint, path)
        functions['load_delta'](model, path, identity)
        assert all(torch.equal(parameter, original[name]) for name, parameter in named)
        with torch.no_grad():
            assert torch.equal(model(), reference), 'In-process D roundtrip output differs'
        fresh, fresh_named, _, _ = new_model(functions, 'D')
        functions['load_delta'](fresh, path, identity)
        assert all(torch.equal(parameter, original[name]) for name, parameter in fresh_named)
        with torch.no_grad():
            assert torch.equal(fresh(), reference), 'Fresh-construction D roundtrip output differs'
        bad_identity = dict(identity, cache_sha256='wrong')
        try:
            functions['load_delta'](fresh, path, bad_identity)
        except AssertionError:
            pass
        else:
            raise AssertionError('Checkpoint identity mismatch accepted')
    return dict(adapter_and_bridge_exact=True, outputs_exact=True,
                fresh_construction_exact=True, rejects_wrong_identity=True)


def test_schedule(functions):
    items = [dict(split='train', category=category, label=label)
             for category in ('a', 'b', 'c') for label in (0, 1) for _ in range(3)]
    items += [dict(split='val', category='a', label=0), dict(split='train', category='oneclass', label=0)]
    schedule = functions['paired_schedule'](items, 100042, 2000)
    assert schedule == functions['paired_schedule'](items, 100042, 2000)
    assert schedule != functions['paired_schedule'](items, 100043, 2000)
    for offset in range(0, len(schedule), 2):
        normal, anomaly = [items[index] for index in schedule[offset:offset+2]]
        assert normal['split'] == anomaly['split'] == 'train'
        assert normal['category'] == anomaly['category']
        assert (normal['label'], anomaly['label']) == (0, 1)
    try:
        functions['paired_schedule'](items, 1, 3)
    except ValueError:
        pass
    else:
        raise AssertionError('Odd exposure count accepted')
    return dict(exposures=len(schedule), exact_seed_reproducibility=True,
                within_category_normal_anomaly_pairs=True, no_val_or_unpaired_category=True)


def main():
    torch.set_num_threads(1)
    functions, source_sha = source_functions()
    results, d_model, d_named = [], None, None
    for arm in ('B', 'C', 'D', 'R'):
        model, named, result = test_arm(functions, arm)
        results.append(result)
        if arm == 'D':
            d_model, d_named = model, named
    output = dict(state='passed', device='cpu', random_tiny_model=True,
                  source_sha256=source_sha,
                  versions=dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
                                peft=__import__('peft').__version__),
                  arms=results, d_checkpoint_roundtrip=test_d_roundtrip(functions, d_model, d_named),
                  paired_schedule=test_schedule(functions))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

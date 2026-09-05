"""Small utilities shared by the single-view baseline and classifier."""
from __future__ import annotations
import torch

def candidate_ids(tokenizer, word: str) -> list[int]:
    ids = {int(encoded[0]) for text in (word, ' ' + word)
           if len(encoded := tokenizer.encode(text, add_special_tokens=False)) == 1}
    if not ids:
        raise ValueError(f"{word!r} is not a single token")
    return sorted(ids)

def binary_scores(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> dict[str, float]:
    values = logits.float()
    yes = torch.logsumexp(values[yes_ids], dim=0)
    no = torch.logsumexp(values[no_ids], dim=0)
    probability = torch.softmax(torch.stack([no, yes]), dim=0)
    return {'p_yes': float(probability[1]), 'p_no': float(probability[0]),
            'yes_logit': float(yes), 'no_logit': float(no),
            'logit_margin_no_minus_yes': float(no - yes)}

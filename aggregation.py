from __future__ import annotations
import torch

def aggregate(hidden_states, attention_mask):
    layer = hidden_states[-1]
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    return layer[last_pos]

def extract_geometric_features(hidden_states, attention_mask):
    return torch.zeros(0)

def aggregation_and_feature_extraction(hidden_states, attention_mask, use_geometric=False):
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        return torch.cat([agg, extract_geometric_features(hidden_states, attention_mask)])
    return agg

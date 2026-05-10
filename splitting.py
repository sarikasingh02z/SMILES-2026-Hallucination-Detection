from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(y, df=None, test_size=0.15, val_size=0.15, random_state=42):
    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y)
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=relative_val, random_state=random_state, stratify=y[idx_trainval])
    return [(idx_train, idx_val, idx_test)]

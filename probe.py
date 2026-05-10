from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

class HallucinationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = None
        self._scaler = StandardScaler()
        self._threshold = 0.5

    def _build_network(self, input_dim):
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self._net(x).squeeze(-1)

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        self._build_network(X_scaled.shape[1])
        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.eval()
        return self

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:,1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0,1,101)]))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs>=t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, float(t)
        self._threshold = best_t
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= self._threshold).astype(int)

    def predict_proba(self, X):
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self(X_t)).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

"""P1 — source x model marginal with shrinkage toward the model marginal (spec §7.3)."""
from __future__ import annotations

import numpy as np

from src.predictors.base import Predictor, empirical_q


class P1SourceModel(Predictor):
    name = "P1"

    def __init__(self, n0: float = 30):
        self.n0 = n0

    def fit(self, train, val=None):
        m = train.meta.model_id.values
        s = train.meta.source.values
        self.q_all = empirical_q(train.y)
        self.q_m = {k: empirical_q(train.y[m == k]) for k in np.unique(m)}
        self.q_sm, self.n_sm = {}, {}
        for key in set(zip(s, m)):
            mask = (s == key[0]) & (m == key[1])
            self.q_sm[key] = empirical_q(train.y[mask])
            self.n_sm[key] = int(mask.sum())
        return self

    def predict_quantiles(self, b):
        out = np.empty((len(b), 99))
        for i, (s, m) in enumerate(zip(b.meta.source.values, b.meta.model_id.values)):
            qm = self.q_m.get(m, self.q_all)
            if (s, m) in self.q_sm:
                n = self.n_sm[(s, m)]
                w = n / (n + self.n0)
                out[i] = w * self.q_sm[(s, m)] + (1 - w) * qm
            else:
                out[i] = qm
        return np.sort(out, axis=1)

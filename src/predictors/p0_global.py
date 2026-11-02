"""P0 — global marginal per model (spec §7.2)."""
from __future__ import annotations

import numpy as np

from src.predictors.base import Predictor, empirical_q


class P0Global(Predictor):
    name = "P0"

    def fit(self, train, val=None):
        m = train.meta.model_id.values
        self.q_m = {k: empirical_q(train.y[m == k]) for k in np.unique(m)}
        self.q_all = empirical_q(train.y)
        return self

    def predict_quantiles(self, b):
        return np.stack([self.q_m.get(k, self.q_all) for k in b.meta.model_id.values])

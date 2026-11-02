"""O2 — prefix quantile GBM (spec §7.11). One P3-style quantile model per k on rows (run, k) of runs with K > k.

The model for a given k is fit by `fit_k`; it can be fit and queried at any k (the base spec's E5 uses the
grid {1, 2, 4, 8, 16, 32}).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.prefix import REGIME_COLS
from src.predictors.base import Bundle
from src.predictors.p3_quantile import P3Quantile


def online_bundle(pre: Bundle, pf: pd.DataFrame, noregime: bool = False) -> Bundle:
    """Join pre-run features (bundle rows) with prefix features (indexed by run_id) at the same k."""
    f = pf.set_index("run_id").loc[pre.meta.run_id.values].reset_index(drop=True)
    cols = [c for c in f.columns if c.startswith("pf_") and not (noregime and c in REGIME_COLS)]
    X = pd.concat([pre.X.reset_index(drop=True), f[cols]], axis=1)
    return Bundle(X, pre.y, pre.meta, None)


class O2Prefix:
    name = "O2"

    def __init__(self, hp, seed=0, noregime=False):
        self.hp, self.seed, self.noregime = hp, seed, noregime
        self.models = {}

    def fit_k(self, k, fit: Bundle, val: Bundle, pf_fit: pd.DataFrame, pf_val: pd.DataFrame):
        """fit/val: pre-run bundles restricted to runs with K > k; pf_*: prefix features at k."""
        tr = online_bundle(fit, pf_fit, self.noregime)
        va = online_bundle(val, pf_val, self.noregime)
        self.models[k] = P3Quantile(self.hp, seed=self.seed).fit(tr, va)
        return self

    def predict_k(self, k, b: Bundle, pf: pd.DataFrame) -> np.ndarray:
        return self.models[k].predict_quantiles(online_bundle(b, pf, self.noregime))

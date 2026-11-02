"""Extension §3.2–3.3: pooled online models that forecast at any decision point.

O2-pooled: one quantile LightGBM per P3 level on rows (run, k), k = 1..min(K-1, 64), with k / log k as features
(prefix features pf_k, pf_log_k), each training run carrying total weight 1 spread equally over its rows.
O1-pooled hazard: one discrete-time hazard model on person-period rows (run, k, j > k) with k as a feature, used
for decision points outside the base k grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.prefix import REGIME_COLS
from src.predictors.base import lgb_train
from src.predictors.p3_quantile import LEVELS, to_grid_fast

KMAX = 64


def cap_k(pf: pd.DataFrame, kmax=KMAX) -> pd.DataFrame:
    pf = pf.copy()
    pf["pf_k"] = np.minimum(pf.pf_k.values, kmax)
    pf["pf_log_k"] = np.log(pf.pf_k.values)
    return pf


def design(Xpre: pd.DataFrame, run_pos: np.ndarray, pf: pd.DataFrame, noregime=False) -> pd.DataFrame:
    """Rows of pre-run features (Xpre.iloc[run_pos]) joined with prefix features (same row order)."""
    cols = [c for c in pf.columns if c.startswith("pf_") and not (noregime and c in REGIME_COLS)]
    X = Xpre.iloc[run_pos].reset_index(drop=True)
    return pd.concat([X, pf[cols].reset_index(drop=True)], axis=1)


def sample_rows(pf_all: pd.DataFrame, K: pd.Series, per_run: int, rng) -> pd.DataFrame:
    """Rows (run, k) with 1 <= k <= min(K-1, 64); at most `per_run` per run, sampled uniformly; weight 1/rows."""
    f = pf_all[(pf_all.k >= 1) & (pf_all.k <= np.minimum(pf_all.run_id.map(K).values - 1, KMAX))]
    f = f.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    f = f.groupby("run_id", sort=False).head(per_run)
    f = f.assign(w=1.0 / f.groupby("run_id").k.transform("size"))
    return f.sort_values(["run_id", "k"]).reset_index(drop=True)


class O2Pooled:
    def __init__(self, hp, seed=0, noregime=False):
        self.hp, self.seed, self.noregime = hp, seed, noregime

    def fit(self, Xtr, wtr, ytr, Xva, wva, yva):
        import lightgbm as lgb
        from src.predictors.base import BASE_PARAMS
        self.models = []
        for a in LEVELS:
            params = dict(BASE_PARAMS, **self.hp, objective="quantile", alpha=float(a), metric="quantile",
                          seed=self.seed)
            dtr = lgb.Dataset(Xtr, ytr, weight=wtr, free_raw_data=False)
            dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr, free_raw_data=False)
            self.models.append(lgb.train(params, dtr, 5000, valid_sets=[dva],
                                         callbacks=[lgb.early_stopping(100, verbose=False)]))
            import time
            print(time.strftime("%H:%M:%S"), f"  level {a}: {self.models[-1].best_iteration} rounds", flush=True)
        return self

    def predict(self, X, chunk=100_000):
        out = np.empty((len(X), 99), dtype=np.float32)
        for s in range(0, len(X), chunk):
            ql = np.sort(np.stack([m.predict(X.iloc[s:s + chunk]) for m in self.models], 1), axis=1)
            out[s:s + chunk] = to_grid_fast(ql, LEVELS)
        return out

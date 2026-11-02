"""P2 — heteroscedastic log-normal with two-stage gradient boosting (spec §7.4)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

from src.eval.metrics import TAUS, qcrps
from src.predictors.base import Predictor, lgb_train, DEFAULT_HP

Z = stats.norm.ppf(TAUS)
LOGCHI2_BIAS = 1.2704


def choose_c(mu, sig, y, grid=(0.5, 2.0, 0.01)):
    cs = np.arange(grid[0], grid[1] + 1e-9, grid[2])
    best = (np.inf, 1.0)
    for c in cs:
        q = mu[:, None] + c * sig[:, None] * Z[None, :]
        s = qcrps(q, y).mean()
        if s < best[0]:
            best = (s, c)
    return best[1], best[0]


class P2Hetero(Predictor):
    name = "P2"

    def __init__(self, hp_mean=None, hp_var=None, seed=0, n_inner=5):
        self.hp_mean = hp_mean or dict(DEFAULT_HP)
        self.hp_var = hp_var or dict(DEFAULT_HP)
        self.seed, self.n_inner = seed, n_inner

    def fit(self, train, val):
        X, y = train.X, train.y
        self.mean = lgb_train(X, y, val.X, val.y, self.hp_mean, seed=self.seed)
        n_r = max(self.mean.best_iteration, 10)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(self.n_inner).split(X, y, groups=train.meta.split_group.values):
            m = lgb_train(X.iloc[tr], y[tr], None, None, self.hp_mean, seed=self.seed, num_rounds=n_r)
            oof[te] = m.predict(X.iloc[te])
        r2 = np.log((y - oof) ** 2 + 1e-6)
        mu_v = self.mean.predict(val.X)
        r2v = np.log((val.y - mu_v) ** 2 + 1e-6)
        self.var = lgb_train(X, r2, val.X, r2v, self.hp_var, seed=self.seed)
        sig_v = np.sqrt(np.exp(self.var.predict(val.X) + LOGCHI2_BIAS))
        self.c, self.val_crps = choose_c(mu_v, sig_v, val.y)
        return self

    def mu_sigma(self, b):
        mu = self.mean.predict(b.X)
        sig = np.sqrt(np.exp(self.var.predict(b.X) + LOGCHI2_BIAS))
        return mu, sig

    def predict_quantiles(self, b):
        mu, sig = self.mu_sigma(b)
        return mu[:, None] + self.c * sig[:, None] * Z[None, :]

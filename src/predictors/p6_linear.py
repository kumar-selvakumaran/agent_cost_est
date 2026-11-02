"""P6 — linear heteroscedastic baseline: P2 with ridge regressions on dense features (spec §7.8)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.eval.metrics import qcrps
from src.predictors.base import Predictor
from src.predictors.p2_hetero import Z, LOGCHI2_BIAS, choose_c


class P6Linear(Predictor):
    name = "P6"

    def __init__(self, alphas=(0.1, 1, 10, 100), n_inner=5):
        self.alphas, self.n_inner = alphas, n_inner

    def fit(self, train, val):
        X, y, Xv, yv = train.Xd, train.y, val.Xd, val.y
        best = None
        for a in self.alphas:
            m = Ridge(alpha=a).fit(X, y)
            mse = np.mean((yv - m.predict(Xv)) ** 2)
            if best is None or mse < best[0]:
                best = (mse, a, m)
        self.alpha_mean, self.mean = best[1], best[2]
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(self.n_inner).split(X, y, groups=train.meta.split_group.values):
            oof[te] = Ridge(alpha=self.alpha_mean).fit(X[tr], y[tr]).predict(X[te])
        r2 = np.log((y - oof) ** 2 + 1e-6)
        mu_v = self.mean.predict(Xv)
        best = None
        for a in self.alphas:
            v = Ridge(alpha=a).fit(X, r2)
            sig_v = np.sqrt(np.exp(v.predict(Xv) + LOGCHI2_BIAS))
            c, s = choose_c(mu_v, sig_v, yv)
            if best is None or s < best[0]:
                best = (s, a, v, c)
        self.val_crps, self.alpha_var, self.var, self.c = best
        return self

    def predict_quantiles(self, b):
        mu = self.mean.predict(b.Xd)
        sig = np.sqrt(np.exp(np.clip(self.var.predict(b.Xd), -30, 30) + LOGCHI2_BIAS))
        return mu[:, None] + self.c * sig[:, None] * Z[None, :]

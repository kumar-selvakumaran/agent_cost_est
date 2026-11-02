"""P3 — direct quantile gradient boosting; P3-EVT — GPD upper tail (spec §7.5)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator

from src.eval.metrics import TAUS
from src.predictors.base import Predictor, lgb_train, DEFAULT_HP

LEVELS = np.array([0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])


def to_grid(qk: np.ndarray, levels=LEVELS) -> np.ndarray:
    qk = np.sort(qk, axis=1)
    out = np.empty((len(qk), 99))
    for i in range(len(qk)):
        x, yv = levels, qk[i]
        # PCHIP needs strictly increasing x; y may tie (fine)
        out[i] = PchipInterpolator(x, yv)(TAUS)
    return np.sort(out, axis=1)


def to_grid_fast(qk: np.ndarray, levels=LEVELS) -> np.ndarray:
    """Vectorised monotone PCHIP (Fritsch-Carlson) over rows."""
    qk = np.sort(qk, axis=1)
    x = levels
    h = np.diff(x)
    d = np.diff(qk, axis=1) / h
    n = len(x)
    m = np.zeros_like(qk)
    # interior slopes: weighted harmonic mean
    for k in range(1, n - 1):
        w1 = 2 * h[k] + h[k - 1]
        w2 = h[k] + 2 * h[k - 1]
        dk0, dk1 = d[:, k - 1], d[:, k]
        same = (dk0 > 0) & (dk1 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w1 + w2) / (w1 / dk0 + w2 / dk1)
        m[:, k] = np.where(same, hm, 0.0)
    # endpoints (scipy's three-point formula, shape-preserving)
    for (k, k1, hk, hk1, dk, dk1) in [(0, 1, h[0], h[1], d[:, 0], d[:, 1]), (n - 1, n - 2, h[-1], h[-2], d[:, -1], d[:, -2])]:
        e = ((2 * hk + hk1) * dk - hk * dk1) / (hk + hk1)
        e = np.where(np.sign(e) != np.sign(dk), 0.0, e)
        e = np.where((np.sign(dk) != np.sign(dk1)) & (np.abs(e) > 3 * np.abs(dk)), 3 * dk, e)
        m[:, k] = e
    j = np.clip(np.searchsorted(x, TAUS, side="right") - 1, 0, n - 2)
    t = (TAUS - x[j]) / h[j]
    y0, y1 = qk[:, j], qk[:, j + 1]
    m0, m1 = m[:, j], m[:, j + 1]
    hj = h[j]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    out = h00 * y0 + h10 * hj * m0 + h01 * y1 + h11 * hj * m1
    return np.sort(out, axis=1)


class P3Quantile(Predictor):
    name = "P3"

    def __init__(self, hp=None, seed=0, levels=LEVELS):
        self.hp = hp or dict(DEFAULT_HP)
        self.seed, self.levels = seed, np.asarray(levels)

    def fit(self, train, val):
        self.models = [lgb_train(train.X, train.y, val.X, val.y, self.hp, objective="quantile", alpha=float(a),
                                 seed=self.seed) for a in self.levels]
        return self

    def raw_levels(self, b) -> np.ndarray:
        return np.sort(np.stack([m.predict(b.X) for m in self.models], 1), axis=1)

    def predict_quantiles(self, b):
        return to_grid_fast(self.raw_levels(b), self.levels)


class P3EVT(Predictor):
    """Wraps a fitted P3; fits a GPD to standardized validation exceedances (McNeil & Frey 2000)."""
    name = "P3-EVT"

    def __init__(self, p3: P3Quantile, threshold_q=0.90, min_exc=200):
        self.p3, self.threshold_q, self.min_exc = p3, threshold_q, min_exc

    def _std(self, ql):
        lv = list(self.p3.levels)
        med = ql[:, lv.index(0.5)]
        scale = np.maximum(ql[:, lv.index(0.9)] - med, 0.05)
        return med, scale

    def fit(self, train, val):
        ql = self.p3.raw_levels(val)
        med, scale = self._std(ql)
        z = (val.y - med) / scale
        self.u = float(np.quantile(z, self.threshold_q))
        exc = z[z > self.u] - self.u
        self.zeta = float((z > self.u).mean())
        self.fallback = len(exc) < self.min_exc
        self.n_exc = int(len(exc))
        if not self.fallback:
            xi, _, beta = stats.genpareto.fit(exc, floc=0)
            self.xi, self.beta = float(xi), float(beta)
        return self

    def predict_quantiles(self, b):
        ql = self.p3.raw_levels(b)
        q = to_grid_fast(ql, self.p3.levels)
        if self.fallback:
            return q
        med, scale = self._std(ql)
        tail = TAUS >= 1 - self.zeta
        p = (1 - TAUS[tail]) / self.zeta
        if abs(self.xi) < 1e-4:
            qz = self.u - self.beta * np.log(p)
        else:
            qz = self.u + self.beta / self.xi * (p ** (-self.xi) - 1)
        q[:, tail] = med[:, None] + scale[:, None] * qz[None, :]
        return np.sort(q, axis=1)

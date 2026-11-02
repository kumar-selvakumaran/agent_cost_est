"""P5 — two-stage structural model: discrete-time hazard on K, then cost given K (spec §7.7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.eval.metrics import TAUS
from src.predictors.base import Predictor, lgb_train, DEFAULT_HP


def j_grid(jmax: int) -> np.ndarray:
    """Evaluation grid for hazards: every j up to 64, then geometric (~4% steps) to jmax."""
    g = list(range(1, min(jmax, 64) + 1))
    j = 64
    while j < jmax:
        j = max(j + 1, int(round(j * 1.04)))
        g.append(min(j, jmax))
    return np.unique(np.array(g))


def person_period(X: pd.DataFrame, K: np.ndarray, kcap: np.ndarray, max_rows: int, rng, run_groups=None,
                  start_after: int = 0):
    """Rows (run, j) for j = start_after+1..K. Returns (Xpp, event, run_index)."""
    K = np.asarray(K).astype(int)
    idx = np.arange(len(K))
    n_rows = np.maximum(K - start_after, 0)
    if n_rows.sum() > max_rows:
        perm = rng.permutation(idx)
        keep = perm[np.cumsum(n_rows[perm]) <= max_rows]
        idx = np.sort(keep)
    reps = n_rows[idx]
    ri = np.repeat(idx, reps)
    j = np.concatenate([np.arange(start_after + 1, k + 1) for k in K[idx]]) if len(idx) else np.array([], int)
    ev = (j == K[ri]).astype(int)
    Xpp = X.iloc[ri].reset_index(drop=True).copy()
    Xpp["j"] = j
    Xpp["log_j"] = np.log(j)
    kc = np.asarray(kcap, dtype=float)[ri]
    Xpp["at_cap"] = (j == kc).astype(int)
    return Xpp, ev, ri


def hazard_matrix(model, X: pd.DataFrame, grid: np.ndarray, kcap: np.ndarray, extra_cols=None) -> np.ndarray:
    """Predict h(j|x) on the grid for every row of X -> (n, len(grid)), interpolated to all j later."""
    n, G = len(X), len(grid)
    out = np.empty((n, G))
    step = max(1, 400_000 // G)
    kc = np.asarray(kcap, dtype=float)
    for s in range(0, n, step):
        Xs = X.iloc[s:s + step]
        m = len(Xs)
        R = Xs.loc[Xs.index.repeat(G)].reset_index(drop=True)
        jj = np.tile(grid, m)
        R["j"] = jj
        R["log_j"] = np.log(jj)
        R["at_cap"] = (jj == np.repeat(kc[s:s + step], G)).astype(int)
        out[s:s + m] = model.predict(R).reshape(m, G)
    return np.clip(out, 1e-6, 1 - 1e-9)


def k_pmf(hz_grid: np.ndarray, grid: np.ndarray, jmax: int, kcap_row: float, start_after: int = 0):
    """Full pmf over j = start_after+1 .. J for one row (vectorised over rows by caller)."""
    raise NotImplementedError


def sample_K(hz_grid, grid, kcap, jmax, M, rng, start_after=0):
    """Sample K for each row given hazards on the grid. Returns (n, M) ints > start_after."""
    n = len(hz_grid)
    J_all = np.arange(1, jmax + 1)
    logg = np.log(grid)
    out = np.empty((n, M), dtype=np.int64)
    for i in range(n):
        h = np.interp(np.log(J_all), logg, hz_grid[i])
        cap = kcap[i]
        if np.isfinite(cap) and cap >= 1:
            Jc = int(cap)
            if Jc <= jmax:
                h = h[:Jc].copy()
                h[-1] = 1.0
            else:
                tail_h = h[-5:].mean()
                h = np.r_[h, np.full(Jc - jmax, tail_h)]
                h[-1] = 1.0
        else:
            tail_h = h[-5:].mean()
            h = np.r_[h, np.full(4 * jmax, tail_h)]  # up to 5 * jmax
            h[-1] = 1.0  # truncation
        h = h[start_after:]
        h[-1] = 1.0
        surv = np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])
        pmf = h * surv
        cdf = np.cumsum(pmf)
        cdf /= cdf[-1]
        u = rng.random(M)
        out[i] = np.searchsorted(cdf, u) + 1 + start_after
    return out


class P5Structural(Predictor):
    name = "P5"

    def __init__(self, hp_haz=None, hp_cost=None, seed=0, M=2000, max_pp=2_000_000):
        self.hp_haz = hp_haz or dict(DEFAULT_HP)
        self.hp_cost = hp_cost or dict(DEFAULT_HP)
        self.seed, self.M, self.max_pp = seed, M, max_pp

    def _xk(self, X, K):
        Xk = X.copy()
        Xk["K"] = K
        Xk["log_K"] = np.log(K)
        return Xk

    def fit(self, train, val):
        rng = np.random.default_rng(self.seed)
        Ktr, Kv = train.meta.K.values, val.meta.K.values
        Xpp, ev, _ = person_period(train.X, Ktr, train.meta.K_cap.values, self.max_pp, rng)
        Xppv, evv, _ = person_period(val.X, Kv, val.meta.K_cap.values, self.max_pp // 4, rng)
        self.haz = lgb_train(Xpp, ev, Xppv, evv, self.hp_haz, objective="binary", seed=self.seed)
        self.jmax = int(np.ceil(1.5 * Ktr.max()))
        self.grid = j_grid(self.jmax)
        Xk, Xkv = self._xk(train.X, Ktr), self._xk(val.X, Kv)
        self.cost = lgb_train(Xk, train.y, Xkv, val.y, self.hp_cost, seed=self.seed)
        n_r = max(self.cost.best_iteration, 10)
        oof = np.zeros(len(train.y))
        for tr, te in GroupKFold(5).split(Xk, train.y, groups=train.meta.split_group.values):
            m = lgb_train(Xk.iloc[tr], train.y[tr], None, None, self.hp_cost, seed=self.seed, num_rounds=n_r)
            oof[te] = m.predict(Xk.iloc[te])
        self.resid = train.y - oof
        self.k_edges = np.unique(np.quantile(Ktr, np.linspace(0, 1, 11)[1:-1]))
        self.kbin_train = np.searchsorted(self.k_edges, Ktr, side="right")
        self.resid_bins = [self.resid[self.kbin_train == b] for b in range(len(self.k_edges) + 1)]
        self.resid_bins = [r if len(r) else self.resid for r in self.resid_bins]
        return self

    def cost_curve(self, X: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
        n, G = len(X), len(grid)
        out = np.empty((n, G))
        step = max(1, 400_000 // G)
        for s in range(0, n, step):
            Xs = X.iloc[s:s + step]
            m = len(Xs)
            R = Xs.loc[Xs.index.repeat(G)].reset_index(drop=True)
            kk = np.tile(grid, m)
            R["K"] = kk
            R["log_K"] = np.log(kk)
            out[s:s + m] = self.cost.predict(R).reshape(m, G)
        return out

    def simulate(self, b, seed_offset=0, return_samples=False):
        rng = np.random.default_rng(self.seed + 7919 * seed_offset)
        X = b.X.reset_index(drop=True)
        kcap = b.meta.K_cap.values.astype(float)
        hz = hazard_matrix(self.haz, X, self.grid, kcap)
        cgrid = np.unique(np.r_[self.grid, np.geomspace(self.jmax, 5 * self.jmax, 20).astype(int)])
        cc = self.cost_curve(X, cgrid)
        Ks = sample_K(hz, self.grid, kcap, self.jmax, self.M, rng)
        out = np.empty((len(X), 99))
        logc = np.log(cgrid)
        for i in range(len(X)):
            yk = np.interp(np.log(Ks[i]), logc, cc[i])
            bins = np.searchsorted(self.k_edges, Ks[i], side="right")
            e = np.empty(self.M)
            for bb in np.unique(bins):
                mk = bins == bb
                pool = self.resid_bins[bb]
                e[mk] = pool[rng.integers(0, len(pool), mk.sum())]
            out[i] = np.quantile(yk + e, TAUS)
        return out

    def predict_quantiles(self, b):
        return np.sort(self.simulate(b), axis=1)

"""O1 — structural online model (spec §7.11).

At decision point k (after observing turns 1..k): the cost so far C_<=k is known; the remaining number of
turns K - k is drawn from a discrete-time hazard model conditioned on K > k (one hazard model per k, fit on
person-period rows j > k with pre-run + prefix features at k); the cost of each future turn is computed
mechanically from the observed context growth. Uncertainty lives only in K. Works at any k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import TAUS
from src.predictors.base import Bundle, lgb_train
from src.predictors.o2_prefix_gbm import online_bundle
from src.predictors.p5_survival import person_period, hazard_matrix, j_grid, sample_K


def remaining_cost_table(I_k, O_k, obs_k, I_1, O_mean, k, max_extra, lam_c=0.1, lam_o=4.0):
    """Mechanical cost of future turns j = k+1 .. k+max_extra for one run; returns cumulative remaining cost
    array R[m] = cost of turns k+1..k+m (m = 1..max_extra).

    I_{k+1} ~= I_k + O_k (spec); I_j = I_{k+1} + (j - k - 1) * g, with g the mean per-turn context growth
    observed so far, g = (I_k + O_k + obs_k - I_1) / k (>= 1); O_j = mean O over turns <= k;
    cached input per §5.2 (perfect prefix caching): Cch_j = min(I_{j-1} + O_{j-1}, I_j)."""
    g = max((I_k + O_k + obs_k - I_1) / k, 1.0)
    m = np.arange(1, max_extra + 1)
    I_next = I_k + O_k
    I = I_next + (m - 1) * g
    O = np.full(max_extra, O_mean)
    prev_I = np.r_[I_k, I[:-1]]
    prev_O = np.r_[O_k, O[:-1]]
    Cch = np.minimum(prev_I + prev_O, I)
    U = I - Cch
    c = U + lam_c * Cch + lam_o * O
    return np.cumsum(c)


class O1Structural:
    name = "O1"

    def __init__(self, hp_haz, seed=0, M=2000, max_pp=2_000_000):
        self.hp_haz, self.seed, self.M, self.max_pp = hp_haz, seed, M, max_pp
        self.haz = {}

    def fit_k(self, k, fit: Bundle, val: Bundle, pf_fit, pf_val):
        rng = np.random.default_rng(self.seed + k)
        tr = online_bundle(fit, pf_fit)
        va = online_bundle(val, pf_val)
        Xpp, ev, _ = person_period(tr.X, fit.meta.K.values, fit.meta.K_cap.values, self.max_pp, rng, start_after=k)
        Xppv, evv, _ = person_period(va.X, val.meta.K.values, val.meta.K_cap.values, self.max_pp // 4, rng,
                                     start_after=k)
        self.haz[k] = lgb_train(Xpp, ev, Xppv, evv, self.hp_haz, objective="binary", seed=self.seed)
        self.jmax = int(np.ceil(1.5 * max(fit.meta.K.max(), k + 2)))
        return self

    def predict_k(self, k, b: Bundle, pf: pd.DataFrame, state: pd.DataFrame, lam_c=0.1, lam_o=4.0):
        """state: per run (indexed by run_id) I_k, O_k, obs_k, I_1, O_mean, C_k at this k."""
        rng = np.random.default_rng(self.seed + 104729 * k)
        ob = online_bundle(b, pf)
        X = ob.X.reset_index(drop=True)
        kcap = b.meta.K_cap.values.astype(float)
        jmax = max(self.jmax, k + 2)
        grid = np.unique(np.r_[j_grid(jmax)])
        grid = grid[grid > k]
        if len(grid) == 0 or grid[0] != k + 1:
            grid = np.unique(np.r_[k + 1, grid])
        hz = hazard_matrix(self.haz[k], X, grid, kcap)
        # hazards are only defined for j > k: prepend placeholder hazards (never used) for j <= k
        full_grid = np.r_[np.arange(1, k + 1), grid]
        hz_full = np.c_[np.full((len(X), k), 0.5), hz]
        Ks = sample_K(hz_full, full_grid, kcap, jmax, self.M, rng, start_after=k)
        st = state.loc[b.meta.run_id.values]
        out = np.empty((len(X), 99))
        for i in range(len(X)):
            s = st.iloc[i]
            extra = Ks[i] - k
            R = remaining_cost_table(s.I_k, s.O_k, s.obs_k, s.I_1, s.O_mean, k, int(extra.max()), lam_c, lam_o)
            y = np.log(s.C_k + R[extra - 1])
            out[i] = np.quantile(y, TAUS)
        return out


def online_state(turns: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-run observable state after k turns (only turns <= k)."""
    t = turns[turns.k <= k]
    g = t.groupby("run_id")
    last = t[t.k == k].set_index("run_id")
    st = pd.DataFrame({"I_k": last.I, "O_k": last.O, "obs_k": last.obs, "C_k": last.cum_cost})
    st["I_1"] = g.I.first().loc[st.index]
    st["O_mean"] = g.O.mean().loc[st.index]
    return st

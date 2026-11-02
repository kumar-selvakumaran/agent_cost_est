"""CQR — conformalized quantile regression wrapper (spec §7.10)."""
from __future__ import annotations

import numpy as np

from src.eval.metrics import TAUS, interp_q

LEVELS_2S = [0.5, 0.8, 0.9, 0.95]
LEVELS_1S = [0.9, 0.95, 0.99]


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    k = int(np.ceil(level * (n + 1)))
    if k > n:
        return float(np.inf)
    return float(np.sort(scores)[k - 1])


def one_per_task(meta, rng) -> np.ndarray:
    """Index of one randomly chosen run per task (calibration exchangeable at task level)."""
    idx = np.arange(len(meta))
    perm = rng.permutation(idx)
    first = meta.task_id.values[perm]
    _, pos = np.unique(first, return_index=True)
    return np.sort(perm[pos])


class CQR:
    """Calibrated on (q_cal, y_cal). Provides symmetric adjustment of every quantile pair on the grid
    (tau, 1 - tau) so that CQR yields a full 99-quantile forecast; plus the spec's level-specific intervals."""

    def __init__(self, mondrian_min=200):
        self.mondrian_min = mondrian_min

    def _fit_adj(self, q, y):
        adj = np.zeros(49)
        for j in range(49):  # pair (tau_{j+1}, tau_{99-j}) -> central level 1 - 2 tau
            lo, hi = q[:, j], q[:, 98 - j]
            s = np.maximum(lo - y, y - hi)
            adj[j] = conformal_quantile(s, 1 - 2 * TAUS[j])
        two = {}
        for L in LEVELS_2S:
            a = 1 - L
            s = np.maximum(interp_q(q, a / 2) - y, y - interp_q(q, 1 - a / 2))
            two[L] = conformal_quantile(s, L)
        one = {}
        for t in LEVELS_1S:
            one[t] = conformal_quantile(y - interp_q(q, t), t)
        return adj, two, one

    def fit(self, q_cal, y_cal, model_cal=None):
        self.pool = self._fit_adj(q_cal, y_cal)
        self.n_cal = len(y_cal)
        self.by_model = {}
        if model_cal is not None:
            for m in np.unique(model_cal):
                mk = model_cal == m
                if mk.sum() >= self.mondrian_min:
                    self.by_model[m] = self._fit_adj(q_cal[mk], y_cal[mk])
        return self

    def _params(self, i_model, mondrian):
        if mondrian and i_model in self.by_model:
            return self.by_model[i_model]
        return self.pool

    def adjust_grid(self, q, models=None, mondrian=False):
        out = q.copy()
        for i in range(len(q)):
            adj = self._params(models[i] if models is not None else None, mondrian)[0]
            out[i, :49] = q[i, :49] - adj
            out[i, 50:] = q[i, 50:] + adj[::-1]
        return np.sort(out, axis=1)

    def intervals(self, q, level, models=None, mondrian=False):
        a = 1 - level
        lo, hi = interp_q(q, a / 2), interp_q(q, 1 - a / 2)
        Q = np.array([self._params(models[i] if models is not None else None, mondrian)[1][level] for i in range(len(q))])
        return lo - Q, hi + Q

    def upper(self, q, tau, models=None, mondrian=False):
        Q = np.array([self._params(models[i] if models is not None else None, mondrian)[2][tau] for i in range(len(q))])
        return interp_q(q, tau) + Q

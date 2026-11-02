"""ACI — adaptive conformal inference (Gibbs & Candes 2021), spec E7."""
from __future__ import annotations

import numpy as np

from src.predictors.cqr import conformal_quantile
from src.eval.metrics import interp_q


def aci_run(q_test, y_test, cal_scores_fn, alpha=0.1, gamma=0.005, clip=(0.001, 0.5)):
    """Process test points in order. cal_scores_fn(alpha_t) -> conformal correction Q_hat at level 1-alpha_t.

    Returns (lo, hi, alpha_path, err)."""
    n = len(y_test)
    a_t = alpha
    lo, hi, path, err = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        Qh, ql, qh = cal_scores_fn(a_t, q_test[t:t + 1])
        lo[t], hi[t] = ql - Qh, qh + Qh
        e = float(not (lo[t] <= y_test[t] <= hi[t]))
        path[t], err[t] = a_t, e
        a_t = float(np.clip(a_t + gamma * (alpha - e), *clip))
    return lo, hi, path, err


class CQRScores:
    """Precomputes calibration nonconformity scores for a central interval at any level."""

    def __init__(self, q_cal, y_cal):
        self.q_cal, self.y_cal = q_cal, y_cal

    def __call__(self, a, q_row):
        ql_c, qh_c = interp_q(self.q_cal, a / 2), interp_q(self.q_cal, 1 - a / 2)
        s = np.maximum(ql_c - self.y_cal, self.y_cal - qh_c)
        Qh = conformal_quantile(s, 1 - a)
        if not np.isfinite(Qh):
            Qh = s.max()
        return Qh, float(interp_q(q_row, a / 2)[0]), float(interp_q(q_row, 1 - a / 2)[0])

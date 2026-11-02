"""Scoring rules on 99-quantile forecasts of y = log Y (spec §9.1–9.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TAUS = np.arange(1, 100) / 100.0
QCOLS = [f"q{j:02d}" for j in range(1, 100)]


def monotone(q: np.ndarray) -> np.ndarray:
    """Monotone rearrangement (Chernozhukov et al. 2010): sort each row."""
    return np.sort(np.asarray(q, dtype=float), axis=1)


def pinball(q: np.ndarray, y: np.ndarray, tau) -> np.ndarray:
    u = np.asarray(y)[:, None] - np.atleast_2d(q).reshape(len(y), -1)
    tau = np.atleast_1d(tau)[None, :]
    return u * (tau - (u < 0))


def qcrps(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """QCRPS(q, y) = (2/99) sum_j rho_tau_j(y - q_j)."""
    return 2.0 / 99.0 * pinball(q, y, TAUS).sum(axis=1)


def q_at(q: np.ndarray, tau: float) -> np.ndarray:
    j = int(round(tau * 100))
    return q[:, j - 1]


def pit(q: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT by linear interpolation of tau between neighbouring quantiles."""
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        qi, yi = q[i], y[i]
        if yi < qi[0]:
            out[i] = rng.uniform(0, 0.01)
        elif yi > qi[-1]:
            out[i] = rng.uniform(0.99, 1.0)
        else:
            # handle ties/flat segments: interpolate over the plateau
            lo = np.searchsorted(qi, yi, side="left")
            hi = np.searchsorted(qi, yi, side="right")
            if hi > lo:  # y equals one or more quantiles exactly
                out[i] = rng.uniform(TAUS[lo], TAUS[hi - 1]) if hi - 1 > lo else TAUS[lo]
            else:
                j = lo  # qi[j-1] < yi < qi[j]
                q0, q1 = qi[j - 1], qi[j]
                out[i] = TAUS[j - 1] + (yi - q0) / (q1 - q0) * 0.01
    return out


def pit_vectorized(q: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    below = (q < y[:, None]).sum(axis=1)       # number of quantiles strictly below y
    eq = (q == y[:, None]).sum(axis=1)
    out = np.empty(n)
    lo_mask = below == 0
    hi_mask = below == 99
    out[lo_mask & (eq == 0)] = rng.uniform(0, 0.01, (lo_mask & (eq == 0)).sum())
    out[hi_mask] = rng.uniform(0.99, 1.0, hi_mask.sum())
    mid = ~(hi_mask | (lo_mask & (eq == 0)))
    idx = np.flatnonzero(mid)
    b = below[idx]
    e = eq[idx]
    # ties: uniform over the tied tau range
    tie = e > 0
    ti = idx[tie]
    lo_t = TAUS[b[tie]]
    hi_t = TAUS[np.minimum(b[tie] + e[tie] - 1, 98)]
    out[ti] = lo_t + rng.uniform(0, 1, len(ti)) * (hi_t - lo_t)
    ni = idx[~tie]
    bn = b[~tie]
    q0 = q[ni, bn - 1]
    q1 = q[ni, bn]
    out[ni] = TAUS[bn - 1] + (y[ni] - q0) / (q1 - q0) * 0.01
    return out


def ks_uniform(p: np.ndarray) -> float:
    return float(stats.kstest(p, "uniform").statistic)


def central_cover(q: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    a = (1 - level) / 2
    lo = np.interp(a, TAUS, np.arange(99))
    # levels used are on the grid (0.5, 0.8, 0.9 -> exact; 0.95 -> 0.025/0.975 interpolate)
    ql = interp_q(q, a)
    qh = interp_q(q, 1 - a)
    return ((y >= ql) & (y <= qh)).astype(float)


def interp_q(q: np.ndarray, tau: float) -> np.ndarray:
    """Quantile at arbitrary tau by linear interpolation in tau (clamped to grid)."""
    tau = min(max(tau, 0.01), 0.99)
    pos = tau * 100 - 1
    j0 = int(np.floor(pos + 1e-9))
    w = pos - j0
    if w < 1e-9:
        return q[:, j0]
    return (1 - w) * q[:, j0] + w * q[:, j0 + 1]


def run_metrics(q: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-run metric columns (aggregated later task-weighted / run-weighted)."""
    d = {
        "qcrps": qcrps(q, y),
        "pin50": pinball(q[:, [49]], y, [0.5])[:, 0],
        "pin90": pinball(q[:, [89]], y, [0.9])[:, 0],
        "pin99": pinball(q[:, [98]], y, [0.99])[:, 0],
        "abs_err50": np.abs(y - q[:, 49]),
        "width80": q[:, 89] - q[:, 9],
        "cov50": central_cover(q, y, 0.5),
        "cov80": central_cover(q, y, 0.8),
        "cov90": central_cover(q, y, 0.9),
        "cov95": central_cover(q, y, 0.95),
        "up90": (y <= q[:, 89]).astype(float),
        "up95": (y <= q[:, 94]).astype(float),
        "up99": (y <= q[:, 98]).astype(float),
    }
    return pd.DataFrame(d)


def task_weighted(values: np.ndarray, task: np.ndarray) -> float:
    s = pd.Series(values).groupby(np.asarray(task)).mean()
    return float(s.mean())


def wasserstein_cells(q: np.ndarray, y: np.ndarray, cell: np.ndarray, min_runs: int = 10) -> pd.DataFrame:
    """W1 between predicted (mean over cell's runs) and empirical type-7 cell quantiles."""
    df = pd.DataFrame({"cell": cell, "i": np.arange(len(y))})
    rows = []
    for c, g in df.groupby("cell"):
        if len(g) < min_runs:
            continue
        idx = g.i.values
        emp = np.quantile(y[idx], TAUS, method="linear")
        pred = q[idx].mean(axis=0)
        rows.append((c, float(np.mean(np.abs(pred - emp))), len(idx)))
    return pd.DataFrame(rows, columns=["cell", "w1", "n"])


def es_ratio(q: np.ndarray, y: np.ndarray) -> float:
    """Realized mean of y above predicted q95 divided by predicted ES_0.95 (on Y scale ratio of logs? -> log scale)."""
    es_pred = q[:, 94:].mean(axis=1)
    hit = y > q[:, 94]
    if hit.sum() == 0:
        return float("nan")
    return float(np.mean(y[hit]) / np.mean(es_pred[hit]))

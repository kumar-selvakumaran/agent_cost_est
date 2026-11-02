"""Calibration backtests (spec §9.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec unconditional coverage LR test; hits = 1{y > q_tau}, p = 1 - tau."""
    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    x = hits.sum()
    def xlogy(a, b):
        return 0.0 if a == 0 else a * np.log(b)
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    phat = x / n
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def dq_test(hits: np.ndarray, p: float, Z: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Cross-sectional DQ-style test: OLS of (h - p) on Z, Wald test all coef = 0,
    task-clustered (CR1) covariance."""
    h = np.asarray(hits, dtype=float) - p
    Z = np.asarray(Z, dtype=float)
    keep = np.isfinite(Z).all(axis=1)
    Z, h, groups = Z[keep], h[keep], np.asarray(groups)[keep]
    # drop constant/duplicate columns (except intercept at col 0)
    sd = Z.std(axis=0)
    cols = [0] + [j for j in range(1, Z.shape[1]) if sd[j] > 1e-12]
    Z = Z[:, cols]
    # remove linearly dependent columns
    q, r = np.linalg.qr(Z)
    ind = np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()
    Z = Z[:, ind]
    ZtZi = np.linalg.pinv(Z.T @ Z)
    beta = ZtZi @ Z.T @ h
    e = h - Z @ beta
    codes, uniq = pd.factorize(groups)
    G = codes.max() + 1
    S = np.zeros((G, Z.shape[1]))
    np.add.at(S, codes, Z * e[:, None])
    n, k = Z.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = adj * ZtZi @ (S.T @ S) @ ZtZi
    try:
        W = float(beta @ np.linalg.pinv(V) @ beta)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), k
    return W, float(stats.chi2.sf(W, k)), k

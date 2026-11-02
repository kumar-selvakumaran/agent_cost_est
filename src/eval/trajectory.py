"""Trajectory-integrated metrics for online cost forecasts (extension §2–§4).

A decision-point table has one row per (run, k), k = 0..K-1 ("before call k+1"), with columns:
run_id, task_id, k, K, C_k (cost after k calls), N_k (tokens after k calls), I_next (= I_{k+1}, input tokens of the
next call; known at d_k), n_next (= n_{k+1} = I_{k+1} + O_{k+1}, used only by NTA), y (= log Y), Y, N.
Forecasts are 99 quantiles of log total cost per row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import qcrps, TAUS, pit_vectorized


def decision_points(turns: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Build the decision-point table from turns (one row per call) — only quantities from turns <= k enter C_k,
    N_k and I_next (I_{k+1} is the size of the history about to be sent at d_k)."""
    t = turns.sort_values(["run_id", "k"])
    g = t.groupby("run_id", sort=False)
    prev_cost = g.cum_cost.shift(1).fillna(0.0).values          # C_{k-1} at call k  -> C at d_{k-1}
    prev_tok = g.cum_tokens.shift(1).fillna(0).values
    dp = pd.DataFrame({"run_id": t.run_id.values, "k": t.k.values - 1, "C_k": prev_cost, "N_k": prev_tok,
                       "I_next": t.I.values, "n_next": t.n_tok.values})
    R = runs.set_index("run_id")
    dp["K"] = R.loc[dp.run_id, "K"].values
    dp["Y"] = R.loc[dp.run_id, "Y"].values
    dp["y"] = np.log(dp.Y.values)
    dp["N"] = g.cum_tokens.transform("max").values
    dp["task_id"] = R.loc[dp.run_id, "task_id"].values
    return dp.reset_index(drop=True)


def truncate(q: np.ndarray, C_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce q_j >= log C_k (cost already spent is a floor); returns corrected quantiles and an infeasible flag."""
    floor = np.log(np.maximum(C_k, 1e-12))[:, None]
    bad = (q < floor).any(axis=1) & (C_k > 0)
    out = np.where((q < floor) & (C_k[:, None] > 0), floor, q)
    return np.sort(out, axis=1), bad


def its(S: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * S) / np.sum(w))


def nta_per_run(dp: pd.DataFrame, q50: np.ndarray) -> pd.Series:
    """NTA_r = sum_k (n_{k+1}/N) |Yhat^(k) - Y| / Y with Yhat = exp(median)."""
    err = dp.n_next.values / dp.N.values * np.abs(np.exp(q50) - dp.Y.values) / dp.Y.values
    return pd.Series(err).groupby(dp.run_id.values).sum()


def area_per_run(dp: pd.DataFrame, yhat: np.ndarray) -> pd.Series:
    """A_r = sum_k n_{k+1} |Yhat^(k) - Y| (tokens x reference cost)."""
    return pd.Series(dp.n_next.values * np.abs(yhat - dp.Y.values)).groupby(dp.run_id.values).sum()


def its_logR(q: np.ndarray, dp: pd.DataFrame) -> np.ndarray:
    """Per-decision-point QCRPS of log remaining cost; q^R = max(exp(q) - C_k, 0), zeros -> log(1); R floored at 1."""
    qR = np.maximum(np.exp(q) - dp.C_k.values[:, None], 0.0)
    lq = np.log(np.maximum(qR, 1.0))
    R = np.maximum(dp.Y.values - dp.C_k.values, 1.0)
    return qcrps(np.sort(lq, axis=1), np.log(R))


def its_level(q: np.ndarray, Y: np.ndarray, chunk=200_000) -> np.ndarray:
    out = np.empty(len(Y))
    for s in range(0, len(Y), chunk):
        out[s:s + chunk] = qcrps(np.exp(q[s:s + chunk]), Y[s:s + chunk])
    return out


def time_to_confidence(dp: pd.DataFrame, q50: np.ndarray, tol=np.log(1.25)) -> pd.Series:
    """Smallest N_k/N such that |q50 - y| <= tol for all later decision points; NaN if never."""
    ok = np.abs(q50 - dp.y.values) <= tol
    df = pd.DataFrame({"run_id": dp.run_id.values, "k": dp.k.values, "pos": dp.N_k.values / dp.N.values, "ok": ok})
    df = df.sort_values(["run_id", "k"], ascending=[True, False])
    # running "all ok from here to the end" from the last decision point backwards
    df["all_ok"] = df.groupby("run_id").ok.cumprod().astype(bool)
    first = df[df.all_ok].groupby("run_id").pos.min()
    return first.reindex(df.run_id.unique())


def weighted_ks(p: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(p)
    cw = np.cumsum(w[o]) / w.sum()
    lo = np.r_[0.0, cw[:-1]]
    ps = p[o]
    return float(max(np.max(np.abs(cw - ps)), np.max(np.abs(lo - ps))))

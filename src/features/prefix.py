"""Prefix features after observing turns 1..k (spec §6.3). Every feature at row (run, k) is a causal
cumulative statistic of turns <= k (cumsum / cummax / current value), so it can be evaluated at any k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_COLS = ["pf_rep_share", "pf_err_count", "pf_err_streak_max", "pf_tc_count", "pf_empty_share",
               "pf_pfail_count"]


def prefix_features(turns: pd.DataFrame) -> pd.DataFrame:
    """turns: rows (run_id, k) sorted by run_id, k with columns I, O, obs, cum_cost, err, rep, empty, tc, pfail.

    Returns one row per input row with features computed from turns <= k of the same run."""
    t = turns.sort_values(["run_id", "k"])
    g = t.groupby("run_id", sort=False)
    k = t.k.values.astype(float)
    out = pd.DataFrame({"run_id": t.run_id.values, "k": t.k.values}, index=t.index)
    out["pf_k"] = k
    out["pf_log_k"] = np.log(k)
    out["pf_log_cum_cost"] = np.log(t.cum_cost.values)
    out["pf_log_ctx"] = np.log(t.I.values + t.O.values)
    out["pf_O_mean"] = np.log1p(g.O.cumsum().values / k)
    out["pf_O_last"] = np.log1p(t.O.values)
    out["pf_obs_mean"] = np.log1p(g.obs.cumsum().values / k)
    out["pf_obs_last"] = np.log1p(t.obs.values)
    # least-squares slope of I_j on j over j <= k, from cumulative sums
    j = k
    sI = g.I.cumsum().values.astype(float)
    sjI = (t.I.values * t.k.values).astype(float)
    sjI = pd.Series(sjI, index=t.index).groupby(t.run_id.values, sort=False).cumsum().values
    sj = k * (k + 1) / 2
    sjj = k * (k + 1) * (2 * k + 1) / 6
    den = k * sjj - sj ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(den > 0, (k * sjI - sj * sI) / den, 0.0)
    out["pf_ctx_growth"] = np.sign(slope) * np.log1p(np.abs(slope))
    out["pf_rep_share"] = g.rep.cumsum().values / k
    out["pf_err_count"] = g.err.cumsum().values
    # longest streak of consecutive error observations so far
    e = t.err.values.astype(int)
    brk = (e == 0).astype(int)
    seg = pd.Series(brk, index=t.index).groupby(t.run_id.values, sort=False).cumsum().values
    streak = pd.Series(e, index=t.index).groupby([t.run_id.values, seg], sort=False).cumsum().values
    out["pf_err_streak_max"] = pd.Series(streak, index=t.index).groupby(t.run_id.values, sort=False).cummax().values
    out["pf_tc_count"] = g.tc.cumsum().values
    out["pf_empty_share"] = g.empty.cumsum().values / k
    out["pf_pfail_count"] = g.pfail.cumsum().values
    return out.loc[turns.index] if out.index.equals(turns.index) else out


def features_at(turns: pd.DataFrame, k_target: dict | int) -> pd.DataFrame:
    """Prefix features of each run at a given k (int for all, or dict run_id -> k). Only turns <= k are used."""
    if isinstance(k_target, int):
        sub = turns[turns.k <= k_target]
        f = prefix_features(sub)
        return f[f.k == k_target].reset_index(drop=True)
    kk = turns.run_id.map(k_target)
    sub = turns[turns.k <= kk]
    f = prefix_features(sub)
    return f[f.k == f.run_id.map(k_target)].reset_index(drop=True)

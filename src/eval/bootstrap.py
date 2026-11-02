"""Task-cluster bootstrap (spec §9.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def task_cluster_indices(task: np.ndarray, B: int, seed: int):
    """Yield, for each replicate, the task-level resampling weights (counts per task)."""
    codes, uniq = pd.factorize(np.asarray(task))
    T = len(uniq)
    rng = np.random.default_rng(seed)
    W = np.zeros((B, T), dtype=np.int32)
    for b in range(B):
        draw = rng.integers(0, T, T)
        W[b] = np.bincount(draw, minlength=T)
    return codes, W


def boot_task_metric(values: np.ndarray, task: np.ndarray, B: int = 1000, seed: int = 0,
                     weighting: str = "task", codes_W=None) -> dict:
    """Point estimate + 95% percentile CI of a mean metric under task-cluster bootstrap.

    weighting='task': mean of per-task means; 'run': mean over runs.
    """
    codes, W = codes_W if codes_W is not None else task_cluster_indices(task, B, seed)
    v = np.asarray(values, dtype=float)
    T = W.shape[1]
    s = np.bincount(codes, weights=v, minlength=T)
    n = np.bincount(codes, minlength=T).astype(float)
    if weighting == "task":
        tm = s / n
        est = tm.mean()
        reps = (W @ tm) / W.sum(axis=1)
    else:
        est = s.sum() / n.sum()
        reps = (W @ s) / (W @ n)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi), reps=reps)


def paired_diff(a: np.ndarray, b: np.ndarray, task: np.ndarray, B: int = 1000, seed: int = 0,
                codes_W=None) -> dict:
    """Bootstrap of task-weighted mean(a) - mean(b); two-sided p-value."""
    r = boot_task_metric(np.asarray(a) - np.asarray(b), task, B, seed, "task", codes_W)
    reps = r["reps"]
    p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
    p = min(1.0, max(p, 1.0 / len(reps)))
    return dict(est=r["est"], lo=r["lo"], hi=r["hi"], p=float(p), reps=reps)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj

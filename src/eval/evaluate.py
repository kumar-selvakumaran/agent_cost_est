"""Full evaluation suite for pooled out-of-fold quantile predictions (spec §9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.backtests import kupiec, dq_test
from src.eval.bootstrap import task_cluster_indices, boot_task_metric, paired_diff, holm
from src.eval.metrics import QCOLS, TAUS, run_metrics, pit_vectorized, ks_uniform, wasserstein_cells, es_ratio

TASK_WT = ["qcrps", "pin50", "pin90", "pin99", "abs_err50", "width80", "cov50", "cov80", "cov90", "cov95",
           "up90", "up95", "up99"]


def qmat(df: pd.DataFrame) -> np.ndarray:
    return df[QCOLS].to_numpy(dtype=float)


class Evaluator:
    """Holds the shared bootstrap resampling (same task draws for every method -> paired comparisons)."""

    def __init__(self, runs: pd.DataFrame, task_ids: np.ndarray, B: int, seed: int):
        self.runs = runs.set_index("run_id")
        self.task_order = np.asarray(task_ids)
        self.B, self.seed = B, seed
        self.codes_W = task_cluster_indices(self.task_order, B, seed)
        self.per_run = {}

    def per_run_metrics(self, method: str, pred: pd.DataFrame) -> pd.DataFrame:
        q, y = qmat(pred), pred.y.values
        m = run_metrics(q, y)
        m.insert(0, "run_id", pred.run_id.values)
        self.per_run[method] = (m, q, y, pred)
        return m

    def summarize(self, method: str, pred: pd.DataFrame, dq=True, extra_ci=True) -> dict:
        assert np.array_equal(pred.task_id.values, self.task_order), "predictions must be aligned to evaluator order"
        m = self.per_run_metrics(method, pred)
        q, y = qmat(pred), pred.y.values
        task = pred.task_id.values
        out = dict(method=method, n_runs=len(y), n_tasks=pd.Series(task).nunique())
        for c in TASK_WT:
            r = boot_task_metric(m[c].values, task, self.B, self.seed, "task", self.codes_W)
            out[c], out[c + "_lo"], out[c + "_hi"] = r["est"], r["lo"], r["hi"]
        r = boot_task_metric(m.qcrps.values, task, self.B, self.seed, "run", self.codes_W)
        out["qcrps_run"], out["qcrps_run_lo"], out["qcrps_run_hi"] = r["est"], r["lo"], r["hi"]
        out["spearman"] = float(stats.spearmanr(q[:, 49], y).statistic)
        out["pearson"] = float(np.corrcoef(q[:, 49], y)[0, 1])
        if extra_ci:
            codes, W = self.codes_W
            sp, pe = [], []
            rng = np.random.default_rng(self.seed + 1)
            for b in range(min(self.B, 200)):
                idx = np.repeat(np.arange(len(y)), W[b][codes])
                sp.append(stats.spearmanr(q[idx, 49], y[idx]).statistic)
                pe.append(np.corrcoef(q[idx, 49], y[idx])[0, 1])
            out["spearman_lo"], out["spearman_hi"] = np.percentile(sp, [2.5, 97.5])
            out["pearson_lo"], out["pearson_hi"] = np.percentile(pe, [2.5, 97.5])
        rng = np.random.default_rng(self.seed + 2)
        p = pit_vectorized(q, y, rng)
        self.per_run[method] = self.per_run[method] + (p,)
        out["pit_ks"] = ks_uniform(p)
        if extra_ci:
            codes, W = self.codes_W
            ks = []
            for b in range(min(self.B, 200)):
                idx = np.repeat(np.arange(len(y)), W[b][codes])
                ks.append(ks_uniform(p[idx]))
            out["pit_ks_lo"], out["pit_ks_hi"] = np.percentile(ks, [2.5, 97.5])
        for t in (0.9, 0.95, 0.99):
            j = int(round(t * 100)) - 1
            lr, pv = kupiec((y > q[:, j]).astype(float), 1 - t)
            out[f"kupiec_lr_{int(t*100)}"], out[f"kupiec_p_{int(t*100)}"] = lr, pv
        if dq:
            meta = self.runs.loc[pred.run_id.values]
            top_m = meta.model_id.value_counts().index[:10]
            fams = sorted(meta.source_family.unique())[1:]
            for t in (0.9, 0.95):
                j = int(round(t * 100)) - 1
                Z = np.c_[np.ones(len(y)), q[:, j],
                          np.stack([(meta.model_id == mm).values for mm in top_m[1:]], 1),
                          np.stack([(meta.source_family == f).values for f in fams], 1),
                          np.log1p(meta.instr_tokens.values)]
                W_, pv, k = dq_test((y > q[:, j]).astype(float), 1 - t, Z, task)
                out[f"dq_stat_{int(t*100)}"], out[f"dq_p_{int(t*100)}"] = W_, pv
        cell = self.runs.loc[pred.run_id.values].cell_id.values
        w1 = wasserstein_cells(q, y, cell, 10)
        out["w1_cells"] = float(w1.w1.mean()) if len(w1) else np.nan
        out["w1_n_cells"] = int(len(w1))
        if len(w1) > 1:
            rng = np.random.default_rng(self.seed + 3)
            bs = [w1.w1.values[rng.integers(0, len(w1), len(w1))].mean() for _ in range(self.B)]
            out["w1_cells_lo"], out["w1_cells_hi"] = np.percentile(bs, [2.5, 97.5])
        out["es_ratio"] = es_ratio(q, y)
        return out

    def compare(self, a: str, b: str) -> dict:
        ma, mb = self.per_run[a][0], self.per_run[b][0]
        task = self.task_order
        r = paired_diff(ma.qcrps.values, mb.qcrps.values, task, self.B, self.seed, self.codes_W)
        return dict(a=a, b=b, diff=r["est"], lo=r["lo"], hi=r["hi"], p=r["p"])

    def rel_gain(self, method: str, ref: str = "P1") -> dict:
        """(CRPS_ref - CRPS_m) / CRPS_ref with bootstrap CI (task-weighted)."""
        codes, W = self.codes_W
        T = W.shape[1]
        def tw(v):
            s = np.bincount(codes, weights=v, minlength=T)
            n = np.bincount(codes, minlength=T)
            tm = s / n
            return tm.mean(), (W @ tm) / W.sum(1)
        e_r, reps_r = tw(self.per_run[ref][0].qcrps.values)
        e_m, reps_m = tw(self.per_run[method][0].qcrps.values)
        g = (e_r - e_m) / e_r
        gr = (reps_r - reps_m) / reps_r
        lo, hi = np.percentile(gr, [2.5, 97.5])
        return dict(gain=g, gain_lo=lo, gain_hi=hi, reps=gr)


def comparisons_table(ev: Evaluator, methods: list[str], ref="P1") -> pd.DataFrame:
    rows = []
    best = min(methods, key=lambda m: ev.per_run[m][0].qcrps.groupby(ev.task_order).mean().mean())
    for m in methods:
        if m != ref:
            rows.append(ev.compare(m, ref))
        if m != best:
            rows.append(ev.compare(m, best))
    df = pd.DataFrame(rows).drop_duplicates(subset=["a", "b"])
    df["p_holm"] = holm(df.p.values)
    df["best_method"] = best
    return df


def pit_by_group(ev: Evaluator, method: str, col: str) -> pd.DataFrame:
    m, q, y, pred, p = ev.per_run[method]
    g = ev.runs.loc[pred.run_id.values, col].values
    rows = []
    for k in np.unique(g):
        mk = g == k
        if mk.sum() >= 30:
            rows.append(dict(method=method, group_col=col, group=k, n=int(mk.sum()), pit_ks=ks_uniform(p[mk]),
                             cov90=float(m.cov90.values[mk].mean())))
    return pd.DataFrame(rows)

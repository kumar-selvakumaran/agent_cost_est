"""E9 evaluation: ITS (primary), variants, NTA, calibration along the trajectory, inference, figures, tables."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.eval.bootstrap import task_cluster_indices, holm
from src.eval.metrics import QCOLS, qcrps, pit_vectorized, central_cover
from src.eval.trajectory import (decision_points, truncate, its_logR, its_level, nta_per_run, time_to_confidence,
                                 weighted_ks)
from src.experiments.common import plt, write_log, color
from src.experiments.e3_cold import load_pred
from src.utils import seed_for

PRED = "data_cache/preds"
METHODS = ["Static", "Static-truncated", "O1", "O2", "O2-noregime"]
COL = {"Static": "#999999", "Static-truncated": "#000000", "O1": "#D55E00", "O2": "#0072B2", "O2-noregime": "#56B4E9"}


def chunked(fn, *arrs, chunk=200_000):
    n = len(arrs[0])
    out = []
    for s in range(0, n, chunk):
        out.append(fn(*[a[s:s + chunk] for a in arrs]))
    return np.concatenate(out)


def forecasts(cfg, runs, dp):
    """Yields (method, q float32 [n_dp, 99], infeasible flag, O1 source) aligned with dp."""
    key = pd.Series(np.arange(len(dp)), index=pd.MultiIndex.from_arrays([dp.run_id.values, dp.k.values]))
    static = np.empty((len(dp), 99), dtype=np.float32)
    rid_pos = dp.groupby("run_id").indices
    for f in range(cfg["cv"]["n_splits"]):
        d = load_pred("e3", "P5", f)
        for rid, q in zip(d["test_run_id"].astype(np.int64), d["test"]):
            static[rid_pos[rid]] = q
    C = dp.C_k.values
    _, bad = truncate(static[:1], C[:1])
    bad_static = chunked(lambda q, c: truncate(q.astype(float), c)[1], static, C)
    yield "Static", static, bad_static, None
    st = chunked(lambda q, c: truncate(q.astype(float), c)[0].astype(np.float32), static, C)
    yield "Static-truncated", st, np.zeros(len(dp), bool), None
    del st
    for name in ["O1", "O2", "O2-noregime"]:
        q = static.copy()
        src = np.zeros(len(dp), dtype=np.int8)
        for f in range(cfg["cv"]["n_splits"]):
            d = np.load(os.path.join(PRED, f"e9__f{f}.npz"))
            idx = key.loc[list(zip(d["dp_run_id"], d["dp_k"]))].values
            q[idx] = d[name]
            if name == "O1":
                src[idx] = d["O1_src"]
        bad = chunked(lambda a, c: truncate(a.astype(float), c)[1], q, C)
        q = chunked(lambda a, c: truncate(a.astype(float), c)[0].astype(np.float32), q, C)
        yield name, q, bad, (src if name == "O1" else None)


def task_reps(codes, W, num, den):
    T = W.shape[1]
    sn = np.bincount(codes, weights=num, minlength=T)
    sd = np.bincount(codes, weights=den, minlength=T)
    return sn.sum() / sd.sum(), (W @ sn) / (W @ sd)


def task_mean_reps(codes, W, v):
    T = W.shape[1]
    s = np.bincount(codes, weights=v, minlength=T)
    n = np.bincount(codes, minlength=T)
    tm = s / n
    return tm.mean(), (W @ tm) / W.sum(1)


def evaluate(cfg, runs, splits, turns):
    os.makedirs("results/predictions", exist_ok=True)
    B = cfg["bootstrap"]["B"]
    seed = seed_for(cfg, "e9_boot")
    dp = decision_points(turns, runs)
    R = runs.set_index("run_id")
    w = dp.I_next.values.astype(float)
    early = (dp.k.values <= 7).astype(float)
    codes, W = task_cluster_indices(dp.task_id.values, B, seed)
    run_tab = dp.groupby("run_id").task_id.first()
    rcodes, RW = task_cluster_indices(run_tab.values, B, seed)   # same seed & task order? map below
    # make run-level bootstrap use the same task draws: map run task to dp task code order
    tcode = pd.Series(codes, index=dp.task_id.values).groupby(level=0).first()
    rcodes = tcode.loc[run_tab.values].values
    RW = W
    fold = splits.set_index("run_id").fold
    summ = R.summ_flag.values
    succ = R.success
    writer = None
    per = {}
    for name, q, bad, src in forecasts(cfg, runs, dp):
        qf = q
        S = chunked(lambda a, y: qcrps(a.astype(float), y), qf, dp.y.values)
        S_level = chunked(lambda a, y: its_level(a.astype(float), y), qf, dp.Y.values)
        S_logR = chunked(lambda a, c, y: its_logR(a.astype(float), pd.DataFrame({"C_k": c, "Y": y})), qf,
                         dp.C_k.values, dp.Y.values)
        rng = np.random.default_rng(seed + 11)
        P = chunked(lambda a, y: pit_vectorized(a.astype(float), y, rng), qf, dp.y.values)
        cov80 = ((dp.y.values >= qf[:, 9]) & (dp.y.values <= qf[:, 89])).astype(float)
        width = (qf[:, 89] - qf[:, 9]).astype(float)
        q50 = qf[:, 49].astype(float)
        per[name] = dict(S=S, S_level=S_level, S_logR=S_logR, P=P, cov80=cov80, width=width, q50=q50, bad=bad, src=src)
        # prediction file (one block per method)
        df = pd.DataFrame(qf, columns=QCOLS)
        meta = pd.DataFrame({"method": name, "run_id": dp.run_id.values, "task_id": dp.task_id.values,
                             "model_id": R.loc[dp.run_id, "model_id"].values, "source": R.loc[dp.run_id, "source"].values,
                             "fold": fold.loc[dp.run_id].values, "k": dp.k.values, "K": dp.K.values,
                             "C_k": dp.C_k.values, "N_k": dp.N_k.values, "I_next": dp.I_next.values})
        tail = pd.DataFrame({"y": dp.y.values, "Y": dp.Y.values, "N": dp.N.values,
                             "success": succ.loc[dp.run_id].values})
        tb = pa.Table.from_pandas(pd.concat([meta, df, tail], axis=1), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter("results/predictions/e9_decision_points.parquet", tb.schema, compression="zstd")
        writer.write_table(tb)
        del df, tb, qf, q
        print("scored", name, flush=True)
    writer.close()

    # ---- metrics with CIs
    rows, reps = [], {}
    run_ids = run_tab.index.values
    for name in METHODS:
        m = per[name]
        r = {"method": name}
        for key, wt, sc in [("ITS", w, "S"), ("ITS_early", w * early, "S"), ("ITS_turn", np.ones(len(w)), "S"),
                            ("ITS_level", w, "S_level"), ("ITS_logR", w, "S_logR")]:
            est, rp = task_reps(codes, W, wt * m[sc], wt)
            lo, hi = np.percentile(rp, [2.5, 97.5])
            r.update({key: est, key + "_lo": lo, key + "_hi": hi})
            reps[(name, key)] = rp
        its_r = pd.Series(w * m["S"]).groupby(dp.run_id.values).sum() / pd.Series(w).groupby(dp.run_id.values).sum()
        est, rp = task_mean_reps(rcodes, RW, its_r.loc[run_ids].values)
        r.update(ITS_rb=est, ITS_rb_lo=np.percentile(rp, 2.5), ITS_rb_hi=np.percentile(rp, 97.5))
        nta = nta_per_run(dp, m["q50"]).loc[run_ids].values
        est, rp = task_mean_reps(rcodes, RW, nta)
        r.update(NTA_mean=est, NTA_lo=np.percentile(rp, 2.5), NTA_hi=np.percentile(rp, 97.5), NTA_median=np.median(nta))
        per[name]["nta"] = nta
        r["PIT_KS_w"] = weighted_ks(m["P"], w)
        r["infeasible_pct"] = 100 * m["bad"][dp.k.values >= 1].mean()
        ttc = time_to_confidence(dp, m["q50"])
        r["ttc_median"] = float(np.nanmedian(ttc.values)) if ttc.notna().any() else np.nan
        r["ttc_never_share"] = float(ttc.isna().mean())
        r["cov80_w"] = float(np.sum(w * m["cov80"]) / w.sum())
        rows.append(r)
    main = pd.DataFrame(rows)
    main.to_csv("results/tables/e9_main.csv", index=False)

    # ---- comparisons (paired, Holm)
    def cmp(a, b, key="ITS"):
        ea = main.set_index("method").loc[a, key]
        eb = main.set_index("method").loc[b, key]
        d = reps[(a, key)] - reps[(b, key)]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return dict(a=a, b=b, metric=key, diff=ea - eb, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                    p=max(p, 1 / len(d)))
    comp = pd.DataFrame([cmp("O2", "Static-truncated"), cmp("O2", "O2-noregime"), cmp("O2", "O2-noregime", "ITS_early"),
                         cmp("O2", "O1"), cmp("O1", "Static-truncated"), cmp("Static-truncated", "Static")])
    comp["p_holm"] = holm(comp.p.values)
    comp.to_csv("results/tables/e9_comparisons.csv", index=False)
    # ---- hypotheses
    c = comp.set_index(["a", "b", "metric"])
    def verdict(row):
        return "supported" if row.hi < 0 and row.p_holm < 0.05 else ("not supported" if row.lo > 0 else "inconclusive")
    h7 = c.loc[("O2", "Static-truncated", "ITS")]
    h9 = c.loc[("O2", "O1", "ITS")]
    h8a = c.loc[("O2", "O2-noregime", "ITS")]
    gap = lambda key: (reps[("O2-noregime", key)] - reps[("O2", key)]) / reps[("O2-noregime", key)]
    m_ = main.set_index("method")
    g_its = (m_.loc["O2-noregime", "ITS"] - m_.loc["O2", "ITS"]) / m_.loc["O2-noregime", "ITS"]
    g_early = (m_.loc["O2-noregime", "ITS_early"] - m_.loc["O2", "ITS_early"]) / m_.loc["O2-noregime", "ITS_early"]
    dg = gap("ITS_early") - gap("ITS")
    dlo, dhi = np.percentile(dg, [2.5, 97.5])
    h8 = "supported" if (verdict(h8a) == "supported" and dlo > 0) else (
        "not supported" if (h8a.lo > 0 or dhi < 0) else "inconclusive")
    hyp = pd.DataFrame([
        dict(hypothesis="H7", verdict=verdict(h7), stat="ITS(O2) - ITS(Static-truncated)", est=h7["diff"], lo=h7.lo, hi=h7.hi, p_holm=h7.p_holm),
        dict(hypothesis="H8", verdict=h8, stat="ITS(O2) - ITS(O2-noregime); rel. gap early - rel. gap ITS",
             est=h8a["diff"], lo=h8a.lo, hi=h8a.hi, p_holm=h8a.p_holm, gap_its=g_its, gap_early=g_early,
             gap_diff=g_early - g_its, gap_diff_lo=dlo, gap_diff_hi=dhi),
        dict(hypothesis="H9", verdict=verdict(h9), stat="ITS(O2) - ITS(O1)", est=h9["diff"], lo=h9.lo, hi=h9.hi, p_holm=h9.p_holm)])
    hyp.to_csv("results/tables/h7_h9.csv", index=False)

    # ---- by model / source family
    for col, fname in [("model_id", "e9_by_model.csv"), ("source_family", "e9_by_source_family.csv")]:
        g = R.loc[dp.run_id, col].values
        out = []
        for name in METHODS:
            S = per[name]["S"]
            for v in np.unique(g):
                mk = g == v
                if mk.sum() < 100:
                    continue
                cw = task_cluster_indices(dp.task_id.values[mk], 200, seed)
                est, rp = task_reps(cw[0], cw[1], (w * S)[mk], w[mk])
                out.append({"method": name, col: v, "n_decision_points": int(mk.sum()), "ITS": est,
                            "lo": np.percentile(rp, 2.5), "hi": np.percentile(rp, 97.5)})
        pd.DataFrame(out).to_csv("results/tables/" + fname, index=False)

    # ---- sensitivity: (a) summarization-flagged runs excluded, (c) K >= 5, (b) secondary targets for Static/-truncated
    sens = []
    for label, mk in [("all decision points", np.ones(len(dp), bool)),
                      ("summarization-flagged runs excluded", ~R.loc[dp.run_id, "summ_flag"].values.astype(bool)),
                      ("runs with K >= 5", dp.K.values >= 5)]:
        cw = task_cluster_indices(dp.task_id.values[mk], B, seed)
        for name in METHODS:
            est, rp = task_reps(cw[0], cw[1], (w * per[name]["S"])[mk], w[mk])
            sens.append(dict(analysis=label, method=name, ITS=est, lo=np.percentile(rp, 2.5), hi=np.percentile(rp, 97.5)))
    for tgt, exp, ccol in [("y_tok", "e3tok", "tok"), ("y_hi", "e3hi", "hi")]:
        tt = turns.sort_values(["run_id", "k"])
        if ccol == "tok":
            cum = tt.groupby("run_id").n_tok.cumsum().values
        else:
            cum = (tt.U + cfg["cost"]["lambda_c"] * tt.Cch + cfg["cost"]["lambda_o_hi"] * tt.O).groupby(tt.run_id).cumsum().values
        prev = pd.Series(cum).groupby(tt.run_id.values).shift(1).fillna(0).values
        yt = R.loc[dp.run_id, tgt].values
        st = np.empty((len(dp), 99), dtype=np.float32)
        pos = dp.groupby("run_id").indices
        for f in range(cfg["cv"]["n_splits"]):
            d = load_pred(exp, "P5", f)
            for rid, q in zip(d["test_run_id"].astype(np.int64), d["test"]):
                st[pos[rid]] = q
        cw = (codes, W)
        for name, qq in [("Static", st), ("Static-truncated",
                                          chunked(lambda a, c: truncate(a.astype(float), c)[0].astype(np.float32), st, prev))]:
            S = chunked(lambda a, y: qcrps(a.astype(float), y), qq, yt)
            est, rp = task_reps(codes, W, w * S, w)
            sens.append(dict(analysis=f"target {tgt} (Static methods only)", method=name, ITS=est,
                             lo=np.percentile(rp, 2.5), hi=np.percentile(rp, 97.5)))
    pd.DataFrame(sens).to_csv("results/tables/e9_sensitivity.csv", index=False)

    # ---- diagnostics + sanity checks
    diag = main[["method", "infeasible_pct", "ttc_median", "ttc_never_share", "PIT_KS_w", "cov80_w"]].copy()
    src = per["O1"]["src"]
    diag["O1_per_k_share"] = np.where(diag.method == "O1", (src == 1).mean(), np.nan)
    diag["O1_pooled_share"] = np.where(diag.method == "O1", (src == 2).mean(), np.nan)
    ncap = sum(int(np.load(os.path.join(PRED, f"e9__f{f}.npz"))["n_capped_k"][0]) for f in range(cfg["cv"]["n_splits"]))
    diag["n_forecasts_k_gt_64"] = ncap
    diag.to_csv("results/tables/e9_diagnostics.csv", index=False)
    san = []
    # 1. Static ITS == pooled mean of pre-run QCRPS weighted by sum_k w_rk
    S0 = per["Static"]["S"][dp.k.values == 0]
    wr = pd.Series(w).groupby(dp.run_id.values).sum().loc[dp.run_id.values[dp.k.values == 0]].values
    v1 = np.sum(wr * S0) / wr.sum()
    san.append(dict(check="1 Static ITS = weighted pre-run QCRPS", value_a=m_.loc["Static", "ITS"], value_b=v1,
                    passed=bool(abs(m_.loc["Static", "ITS"] - v1) < 1e-9)))
    # 4. O2 at grid k: (a) the base per-k forecasts (E5) scored through the E9 pipeline reproduce E5's per-k QCRPS;
    #    (b) informative: pooled O2 (used in E9) vs the base per-k O2 on the same decision points.
    e5 = pd.read_csv("results/tables/e5_online.csv")
    e5p = pd.read_parquet("results/predictions/e5_online.parquet")
    e5p = e5p[e5p.method == "O2"]
    key = pd.Series(np.arange(len(dp)), index=pd.MultiIndex.from_arrays([dp.run_id.values, dp.k.values]))
    for k in [1, 2, 4, 8, 16, 32]:
        sub = e5p[(e5p.k == k) & e5p.run_id.isin(dp.run_id.values)]
        idx = key.loc[list(zip(sub.run_id.values, np.full(len(sub), k)))].values
        Sb = qcrps(sub[QCOLS].to_numpy(float), dp.y.values[idx])
        cw = task_cluster_indices(dp.task_id.values[idx], B, seed)
        eb, rb = task_mean_reps(cw[0], cw[1], Sb)
        ep, rp = task_mean_reps(cw[0], cw[1], per["O2"]["S"][idx])
        base = e5[(e5.population == "eligible") & (e5.method == "O2") & (e5.k == k)].iloc[0]
        san.append(dict(check=f"4a O2 per-k at k={k}: E5 table vs E9 pipeline on the same forecasts",
                        value_a=base.qcrps, value_b=eb, lo_b=np.percentile(rb, 2.5), hi_b=np.percentile(rb, 97.5),
                        passed=bool(np.percentile(rb, 2.5) <= base.qcrps <= np.percentile(rb, 97.5))))
        d = rp - rb
        san.append(dict(check=f"4b (info) pooled O2 minus per-k O2 at k={k}", value_a=eb, value_b=ep,
                        lo_b=np.percentile(d, 2.5), hi_b=np.percentile(d, 97.5), passed=np.nan))
    san.append(dict(check="5 rows = sum of K over test runs", value_a=len(dp), value_b=float(runs.K.sum()),
                    passed=bool(len(dp) == runs.K.sum())))
    pd.DataFrame(san).to_csv("results/tables/e9_sanity.csv", index=False)
    figures(cfg, dp, per, runs)
    write_e9_log(main, comp, hyp, pd.DataFrame(san), diag)


def figures(cfg, dp, per, runs):
    prog = dp.N_k.values / dp.N.values
    dec = np.minimum((prog * 10).astype(int), 9)
    w = dp.I_next.values.astype(float)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for name in METHODS:
        m = per[name]
        e, c, wd = [], [], []
        for d in range(10):
            mk = dec == d
            e.append(m["S"][mk].mean())
            c.append(m["cov80"][mk].mean())
            wd.append(m["width"][mk].mean())
        x = (np.arange(10) + 0.5) / 10
        ax1.plot(x, e, "o-", color=COL[name], label=name)
        ax2.plot(x, c, "o-", color=COL[name], label=name)
        ax3.plot(x, wd, "o-", color=COL[name], label=name)
    ax2.axhline(0.8, color="k", ls="--", lw=1)
    for ax, yl, t, fn, f in [(ax1, "mean QCRPS of log total cost", "Error vs progress", "fig_e9_error_vs_progress.png", fig1),
                             (ax2, "coverage of central 80% interval", "Coverage vs progress", "fig_e9_coverage_vs_progress.png", fig2),
                             (ax3, "mean width q90 − q10 (log units)", "Sharpness vs progress", "fig_e9_width_vs_progress.png", fig3)]:
        ax.set_xlabel("normalized token clock N_k / N (decile midpoints)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.legend(fontsize=7)
        f.tight_layout()
        f.savefig("results/figures/" + fn)
        plt.close(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in METHODS:
        v = np.sort(per[name]["nta"])
        ax.plot(v, np.linspace(0, 1, len(v)), color=COL[name], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("NTA per run (log scale)")
    ax.set_ylabel("cumulative share of runs")
    ax.set_title("Distribution of NTA")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/figures/fig_e9_nta_distribution.png")
    plt.close(fig)
    # example trajectories: one run per sextile of K, seed SEED + 9 (never chosen by error)
    rng = np.random.default_rng(cfg["seed"] + 9)
    K = runs.set_index("run_id").K
    edges = np.quantile(K.values, np.linspace(0, 1, 7))
    picks = []
    for i in range(6):
        cand = K[(K >= edges[i]) & ((K < edges[i + 1]) if i < 5 else (K <= edges[i + 1]))].index.values
        cand = cand[~np.isin(cand, picks)]
        picks.append(int(rng.choice(cand)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, rid in zip(axes.flat, picks):
        mk = np.flatnonzero(dp.run_id.values == rid)
        d = dp.iloc[mk]
        Y, N = d.Y.iloc[0], d.N.iloc[0]
        Nk = np.r_[d.N_k.values, N]
        Ck = np.r_[d.C_k.values, Y]
        ax.plot(Nk / N, (Y - Ck) / Y, color="k", lw=2, label="true remaining R(t)")
        for name in ["Static-truncated", "O1", "O2"]:
            q50 = np.exp(per[name]["q50"][mk])
            m = per[name]
            lo = np.exp(np.array([0.0]))  # placeholder
            # forecast remaining-cost median held over (N_k, N_{k+1}] -> step function, normalized by Y
            xs = np.repeat(Nk, 2)[1:-1] / N
            Ct = np.repeat(Ck[:-1], 2)
            Ct = np.interp(xs * N, Nk, Ck)
            med = np.repeat(q50, 2)
            ax.plot(xs, np.maximum(med - Ct, 0) / Y, color=COL[name], lw=1.2, label=f"{name} median")
            if name == "O2":
                ax.fill_between(xs, np.maximum(med - Ct, 0) / Y, (Y - Ct) / Y, color=COL[name], alpha=0.15,
                                label="area (NTA)")
        ax.set_title(f"run {rid}: K = {int(d.K.iloc[0])}", fontsize=8)
        ax.set_xlabel("normalized token clock t / N")
        ax.set_ylabel("remaining cost / Y")
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Example trajectories (one random run per sextile of K): true vs forecast remaining cost")
    fig.tight_layout()
    fig.savefig("results/figures/fig_e9_example_trajectories.png")
    plt.close(fig)


def write_e9_log(main, comp, hyp, san, diag):
    L = ["# E9 — Trajectory-integrated evaluation\n",
         "ITS: token-weighted (weight = input tokens of the next call, known when the forecast is issued) average "
         "QCRPS of log total cost over every decision point of every run; proper. NTA: the author's normalized area "
         "between forecast and true remaining-cost curves (median forecast); descriptive and improper.\n",
         "## Main table\n", main.round(4).to_markdown(index=False) + "\n",
         "## Comparisons (paired task-cluster bootstrap, Holm)\n", comp.round(4).to_markdown(index=False) + "\n",
         "## Hypotheses\n", hyp.round(4).to_markdown(index=False) + "\n",
         "## Sanity checks\n", san.round(4).to_markdown(index=False) + "\n",
         "## Diagnostics\n", diag.round(4).to_markdown(index=False) + "\n"]
    write_log("e9", L)
    print("\n".join(L))

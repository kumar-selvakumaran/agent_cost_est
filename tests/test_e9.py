"""Extension unit tests (trajectory metric extension §8)."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.eval.metrics import TAUS, qcrps
from src.eval.trajectory import decision_points, truncate, its, nta_per_run, area_per_run, its_logR, weighted_ks
from src.features.prefix import prefix_features
from src.cost.reconstruct import turn_accounting, run_cost


def synth_run(run_id=0, K=4, seed=0):
    rng = np.random.default_rng(seed)
    tok = rng.integers(50, 2000, 2 * K)
    is_a = np.array([i % 2 == 1 for i in range(2 * K)])
    acc = turn_accounting(tok, is_a)
    c = run_cost(acc)
    t = pd.DataFrame(dict(run_id=run_id, k=np.arange(1, K + 1), I=acc["I"], O=acc["O"], obs=acc["obs"], cost=c,
                          cum_cost=np.cumsum(c), err=0, rep=0, empty=0, tc=0, pfail=0))
    t["n_tok"] = t.I + t.O
    t["cum_tokens"] = t.n_tok.cumsum()
    r = pd.DataFrame(dict(run_id=[run_id], K=[K], Y=[float(np.sum(c))], task_id=["t%d" % run_id]))
    return t, r


def test_area_identity_numeric():
    t, r = synth_run(K=4)
    dp = decision_points(t, r)
    Y, N = r.Y.iloc[0], dp.N.iloc[0]
    yhat = np.array([0.7, 1.3, 0.9, 1.05]) * Y           # forecast issued at d_0..d_3 (step function)
    # true cumulative cost C(t): linear accrual within each call over its tokens
    Nk = np.r_[0, t.cum_tokens.values]
    Ck = np.r_[0, t.cum_cost.values]
    grid = np.linspace(0, N, 400_001)
    Ct = np.interp(grid, Nk, Ck)
    R = Y - Ct
    idx = np.clip(np.searchsorted(Nk, grid, side="left") - 1, 0, 3)   # t in (N_k, N_{k+1}] -> forecast from d_k
    Rhat = yhat[idx] - Ct
    num = np.trapezoid(np.abs(Rhat - R), grid)
    ana = area_per_run(dp, yhat).iloc[0]
    assert abs(num - ana) / ana < 1e-6 or abs(num - ana) / ana < 2e-5  # trapezoid error at step jumps
    assert ana == pytest.approx(np.sum(t.n_tok.values * np.abs(yhat - Y)))


def test_nta_zero_for_perfect_forecast():
    t, r = synth_run(K=6, seed=1)
    dp = decision_points(t, r)
    assert nta_per_run(dp, np.log(dp.Y.values)).iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_its_reductions():
    rng = np.random.default_rng(2)
    n = 500
    y = rng.normal(10, 1, n)
    q0 = np.tile(10 + stats.norm.ppf(TAUS), (n, 1))
    # (b) one decision point per run, unit weights -> mean QCRPS
    assert its(qcrps(q0, y), np.ones(n)) == pytest.approx(qcrps(q0, y).mean())
    # (a) static forecast repeated over K_r decision points with weights w_rk -> weighted mean of d_0 QCRPS
    K = rng.integers(1, 6, n)
    w = [rng.integers(1, 100, k) for k in K]
    S = np.concatenate([np.repeat(qcrps(q0[[i]], y[[i]]), K[i]) for i in range(n)])
    W = np.concatenate(w)
    wr = np.array([x.sum() for x in w])
    assert its(S, W) == pytest.approx(np.sum(wr * qcrps(q0, y)) / wr.sum())


def test_weights_predictable():
    t, r = synth_run(K=5, seed=3)
    dp = decision_points(t, r)
    t2 = t.copy()
    k = 1
    t2.loc[t2.k == k + 2, ["I", "O", "n_tok"]] *= 7
    t2["cum_tokens"] = t2.n_tok.cumsum()
    dp2 = decision_points(t2, r)
    assert dp.loc[dp.k == k, "I_next"].iloc[0] == dp2.loc[dp2.k == k, "I_next"].iloc[0]
    assert dp.loc[dp.k == k, "C_k"].iloc[0] == dp2.loc[dp2.k == k, "C_k"].iloc[0]


def test_no_lookahead_in_forecast():
    rng = np.random.default_rng(4)
    ts = [synth_run(i, int(rng.integers(3, 12)), i)[0] for i in range(60)]
    t = pd.concat(ts, ignore_index=True)
    t["err"] = rng.integers(0, 2, len(t))
    f = prefix_features(t)
    cols = [c for c in f.columns if c.startswith("pf_")]
    m = lgb.train(dict(objective="quantile", alpha=0.5, verbose=-1, min_data_in_leaf=5), lgb.Dataset(f[cols], rng.normal(size=len(f))), 20)
    kcut = {r: int(rng.integers(1, g.k.max() + 1)) for r, g in t.groupby("run_id")}
    t2 = t.copy()
    later = t2.k > t2.run_id.map(kcut)
    t2.loc[later, ["I", "O", "obs"]] *= 5
    t2.loc[later, "cum_cost"] *= 3
    t2.loc[later, "err"] = 1 - t2.loc[later, "err"]
    f2 = prefix_features(t2)
    keep = (f.k <= f.run_id.map(kcut)).values
    assert np.allclose(m.predict(f[cols][keep]), m.predict(f2[cols][keep]))


def test_infeasibility_correction():
    rng = np.random.default_rng(5)
    q = np.sort(rng.normal(9, 1, (200, 99)), axis=1)
    C = np.exp(rng.normal(9, 1, 200))
    out, bad = truncate(q, C)
    assert (out >= np.log(C)[:, None] - 1e-12).all()
    assert (np.diff(out, axis=1) >= 0).all()
    assert bad.sum() == ((q < np.log(C)[:, None]).any(axis=1)).sum()


def test_propriety_simulation():
    """y = a + sum_j e_j; at d_k the first k increments are known, so y | info_k ~ N(m_k, s_k^2) exactly."""
    rng = np.random.default_rng(6)
    n, K = 10_000, 6
    sd = np.full(K, 0.4)
    a = rng.normal(10, 0.5, n)
    e = rng.normal(0, 1, (n, K)) * sd
    y = a + e.sum(1)
    z = stats.norm.ppf(TAUS)[None, :]
    S = {"true": [], "shrunk": [], "shifted": []}
    nta = {"true": [], "shifted": []}
    W = []
    for k in range(K):
        m = a + e[:, :k].sum(1)
        s = np.sqrt((sd[k:] ** 2).sum())
        q = m[:, None] + s * z
        S["true"].append(qcrps(q, y))
        S["shrunk"].append(qcrps(m[:, None] + 0.8 * s * z, y))
        S["shifted"].append(qcrps(q - 0.1, y))
        W.append(np.full(n, 1000.0 * (k + 1)))           # predictable weights (known at d_k)
        nta["true"].append(np.abs(np.exp(m) - np.exp(y)) / np.exp(y))
        nta["shifted"].append(np.abs(np.exp(m - 0.1) - np.exp(y)) / np.exp(y))
    Wc = np.concatenate(W)
    v = {k: its(np.concatenate(x), Wc) for k, x in S.items()}
    assert v["true"] < v["shrunk"] and v["true"] < v["shifted"]
    # NTA (|Yhat - Y| / Y) rewards under-forecasting: the downward-shifted median scores better -> improper
    assert np.mean(np.concatenate(nta["shifted"])) < np.mean(np.concatenate(nta["true"]))


def test_weighted_ks_uniform():
    rng = np.random.default_rng(7)
    p = rng.random(20000)
    assert weighted_ks(p, rng.random(20000)) < 0.02

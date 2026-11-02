import numpy as np
import pytest
from scipy import stats

from src.eval.metrics import TAUS, qcrps, monotone, pit_vectorized, pit, ks_uniform, run_metrics
from src.eval.backtests import kupiec, dq_test
from src.eval.bootstrap import boot_task_metric, holm


def gauss_crps(mu, s, y):
    z = (y - mu) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_qcrps_matches_gaussian_closed_form():
    rng = np.random.default_rng(1)
    mu, s = 10.0, 0.8
    y = rng.normal(mu, s, 20000)
    q = np.tile(mu + s * stats.norm.ppf(TAUS), (len(y), 1))
    a = qcrps(q, y).mean()
    b = gauss_crps(mu, s, y).mean()
    assert abs(a - b) / b < 0.01


def test_monotone():
    q = np.random.default_rng(2).normal(size=(50, 99))
    m = monotone(q)
    assert (np.diff(m, axis=1) >= 0).all()


def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(3)
    n = 20000
    mu = rng.normal(10, 1, n)
    s = rng.uniform(0.2, 1.5, n)
    y = rng.normal(mu, s)
    q = mu[:, None] + s[:, None] * stats.norm.ppf(TAUS)[None, :]
    p = pit_vectorized(q, y, rng)
    assert ks_uniform(p) < 0.015
    hist = np.histogram(p, bins=10, range=(0, 1))[0] / n
    assert np.all(np.abs(hist - 0.1) < 0.01)
    p2 = pit(q[:500], y[:500], np.random.default_rng(0))
    p3 = pit_vectorized(q[:500], y[:500], np.random.default_rng(0))
    inner = (y[:500] > q[:500, 0]) & (y[:500] < q[:500, -1])
    assert np.allclose(p2[inner], p3[inner])


def test_pit_misspecified_detected():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 2, 5000)
    q = np.tile(stats.norm.ppf(TAUS), (5000, 1))  # too narrow
    assert ks_uniform(pit_vectorized(q, y, rng)) > 0.1


def test_kupiec_known_example():
    # n=250, x=5 at p=0.01 -> LR approx 1.9568 (textbook value)
    hits = np.zeros(250); hits[:5] = 1
    lr, p = kupiec(hits, 0.01)
    assert lr == pytest.approx(1.9568, abs=1e-3)
    lr0, p0 = kupiec(np.r_[np.ones(10), np.zeros(90)], 0.1)
    assert lr0 == pytest.approx(0.0, abs=1e-9)


def test_dq_detects_predictable_exceedances():
    rng = np.random.default_rng(5)
    n = 4000
    x = rng.normal(size=n)
    hits = (rng.random(n) < np.where(x > 0, 0.2, 0.0)).astype(float)
    Z = np.c_[np.ones(n), x]
    W, p, k = dq_test(hits, 0.1, Z, np.arange(n) // 4)
    assert p < 1e-6
    hits2 = (rng.random(n) < 0.1).astype(float)
    W2, p2, _ = dq_test(hits2, 0.1, Z, np.arange(n) // 4)
    assert p2 > 0.001


def test_coverage_and_bootstrap():
    rng = np.random.default_rng(6)
    n = 5000
    y = rng.normal(size=n)
    q = np.tile(stats.norm.ppf(TAUS), (n, 1))
    m = run_metrics(q, y)
    assert abs(m.cov90.mean() - 0.9) < 0.02
    assert abs(m.cov95.mean() - 0.95) < 0.02
    task = np.arange(n) // 5
    r = boot_task_metric(m.qcrps.values, task, B=200, seed=1)
    assert r["lo"] <= r["est"] <= r["hi"]


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])

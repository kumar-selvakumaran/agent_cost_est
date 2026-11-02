"""No look-ahead in prefix features (extension §8 item 5; base §6.3)."""
import os

import numpy as np
import pandas as pd
import pytest

from src.features.prefix import prefix_features, features_at

TURNS = "data_cache/turns.parquet"


def _synthetic():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(30):
        K = int(rng.integers(1, 40))
        I = np.cumsum(rng.integers(100, 3000, K))
        for k in range(1, K + 1):
            rows.append(dict(run_id=r, k=k, I=int(I[k - 1]), O=int(rng.integers(10, 900)), obs=int(rng.integers(0, 4000)),
                             cum_cost=float(I[k - 1] * 2 + k), err=int(rng.random() < 0.3), rep=int(rng.random() < 0.2),
                             empty=int(rng.random() < 0.1), tc=int(rng.random() < 0.1), pfail=int(rng.random() < 0.05)))
    return pd.DataFrame(rows)


def _perturb_after(t, kcut, rng):
    t = t.copy()
    m = t.k > t.run_id.map(kcut)
    for c in ["I", "O", "obs"]:
        t.loc[m, c] = (t.loc[m, c] * rng.uniform(0.1, 10, m.sum())).astype(int) + 1
    t.loc[m, "cum_cost"] = t.loc[m, "cum_cost"] * 3.7
    for c in ["err", "rep", "empty", "tc", "pfail"]:
        t.loc[m, c] = 1 - t.loc[m, c]
    return t


def _check(t, seed):
    rng = np.random.default_rng(seed)
    K = t.groupby("run_id").k.max()
    kcut = {r: int(rng.integers(1, Kr + 1)) for r, Kr in K.items()}
    a = prefix_features(t)
    b = prefix_features(_perturb_after(t, kcut, rng))
    keep = a.k <= a.run_id.map(kcut)
    pd.testing.assert_frame_equal(a[keep].reset_index(drop=True), b[keep].reset_index(drop=True))
    fa = features_at(t, kcut)
    fb = features_at(_perturb_after(t, kcut, rng), kcut)
    pd.testing.assert_frame_equal(fa, fb)
    # features_at on the truncated history equals the full-history row at k
    full = a[a.k == a.run_id.map(kcut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(fa, full)


def test_no_lookahead_synthetic():
    _check(_synthetic(), 1)


@pytest.mark.skipif(not os.path.exists(TURNS), reason="turns cache not built")
def test_no_lookahead_real_runs():
    t = pd.read_parquet(TURNS, columns=["run_id", "k", "I", "O", "obs", "cum_cost", "err", "rep", "empty", "tc", "pfail"])
    rng = np.random.default_rng(5)
    ids = rng.choice(t.run_id.unique(), 50, replace=False)
    _check(t[t.run_id.isin(ids)].reset_index(drop=True), 2)

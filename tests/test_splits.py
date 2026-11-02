"""Leakage tests (spec §11 item 7, §A.6 item 7). Skipped if caches are not built."""
import os

import numpy as np
import pandas as pd
import pytest

SPLITS = "data_cache/splits.parquet"
RUNS = "data_cache/runs.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(SPLITS), reason="caches not built")


@pytest.fixture(scope="module")
def sp():
    s = pd.read_parquet(SPLITS)
    r = pd.read_parquet(RUNS, columns=["run_id", "task_name", "cell_id"])
    return s.merge(r, on="run_id")


def test_no_task_overlap_between_train_and_test(sp):
    for f in sorted(sp.fold.unique()):
        te, tr = sp[sp.fold == f], sp[sp.fold != f]
        assert not set(te.task_id) & set(tr.task_id)
        assert not set(te.split_group) & set(tr.split_group)
        assert not set(te.task_name) & set(tr.task_name)


def test_inner_roles_disjoint_by_group(sp):
    for f in sorted(sp.fold.unique()):
        role = sp[f"role_{f}"]
        assert (role[sp.fold == f] == "test").all()
        assert set(role[sp.fold != f]) <= {"fit", "val", "cal"}
        groups = {r: set(sp.split_group[role == r]) for r in ["fit", "val", "cal", "test"]}
        for a in groups:
            for b in groups:
                if a < b:
                    assert not groups[a] & groups[b], (f, a, b)


def test_warm_halves(sp):
    w = sp[sp.warm_half != ""]
    g = w.groupby("cell_id").warm_half.agg(lambda s: ((s == "H").sum(), (s == "T").sum()))
    for h, t in g:
        assert h >= t >= 3 and h - t <= 1


def test_every_run_has_one_test_fold(sp):
    assert sp.fold.between(0, 4).all()

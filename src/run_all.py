"""Run every step after data caching, in spec §13 order.

    python -m src.run_all --config config.yaml            # all experiments + report
    python -m src.run_all --config config.yaml --only e3  # one step

Data caching (download, Pass 1 scan, selection, transcript loading, cache build) is done once with:
    python -m src.data.audit; python -m src.data.select; python -m src.data.load;
    python -m src.experiments.e0_audit (needs runs_raw); python -m src.data.build
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

STEPS = [
    ("tests", None),
    ("e0", "src.experiments.e0_audit"),
    ("e1", "src.experiments.e1_stylized"),
    ("e2", "src.experiments.e2_variance"),
    ("e3", "src.experiments.e3_cold"),
    ("e3_eval", "src.experiments.e3_eval"),
    ("e3_sens", "src.experiments.e3_sensitivity"),
    ("e4", "src.experiments.e4_warm"),
    ("e5", "src.experiments.e5_online"),
    ("e7", "src.experiments.e7_shift"),
    ("e8", "src.experiments.e8_replay"),
    ("e6", "src.experiments.e6_llm"),
    ("report", "src.report.make_report"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", default=None, help="comma-separated step names")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    times = {}
    if os.path.exists("results/tables/run_times.json"):
        times = json.load(open("results/tables/run_times.json"))
    for name, mod in STEPS:
        if only and name not in only:
            continue
        t = time.time()
        print(f"=== {name}", flush=True)
        if name == "tests":
            import pytest
            if pytest.main(["-q", "tests"]) != 0:
                sys.exit("unit tests failed")
        else:
            sys.argv = [mod]
            importlib.import_module(mod).main()
        times[name] = round(time.time() - t, 1)
        os.makedirs("results/tables", exist_ok=True)
        json.dump(times, open("results/tables/run_times.json", "w"), indent=1)


if __name__ == "__main__":
    main()

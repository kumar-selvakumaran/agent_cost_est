"""Shared experiment helpers: data loading, plotting style, formatting."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_config

# Okabe-Ito colorblind-safe palette; fixed method -> color mapping across all figures
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#999999"]
METHOD_COLORS = {
    "P0": "#999999", "P1": "#000000", "P2": "#0072B2", "P3": "#56B4E9", "P3-EVT": "#009E73",
    "P4": "#D55E00", "P4-K1": "#E69F00", "P4-K2": "#CC79A7", "P5": "#F0E442", "P6": "#8C564B",
    "CQR(P3-EVT)": "#117733", "CQR(best)": "#882255", "O1": "#D55E00", "O2": "#0072B2", "O2-noregime": "#56B4E9",
    "oracle": "#000000",
}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 9})


def color(m):
    return METHOD_COLORS.get(m, OKABE[hash(m) % len(OKABE)])


def load_data(cfg=None):
    cfg = cfg or load_config()
    cache = cfg["data"]["cache_dir"]
    runs = pd.read_parquet(os.path.join(cache, "runs.parquet"))
    splits = pd.read_parquet(os.path.join(cache, "splits.parquet"))
    return cfg, runs, splits


def sig3(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if x == 0:
        return "0"
    return f"{x:.3g}"


def ci(est, lo, hi, f=sig3):
    return f"{f(est)} ({f(lo)}, {f(hi)})"


def write_log(name: str, lines: list[str]):
    os.makedirs("results/logs", exist_ok=True)
    with open(f"results/logs/{name}.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")

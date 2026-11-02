"""Consistency check (spec §A.6 item 6): every number printed in REPORT.md must match a value in a CSV table
under one of the report's formatting rules (3 significant figures, 3 decimals, percent with 0–2 decimals,
thousands separators), or be a configuration constant / small integer label (k, τ, counts of methods...)."""
from __future__ import annotations

import glob
import re
import sys

import numpy as np
import pandas as pd

from src.utils import load_config

NUM = re.compile(r"(?<![\w.#/-])-?\d[\d,]*(?:\.\d+)?%?(?![\w/])")


def csv_values() -> set[str]:
    vals = set()
    for f in glob.glob("results/tables/*.csv"):
        df = pd.read_csv(f)
        for c in df.columns:
            for v in pd.to_numeric(df[c], errors="coerce").dropna().values:
                vals |= renderings(float(v))
            if df[c].dtype == object:  # numbers embedded in strings (e.g. seed lists)
                for s in df[c].dropna().astype(str):
                    for t in re.findall(r"-?\d+\.\d+", s):
                        vals |= renderings(float(t))
    return vals


def renderings(v: float) -> set[str]:
    out = set()
    for fmt in ("{:.3g}", "{:.3f}", "{:.4f}", "{:.2f}", "{:.1f}", "{:.0f}", "{:,.0f}", "{:.2g}", "{:.4g}"):
        out.add(fmt.format(v))
    for nd in (0, 1, 2):
        out.add(f"{100 * v:.{nd}f}%")
    if float(v).is_integer():
        out.add(f"{int(v):,}")
        out.add(str(int(v)))
    return {s.replace("-0.000", "0.000") for s in out} | out


def main():
    cfg = load_config()
    vals = csv_values()
    txt = open("results/REPORT.md").read()
    # drop image links and file paths
    txt = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", txt)
    txt = re.sub(r"`[^`]*`", "", txt)
    allowed_small = {str(i) for i in range(0, 101)}
    consts = {"1,000", "1000", "5,000", "2,000", "200", "500", "99", "0.1", "0.5", "0.8", "0.9", "0.95", "0.99",
              "0.025", "0.975", "0.01", "100k", str(cfg["seed"]), "10,000", "16", "64", "15", "1.5", "1e-3"}
    bad = []
    for m in NUM.finditer(txt):
        tok = m.group(0).rstrip(",")
        if tok in allowed_small or tok in consts or tok in vals:
            continue
        plain = tok.replace(",", "")
        if plain in vals or plain in allowed_small:
            continue
        ctx = txt[max(0, m.start() - 40): m.end() + 20].replace("\n", " ")
        bad.append((tok, ctx))
    print(f"numbers checked: {len(NUM.findall(txt))}; unmatched: {len(bad)}")
    for tok, ctx in bad[:60]:
        print(f"  {tok!r}: …{ctx}…")
    with open("results/logs/report_check.md", "w") as f:
        f.write(f"# REPORT.md consistency check\n\nNumbers checked: {len(NUM.findall(txt))}; unmatched: {len(bad)}\n\n")
        for tok, ctx in bad:
            f.write(f"- `{tok}`: …{ctx}…\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

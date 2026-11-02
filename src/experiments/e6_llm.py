"""E6 — LLM self-report predictor (optional; spec §7.9, §10 E6). Off by default (config llm_predictor.enabled)."""
from __future__ import annotations

import os

from src.experiments.common import load_data, write_log


def main():
    cfg, _, _ = load_data()
    enabled = cfg["llm_predictor"]["enabled"]
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    reason = []
    if not enabled:
        reason.append("`llm_predictor.enabled` is false (default; the user did not enable it)")
    if not key:
        reason.append("no API key is configured in the environment")
    if reason:
        write_log("e6", ["# E6 — Predictor capacity (skipped)\n",
                         "E6 was not run: " + "; ".join(reason) + ". H5 is therefore recorded as *skipped*.\n"])
        print("E6 skipped:", "; ".join(reason))
        return
    raise NotImplementedError("E6 enabled but not implemented in this run")


if __name__ == "__main__":
    main()

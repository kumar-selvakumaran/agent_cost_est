# AgentTrove token-cost prediction benchmark

Benchmarks methods that forecast the **probability distribution** of an LLM agent run's token cost, using the
terminus-2 trajectories in [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove).
Implements `agenttrove_cost_benchmark_spec.md` v1.0. Results: [`results/REPORT.md`](results/REPORT.md);
departures from the spec: [`results/DEVIATIONS.md`](results/DEVIATIONS.md).

## Layout
```
config.yaml            every constant (spec §14); deviations marked DEV
src/data/              audit.py (Pass 1 scan), ids.py, select.py (Pass 2), parse.py, load.py, build.py
src/cost/              tokenizers.py (o200k_base), reconstruct.py (turn accounting, reference cost)
src/features/          prerun.py, text.py (TF-IDF/SVD), history.py (S2), prefix.py (S3, causal)
src/predictors/        P0–P6, P3-EVT, P4 MDN, P5 structural, CQR, ACI, O1, O2
src/eval/              metrics.py (QCRPS, PIT, coverage), backtests.py (Kupiec, DQ), bootstrap.py, decomposition.py
src/experiments/       e0_audit … e8_replay, e3_eval, e3_sensitivity
src/report/            make_report.py, check_report.py (number consistency)
tests/                 unit tests (cost reconstruction, metrics, leakage, no look-ahead)
```

## Reproduce
```bash
pip install -r requirements.txt
# one-time data caching (downloads 19.5 GB of Parquet shards to data.raw_dir)
python -m src.data.audit && python -m src.data.select && python -m src.data.load
python -c "from src.data.build import concat_parts; concat_parts('data_cache')"
python -m src.experiments.e0_audit && python -m src.data.build
# every experiment, table, figure and the report
python -m src.run_all --config config.yaml
```

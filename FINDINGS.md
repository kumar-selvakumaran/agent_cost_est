# Findings — hypotheses and verdicts

Error = CRPS of the forecast of log cost (scores the whole predicted distribution; lower is better).
CI = 95% task-cluster bootstrap interval. Full details: `results/REPORT.md`; departures from the spec: `results/DEVIATIONS.md`.
Sample: 98,501 terminus-2 runs from AgentTrove (20,848 tasks, 10 models, 21 sources).

| # | Hypothesis (plain) | Verdict | Evidence |
|---|---|---|---|
| H1 | How much a task's cost varies run to run depends more on the task than on the model. | **Not supported** | The model explains 39% of the variation in run-to-run spread, the task 3% (difference −36 pts, CI −42 to −32). About half (49%) of all cost variation is noise between repeated runs of the same task and model. |
| H2 | Before a run, the task text improves the forecast beyond knowing the benchmark and model. | **Supported** | Best text model (P5) is 14.8% more accurate than the benchmark × model baseline (CI 14.3–15.4%). |
| H3 | Forecasts that don't force a bell shape on log cost are better calibrated. | **Inconclusive** | Calibration distance (PIT KS, lower = better): bell-shaped P2 0.016; P5 0.010 (better); P4 0.023 and P3-EVT 0.033 (worse). |
| H4 | Forecast error falls steadily with each turn observed. | **Inconclusive** | Falls at every checkpoint on a fixed set of long runs; on all still-running runs it falls until turn 8, then rises (remaining runs are the hard ones). After 4 turns, error is 73% of pre-run (CI 72–73%); halves only by turn 32. |
| H5 | A small LLM forecasts as well as a large one. | **Not tested** | E6 off by default; no API key. |
| H6 | Past runs of the same task help more than the task text. | **Supported** | History: 25% better than baseline; text: 15%. Difference +10.3 pts (CI 8.4–12.1). |
| H7 | Updating during the run beats a static forecast that only knows the cost already spent. | **Supported** | Whole-run error (ITS) 0.203 vs 0.297 (difference −0.094, CI −0.096 to −0.091). |
| H8 | Detecting loops/errors helps, more so early in the run. | **Supported (small)** | 0.203 vs 0.211; gain 5.0% in the first 8 calls vs 3.8% overall (difference +1.2 pts, CI 0.9–1.3). |
| H9 | The learned online model beats mechanical per-turn extrapolation. | **Supported (small)** | 0.203 vs 0.210 (difference −0.007, CI −0.009 to −0.005). The mechanical model is still better at estimating remaining cost late in a run. |

## Other key results
- **Calibration:** conformal correction (CQR) brings 90% intervals to 88–89% coverage. Under a shift to unseen sources, static intervals fall to 70% coverage; adaptive conformal (ACI) keeps 89%.
- **Budget caps** from the forecast do not beat the simple baseline at matched loss of completed runs (6.5% vs 7.4% cost saved; success measured with an exploratory proxy because the dataset's success label is degenerate).
- **Exploratory — model vs lookup tables (after 4 turns):** median miss on final cost is 18% for the online model vs 29% for a source × model lookup table and 43% for a single constant. The advantage narrows for very long runs (turn 32: 25% vs 27–31%).

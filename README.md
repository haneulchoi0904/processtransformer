# STLFormer (ProcessTransformer) replication

Part of a replication of Wang et al., "MTLFormer: Multi-task learning guided transformer network for business process prediction", IEEE Access 11 (2023) (course 2IMI00, TU/e). This repository reproduces the **STLFormer baseline, Table 3 of the paper: next-activity accuracy, next-event time MAE and remaining-time MAE** on Helpdesk and BPIC2012 A, O, W. STLFormer is the single-task ProcessTransformer of Bukhsh et al. (one model per task).

This is a patched copy of ProcessTransformer (Apache 2.0, base commit `3f041e4`, cloned 2026-10-19) with the changes listed below. 

## Environment
Tested on Windows, **CPU only** (AMD Ryzen 7 PRO 8845HS, 31 GB RAM): Python 3.10.2, TensorFlow 2.21.0, Keras 3.12.4, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 (see `requirements.txt`).

## Data
`data/<name>/<name>.csv` for helpdesk, BPIC2012_A, BPIC2012_O, BPIC2012_W: the CSV logs shipped with the MTLFormer repository, with columns `Case ID, Activity, Complete Timestamp`. The shipped BPIC2012 files differ from the public XES logs. See `data/DATA.md`. If you only have the full BPIC2012 log, `split_bpic2012.py` splits it by the initial letter of the activity name.

## Setup
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run
Run from the repository root. Preprocessing is done once per log and task; run `r` writes to `results/seeded_run<r>`. Finished runs are skipped, so a command can be stopped and restarted.
```powershell
python run_stlformer.py --repo . --logs_dir data --datasets helpdesk BPIC2012_A BPIC2012_O BPIC2012_W --epochs 100 --runs 3 --gpu 0 --seed 42
```
Run `r` uses seed `42 + r` (43, 44, 45). Each task script is launched through `run_seeded.py`, which seeds Python, NumPy and TensorFlow and turns on TensorFlow's deterministic ops. Quick check on one log (one run): `--datasets helpdesk --runs 1`.

Settings: 100 epochs, batch size 12, learning rate 0.001. These are the repository defaults; the paper reports no STLFormer configuration (it reports batch 64 and learning rate 0.002 for MTLFormer).

## Output
`results/seeded_run<r>/<dataset>/results_{next_activity,next_time,remaining_time}.csv`: one row per prefix length k, and as the last row the original script's own summary. Summarise over runs (drops that last row and recomputes every average from the per-k rows, equal weight per k; mean and sample SD over runs):
```powershell
python summarize_results.py . "seeded_run*"
```
This writes `summary_seeded_run.csv`.

## Expected results
Mean ± sample SD over 3 runs, mean over prefix lengths k:

| | Helpdesk | BPIC2012_A | BPIC2012_O | BPIC2012_W |
|---|---|---|---|---|
| Accuracy (%) | 82.66 ± 0.96 | 74.81 ± 0.00 | 86.79 ± 0.34 | 89.89 ± 0.01 |
| Next-event time MAE (days) | 2.98 ± 0.23 | 1.04 ± 0.04 | 1.14 ± 0.12 | 0.41 ± 0.01 |
| Remaining time MAE (days) | 3.63 ± 0.12 | 4.94 ± 0.22 | 5.24 ± 0.33 | 4.78 ± 0.36 |

Paper (Table 3, STLFormer): accuracy 86 / 77 / 87 / 90; next-event time 3.14 / 1.12 / 1.13 / 0.40; remaining time 4.04 / 4.72 / 5.28 / 5.03. Numbers can differ in the last digits on other hardware or library versions. A full run takes roughly 20 hours per seed (all logs and tasks, CPU), mostly BPIC2012_W.

## Changes to the original code
1. `processtransformer/data/processor.py`: timestamps parsed with `dayfirst=False, format="mixed"` (the logs are year/month/day).
2. `processtransformer/data/processor.py`: the `np.array_split` + `multiprocessing.Pool` preprocessing is replaced by a direct call of the helper functions (it breaks with the installed pandas/numpy).
3. `processtransformer/models/transformer.py`: `call(self, inputs, training=None)` (Keras 3).
4. `next_activity.py`, `next_time.py`, `remaining_time.py`: checkpoint names end in `.weights.h5` (Keras 3).
5. Added `run_stlformer.py` (copies the log, preprocesses, runs the three tasks, seeded runs), `split_bpic2012.py`, `summarize_results.py`. The original `next_activity.py` summary row computes accuracy as `np.mean(accuracy)` (the last prefix) instead of `np.mean(accuracies)`; `summarize_results.py` therefore drops that row and recomputes all averages.

## Known issues
- BPIC2012_A has zero SD over the three runs.
- Recall equals accuracy (class-weighted averaging).
- Seeded runs on other hardware or library versions may differ in the last digits.

## Licence
Apache 2.0, see `LICENSE.txt` and `THIRD_PARTY.md`. Cite Bukhsh, Saeed, Dijkman, "ProcessTransformer: Predictive business process monitoring with transformer network", arXiv:2104.00721 (2021).

"""Average the per-prefix results over prefix lengths k, then mean/std over runs.

Usage: python summarize_results.py /path/to/processtransformer-main [run_folder_pattern]
  default pattern "run*" (unseeded runs) -> summary_over_runs.csv
  e.g. "seeded_run*"               -> summary_seeded_run.csv

The last row of every results_*.csv is the script's own summary row. For next_activity its
accuracy is wrong (np.mean(accuracy) instead of np.mean(accuracies)), so this script drops that
row and recomputes every average itself from the per-k rows (each k weighted equally, like the repo).
"""
import sys, glob, pathlib
import pandas as pd

root = pathlib.Path(sys.argv[1])
pattern = sys.argv[2] if len(sys.argv) > 2 else "run*"
outname = "summary_over_runs.csv" if pattern == "run*" else "summary_" + pattern.replace("*", "") + ".csv"
cols = {"next_activity": ["accuracy", "fscore", "precision", "recall"],
        "next_time": ["mean_absolute_error"], "remaining_time": ["mean_absolute_error"]}
rows = []
for f in sorted(glob.glob(str(root / "results" / pattern / "*" / "results_*.csv"))):
    p = pathlib.Path(f)
    run, dataset, task = p.parts[-3], p.parts[-2], p.stem.replace("results_", "")
    df = pd.read_csv(f).iloc[:-1]            # drop the script's own summary row
    for c in cols[task]:
        rows.append((dataset, task, c, run, df[c].mean(), len(df)))
res = pd.DataFrame(rows, columns=["dataset", "task", "metric", "run", "value", "n_prefix_lengths"])
out = res.groupby(["dataset", "task", "metric"]).agg(mean=("value", "mean"), std=("value", "std"), runs=("run", "nunique")).reset_index()
print(out.to_string(index=False))
out.to_csv(root / outname, index=False)
print("written:", root / outname)

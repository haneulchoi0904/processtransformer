"""Run STLFormer (ProcessTransformer) end to end on one or more event logs.

Example:
  python run_stlformer.py --repo ./processtransformer-main \
      --logs_dir "./MTLFormer-main/code/MULTI TASK/datasets" \
      --datasets helpdesk BPIC2012_A BPIC2012_O BPIC2012_W \
      --epochs 100 --runs 3 --gpu 0

Expects <logs_dir>/<name>/<name>.csv with columns: Case ID, Activity, Complete Timestamp.
Preprocessing is done once per dataset and task; every run gets its own models/ and results/ folder.

Seeded, reproducible runs (separate folders models/seeded_run<r> and results/seeded_run<r>, so earlier
unseeded runs are never touched or skipped):
  python run_stlformer.py --repo . --logs_dir data --datasets helpdesk --epochs 100 --runs 3 --gpu 0 --seed 42
Run r uses seed (42 + r). Every task script is launched through run_seeded.py, which seeds Python, NumPy and
TensorFlow and turns on TensorFlow's deterministic ops before the original, unmodified script runs.
"""
import argparse, os, pathlib, shutil, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--logs_dir", required=True)
ap.add_argument("--datasets", nargs="+", default=["helpdesk", "BPIC2012_A", "BPIC2012_O", "BPIC2012_W"])
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--batch_size", type=int, default=12)        # repo default, NOT necessarily what the paper used
ap.add_argument("--learning_rate", type=float, default=0.001)  # repo default
ap.add_argument("--seed", type=int, default=None,
                help="base seed; run r uses seed+r. Enables deterministic mode and writes to seeded_run<r> folders")
args = ap.parse_args()

repo = pathlib.Path(args.repo).resolve()
logs = pathlib.Path(args.logs_dir).resolve()
tasks = ["next_activity", "next_time", "remaining_time"]

NEED = ["Case ID", "Activity", "Complete Timestamp"]
XES = {"case:concept:name": "Case ID", "concept:name": "Activity", "time:timestamp": "Complete Timestamp"}

def find_log(name):
    """Accept <logs>/<name>/<name>.csv, <logs>/<name>.csv, or any <name>.csv below logs (case-insensitive)."""
    for c in (logs / name / f"{name}.csv", logs / f"{name}.csv", repo / "datasets" / name / f"{name}.csv"):
        if c.exists():
            return c
    for q in logs.rglob("*.csv"):
        if q.stem.lower() == name.lower():
            return q
    return None

found = {n: find_log(n) for n in args.datasets}
missing = [n for n, f in found.items() if f is None]
if missing:
    sys.exit("Cannot find an event log named <name>.csv for: " + ", ".join(missing) +
             f"\nSearched under: {logs}\n"
             "Files should be called helpdesk.csv, BPIC2012_A.csv, BPIC2012_O.csv, BPIC2012_W.csv (any folder layout, any capitalisation).\n"
             "If you only have the full BPIC2012 log, split it first with split_bpic2012.py.\n"
             "Tip (PowerShell): Get-ChildItem -Path $HOME\\Downloads -Recurse -Filter *.csv")

def install_log(src, dst):
    """Copy the log into <repo>/datasets/<name>/ and make sure it has the three column names the code expects."""
    import pandas as pd
    cols = list(pd.read_csv(src, nrows=0).columns)
    if all(c in cols for c in NEED):
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
    elif all(c in cols for c in XES):
        pd.read_csv(src).rename(columns=XES)[NEED].to_csv(dst, index=False)
        print(f"   renamed XES column names in {src.name}")
    else:
        sys.exit(f"{src} has columns {cols}\nExpected {NEED} (or the XES names {list(XES)}). "
                 "If the file uses ';' as separator, convert it to a comma separated CSV first.")

SEEDED_WRAPPER = '''"""Launches an unmodified task script with all random number generators seeded (seed in env STL_SEED)."""
import os, sys, random, runpy
seed = int(os.environ["STL_SEED"])
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
try:
    import keras
    keras.utils.set_random_seed(seed)          # seeds python, numpy and the backend together
except Exception:
    pass
tf.config.experimental.enable_op_determinism()
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
'''

def run(cmd, seed=None):
    print(">>", " ".join(cmd), flush=True)
    t = time.time()
    env = None
    if seed is not None:
        env = dict(os.environ, STL_SEED=str(seed), PYTHONHASHSEED=str(seed),
                   TF_DETERMINISTIC_OPS="1", TF_ENABLE_ONEDNN_OPTS="0")
    subprocess.run(cmd, cwd=repo, check=True, env=env)
    print(f"   ({time.time() - t:.0f}s)", flush=True)

if args.seed is not None:
    (repo / "run_seeded.py").write_text(SEEDED_WRAPPER, encoding="utf-8")
    print(f"seeded mode: base seed {args.seed}, run r uses {args.seed}+r, deterministic TensorFlow ops")

for name in args.datasets:
    dst = repo / "datasets" / name
    dst.mkdir(parents=True, exist_ok=True)
    csv = dst / f"{name}.csv"
    if not csv.exists():
        install_log(found[name], csv)
    processed = dst / "processed"
    for task in tasks:   # preprocessing only once per dataset+task
        if not (processed / f"{task}_train.csv").exists():
            run([sys.executable, "data_processing.py", "--dataset", name,
                 "--raw_log_file", f"./datasets/{name}/{name}.csv", "--task", task])
    for r in range(1, args.runs + 1):
        tag = f"seeded_run{r}" if args.seed is not None else f"run{r}"
        seed = None if args.seed is None else args.seed + r
        for task in tasks:
            if (repo / "results" / tag / name / f"results_{task}.csv").exists():
                print(f"skip {name} {tag} {task} (already done)")
                continue
            launcher = [sys.executable] + (["run_seeded.py"] if seed is not None else []) + [f"{task}.py"]
            run(launcher + ["--dataset", name,
                 "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
                 "--learning_rate", str(args.learning_rate), "--gpu", str(args.gpu),
                 "--model_dir", f"./models/{tag}", "--result_dir", f"./results/{tag}"], seed=seed)
print("all done")

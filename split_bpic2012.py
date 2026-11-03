"""OPTIONAL. The A/O/W logs already exist in the MTLFormer repo, you only need this if you start from the full BPIC2012 log.

Splits one CSV (columns Case ID, Activity, Complete Timestamp) into BPIC2012_A / _O / _W by activity prefix (A_, O_, W_).
Usage: python split_bpic2012.py full_bpic2012.csv out_dir
Keeps row order and all events, nothing is dropped. If your CSV uses the XES names
(case:concept:name, concept:name, time:timestamp) they are renamed first.
"""
import sys, pathlib
import pandas as pd

src, out = sys.argv[1], pathlib.Path(sys.argv[2])
df = pd.read_csv(src).rename(columns={"case:concept:name": "Case ID", "concept:name": "Activity", "time:timestamp": "Complete Timestamp"})
for letter in "AOW":
    sub = df[df["Activity"].astype(str).str.startswith(letter + "_")]
    d = out / f"BPIC2012_{letter}"
    d.mkdir(parents=True, exist_ok=True)
    sub[["Case ID", "Activity", "Complete Timestamp"]].to_csv(d / f"BPIC2012_{letter}.csv", index=False)
    print(letter, len(sub), "events,", sub["Case ID"].nunique(), "cases")
print("events not in A/O/W:", len(df) - sum(df["Activity"].astype(str).str.startswith(l + "_").sum() for l in "AOW"))

"""Pilot: 33 reports per source through Qwen3.5-27B on the Ollama servers; speed + samples."""
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/jovyan/work/cis634/scripts")
spec = importlib.util.spec_from_file_location("sr", "/home/jovyan/work/cis634/scripts/summarise_reports.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
import pandas as pd

hosts = sys.argv[1].split(",")
r = pd.read_parquet("/home/jovyan/work/cis634/data/processed_v2/reports.parquet")
groups = {"ASRS": r[r.acn.str.startswith("ASRS:")], "NTSB": r[r.acn.str.startswith("NTSB:")],
          "ASRSN": r[r.acn.str.startswith("ASRSN:")]}
sample = pd.concat([g.sample(33, random_state=0) for g in groups.values()])
t0 = time.time()
with ThreadPoolExecutor(len(hosts) * 8) as pool:
    outs = list(pool.map(lambda kv: sr.llamacpp_summary(hosts[kv[0] % len(hosts)], "qwen3.5-27b-q6k", kv[1], 96),
                         enumerate(sample["narrative"])))
dt = time.time() - t0
sample = sample.assign(summary=[" ".join(o.split()) for o in outs])
words = sample["summary"].str.split().str.len()
print(f"{len(sample)} summaries in {dt:.0f}s = {len(sample) / dt:.2f} reports/s on {len(hosts)} servers; "
      f"words median {words.median():.0f} max {words.max()}; empty {int((words == 0).sum())}; "
      f"think tags {int(sample.summary.str.contains('<think>').sum())}")
sample[["acn", "narrative", "synopsis", "summary"]].to_json("/home/jovyan/work/scratch/pilot_27b.jsonl",
                                                            orient="records", lines=True)

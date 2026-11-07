"""Throughput of one llama-server: N reports with C requests in flight."""
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

spec = importlib.util.spec_from_file_location("sr", "/home/jovyan/work/cis634/scripts/summarise_reports.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
host, conc, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = pd.read_parquet("/home/jovyan/work/cis634/data/processed_v2/reports.parquet").sample(n, random_state=1)
t0 = time.time()
with ThreadPoolExecutor(conc) as pool:
    list(pool.map(lambda x: sr.llamacpp_summary(host, "q", x, 96), r["narrative"]))
print(f"{host} concurrency {conc}: {n / (time.time() - t0):.2f} reports/s")

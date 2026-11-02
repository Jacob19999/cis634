"""Blind relevance judging: pooling, label import and inter-annotator agreement.

Workflow
--------
1. ``asrsbench pool``: merge the top-``depth`` reports of every system for each topical
   query, shuffle, hide system names and write one CSV per annotator. A shared subset of
   queries (``--overlap``) goes to both annotators for agreement.
2. Annotators fill the ``grade`` column with 2 / 1 / 0 (see GRADING_GUIDE).
3. ``asrsbench labels``: report Cohen's kappa on the overlap and write TREC qrels.
   Disagreements are resolved conservatively (lower grade) unless an adjudicated
   ``grade_final`` column is filled in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .benchmark import read_run
from .data import ChunkConfig, load_reports
from .tracks import write_qrels_tsv

GRADING_GUIDE = (
    "2 = same operational issue as the query (same event and same cause/mechanism); "
    "1 = related (same kind of event, different cause, or the issue is secondary); "
    "0 = only shares an airport, aircraft type or keyword."
)


def build_pool(run_dir: str | Path, queries_path: str | Path, out_dir: str | Path,
               depth: int = 10, overlap: float = 0.25, annotators: tuple[str, str] = ("A", "B"),
               data_dir: str = "data/processed", seed: int = 0) -> dict[str, Path]:
    """Write one blind judging CSV per annotator from a run's topical top-``depth`` results."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.concat([read_run(p) for p in sorted((run_dir / "runs" / "topical").glob("*.tsv"))],
                     ignore_index=True)
    runs = runs[runs["rank"] <= depth]
    queries = pd.DataFrame(yaml.safe_load(Path(queries_path).read_text())["queries"])
    queries = queries[["qid", "text"]].rename(columns={"text": "query"})

    # One row per (query, report); keep the passage from the best-ranked occurrence.
    pool = (runs.sort_values("rank").drop_duplicates(["qid", "acn"])
            .loc[:, ["qid", "acn", "unit_id"]])

    reports = load_reports(data_dir).set_index("acn")
    # Passages come from the chunking the run was made with.
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    chunk_tag = ChunkConfig(**run_cfg.get("chunking", {})).tag
    passages = pd.read_parquet(Path(data_dir) / f"chunks_{chunk_tag}.parquet").set_index("chunk_id")["text"]
    pool["matched_passage"] = pool["unit_id"].map(passages).fillna(
        pool["acn"].map(reports["narrative"]).str.slice(0, 1200))
    pool["narrative"] = pool["acn"].map(reports["narrative"])
    pool = pool.merge(queries, on="qid").drop(columns=["unit_id"])

    rng = np.random.default_rng(seed)
    pool = pool.iloc[rng.permutation(len(pool))]  # hide system order
    pool = pool.sort_values("qid", kind="stable")
    pool.insert(0, "item_id", [f"J{i:05d}" for i in range(len(pool))])
    pool["grade"] = ""
    pool["notes"] = ""
    pool = pool[["item_id", "qid", "query", "acn", "grade", "notes", "matched_passage", "narrative"]]

    # Some queries go to both annotators (for agreement); the rest alternate between them.
    qids = sorted(pool["qid"].unique())
    shared = {str(q) for q in rng.choice(qids, size=max(1, int(round(overlap * len(qids)))), replace=False)}
    half = [q for q in qids if q not in shared]
    a_only, b_only = set(half[::2]), set(half[1::2])
    paths = {}
    for name, own in zip(annotators, (a_only, b_only)):
        sel = pool[pool["qid"].isin(own | shared)]
        path = out_dir / f"pool_{name}.csv"
        sel.to_csv(path, index=False)
        paths[name] = path
    (out_dir / "README.txt").write_text(
        f"Grade each row's report against its query. {GRADING_GUIDE}\n"
        f"Queries judged by both annotators (agreement): {sorted(shared)}\n")
    return paths


def import_labels(paths: list[str | Path], out_path: str | Path) -> dict:
    """Read graded pools, measure agreement (Cohen's kappa) and write the final qrels."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"acn": str})
        df["annotator"] = Path(p).stem.replace("pool_", "")
        frames.append(df[df["grade"].notna() & (df["grade"].astype(str).str.strip() != "")])
    labels = pd.concat(frames, ignore_index=True)
    labels["grade"] = labels["grade"].astype(float).astype(int)

    stats: dict = {"n_labels": int(len(labels))}
    # One row per (query, report), one column per annotator.
    wide = labels.pivot_table(index=["qid", "acn"], columns="annotator", values="grade", aggfunc="first")
    if wide.shape[1] >= 2:
        a, b = wide.columns[:2]
        both = wide[[a, b]].dropna()
        if len(both):
            from sklearn.metrics import cohen_kappa_score

            stats["n_double_labeled"] = int(len(both))
            stats["raw_agreement"] = float((both[a] == both[b]).mean())
            stats["cohen_kappa"] = float(cohen_kappa_score(both[a], both[b]))
            stats["weighted_kappa_linear"] = float(cohen_kappa_score(both[a], both[b], weights="linear"))
    final = wide.min(axis=1)  # conservative resolution
    if "grade_final" in labels:
        adj = labels.dropna(subset=["grade_final"]).set_index(["qid", "acn"])["grade_final"]
        final.loc[adj.index] = adj.astype(int)
    qrels: dict = {}
    for (qid, acn), g in final.dropna().items():
        qrels.setdefault(qid, {})[acn] = int(g)
    write_qrels_tsv(qrels, out_path)
    stats["n_queries"] = len(qrels)
    return stats

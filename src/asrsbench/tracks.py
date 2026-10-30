"""Evaluation tracks.

* ``synopsis``  -- known-item paraphrase retrieval. The query is NASA's analyst synopsis of a
  report, written independently of the reporter's narrative; the relevant documents are that
  report and any report of the same incident. Fully automatic and large-scale.
* ``topical``   -- analyst-style issue descriptions (``data/queries/*.yaml``) with graded
  relevance: silver labels from analyst fields and, once available, human labels.
* ``contrast``  -- the meaning-change test: does a system score a same-meaning paraphrase
  above a minimally edited, meaning-changed version of the same sentence?
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .metrics import Qrels


@dataclass
class QuerySet:
    name: str
    queries: pd.DataFrame  # columns: qid, text, (+ any metadata)
    qrels: dict[str, Qrels] = field(default_factory=dict)  # label source -> qrels
    metrics: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Synopsis (known-item) track
# --------------------------------------------------------------------------------------


def synopsis_track(reports: pd.DataFrame, split: str = "validation", n_queries: int = 1000,
                   seed: int = 13, min_words: int = 8) -> QuerySet:
    pool = reports[(reports["split"] == split) & (reports["synopsis"].str.split().str.len() >= min_words)]
    sample = pool.sample(n=min(n_queries, len(pool)), random_state=seed).sort_values("acn")
    groups = reports.groupby("group_id")["acn"].apply(list).to_dict()
    queries = pd.DataFrame({"qid": "S" + sample["acn"], "text": sample["synopsis"],
                            "acn": sample["acn"]})
    qrels = {f"S{acn}": {d: 2 for d in groups[g]} for acn, g in zip(sample["acn"], sample["group_id"])}
    return QuerySet(f"synopsis_{split}", queries.reset_index(drop=True), {"auto": qrels},
                    ["MRR@10", "R@1", "R@10", "R@100", "nDCG@10"])


# --------------------------------------------------------------------------------------
# Topical track
# --------------------------------------------------------------------------------------

RULE_FIELDS = {"anomaly", "synopsis", "human_factors", "comm_breakdown", "flight_phase", "result",
               "detector", "primary_problem", "contributing_factors", "reporter_function"}


def _clause_mask(reports: pd.DataFrame, clause: dict[str, str]) -> np.ndarray:
    mask = np.ones(len(reports), dtype=bool)
    for fld, pattern in clause.items():
        if fld not in RULE_FIELDS:
            raise ValueError(f"unknown silver-rule field {fld!r}")
        with warnings.catch_warnings():  # rules may use capture groups; we only need matches
            warnings.simplefilter("ignore", UserWarning)
            hit = reports[fld].str.contains(pattern, flags=re.IGNORECASE, regex=True)
        mask &= hit.to_numpy()
    return mask


def silver_qrels(reports: pd.DataFrame, rules: dict) -> dict[str, int]:
    """Apply one query's silver rule; returns {acn: grade} for grade >= 1."""
    grades = np.zeros(len(reports), dtype=np.int8)
    for grade in sorted(rules, key=int):  # lower grades first, higher grades overwrite
        clauses = rules[grade]
        mask = np.zeros(len(reports), dtype=bool)
        for clause in clauses:
            mask |= _clause_mask(reports, clause)
        grades[mask] = np.maximum(grades[mask], int(grade))
    acns = reports["acn"].to_numpy()
    nz = np.flatnonzero(grades)
    return dict(zip(acns[nz], grades[nz].tolist()))


def read_qrels_tsv(path: str | Path) -> Qrels:
    """TREC-style qrels: ``qid  0  doc_id  grade`` (whitespace separated)."""
    qrels: Qrels = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        qid, _, doc, grade = line.split()[:4]
        qrels.setdefault(qid, {})[doc] = int(grade)
    return qrels


def write_qrels_tsv(qrels: Qrels, path: str | Path) -> None:
    with open(path, "w") as f:
        for qid in sorted(qrels):
            for doc, g in sorted(qrels[qid].items()):
                f.write(f"{qid}\t0\t{doc}\t{g}\n")


def topical_track(reports: pd.DataFrame, path: str | Path, human_qrels: list[str] | None = None,
                  splits: list[str] | None = None) -> QuerySet:
    spec = yaml.safe_load(Path(path).read_text())
    qs = spec["queries"]
    if splits:
        qs = [q for q in qs if q.get("split") in splits]
    queries = pd.DataFrame([{k: v for k, v in q.items() if k != "silver"} for q in qs])
    qrels = {"silver": {q["qid"]: silver_qrels(reports, q["silver"]) for q in qs if "silver" in q}}
    for p in human_qrels or []:
        if Path(p).exists():
            qrels[f"human:{Path(p).stem}"] = read_qrels_tsv(p)
    return QuerySet(Path(path).stem, queries, qrels,
                    ["P@10", "P@10_lenient", "nDCG@10", "MRR@10", "judged@10"])


# --------------------------------------------------------------------------------------
# Contrast (meaning-change) track
# --------------------------------------------------------------------------------------

CONTRAST_FIELDS = ("id", "category", "original", "paraphrase", "changed")


def load_contrast(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    missing = set(CONTRAST_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"contrast set missing fields: {missing}")
    return df

"""Ranking metrics and significance tests.

Relevance grades: 2 = same issue, 1 = related, 0 = not relevant.
nDCG uses linear gains (gain = grade), as in trec_eval.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

Qrels = dict[str, dict[str, int]]  # qid -> {doc_id: grade}
Run = dict[str, list[str]]  # qid -> ranked doc ids


def precision_at_k(ranked: list[str], rel: dict[str, int], k: int, min_grade: int = 2) -> float:
    return sum(rel.get(d, 0) >= min_grade for d in ranked[:k]) / k


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int, min_grade: int = 1) -> float:
    n_rel = sum(g >= min_grade for g in rel.values())
    if n_rel == 0:
        return float("nan")
    return sum(rel.get(d, 0) >= min_grade for d in ranked[:k]) / n_rel


def reciprocal_rank(ranked: list[str], rel: dict[str, int], k: int, min_grade: int = 1) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) >= min_grade:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else float("nan")


def judged_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Share of the top k that has any label (matters for incomplete human qrels)."""
    return sum(d in rel for d in ranked[:k]) / k


METRICS = {
    "P@10": lambda r, q: precision_at_k(r, q, 10, 2),
    "P@10_lenient": lambda r, q: precision_at_k(r, q, 10, 1),
    "nDCG@10": lambda r, q: ndcg_at_k(r, q, 10),
    "MRR@10": lambda r, q: reciprocal_rank(r, q, 10, 1),
    "R@1": lambda r, q: recall_at_k(r, q, 1),
    "R@10": lambda r, q: recall_at_k(r, q, 10),
    "R@100": lambda r, q: recall_at_k(r, q, 100),
    "judged@10": lambda r, q: judged_at_k(r, q, 10),
}


def evaluate_run(run: Run, qrels: Qrels, metrics: list[str]) -> pd.DataFrame:
    """Per-query metric table (index: qid). Queries without qrels are skipped."""
    rows = {}
    for qid, ranked in run.items():
        if qid not in qrels:
            continue
        rows[qid] = {m: METRICS[m](ranked, qrels[qid]) for m in metrics}
    return pd.DataFrame.from_dict(rows, orient="index")


def bootstrap_ci(values: np.ndarray, n: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_randomization_test(a: np.ndarray, b: np.ndarray, n: int = 10_000,
                              seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) == 0 under random sign flips."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

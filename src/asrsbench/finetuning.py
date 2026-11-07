"""Deterministic, incident-safe data preparation for embedding fine-tuning.

This module has no model or network dependency. Evaluation targets are never used to
generate synthetic questions; they supply supervised queries for training reports only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .data import ChunkConfig, chunk_text

SEED = 634
WEIGHTS = {"asrs_synopsis": .35, "synthetic": .30, "ntsb_cause": .25,
           "second_narrative": .05, "faa": .05}
QUERY_PREFIX = ("Instruct: Given a web search query, retrieve relevant passages that answer "
                "the query\nQuery:")


def digest(text: str | bytes) -> str:
    """Stable SHA-256 digest for provenance and deterministic sampling."""
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def tokens(text: str) -> list[str]:
    """Case-insensitive alphanumeric words, independent of punctuation and spacing."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def bucket(key: str, modulo: int = 1000) -> int:
    """Stable bucket that does not depend on Python's randomised hash seed."""
    return int(digest(f"{SEED}:{key}")[:16], 16) % modulo


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Atomically write UTF-8 JSON lines to a regenerable output file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSON lines, rejecting malformed complete files."""
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def training_partitions(reports: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge known incidents and text duplicates, then protect held-out components.

    All known group links and identical narrative components are unioned. Near duplicates
    are candidates when they share a bottom-eight five-word-shingle hash, and are merged
    only at >= .85 shingle Jaccard with a >= .85 length ratio. This is a conservative
    duplicate screen, not a claim to identify every same-incident report across sources.
    Any component touching an official validation/test row is excluded from training.
    Ten percent of the remaining components become development data.
    """
    r = reports.fillna("").copy().reset_index(drop=True)
    if r.acn.duplicated().any():
        raise ValueError("duplicate document ids")
    parent = list(range(len(r)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    known, exact, sketches = {}, {}, defaultdict(list)
    shingle_sets: list[set[int] | None] = [None] * len(r)
    near_links = 0
    for i, row in enumerate(r.to_dict("records")):
        group = row["group_id"]
        if group:
            if group in known:
                union(i, known[group])
            known[group] = i
        for field in ("narrative", "narrative_1", "narrative_2"):
            ws = tokens(row.get(field, ""))
            if len(ws) >= 20:
                key = digest(" ".join(ws))
                if key in exact:
                    union(i, exact[key])
                exact[key] = i
        ws = tokens(row["narrative"])
        if len(ws) < 40:
            continue
        shingles = {int.from_bytes(hashlib.blake2b(" ".join(ws[k:k + 5]).encode(),
                                                  digest_size=8).digest(), "big")
                    for k in range(len(ws) - 4)}
        shingle_sets[i] = shingles
        sketch = sorted(shingles)[:8]
        candidates = {j for h in sketch for j in sketches[h]}
        for j in candidates:
            other = shingle_sets[j]
            if min(len(shingles), len(other)) / max(len(shingles), len(other)) < .85:
                continue
            if len(shingles & other) / len(shingles | other) >= .85:
                union(i, j)
                near_links += 1
        for h in sketch:
            sketches[h].append(i)
    roots = [find(i) for i in range(len(r))]
    names = {}
    for root, acn in zip(roots, r.acn):
        names[root] = min(names.get(root, acn), acn)
    r["ft_group"] = [names[root] for root in roots]
    protected = set(r.loc[r.split != "train", "ft_group"])
    r["ft_split"] = ["protected" if g in protected else "dev" if bucket(g) < 100 else "train"
                     for g in r.ft_group]
    counts = r.groupby([r.acn.str.split(":").str[0], "ft_split"]).size()
    audit = {"seed": SEED, "near_duplicate_links": near_links,
             "train_rows_excluded_by_protected_group": int(
                 ((r.split == "train") & (r.ft_split == "protected")).sum()),
             "counts": {f"{src}/{split}": int(n) for (src, split), n in counts.items()}}
    return r, audit


def support_overlap(query: str, passage: str) -> float:
    """Fraction of substantive query words present in a candidate passage."""
    stop = {"the", "and", "was", "were", "with", "from", "that", "this", "for", "into",
            "after", "before", "during", "when", "which", "their", "they", "have", "had",
            "what", "how", "does", "would", "could", "should", "reports", "report"}
    q = {w for w in tokens(query) if len(w) > 2 and w not in stop}
    return len(q & set(tokens(passage))) / max(1, len(q))


def best_passage(query: str, narrative: str) -> tuple[str, float]:
    """Select the most lexically supported existing chunk within the positive report."""
    chunks = chunk_text(narrative, ChunkConfig())
    if not chunks:
        return "", 0.
    scores = [support_overlap(query, p) for p in chunks]
    i = max(range(len(chunks)), key=lambda i: scores[i])
    return chunks[i], scores[i]


def supervised_pairs(reports: pd.DataFrame) -> tuple[list[dict], dict]:
    """Build labelled training/development queries without mixing reporter narratives."""
    pairs, rejected = [], defaultdict(int)
    for row in reports.to_dict("records"):
        if row["ft_split"] not in ("train", "dev"):
            continue
        src = row["acn"].split(":")[0]
        kind = "ntsb_cause" if src == "NTSB" else "asrs_synopsis"
        query = row.get("synopsis", "").strip()
        if query:
            positive, overlap = best_passage(query, row["narrative"])
            if len(tokens(query)) >= 5 and overlap >= .20 and len(tokens(positive)) >= 30:
                pairs.append({"query": query, "positive": positive, "kind": kind,
                              "doc_id": row["acn"], "group_id": row["ft_group"],
                              "split": row["ft_split"], "support_overlap": overlap})
            else:
                rejected[kind] += 1
        n1, n2 = row.get("narrative_1", ""), row.get("narrative_2", "")
        if src in ("ASRS", "ASRSN") and len(tokens(n2)) >= 40:
            query = chunk_text(n2, ChunkConfig(max_words=100, min_tail_words=20))[0]
            positive, overlap = best_passage(query, n1)
            if len(tokens(positive)) >= 30 and overlap >= .20 and tokens(query) != tokens(positive):
                pairs.append({"query": query, "positive": positive, "kind": "second_narrative",
                              "doc_id": row["acn"], "group_id": row["ft_group"],
                              "split": row["ft_split"], "support_overlap": overlap})
            else:
                rejected["second_narrative"] += 1
    return pairs, dict(rejected)


def validate_generated(item: dict, passage: str) -> tuple[bool, str]:
    """Require a useful query and an exact evidence quotation from its source passage.

    These automated checks provide traceability, not a human faithfulness judgement.
    """
    q, evidence = item.get("query", ""), item.get("evidence", "")
    if not isinstance(q, str) or not isinstance(evidence, str):
        return False, "type"
    if not 6 <= len(tokens(q)) <= 50 or not 5 <= len(tokens(evidence)) <= 60:
        return False, "length"
    if " ".join(tokens(evidence)) not in " ".join(tokens(passage)):
        return False, "evidence_not_in_passage"
    if support_overlap(q, passage) < .20:
        return False, "weak_support"
    q_nums = set(re.findall(r"\d+(?:\.\d+)?", q))
    p_nums = set(re.findall(r"\d+(?:\.\d+)?", passage))
    if q_nums - p_nums:
        return False, "unsupported_number"
    if re.search(r"<think>|as an ai|here are|according to the passage", q, re.I):
        return False, "meta_text"
    return True, "ok"


def balanced_epoch(pools: dict[str, list[dict]], size: int, seed: int = SEED) -> list[dict]:
    """Sample exact mixture counts, cycling shuffled pools when oversampling is needed."""
    import random

    rng = random.Random(seed)
    rows = []
    counts = {k: int(size * w) for k, w in WEIGHTS.items()}
    counts["asrs_synopsis"] += size - sum(counts.values())
    for kind, n in counts.items():
        pool = pools.get(kind, [])
        if n and not pool:
            raise ValueError(f"empty approved training stratum: {kind}")
        pending = []
        while len(pending) < n:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            pending.extend(shuffled[:n - len(pending)])
        rows.extend(pending)
    rng.shuffle(rows)
    return rows

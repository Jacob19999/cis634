"""LLM query rewrites: look up a query's pre-computed rewrite.

``scripts/rewrite_queries.py`` writes one JSON line per query (``{"text": ..., "rewrite":
...}``) to ``data/queries/rewrites/``. A retriever whose config sets ``query_rewrites: <file>``
searches with the rewrite instead of, or appended to, the original query
(``rewrite_mode: replace | append``). Rewrites are computed ahead of time so that a
benchmark run needs no LLM and every run sees the same text.
"""

from __future__ import annotations

import json
from functools import lru_cache

REWRITE_MODES = ("append", "replace")


@lru_cache(maxsize=None)
def load_rewrites(path: str) -> dict[str, str]:
    """Query text -> rewrite; the last non-empty rewrite of a query wins."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("rewrite"):
                    out[rec["text"]] = rec["rewrite"]
    return out


def apply_rewrites(queries: list[str], rewrites: dict[str, str], mode: str = "append",
                   source: str = "") -> list[str]:
    """Rewritten queries. Every query must have a rewrite: a silent fallback to the original
    would mix two conditions in one run."""
    if mode not in REWRITE_MODES:
        raise ValueError(f"rewrite_mode must be one of {REWRITE_MODES}, got {mode!r}")
    missing = [q for q in queries if q not in rewrites]
    if missing:
        raise KeyError(f"{len(missing)} queries have no rewrite in {source or 'the file'}, "
                       f"e.g. {missing[0][:80]!r}; run scripts/rewrite_queries.py on this config")
    if mode == "replace":
        return [rewrites[q] for q in queries]
    return [f"{q} {rewrites[q]}" for q in queries]

"""Fusion of two or more report-level rankings.

``rrf``: reciprocal rank fusion, score(d) = sum_s 1 / (k + rank_s(d))  (Cormack et al., 2009).
``convex``: alpha-weighted sum of per-query min-max normalised scores.
``max``: a report's best raw score over the components. Only meaningful when the components
score on one scale, e.g. one encoder over chunks and over LLM summaries: it then equals MaxP
over the union of their units, without re-encoding the union (exact for k <= depth).
A document missing from a component's top-``depth`` list gets no contribution from it.
"""

from __future__ import annotations

import numpy as np

from .base import Retriever, SearchResult, Units


class HybridRetriever(Retriever):
    """Fuses the report rankings of systems built earlier, e.g. BM25 + MiniLM."""

    supports_pairs = False

    def __init__(self, name: str, components: list[Retriever], method: str = "rrf",
                 k: int = 60, weights: list[float] | None = None, depth: int = 1000, **kw):
        kw.pop("doc_mode", None)
        if kw.get("expand") or kw.get("query_rewrites"):
            raise ValueError("a hybrid fuses rankings; set `expand` / `query_rewrites` on its "
                             "components instead")
        super().__init__(name, doc_mode="chunk_max", **kw)
        if method not in ("rrf", "convex", "max"):
            raise ValueError(f"unknown fusion method {method!r}")
        self.components = components
        self.method, self.k, self.depth = method, k, depth
        self.weights = weights or [1.0 / len(components)] * len(components)

    def build(self, units: Units) -> None:  # components are indexed by the registry
        pass

    def index(self, units: Units) -> None:
        self.units = units

    def score_units(self, queries):  # not used: fusion happens at report level
        raise NotImplementedError

    def search(self, queries: list[str], k: int = 100) -> list[SearchResult]:
        # Each component returns its top ``depth`` reports for every query.
        runs = [c.search(queries, k=self.depth) for c in self.components]
        out = []
        for per_q in zip(*runs):
            fused: dict[str, float] = {}
            passage: dict[str, str] = {}
            for w, res in zip(self.weights, per_q):
                if self.method == "max":
                    for d, s, u in zip(res.doc_ids, res.scores, res.best_units):
                        if d not in fused or s > fused[d]:
                            fused[d], passage[d] = float(s), u  # passage of the best unit
                    continue
                if self.method == "rrf":
                    # RRF: the report at rank r gets 1 / (k + r). Equal weights make w * n = 1.
                    contrib = w * len(self.weights) / (self.k + np.arange(1, len(res.doc_ids) + 1))
                else:
                    # Convex: rescale this list's scores to 0..1, then weight them.
                    s = res.scores.astype(np.float64)
                    span = s.max() - s.min()
                    contrib = w * ((s - s.min()) / span if span > 0 else np.ones_like(s))
                for d, c, u in zip(res.doc_ids, contrib, res.best_units):
                    fused[d] = fused.get(d, 0.0) + float(c)
                    passage.setdefault(d, u)  # passage from the first component that found it
            # Highest fused score first.
            docs = sorted(fused, key=fused.get, reverse=True)[:k]
            out.append(SearchResult(np.array(docs), np.array([fused[d] for d in docs]),
                                    np.array([passage[d] for d in docs])))
        return out

    def describe(self) -> dict:
        return {"name": self.name, "class": "HybridRetriever", "method": self.method,
                "k": self.k, "weights": self.weights, "depth": self.depth,
                "components": [c.name for c in self.components]}

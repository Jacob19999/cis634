"""Common retriever interface.

Every system scores *units* -- chunks of a narrative (``doc_mode: chunk_max``), whole
narratives (``doc_mode: whole``), or LLM summaries alone or next to the chunks (``summary``,
``chunk_summary``). Unit scores are reduced to one score per report by
taking the maximum over that report's units (MaxP), and the best-scoring unit is returned
as the matched passage. Adding a new system only requires implementing ``score_units``
(and optionally ``pair_scores`` for the meaning-change test).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..expand import AcronymExpander
from ..rewrites import REWRITE_MODES, apply_rewrites, load_rewrites

# chunk_max: chunks of the narrative; whole: the whole narrative; summary: an LLM summary of
# it; chunk_summary: the chunks plus the summary (see benchmark._unit_sets).
DOC_MODES = ("chunk_max", "whole", "summary", "chunk_summary")


@dataclass
class Units:
    """Texts that a retriever indexes, sorted so each report's units are contiguous."""

    texts: list[str]
    unit_ids: np.ndarray  # chunk_id, or acn for whole-report units
    doc_ids: np.ndarray  # acn of each unit's report (one entry per report, in order)
    doc_starts: np.ndarray  # index of each report's first unit
    tag: str  # identifies the unit set, used for embedding caches

    @classmethod
    def from_chunks(cls, chunks: pd.DataFrame, tag: str) -> "Units":
        chunks = chunks.sort_values(["acn", "narrative_idx", "chunk_idx"], kind="stable")
        acn = chunks["acn"].to_numpy()
        starts = np.flatnonzero(np.r_[True, acn[1:] != acn[:-1]])
        return cls(chunks["text"].tolist(), chunks["chunk_id"].to_numpy(), acn[starts], starts, tag)

    @classmethod
    def from_reports(cls, reports: pd.DataFrame, tag: str = "whole") -> "Units":
        reports = reports.sort_values("acn", kind="stable")
        acn = reports["acn"].to_numpy()
        return cls(reports["narrative"].tolist(), acn, acn, np.arange(len(acn)), tag)

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class SearchResult:
    """Top-k reports for one query."""

    doc_ids: np.ndarray
    scores: np.ndarray
    best_units: np.ndarray  # unit id of the best-matching passage per report


class Retriever(ABC):
    """Base class: subclasses implement ``build`` and ``score_units``."""

    #: whether ``pair_scores`` is meaningful (used by the contrast test)
    supports_pairs = True

    def __init__(self, name: str, doc_mode: str = "chunk_max", query_batch_size: int = 64,
                 expand: str | None = None, query_rewrites: str | None = None,
                 rewrite_mode: str = "append"):
        if doc_mode not in DOC_MODES:
            raise ValueError(f"doc_mode must be one of {DOC_MODES}, got {doc_mode!r}")
        if rewrite_mode not in REWRITE_MODES:
            raise ValueError(f"rewrite_mode must be one of {REWRITE_MODES}, got {rewrite_mode!r}")
        self.name = name
        self.doc_mode = doc_mode
        self.query_batch_size = query_batch_size
        # Optional acronym glossary applied to units and queries alike (see ``prepare``).
        self.expand = expand
        self._expander = AcronymExpander.load(expand) if expand else None
        # Optional pre-computed LLM rewrites of the search queries (see ``rewrite``).
        self.query_rewrites, self.rewrite_mode = query_rewrites, rewrite_mode
        self.units: Units | None = None

    # ---- to implement -------------------------------------------------------------------
    @abstractmethod
    def build(self, units: Units) -> None:
        """Index the units."""

    @abstractmethod
    def score_units(self, queries: list[str]) -> np.ndarray:
        """Return a (len(queries), len(units)) float32 array of relevance scores."""

    def pair_scores(self, queries: list[str], texts: list[str]) -> np.ndarray:
        """Score ``texts[i]`` against ``queries[i]`` (used by the meaning-change test)."""
        raise NotImplementedError

    # ---- shared -------------------------------------------------------------------------
    def prepare(self, texts: list[str]) -> list[str]:
        """Texts as this system sees them: acronyms expanded when ``expand`` is set.

        Applied to the units in ``index``, to queries in ``search``, and to both sides of
        the contrast test's pairs, so documents and queries always get the same treatment.
        """
        return self._expander.apply(list(texts)) if self._expander else list(texts)

    def rewrite(self, queries: list[str]) -> list[str]:
        """Search queries replaced by, or extended with, their LLM rewrites when configured.

        Only ``search`` uses it: the contrast test scores sentence pairs, not queries.
        """
        if not self.query_rewrites:
            return list(queries)
        return apply_rewrites(list(queries), load_rewrites(self.query_rewrites),
                              self.rewrite_mode, self.query_rewrites)

    def index(self, units: Units) -> None:
        """Keep the units and build the index over them."""
        if self._expander:
            # A new tag keeps the expanded units' embedding cache apart from the plain one.
            units = replace(units, texts=self.prepare(units.texts),
                            tag=f"{units.tag}+{self._expander.name}")
        self.units = units
        self.build(units)

    def search(self, queries: list[str], k: int = 100) -> list[SearchResult]:
        """Top-k reports per query: score every unit, then keep each report's best unit (MaxP)."""
        assert self.units is not None, "call index() first"
        queries = self.prepare(self.rewrite(queries))
        u = self.units
        out: list[SearchResult] = []
        for i in range(0, len(queries), self.query_batch_size):
            # Score a batch of queries against every unit: shape (batch, n_units).
            unit_scores = self.score_units(queries[i : i + self.query_batch_size])
            for row in unit_scores:
                # MaxP: a report's score is the max over its units (which are contiguous).
                doc_scores = np.maximum.reduceat(row, u.doc_starts)
                # Top k reports, best first (argpartition avoids sorting every report).
                kk = min(k, len(doc_scores))
                top = np.argpartition(-doc_scores, kk - 1)[:kk]
                top = top[np.argsort(-doc_scores[top], kind="stable")]
                # Remember which unit matched best: it is shown as the matched passage.
                best = self._best_units(row, top)
                out.append(SearchResult(u.doc_ids[top], doc_scores[top], best))
        return out

    def _best_units(self, row: np.ndarray, top_docs: np.ndarray) -> np.ndarray:
        """Id of the highest-scoring unit inside each of the top reports."""
        u = self.units
        ends = np.r_[u.doc_starts[1:], len(u)]
        best = [u.doc_starts[d] + int(np.argmax(row[u.doc_starts[d] : ends[d]])) for d in top_docs]
        return u.unit_ids[best]

    def describe(self) -> dict:
        out = {"name": self.name, "class": type(self).__name__, "doc_mode": self.doc_mode}
        if self.expand:
            out["expand"] = self.expand
        if self.query_rewrites:
            out.update(query_rewrites=self.query_rewrites, rewrite_mode=self.rewrite_mode)
        return out

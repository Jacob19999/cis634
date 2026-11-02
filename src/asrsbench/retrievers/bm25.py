"""Okapi BM25 over a sparse term matrix.

We implement BM25 directly (Lucene variant of the IDF) rather than calling a library so the
same corpus statistics can also score arbitrary text pairs in the meaning-change test.
``tests/test_retrievers.py`` checks the rankings against the ``bm25s`` library.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
import Stemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from .base import Retriever, Units

_TOKEN = re.compile(r"[a-z0-9]+")
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere",
             "cannot", "without"}

STOPWORD_SETS = {
    "english": frozenset(ENGLISH_STOP_WORDS),
    # Standard list minus negation words, so "did not descend" keeps its "not".
    "english_keep_negation": frozenset(ENGLISH_STOP_WORDS - NEGATIONS),
    "none": frozenset(),
}


class Analyzer:
    """lowercase -> alnum tokens -> stopword removal -> Snowball (Porter2) stemming."""

    def __init__(self, stopwords: str = "english", stem: bool = True):
        self.stopwords = STOPWORD_SETS[stopwords]
        self.stemmer = Stemmer.Stemmer("english") if stem else None

    def __call__(self, text: str) -> list[str]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in self.stopwords]
        return self.stemmer.stemWords(toks) if self.stemmer else toks


class BM25Retriever(Retriever):
    """Keyword retrieval: rare query words that appear often in a unit give a high score."""

    def __init__(self, name: str = "bm25", k1: float = 1.2, b: float = 0.75,
                 stopwords: str = "english", stem: bool = True, **kw):
        super().__init__(name, **kw)
        self.k1, self.b = k1, b
        self.stopwords, self.stem = stopwords, stem
        self.analyzer = Analyzer(stopwords, stem)

    def build(self, units: Units) -> None:
        # Term counts per unit: rows = units, columns = vocabulary terms.
        self.vectorizer = CountVectorizer(analyzer=self.analyzer, dtype=np.float32)
        tf = self.vectorizer.fit_transform(units.texts).tocsr()  # (U, V)
        n_docs = tf.shape[0]
        # IDF (Lucene): log(1 + (N - df + 0.5) / (df + 0.5)), so rare terms weigh more.
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Unit lengths: long units are penalised relative to the average length.
        dl = np.asarray(tf.sum(axis=1)).ravel()
        self.avgdl = float(dl.mean())
        self.W = self._weights(tf, dl).T.tocsr()  # (V, U): fast row slicing by query term

    def _weights(self, tf: sp.csr_matrix, dl: np.ndarray) -> sp.csr_matrix:
        """BM25 term weight for every (unit, term) entry of a tf matrix."""
        tf = tf.tocsr(copy=True)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        row_norm = np.repeat(norm, np.diff(tf.indptr)).astype(np.float32)
        # tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len)): repeats help, but saturate.
        tf.data = tf.data * (self.k1 + 1) / (tf.data + row_norm)
        return tf.multiply(self.idf[None, :]).tocsr()

    def _query_matrix(self, queries: list[str]) -> sp.csr_matrix:
        # Query term frequency counts (repeated terms count again, as in Lucene/bm25s).
        return self.vectorizer.transform(queries).tocsr()

    def score_units(self, queries: list[str]) -> np.ndarray:
        # One sparse matrix product sums the weights of the query terms in every unit.
        q = self._query_matrix(queries)  # (Q, V)
        return np.asarray((q @ self.W).todense(), dtype=np.float32)

    def pair_scores(self, queries: list[str], texts: list[str]) -> np.ndarray:
        # Score each sentence against its partner with the corpus IDF (meaning-change test).
        q = self._query_matrix(queries)
        tf = self.vectorizer.transform(texts).tocsr()
        w = self._weights(tf, np.asarray(tf.sum(axis=1)).ravel())
        return np.asarray(q.multiply(w).sum(axis=1)).ravel()

    def describe(self) -> dict:
        return {**super().describe(), "k1": self.k1, "b": self.b,
                "stopwords": self.stopwords, "stem": self.stem}

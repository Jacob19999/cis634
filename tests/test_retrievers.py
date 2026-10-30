import numpy as np
import pandas as pd
import pytest

from asrsbench.retrievers import BM25Retriever, HybridRetriever, Units
from asrsbench.retrievers.base import SearchResult

DOCS = [
    "The crew descended below the assigned altitude after a readback error.",
    "TCAS issued a resolution advisory and the crew climbed.",
    "Bird strike on takeoff roll; the captain rejected the takeoff.",
    "Controller issued a clearance to descend; the readback was not corrected.",
    "Fuel imbalance noted in cruise; the crew followed the checklist.",
]


def _chunks(docs):
    rows = []
    for i, d in enumerate(docs):
        acn = f"{i:03d}"
        # two chunks per doc: the text and an unrelated filler chunk
        rows.append((f"{acn}_1_0", acn, 1, 0, d))
        rows.append((f"{acn}_1_1", acn, 1, 1, "weather was clear"))
    return pd.DataFrame(rows, columns=["chunk_id", "acn", "narrative_idx", "chunk_idx", "text"])


def test_units_contiguous_per_report():
    u = Units.from_chunks(_chunks(DOCS).sample(frac=1, random_state=0), "t")
    assert list(u.doc_ids) == [f"{i:03d}" for i in range(5)]
    assert list(u.doc_starts) == [0, 2, 4, 6, 8]


def test_bm25_ranks_and_returns_best_passage():
    bm = BM25Retriever(stopwords="english")
    bm.index(Units.from_chunks(_chunks(DOCS), "t"))
    [res] = bm.search(["readback error altitude"], k=3)
    assert res.doc_ids[0] == "000"
    assert res.best_units[0] == "000_1_0"
    assert np.all(np.diff(res.scores) <= 0)


def test_bm25_matches_bm25s_library():
    bm25s = pytest.importorskip("bm25s")
    import Stemmer

    # No stopwords and no 1-letter tokens, so both libraries see identical token streams.
    corpus = [d.replace(" a ", " the ") for d in DOCS] * 3 + [
        "altitude altitude altitude deviation", "readback readback"]
    bm = BM25Retriever(stopwords="none")
    ids = np.arange(len(corpus)).astype(str)
    bm.index(Units(corpus, ids, ids, np.arange(len(corpus)), "t"))
    query = "readback altitude deviation"
    ours = bm.score_units([query])[0]

    stemmer = Stemmer.Stemmer("english")
    tok = dict(stopwords=None, stemmer=stemmer, show_progress=False)
    ref = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    ref.index(bm25s.tokenize(corpus, **tok), show_progress=False)
    theirs = ref.get_scores(bm25s.tokenize(query, return_ids=False, **tok)[0])
    # Lucene drops the constant (k1 + 1) numerator; it does not change rankings.
    np.testing.assert_allclose(ours / (1.2 + 1), theirs, rtol=1e-4, atol=1e-5)


def test_bm25_pair_scores_agree_with_index_scores():
    bm = BM25Retriever()
    bm.index(Units.from_reports(pd.DataFrame({"acn": ["a", "b"], "narrative": DOCS[:2]})))
    q = "crew altitude"
    idx = bm.score_units([q])[0]
    pair = bm.pair_scores([q, q], DOCS[:2])
    # Same idf; avgdl is shared, so scores for indexed texts must match.
    np.testing.assert_allclose(idx, pair, rtol=1e-5)


def test_negation_kept_only_with_keep_negation_stopwords():
    std = BM25Retriever(stopwords="english").analyzer("did not descend")
    neg = BM25Retriever(stopwords="english_keep_negation").analyzer("did not descend")
    assert "not" not in std and "not" in neg


class _Fixed:
    def __init__(self, name, docs):
        self.name, self.docs = name, docs

    def search(self, queries, k):
        n = len(self.docs)
        return [SearchResult(np.array(self.docs), np.linspace(1, 0, n), np.array(self.docs))
                for _ in queries]


def test_rrf_fusion():
    a = _Fixed("a", ["x", "y", "z"])
    b = _Fixed("b", ["y", "x", "w"])
    h = HybridRetriever("h", [a, b], method="rrf", k=60)
    [res] = h.search(["q"], k=4)
    assert set(res.doc_ids[:2]) == {"x", "y"}
    assert res.scores[0] == pytest.approx(1 / 61 + 1 / 62)
    assert list(res.doc_ids[2:]) == ["z", "w"]

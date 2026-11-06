import csv

import pandas as pd

from asrsbench.expand import AcronymExpander
from asrsbench.retrievers import BM25Retriever, Units

GLOSSARY = {"RA": "resolution advisory", "TCAS": "traffic alert and collision avoidance system",
            "FO": "first officer"}


def test_expands_first_use_only():
    ex = AcronymExpander(GLOSSARY)
    out = ex("TCAS gave an RA. We followed the RA.")
    assert out == ("TCAS (traffic alert and collision avoidance system) gave an "
                   "RA (resolution advisory). We followed the RA.")


def test_whole_word_case_sensitive_and_plural():
    ex = AcronymExpander(GLOSSARY)
    assert ex("RAIN and Ra and the FOR") == "RAIN and Ra and the FOR"
    assert ex("two RAs") == "two RAs (resolution advisory)"


def test_skips_when_already_spelled_out():
    ex = AcronymExpander(GLOSSARY)
    assert ex("a resolution advisory (RA) sounded") == "a resolution advisory (RA) sounded"


def test_committed_glossary_loads_and_has_no_duplicates(tmp_path):
    path = "data/glossary/asrs_acronyms_v1.tsv"
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    assert len({r["abbr"] for r in rows}) == len(rows)
    assert {r["source"] for r in rows} <= {"pcg", "jo7340", "manual"}
    ex = AcronymExpander.load(path)
    assert ex.name == "asrs_acronyms_v1" and ex("RA") == "RA (resolution advisory)"


def test_retriever_expands_units_and_queries(tmp_path):
    gl = tmp_path / "g.tsv"
    gl.write_text("abbr\texpansion\tsource\nRA\tresolution advisory\tmanual\n")
    rows = [("000_1_0", "000", 1, 0, "We got an RA and climbed."),
            ("001_1_0", "001", 1, 0, "The resolution advisory said descend."),
            ("002_1_0", "002", 1, 0, "Bird strike on takeoff.")]
    chunks = pd.DataFrame(rows, columns=["chunk_id", "acn", "narrative_idx", "chunk_idx", "text"])
    units = Units.from_chunks(chunks, "t")

    plain = BM25Retriever(stopwords="english")
    plain.index(units)
    [res] = plain.search(["resolution advisory"], k=3)
    assert "000" not in list(res.doc_ids[res.scores > 0])  # "RA" alone doesn't match

    bm = BM25Retriever(stopwords="english", expand=str(gl))
    bm.index(units)
    assert bm.units.tag == "t+g"  # separate embedding-cache key from the plain units
    assert units.texts[0] == "We got an RA and climbed."  # caller's units are untouched
    [res] = bm.search(["RA"], k=3)  # the query is expanded too
    assert set(res.doc_ids[res.scores > 0]) == {"000", "001"}
    assert bm.describe()["expand"] == str(gl)


def test_query_rewrites_append_replace_and_missing(tmp_path):
    import json

    import pytest

    from asrsbench.rewrites import apply_rewrites, load_rewrites

    f = tmp_path / "rw.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"text": "q1", "rewrite": ""},          # failed generation: ignored
        {"text": "q1", "rewrite": "first try"},
        {"text": "q1", "rewrite": "TCAS RA climb"},  # last one wins
    ]) + "\n")
    rw = load_rewrites(str(f))
    assert rw == {"q1": "TCAS RA climb"}
    assert apply_rewrites(["q1"], rw, "append") == ["q1 TCAS RA climb"]
    assert apply_rewrites(["q1"], rw, "replace") == ["TCAS RA climb"]
    with pytest.raises(KeyError):
        apply_rewrites(["q1", "q2"], rw)

    rows = [("000_1_0", "000", 1, 0, "TCAS RA so we climbed."),
            ("001_1_0", "001", 1, 0, "Bird strike on takeoff.")]
    chunks = pd.DataFrame(rows, columns=["chunk_id", "acn", "narrative_idx", "chunk_idx", "text"])
    bm = BM25Retriever(stopwords="english", query_rewrites=str(f), rewrite_mode="replace")
    bm.index(Units.from_chunks(chunks, "t"))
    [res] = bm.search(["q1"], k=2)
    assert res.doc_ids[0] == "000" and res.scores[0] > 0

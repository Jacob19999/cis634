import pandas as pd

from asrsbench.data import (
    ChunkConfig, assign_incident_groups, chunk_text, is_placeholder, normalize_text,
)


def test_normalize_undoes_semicolon_artifact():
    assert normalize_text("cabin altitude climbing past 10;000 FT at FL240") == (
        "cabin altitude climbing past 10,000 FT at FL240"
    )
    assert normalize_text("crossing KLEVR at 5000 FT IAF; we set zeros") == (
        "crossing KLEVR at 5000 FT IAF, we set zeros"
    )
    assert normalize_text("  a \n b  ") == "a b"
    assert normalize_text(None) == ""


def test_chunking_respects_window_and_overlap():
    sents = [f"Sentence number {i} has exactly seven words." for i in range(10)]
    text = " ".join(sents)
    chunks = chunk_text(text, ChunkConfig(max_words=20, overlap_sentences=1, min_tail_words=0))
    assert all(len(c.split()) <= 20 for c in chunks)
    # Every sentence is covered.
    joined = " ".join(chunks)
    assert all(s in joined for s in sents)
    # Consecutive chunks overlap by one sentence.
    assert chunks[1].startswith(sents[1])


def test_chunking_hard_splits_long_sentence():
    text = " ".join(["word"] * 45) + "."
    chunks = chunk_text(text, ChunkConfig(max_words=20, overlap_sentences=0, min_tail_words=0))
    assert [len(c.split()) for c in chunks] == [20, 20, 5]


def test_incident_groups_link_shared_accessions_and_duplicates():
    df = pd.DataFrame({
        "acn": ["100", "200", "300", "400"],
        "accession_1": ["100", "200", "300", "400"],
        "accession_2": ["", "100", "", ""],
        "narrative_1": ["a", "b", "same text", "Same text"],
    })
    groups = assign_incident_groups(df).tolist()
    assert groups == ["100", "100", "300", "300"]


def test_short_tail_is_merged_into_previous_chunk():
    text = " ".join(["Alpha beta gamma delta epsilon zeta eta theta."] * 2) + " Short end."
    no_merge = chunk_text(text, ChunkConfig(max_words=16, overlap_sentences=0, min_tail_words=0))
    merged = chunk_text(text, ChunkConfig(max_words=16, overlap_sentences=0, min_tail_words=5))
    assert no_merge[-1] == "Short end."
    assert len(merged) == 1 and len(merged[0].split()) == 18


def test_placeholder_narratives_detected():
    for t in ["[Report narrative contained no additional information.]",
              "[Narrative #2 had no additional information]",
              "[No substantive additional information was provided in the secondary narrative.]"]:
        assert is_placeholder(t)
    assert not is_placeholder("We had no additional information about the traffic, so we climbed.")
    assert not is_placeholder("")


def test_summary_units_sort_first_and_require_every_report(tmp_path):
    import json

    import pytest

    from asrsbench.data import load_summaries, summary_units
    from asrsbench.retrievers import Units

    (tmp_path / "s_v1.0.jsonl").write_text(json.dumps({"acn": "1", "summary": "A"}) + "\n{broken")
    (tmp_path / "s_v1.1.jsonl").write_text(json.dumps({"acn": "2", "summary": "B"}) + "\n")
    summ = load_summaries(tmp_path / "s_v1.jsonl")  # reads both shards, skips the broken line
    assert summ == {"1": "A", "2": "B"}

    reports = pd.DataFrame({"acn": ["1", "2"]})
    chunks = pd.DataFrame({"chunk_id": ["1_1_0", "2_1_0", "2_1_1"], "acn": ["1", "2", "2"],
                           "narrative_idx": 1, "chunk_idx": [0, 0, 1], "text": ["a", "b", "c"]})
    u = Units.from_chunks(pd.concat([summary_units(reports, summ), chunks]), "t")
    assert u.texts == ["A", "a", "B", "b", "c"]  # each summary leads its report's units
    assert list(u.doc_starts) == [0, 2]
    with pytest.raises(ValueError):
        summary_units(pd.DataFrame({"acn": ["1", "3"]}), summ)

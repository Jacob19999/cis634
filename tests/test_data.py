import pandas as pd

from asrsbench.data import ChunkConfig, assign_incident_groups, chunk_text, normalize_text


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
    chunks = chunk_text(text, ChunkConfig(max_words=20, overlap_sentences=1))
    assert all(len(c.split()) <= 20 for c in chunks)
    # Every sentence is covered.
    joined = " ".join(chunks)
    assert all(s in joined for s in sents)
    # Consecutive chunks overlap by one sentence.
    assert chunks[1].startswith(sents[1])


def test_chunking_hard_splits_long_sentence():
    text = " ".join(["word"] * 45) + "."
    chunks = chunk_text(text, ChunkConfig(max_words=20, overlap_sentences=0))
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

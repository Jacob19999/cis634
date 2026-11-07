"""Offline checks for fine-tuning split isolation, positive support and sampling."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from asrsbench.finetuning import (WEIGHTS, balanced_epoch, bucket, supervised_pairs,
                                 training_partitions, validate_generated)


def row(acn, text, split="train", group=None, n2=""):
    """Build the smallest corpus-v2 row needed for these tests."""
    return {"acn": acn, "group_id": group or acn, "split": split, "narrative": text,
            "narrative_1": text, "narrative_2": n2, "synopsis": ""}


def test_protected_incident_and_duplicate_components_cannot_train():
    text = " ".join(f"word{i}" for i in range(100))
    r = pd.DataFrame([row("ASRS:1", text, group="g"),
                      row("ASRS:2", text + " tail", "test", group="g"),
                      row("ASRSN:3", text.upper())])
    partitioned, _ = training_partitions(r)
    assert set(partitioned.ft_split) == {"protected"}
    assert partitioned.ft_group.nunique() == 1


def test_near_duplicate_across_sources_protects_training_member():
    words = [f"token{i}" for i in range(200)]
    altered = words.copy()
    altered[100] = "different"
    r, audit = training_partitions(pd.DataFrame([
        row("ASRSN:1", " ".join(words)),
        row("NTSB:2", " ".join(altered), "validation")]))
    assert r.ft_group.nunique() == 1
    assert set(r.ft_split) == {"protected"}
    assert audit["near_duplicate_links"] == 1


def test_group_assignment_is_deterministic_under_row_reordering():
    r = pd.DataFrame([row("ASRS:1", "unique one " * 30, group="g"),
                      row("ASRS:2", "unique two " * 30, group="g")])
    a, _ = training_partitions(r)
    b, _ = training_partitions(r.iloc[::-1])
    assert a.set_index("acn").ft_split.to_dict() == b.set_index("acn").ft_split.to_dict()


def test_second_narrative_positive_does_not_include_the_query_narrative():
    n1 = "Pilot reported engine vibration after takeoff and returned to land. " * 8
    n2 = "Captain noticed engine vibration during climb and requested a return. " * 8
    r = pd.DataFrame([{**row("ASRS:1", n1, n2=n2), "ft_group": "g", "ft_split": "train"}])
    pairs, _ = supervised_pairs(r)
    assert len(pairs) == 1
    assert pairs[0]["kind"] == "second_narrative"
    assert "Captain" not in pairs[0]["positive"]


def test_generated_evidence_and_numbers_must_come_from_passage():
    passage = "The pilot climbed to 5000 feet after receiving an amended clearance from the controller."
    item = {"query": "Find reports of a pilot climbing after an amended clearance",
            "evidence": "climbed to 5000 feet after receiving an amended clearance"}
    assert validate_generated(item, passage)[0]
    assert not validate_generated({**item, "evidence": "invented quotation not in this passage"}, passage)[0]
    assert not validate_generated({**item, "query": item["query"] + " to 9000 feet"}, passage)[0]


def test_exact_weights_and_empty_strata_fail_closed():
    pools = {k: [{"kind": k, "query": k}] for k in WEIGHTS}
    a = balanced_epoch(pools, 100)
    assert Counter(p["kind"] for p in a) == {k: int(w * 100) for k, w in WEIGHTS.items()}
    assert a == balanced_epoch(pools, 100)
    with pytest.raises(ValueError, match="empty approved"):
        balanced_epoch({**pools, "faa": []}, 100)

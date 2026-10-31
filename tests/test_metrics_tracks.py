import math

import numpy as np
import pandas as pd
import pytest

from asrsbench.metrics import (
    evaluate_run, ndcg_at_k, paired_randomization_test, precision_at_k, reciprocal_rank,
    recall_at_k, wilson_ci,
)
from asrsbench.tracks import load_contrast, silver_qrels, synopsis_track


def test_ndcg_perfect_and_partial():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rel, 10) == pytest.approx(1.0)
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], rel, 10) == pytest.approx(dcg / idcg)
    assert math.isnan(ndcg_at_k(["a"], {}, 10))


def test_precision_recall_rr():
    rel = {"a": 2, "b": 1, "z": 2}
    ranked = ["x", "a", "b"]
    assert precision_at_k(ranked, rel, 10, 2) == 0.1
    assert precision_at_k(ranked, rel, 10, 1) == 0.2
    assert reciprocal_rank(ranked, rel, 10) == 0.5
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)


def test_evaluate_run_skips_unjudged_queries():
    df = evaluate_run({"q1": ["a"], "q2": ["b"]}, {"q1": {"a": 2}}, ["MRR@10", "P@10"])
    assert list(df.index) == ["q1"]
    assert df.loc["q1", "MRR@10"] == 1.0


def test_randomization_and_wilson():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 200)
    assert paired_randomization_test(a, a) == 1.0
    assert paired_randomization_test(a + 0.1, a) < 0.01
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


REPORTS = pd.DataFrame({
    "acn": ["1", "2", "3", "4"],
    "group_id": ["1", "1", "3", "4"],
    "split": ["validation"] * 4,
    "synopsis": ["A crew reported an altitude deviation after a readback error on approach."] * 4,
    "anomaly": ["Deviation - Altitude Overshoot", "Deviation - Altitude Undershoot",
                "Conflict NMAC", "Deviation - Altitude Overshoot; Conflict NMAC"],
    "human_factors": ["Distraction", "", "", ""],
})


def test_silver_rules_take_highest_matching_grade():
    rules = {"2": [{"anomaly": "Overshoot", "human_factors": "Distraction"}],
             "1": [{"anomaly": "Deviation - Altitude"}]}
    assert silver_qrels(REPORTS, rules) == {"1": 2, "2": 1, "4": 1}


def test_silver_patterns_match_at_word_starts():
    reports = REPORTS.assign(synopsis=["Center issued a clearance.", "Crew entered the wrong altitude.",
                                       "Cabin pressurization failed.", "No match here."])
    assert silver_qrels(reports, {"2": [{"synopsis": "enter|pressuri[sz]"}]}) == {"2": 2, "3": 2}


def test_silver_rejects_unknown_fields():
    with pytest.raises(ValueError):
        silver_qrels(REPORTS, {"2": [{"narrative": "x"}]})


def test_synopsis_track_counts_same_incident_as_relevant():
    qs = synopsis_track(REPORTS, n_queries=4)
    assert qs.qrels["auto"]["S1"] == {"1": 2, "2": 2}
    assert qs.qrels["auto"]["S3"] == {"3": 2}


def test_contrast_set_is_well_formed():
    df = load_contrast("data/contrast/contrast_v1.jsonl")
    assert df["id"].is_unique
    assert df["category"].value_counts().to_dict() == {
        "negation": 25, "altitude": 25, "direction": 25, "event_order": 25}
    assert (df["original"] != df["paraphrase"]).all() and (df["original"] != df["changed"]).all()

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_v2", Path(__file__).resolve().parents[1] / "scripts" / "export_precedent_graph_v2.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_ntsb_phase_mapping():
    assert ex.ntsb_phase("Landing-flare/touchdown") == "Landing"
    assert ex.ntsb_phase("Enroute-cruise") == "Cruise"
    assert ex.ntsb_phase("Enroute-climb to cruise") == "Climb"
    assert ex.ntsb_phase("Approach-IFR final approach") == "Final Approach"
    assert ex.ntsb_phase("Approach-VFR pattern downwind") == "Initial Approach"
    assert ex.ntsb_phase("Something else") is None


def test_ntsb_facets_map_codes_onto_page_layers():
    codes = {"occurrences": ["Landing-flare/touchdown: Hard landing"],
             "findings": [{"finding": "Personnel issues-Task performance-Use of equip/info-Aircraft control-Pilot - C"},
                          {"finding": "Aircraft-Aircraft power plant-Engine (reciprocating)-Recip engine power section - C"},
                          {"finding": "Environmental issues-Conditions/weather/phenomena-Wind-Crosswind - F"}],
             "weather": "VMC", "light": "NITE", "highest_injury": "FATL", "damage": "SUBS", "far_part": "091"}
    got = set(ex.ntsb_facets(codes, "Cessna 172P", "BKV"))
    assert ("event", "Hard landing") in got and ("phase", "Landing") in got
    assert ("human", "Finding: Use of equip/info / Aircraft control") in got
    assert ("system", "ATA 71-72 Engine") in got
    assert ("weather", "Wind / gusts") in got and ("weather", "Night") in got and ("weather", "VMC") in got
    assert ("response", "Fatal injuries") in got and ("response", "Substantial damage") in got
    assert ("context", "Aircraft: Cessna 172") in got and ("context", "Operating under Part 91") in got
    assert ("context", "Location: BKV") in got


def test_tod_bucket():
    assert ex.tod_bucket(0, "1201-1800", None) == 2
    assert ex.tod_bucket(1, "", "0545") == 0 and ex.tod_bucket(1, "", 1830) == 3
    assert ex.tod_bucket(1, "", None) is None and ex.tod_bucket(2, "", None) is None


def test_fatality_counts_preserve_zero_and_reject_missing_or_invalid():
    for value in [None, "", "UNK", float("nan"), float("inf"), -1, 1.5, True]:
        assert ex.fatality_count(value) is None
    assert ex.fatality_count(0) == 0
    assert ex.fatality_count("12") == 12


def test_incremental_fatalities_use_event_id_without_changing_other_fields(tmp_path):
    import json
    import pandas as pd

    folder = tmp_path / "data" / "reports"
    folder.mkdir(parents=True)
    rows = [{"src": 1, "id": "NTSB:event-a", "acn": "case-a", "nar": [[1, .9]]},
            {"src": 1, "id": "NTSB:event-b"}, {"src": 1, "id": "NTSB:missing"},
            {"src": 0, "id": "ASRS:event-a"}, {"src": 2, "id": "ASRSN:event-b"}]
    path = folder / "0.json"
    path.write_text(json.dumps(rows))
    events = pd.DataFrame({"ev_id": ["event-a", "event-b"], "inj_tot_f": [0, 12]})
    ex.update_fatalities(tmp_path, events)
    expected = [{**r, **({"fatalities": [0, 12][i]} if i < 2 else {})}
                for i, r in enumerate(rows)]
    assert json.loads(path.read_text()) == expected


def test_cross_source_neighbours_exclude_both_asrs_variants_and_self():
    import numpy as np

    vectors = np.array([[1., 0.], [1., 0.], [0.8, 0.6], [0., 1.]], dtype=np.float32)
    indices, scores = ex.cross_source_neighbours(vectors, np.array([0, 2, 1, 1]), k=3)
    assert list(indices[0]) == [2, -1, -1]
    assert list(indices[1]) == [2, -1, -1]
    assert set(indices[2]) == {0, 1, -1}
    assert (indices[3] == -1).all()
    assert abs(scores[0, 0] - 0.8) < 1e-6


def test_generated_summary_is_not_labelled_as_probable_cause():
    assert ex.synopsis_kind(1, '') == 'generated'
    assert ex.synopsis_kind(1, 'A probable cause') == 'probable_cause'
    assert ex.synopsis_kind(0, 'An analyst synopsis') == 'analyst'
    assert ex.synopsis_kind(2, '') == 'generated'


def test_export_missing_cache_fails_before_encoding(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import pytest

    units = ex.Units(['narrative'], ['NTSB:1:0'], ['NTSB:1'], np.array([0]), 'test')
    monkeypatch.setattr(ex, 'load_config', lambda _: {
        'cache_dir': str(tmp_path), 'systems': [{
            'name': 'qwen3-8b', 'params': {'model': 'offline-test', 'max_seq_length': 16}}]})
    monkeypatch.setattr(ex, '_unit_sets', lambda *_: {'chunk_max': units})

    def unexpected_encoding(*args, **kwargs):
        pytest.fail('Exporter must not start encoding when the cache is absent')

    monkeypatch.setattr(ex.DenseRetriever, 'index', unexpected_encoding)
    with pytest.raises(RuntimeError, match='export never encodes'):
        ex.report_vectors('unused.yaml', pd.DataFrame({'acn': ['NTSB:1']}))

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

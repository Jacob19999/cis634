import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "build_corpus_v2", Path(__file__).resolve().parents[1] / "scripts" / "build_corpus_v2.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def test_hash_split_is_fixed_and_roughly_80_10_10():
    ids = [f"NTSB:{i}" for i in range(20000)]
    splits = pd.Series([bc.hash_split(i) for i in ids]).value_counts(normalize=True)
    assert bc.hash_split("NTSB:42") == bc.hash_split("NTSB:42")
    assert abs(splits["train"] - 0.8) < 0.02 and abs(splits["test"] - 0.1) < 0.02


def test_split_hf_narratives_and_comma_repair():
    text = "Narrative 1 - 'We descended; then climbed to 10;000 ft.'\nNarrative 2 - 'I agree.'"
    n1, n2 = bc.split_hf_narratives(text)
    assert n1 == "We descended, then climbed to 10,000 ft."
    assert n2 == "I agree."
    assert bc.split_hf_narratives("Narrative 1 - 'Only one.'") == ("Only one.", "")


def test_month_round_trip_and_intake_date_model(tmp_path):
    assert bc.yyyymm(bc.month_index("202203")) == "202203"
    # 24 months of intake, growing from 7,000 to 9,300 reports a month.
    months = pd.period_range("2019-01", periods=24, freq="M")
    intake = {m.strftime("%B %Y"): 7000 + 100 * i for i, m in enumerate(months)}
    path = tmp_path / "intake.json"
    path.write_text(__import__("json").dumps(intake))
    idx, cum = bc.cumulative_intake(path)
    # Reports: ACN = 1,500,000 + reports received by their event month, plus receipt noise.
    rng = np.random.default_rng(0)
    m = rng.integers(idx[0], idx[-1] + 1, 4000).astype(float)
    acn = 1_500_000 + np.interp(m, idx, cum) + rng.integers(0, 3000, 4000)
    c, s = bc.fit_intake_model(acn, m, idx, cum, bin_size=2000)
    est = bc.acn_to_month(acn, c, s, idx, cum)
    assert np.mean(np.abs(est - m)) < 0.6


def test_phase_names_from_shared_description_prefix():
    codes = pd.Series(["550250", "551230", "551240", "350250", "350260"])
    desc = pd.Series(["Landing Midair collision",
                      "Landing-flare/touchdown Loss of control on ground",
                      "Landing-flare/touchdown Hard landing",
                      "Initial climb Midair collision", "Initial climb Loss of engine power"])
    names = bc.phase_names(codes, desc)
    assert names["551"] == "Landing-flare/touchdown"
    assert names["350"] == "Initial climb"
    assert names["550"] == "Landing"  # single description: drop its event words


def test_fix_ntsb_text_restores_typographic_characters():
    raw = "The pilot\x19 s failure; he said \x1c visibility was poor.\x1d A Special \x13  Light Sport, on\x1egoing.\x0b End"
    assert bc.fix_ntsb_text(raw) == ("The pilot's failure; he said \"visibility was poor.\" A Special - "
                                     "Light Sport, on-going. End")

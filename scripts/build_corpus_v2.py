"""Build corpus v2: ASRS and NTSB reports for 2010–2026 in one table, for the Precedent Graph page.

usage: python scripts/build_corpus_v2.py [--out data/processed_v2]

Inputs (git-ignored under ``data/external/``):

* ASRS v1: ``data/processed/reports.parquet`` (``asrsbench prepare``), all 47,723 reports
  (Jan 2012 – Mar 2022), already cleaned. Place and state come from the raw HF dataset.
* NTSB: ``ntsb/*.parquet``, tables exported from ``avall.mdb`` (``avall.zip`` from
  data.ntsb.gov/avdata, rebuilt 2026-11-01, sha256 below) by ``--export-ntsb``. Events from
  2010 on that have a narrative: one document per event.
* Newer ASRS narratives: ``asrs_hf_narratives.parquet`` (``sookiemonster/asrs-narratives``,
  ACNs 1.82M–2.32M). Only ACNs not in v1. They come with a primary-problem label but no date,
  synopsis or codes; their month is estimated from the ACN and ASRS's monthly intake
  (``data/asrs/callback_intake_v1.json``, see ``fit_intake_model``) and flagged
  ``date_estimated``. The dataset
  states no license; the text itself is public NASA ASRS data.

Writes ``reports.parquet`` (schema in ``COLUMNS``), ``chunks_<tag>.parquet`` (the v1 chunking)
and ``stats.json``. ``acn`` holds the document id (``ASRS:<acn>``, ``NTSB:<ev_id>``,
``ASRSN:<acn>``), so the v1 chunking, unit and summary code works unchanged. Corpus v1 and
its results are not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from asrsbench.data import (HF_DATASET, ChunkConfig, build_chunks, is_placeholder, load_reports,
                            normalize_text)

AVALL_SHA256 = "c1a447835b41104737f8c3fcf6b2bbaa4b6a00f506fe7668b528d3b3a2f78bbc"
NTSB_TABLES = ["events", "aircraft", "narratives", "Findings", "Events_Sequence"]
YEARS = (2010, 2026)
COLUMNS = ["acn", "source", "source_id", "display_id", "date", "date_estimated", "aircraft", "flight_phase",
           "state", "location", "narrative_1", "narrative_2", "narrative", "synopsis", "label",
           "codes", "n_words", "split", "group_id"]


# ---------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------

def hash_split(doc_id: str) -> str:
    """Fixed 80/10/10 train/validation/test split from a hash of the document id."""
    b = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 10
    return "train" if b < 8 else "validation" if b == 8 else "test"


def month_index(yyyymm: str) -> int:
    """Months since January 2000 for a ``YYYYMM`` string."""
    return (int(yyyymm[:4]) - 2000) * 12 + int(yyyymm[4:6]) - 1


def yyyymm(m: float) -> str:
    """Inverse of ``month_index`` (rounded)."""
    m = int(round(m))
    return f"{2000 + m // 12:04d}{m % 12 + 1:02d}"


def cumulative_intake(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Month index of each CALLBACK intake month and reports received up to its end."""
    d = json.loads(Path(path).read_text())
    months = sorted(d, key=lambda k: pd.Timestamp(k))
    idx = np.array([month_index(pd.Timestamp(m).strftime("%Y%m")) for m in months], float)
    if not np.all(np.diff(idx) == 1):
        raise ValueError("CALLBACK intake months are not consecutive")
    return idx, np.cumsum([d[m] for m in months]).astype(float)


def fit_intake_model(acn: np.ndarray, months: np.ndarray, idx: np.ndarray, cum: np.ndarray,
                     bin_size: int = 5_000) -> tuple[float, float]:
    """Fit ACN = c + s * (reports received by the end of the event month) on dated reports.

    ASRS numbers reports in order of receipt, so ACNs grow with cumulative intake; ``c`` and
    ``s`` absorb the numbering offset and the reporting lag (event month vs receipt). The
    median month of each ACN bin damps the per-report lag before the least-squares fit.
    """
    keep = (months >= idx[0]) & (months <= idx[-1])
    df = pd.DataFrame({"bin": acn[keep] // bin_size, "acn": acn[keep], "m": months[keep]})
    g = df.groupby("bin").agg(acn=("acn", "median"), m=("m", "median"), n=("m", "size"))
    g = g[g["n"] >= 20]
    s, c = np.polyfit(np.interp(g["m"], idx, cum), g["acn"], 1)
    return float(c), float(s)


def acn_to_month(acn: np.ndarray, c: float, s: float, idx: np.ndarray, cum: np.ndarray) -> np.ndarray:
    """Estimated event month index for ACNs (inverse of ``fit_intake_model``; clipped to the
    months the intake table covers)."""
    return np.interp((np.asarray(acn, float) - c) / s, cum, idx)


def split_hf_narratives(text: str) -> tuple[str, str]:
    """``"Narrative 1 - '...'\\nNarrative 2 - '...'"`` -> (narrative 1, narrative 2)."""
    parts = re.split(r"(?:^|\n)Narrative (\d) - '", text)
    found = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    clean = lambda s: normalize_text(s.strip().rstrip("'")) if s else ""
    return clean(found.get("1", "")), clean(found.get("2", ""))


# ---------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------

def asrs_v1(v1_dir: str) -> pd.DataFrame:
    """The 47,723 v1 reports, with place and state from the raw dataset."""
    from datasets import load_dataset

    r = load_reports(v1_dir)
    raw = pd.concat([d.to_pandas() for d in load_dataset(HF_DATASET).values()])
    raw = raw.astype({"acn_num_ACN": str}).set_index("acn_num_ACN")
    state = raw["Place.1_State Reference"].reindex(r["acn"]).fillna("").astype(str).str.strip()
    loc = raw["Place_Locale Reference"].reindex(r["acn"]).fillna("").astype(str)
    loc = loc.str.split(".").str[0].str.strip()
    return pd.DataFrame({
        "acn": "ASRS:" + r["acn"], "source": "ASRS", "source_id": r["acn"], "display_id": r["acn"],
        "date": r["date"].astype(str), "date_estimated": False, "aircraft": r["aircraft"],
        "flight_phase": r["flight_phase"], "state": state.to_numpy(),
        "location": np.where(loc.str.startswith("ZZZ"), "", loc.to_numpy()),
        "narrative_1": r["narrative_1"], "narrative_2": r["narrative_2"],
        "narrative": r["narrative"], "synopsis": r["synopsis"], "label": r["primary_problem"],
        "codes": [json.dumps({"anomaly": a, "primary_problem": p, "human_factors": h})
                  for a, p, h in zip(r["anomaly"], r["primary_problem"], r["human_factors"])],
        "n_words": r["n_words"], "split": r["split"], "group_id": "ASRS:" + r["group_id"].astype(str),
    })


def _first_text(series: pd.Series) -> str:
    return " ".join(s for s in series.fillna("").astype(str).str.strip() if s and s != "None")


def _clean(series: pd.Series) -> pd.Series:
    """NTSB nulls arrive as None or the string "None"."""
    s = series.fillna("").astype(str).str.strip()
    return s.where(~s.isin(["None", "nan"]), "")


def phase_names(codes: pd.Series, descriptions: pd.Series) -> dict[str, str]:
    """Phase-of-flight name for each 3-digit phase prefix of an NTSB occurrence code.

    Codes are phase + event (``453342`` = "Maneuvering-hover" + "Loss of engine power
    (partial)") and the description is the two names joined by a space, so a phase's name is
    the words that every description with that prefix starts with.
    """
    def common(word_lists: list[list[str]]) -> list[str]:
        out = []
        for ws in zip(*word_lists):
            if len(set(ws)) > 1:
                break
            out.append(ws[0])
        return out

    df = pd.DataFrame({"p": codes.str[:3], "e": codes.str[3:], "d": descriptions}).drop_duplicates()
    # An event's name is the words its descriptions share at the end, across phases.
    events = {e: common([x.split()[::-1] for x in g.unique()])[::-1]
              for e, g in df.groupby("e")["d"] if g.nunique() > 1}
    out = {}
    for p, g in df.groupby("p"):
        descs = g["d"].unique()
        if len(descs) > 1:
            out[p] = " ".join(common([x.split() for x in descs]))
            continue
        # A phase seen with one event only: strip that event's name, if known.
        words, ev = descs[0].split(), events.get(g["e"].iloc[0], [])
        out[p] = " ".join(words[: len(words) - len(ev)] if ev and words[-len(ev):] == ev else words[:1])
    return out


def ntsb(ntsb_dir: Path) -> pd.DataFrame:
    """One document per NTSB event from 2010 on with a narrative."""
    t = {n: pd.read_parquet(ntsb_dir / f"{n}.parquet") for n in NTSB_TABLES}
    ev = t["events"].copy()
    ev["year"] = pd.to_numeric(ev["ev_year"], errors="coerce")
    ev = ev[ev["year"].between(*YEARS)].set_index("ev_id")
    nar = t["narratives"].groupby("ev_id").agg(accp=("narr_accp", _first_text),
                                               accf=("narr_accf", _first_text),
                                               inc=("narr_inc", _first_text),
                                               cause=("narr_cause", _first_text))
    ac = t["aircraft"].sort_values("Aircraft_Key").groupby("ev_id").first()
    seq = t["Events_Sequence"].sort_values(["Aircraft_Key", "Occurrence_No"]).copy()
    seq["code"], seq["desc"] = _clean(seq["Occurrence_Code"]), _clean(seq["Occurrence_Description"])
    seq = seq[(seq["code"].str.len() == 6) & (seq["desc"] != "")]
    names = phase_names(seq["code"], seq["desc"])
    seq["phase"] = seq["code"].str[:3].map(names)
    # "Phase: event", e.g. "Landing-flare/touchdown: Loss of control on ground".
    seq["occ"] = [f"{p}: {d[len(p):].strip()}" if p and d.startswith(p) else d
                  for p, d in zip(seq["phase"], seq["desc"])]
    occ = seq.groupby("ev_id")["occ"].apply(lambda s: list(dict.fromkeys(s)))
    is_def = seq["Defining_ev"].astype(str).isin(["True", "1", "Y"])
    defining = seq[is_def].groupby("ev_id")["occ"].first()
    def_phase = seq[is_def].groupby("ev_id")["phase"].first()
    fnd = t["Findings"].assign(cf=_clean(t["Findings"]["Cause_Factor"]))
    findings = fnd.groupby("ev_id").apply(
        lambda g: [{"finding": d, "cause_factor": c} for d, c in
                   zip(g["finding_description"].fillna("").astype(str), g["cf"]) if d],
        include_groups=False)
    d = ev.join(nar, how="inner").join(ac, rsuffix="_ac")
    # Factual narrative if there is one, else the preliminary one; incidents use narr_inc.
    d["text"] = np.where(d["accf"].str.len() > 50, d["accf"],
                         np.where(d["accp"].str.len() > 50, d["accp"], d["inc"]))
    d["text"] = d["text"].map(fix_ntsb_text)
    d = d[d["text"].str.len() > 50]
    phase = def_phase.reindex(d.index).fillna("")
    ids = d.index.astype(str)
    make_model = (_clean(d["acft_make"]).str.title() + " " + _clean(d["acft_model"])).str.strip()
    cols = {c: _clean(d[c]) for c in ["ev_type", "ev_highest_injury", "damage", "light_cond",
                                      "wx_cond_basic", "far_part"]}
    codes = [json.dumps({
        "ev_type": e, "highest_injury": inj, "damage": dmg, "light": lc, "weather": wx,
        "far_part": far, "occurrences": occ.get(i, []), "defining_event": defining.get(i, ""),
        "findings": findings.get(i, [])}, default=str)
        for i, e, inj, dmg, lc, wx, far in zip(d.index, *cols.values())]
    dates = pd.to_datetime(d["ev_date"], errors="coerce")
    return pd.DataFrame({
        "acn": "NTSB:" + ids, "source": "NTSB", "source_id": ids,
        # The NTSB case number (e.g. CEN19LA160) is what investigators and the public cite.
        "display_id": _clean(d["ntsb_no"]).where(_clean(d["ntsb_no"]) != "", pd.Series(ids, index=d.index)).to_numpy(),
        "date": dates.dt.strftime("%Y%m").fillna(d["year"].astype(int).astype(str) + "01").to_numpy(),
        "date_estimated": False, "aircraft": make_model.to_numpy(),
        "flight_phase": phase.to_numpy(),
        "state": _clean(d["ev_state"]).str.upper().to_numpy(),
        "location": _clean(d["ev_nr_apt_id"]).str.upper().to_numpy(),
        "narrative_1": d["text"].to_numpy(), "narrative_2": "", "narrative": d["text"].to_numpy(),
        "synopsis": d["cause"].map(fix_ntsb_text).to_numpy(), "label": cols["ev_type"].to_numpy(),
        "codes": codes, "n_words": d["text"].str.split().str.len().to_numpy(),
        "split": [hash_split("NTSB:" + i) for i in ids], "group_id": ("NTSB:" + ids).to_numpy(),
    })


_NTSB_CHARS = [
    # The Access export keeps only the low byte of typographic characters, often followed by a
    # stray space: U+2019 arrives as "\x19 " ("pilot\x19 s failure"), U+201C as "\x1c ", etc.
    (re.compile(r"[\x18\x19] ?s\b"), "'s"), (re.compile(r"[\x18\x19]"), "'"),
    (re.compile(r"\x1c ?"), '"'), (re.compile(r"\x1d"), '"'),
    (re.compile(r" ?[\x13\x14] ?"), " - "), (re.compile(r"\x1e"), "-"),
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"), " "),
]


def fix_ntsb_text(s: str) -> str:
    """Repair the NTSB export's typographic-character artefacts and collapse whitespace."""
    s = s or ""
    for rx, rep in _NTSB_CHARS:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s).strip()


def asrs_hf(path: Path, v1: pd.DataFrame, labels: list[str],
            intake_path: Path) -> tuple[pd.DataFrame, dict]:
    """Newer ASRS narratives not in v1, with an ACN-based date estimate from CALLBACK intake."""
    h = pd.read_parquet(path)
    h["acn_int"] = h["acn"].astype(int)
    v1_acn = v1["source_id"].astype(int).to_numpy()
    h = h[~h["acn_int"].isin(set(v1_acn))].drop_duplicates("acn_int").sort_values("acn_int")
    months = v1["date"].map(month_index).to_numpy().astype(float)
    idx, cum = cumulative_intake(intake_path)
    # Check on v1 itself: fit on events up to 2020, predict the 2021–2022 ones.
    old, new = months < month_index("202101"), months >= month_index("202101")
    c, s = fit_intake_model(v1_acn[old], months[old], idx, cum)
    err = np.abs(acn_to_month(v1_acn[new], c, s, idx, cum) - months[new])
    c, s = fit_intake_model(v1_acn, months, idx, cum)
    est = acn_to_month(h["acn_int"].to_numpy(), c, s, idx, cum)
    n12 = h["text"].map(split_hf_narratives)
    n1 = n12.str[0]
    n2 = n12.str[1].map(lambda s: "" if is_placeholder(s) else s)
    acn = h["acn_int"].astype(str)
    narrative = (n1 + " " + n2).str.strip()
    df = pd.DataFrame({
        "acn": ("ASRSN:" + acn).to_numpy(), "source": "ASRS", "source_id": acn.to_numpy(),
        "display_id": acn.to_numpy(),
        "date": [yyyymm(m) for m in est], "date_estimated": True,
        "aircraft": "", "flight_phase": "", "state": "", "location": "",
        "narrative_1": n1.to_numpy(), "narrative_2": n2.to_numpy(), "narrative": narrative.to_numpy(),
        "synopsis": "", "label": [labels[i] for i in h["label"]], "codes": "{}",
        "n_words": narrative.str.split().str.len().to_numpy(),
        "split": [hash_split("ASRSN:" + x) for x in acn], "group_id": ("ASRSN:" + acn).to_numpy(),
    })
    df = df[df["narrative"].str.len() > 50]
    return df, {"model": "ACN = c + s * cumulative CALLBACK intake (data/asrs/callback_intake_v1.json)",
                "fit": {"c": c, "s": s},
                "check_fit_to_2020_predict_v1_2021_2022": {
                    "n": int(new.sum()), "mae_months": round(float(err.mean()), 2),
                    "p90_abs_months": round(float(np.quantile(err, 0.9)), 2)},
                "estimated_range": [yyyymm(est.min()), yyyymm(est.max())],
                "at_intake_table_end": int((est >= idx[-1]).sum())}


def export_ntsb(mdb: Path, out_dir: Path) -> None:
    """Dump the NTSB tables used here from ``avall.mdb`` to parquet (slow; run once)."""
    from access_parser import AccessParser

    if hashlib.sha256(mdb.with_suffix(".zip").read_bytes()).hexdigest() != AVALL_SHA256:
        print("warning: avall.zip differs from the recorded release (a newer NTSB rebuild?)")
    db = AccessParser(str(mdb))
    for name in NTSB_TABLES:
        pd.DataFrame(db.parse_table(name)).to_parquet(out_dir / f"{name}.parquet", index=False)


def stats(r: pd.DataFrame, chunks: pd.DataFrame, extra: dict) -> dict:
    """Counts by source and year, split sizes and lengths (saved to stats.json)."""
    r = r.assign(year=r["date"].str[:4], kind=np.where(r["acn"].str.startswith("ASRSN:"),
                                                       "ASRS (narrative only)", r["source"]))
    return {
        "n_reports": int(len(r)), "n_chunks": int(len(chunks)),
        "by_source": r["kind"].value_counts().to_dict(),
        "by_source_year": {k: g["year"].value_counts().sort_index().to_dict()
                           for k, g in r.groupby("kind")},
        "split_sizes": {k: g["split"].value_counts().to_dict() for k, g in r.groupby("kind")},
        "words_per_report_median": {k: float(g["n_words"].median()) for k, g in r.groupby("kind")},
        "with_synopsis_or_cause": {k: int((g["synopsis"] != "").sum()) for k, g in r.groupby("kind")},
        "date_estimated": int(r["date_estimated"].sum()), **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed_v2")
    ap.add_argument("--v1", default="data/processed")
    ap.add_argument("--external", default="data/external")
    ap.add_argument("--export-ntsb", action="store_true", help="first dump avall.mdb to parquet")
    args = ap.parse_args()
    ext, out = Path(args.external), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.export_ntsb:
        export_ntsb(ext / "ntsb" / "avall.mdb", ext / "ntsb")

    from datasets import load_dataset_builder
    labels = load_dataset_builder("sookiemonster/asrs-narratives").info.features["label"].names

    v1 = asrs_v1(args.v1)
    nt = ntsb(ext / "ntsb")
    hf, date_fit = asrs_hf(ext / "asrs_hf_narratives.parquet", v1, labels,
                           Path("data/asrs/callback_intake_v1.json"))
    r = pd.concat([v1, nt, hf], ignore_index=True)[COLUMNS]
    assert r["acn"].is_unique, "duplicate document ids"
    assert r["display_id"].is_unique, "duplicate display ids"
    r = r.sort_values("acn", kind="stable").reset_index(drop=True)
    cfg = ChunkConfig()
    chunks = build_chunks(r, cfg)
    r.to_parquet(out / "reports.parquet", index=False)
    chunks.to_parquet(out / f"chunks_{cfg.tag}.parquet", index=False)
    s = stats(r, chunks, {"hf_date_estimate": date_fit, "ntsb_years": list(YEARS)})
    (out / "stats.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s[k] for k in ("n_reports", "n_chunks", "by_source", "hf_date_estimate")},
                     indent=2))


if __name__ == "__main__":
    main()

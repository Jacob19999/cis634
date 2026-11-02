"""Error analyses quoted in the milestone report (Sections III-B, III-D and III-E).

usage: python scripts/analysis.py results/milestone [--v1 results/milestone_v1_rawchunks]

Writes <run>/analysis.json. Needs data/processed (``asrsbench prepare``), the run files that
``asrsbench run`` writes to <run>/runs/ (not committed) and the cached embeddings of the
run's E5 system. ``--v1`` adds the pre-processing comparison; its run files were made with
pre-processing v1 (see results/milestone_v1_rawchunks/NOTE.md).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from asrsbench.benchmark import read_run
from asrsbench.data import ChunkConfig, load_chunks, load_reports
from asrsbench.metrics import evaluate_run
from asrsbench.retrievers import DenseRetriever, Units
from asrsbench.retrievers.bm25 import Analyzer
from asrsbench.tracks import load_contrast, synopsis_track, topical_track

QUERIES = "data/queries/topical_v1.yaml"
SHORT_WORDS = 30  # a "short unit" has fewer words than this

# Identifiers a synopsis can copy from its narrative: aircraft type designators (B737-800,
# CE560), runways, and 3-4 letter codes (airports, fixes) other than common acronyms.
AIRCRAFT = r"\b[A-Z]{1,3}-?\d{2,4}[A-Z]?(?:-\d{1,3}[A-Z]?)?\b"
RUNWAY = r"\b(?:Runway|RWY|Rwy)\s?\d{1,2}[LRC]?\b"
CODE = r"\b[A-Z]{3,4}\b"
NOT_CODES = frozenset(
    "ATC CRM EGT KTS TCAS ATIS DME FMS FMC MCP PIC SID IFR VFR IMC VMC ILS RNAV GPS VOR NDB APU "
    "MEL FAA NAS ARTCC TRACON TWR GND CTAF UNICOM NOTAM ETA ETD ATP CFI FBO PAX FAR LOC MSA MDA "
    "TOD ACARS EICAS ECAM CAS AOA PFD MFD NAV COM ADS NTSB RVR NMAC IAF FAF ZZZ ERAM ICAO ATR DPS "
    "STAR ODP".split())


def _run_config(run: Path) -> dict:
    """The resolved config saved with the run."""
    return yaml.safe_load((run / "config.yaml").read_text())


def _chunk_units(run: Path, reports: pd.DataFrame, **chunking) -> Units:
    """The chunks the run indexed (or another chunking setting, if given)."""
    cfg = _run_config(run)
    ccfg = ChunkConfig(**{**cfg.get("chunking", {}), **chunking})
    return Units.from_chunks(load_chunks(ccfg, cfg.get("data_dir", "data/processed"), reports), ccfg.tag)


def synopsis_overlap(reports: pd.DataFrame, run: Path) -> dict:
    """Share of a synopsis's content stems that occur in its own vs. a random narrative."""
    qs = synopsis_track(reports, **_run_config(run)["evaluation"]["synopsis"])
    analyze, narrative = Analyzer("english"), reports.set_index("acn")["narrative"]
    others = np.random.default_rng(0).choice(reports["acn"].to_numpy(), len(qs.queries))
    own, rand = [], []
    for acn, other, synopsis in zip(qs.queries["acn"], others, qs.queries["text"]):
        stems = set(analyze(synopsis))
        if stems:
            own.append(len(stems & set(analyze(narrative[acn]))) / len(stems))
            rand.append(len(stems & set(analyze(narrative[other]))) / len(stems))
    return {"own_narrative": float(np.mean(own)), "random_narrative": float(np.mean(rand))}


def identifier_split(reports: pd.DataFrame, run: Path) -> dict:
    """Synopsis MRR@10 with and without an identifier repeated from the narrative."""
    qs = synopsis_track(reports, **_run_config(run)["evaluation"]["synopsis"])
    narrative = reports.set_index("acn")["narrative"]

    def repeats(acn: str, synopsis: str) -> bool:
        ids = set(re.findall(AIRCRAFT, synopsis)) | set(re.findall(RUNWAY, synopsis))
        ids |= {t for t in re.findall(CODE, synopsis) if t not in NOT_CODES}
        return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(i)}(?![A-Za-z0-9])", narrative[acn])
                   for i in ids)

    flag = {f"S{a}": repeats(a, s) for a, s in zip(qs.queries["acn"], qs.queries["text"])}
    pq = pd.read_csv(run / "per_query_synopsis.csv")
    pq = pq[pq["labels"] == "auto"].assign(repeats=lambda d: d["qid"].map(flag))
    mrr = pq.groupby(["repeats", "system"])["MRR@10"].mean().unstack(0)
    return {"share_repeating_identifier": float(np.mean(list(flag.values()))),
            "MRR@10_with": mrr[True].to_dict(), "MRR@10_without": mrr[False].to_dict()}


def silver_label_stats(reports: pd.DataFrame) -> dict:
    """Size of the silver grade-2 sets, and how many rules use synopsis keywords."""
    spec = yaml.safe_load(Path(QUERIES).read_text())["queries"]
    qrels = topical_track(reports, QUERIES).qrels["silver"]
    n2 = [sum(g == 2 for g in qrels[q["qid"]].values()) for q in spec]
    keyword = sum(any("synopsis" in c for c in q["silver"].get("2", [])) for q in spec)
    return {"n_queries": len(spec), "grade2_rule_uses_synopsis": keyword,
            "grade2_per_query_median": float(np.median(n2)), "grade2_per_query_min": int(min(n2))}


def contrast_overlap() -> dict:
    """Word Jaccard of the paraphrase (controlled anchor) with the original and the edit."""
    items = load_contrast("data/contrast/contrast_v1.jsonl")

    def jaccard(a: str, b: str) -> float:
        a, b = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
        return len(a & b) / len(a | b)

    items["with_original"] = [jaccard(p, o) for p, o in zip(items["paraphrase"], items["original"])]
    items["with_changed"] = [jaccard(p, c) for p, c in zip(items["paraphrase"], items["changed"])]
    cols = ["with_original", "with_changed"]
    table = items.groupby("category")[cols].mean()
    table.loc["all"] = items[cols].mean()
    return table.round(3).to_dict("index")


def short_hits(run: Path, reports: pd.DataFrame, systems=("e5-small", "minilm"),
               **chunking) -> dict:
    """How often short units are top-10 matched passages on topical queries, and how relevant."""
    units = _chunk_units(run, reports, **chunking)
    short = pd.Series([len(t.split()) < SHORT_WORDS for t in units.texts], index=units.unit_ids)
    qrels = topical_track(reports, QUERIES).qrels["silver"]
    out: dict = {"index_share_short": float(short.mean())}
    for s in systems:
        run_df = read_run(run / "runs" / "topical" / f"{s}.tsv")
        ranked = run_df.sort_values("rank").groupby("qid")["acn"].apply(list).to_dict()
        top = run_df[run_df["rank"] <= 10]
        top = top.assign(short=top["unit_id"].map(short),
                         grade2=[qrels[q].get(a, 0) == 2 for q, a in zip(top["qid"], top["acn"])])
        rate = top.groupby("short")["grade2"].mean()
        out[s] = {"top10_share_short": float(top["short"].mean()),
                  "grade2_rate_short": float(rate[True]), "grade2_rate_other": float(rate[False]),
                  # P@10 gain if every short hit were as relevant as the system's other hits
                  "P@10_gain_bound": float(top["short"].mean() * (rate[False] - rate[True])),
                  "nDCG@10": float(evaluate_run(ranked, qrels, ["nDCG@10"])["nDCG@10"].mean())}
    return out


def e5_score_spread(run: Path, reports: pd.DataFrame, system: str = "e5-small") -> dict:
    """Per-query z-scores of E5 similarities for short units vs. all units (topical queries)."""
    cfg = _run_config(run)
    spec = next(s for s in cfg["systems"] if s["name"] == system)
    units = _chunk_units(run, reports)
    dense = DenseRetriever(system, device="cpu", cache_dir=cfg.get("cache_dir", "cache"), **spec["params"])
    dense.index(units)  # loads the cached embeddings
    queries = [q["text"] for q in yaml.safe_load(Path(QUERIES).read_text())["queries"]]
    scores = dense.score_units(queries)
    short = np.array([len(t.split()) < SHORT_WORDS for t in units.texts])
    z = (scores - scores.mean(1, keepdims=True)) / scores.std(1, keepdims=True)
    return {"mean_z_short": float(z[:, short].mean()),
            "score_sd_short": float(scores[:, short].std(1).mean()),
            "score_sd_other": float(scores[:, ~short].std(1).mean())}


def whole_report_truncation(reports: pd.DataFrame, config: Path) -> dict:
    """Share of reports longer than each whole-report encoder's input limit."""
    from transformers import AutoTokenizer

    out = {}
    for spec in yaml.safe_load(config.read_text())["systems"]:
        p = spec.get("params", {})
        if spec["type"] != "dense" or p.get("doc_mode") != "whole":
            continue
        tok = AutoTokenizer.from_pretrained(p["model"])
        texts = [p.get("passage_prefix", "") + t for t in reports["narrative"]]
        lens = np.array([len(ids) for ids in tok(texts, verbose=False)["input_ids"]])
        limit = p["max_seq_length"]
        out[spec["name"]] = {"max_seq_length": limit, "share_truncated": float((lens > limit).mean()),
                             "share_tokens_kept": float(np.minimum(lens, limit).sum() / lens.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--v1", type=Path, help="results folder of the pre-processing v1 run")
    ap.add_argument("--whole-config", type=Path, default=Path("configs/extended.yaml"))
    a = ap.parse_args()
    reports = load_reports(_run_config(a.run).get("data_dir", "data/processed"))
    out = {
        "synopsis_overlap": synopsis_overlap(reports, a.run),
        "identifier_split": identifier_split(reports, a.run),
        "silver_labels": silver_label_stats(reports),
        "contrast_overlap": contrast_overlap(),
        "short_hits": short_hits(a.run, reports),
        "e5_score_spread": e5_score_spread(a.run, reports),
        "whole_report_truncation": whole_report_truncation(reports, a.whole_config),
    }
    if a.v1:
        # v1 predates tail merging, so its config has no min_tail_words (default 40)
        out["short_hits_v1"] = short_hits(a.v1, reports, min_tail_words=0)
    (a.run / "analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

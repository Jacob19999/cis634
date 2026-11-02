"""Run every configured system on every configured track and write a results folder.

results/<run_name>/
    config.yaml                    resolved config
    meta.json                      system descriptions, index/query timings, environment
    metrics_<track>.csv            one row per (system, label source): mean metrics + 95% CIs
    per_query_<track>.csv          per-query metrics (for significance tests / error analysis)
    significance_<track>.csv       paired randomization tests on the primary metric
    contrast_items.csv             per-item scores for the meaning-change test
    metrics_contrast.csv           accuracy per system x category (+ Wilson CIs)
    summary.md                     human-readable tables
    runs/<track>/<system>.tsv      TREC run files with matched passage ids (not committed)
"""

from __future__ import annotations

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .data import ChunkConfig, load_chunks, load_reports
from .metrics import bootstrap_ci, evaluate_run, paired_randomization_test, wilson_ci
from .retrievers import Retriever, Units, build_systems
from .tracks import QuerySet, load_contrast, synopsis_track, topical_track

PRIMARY_METRIC = {"synopsis": "MRR@10", "topical": "nDCG@10"}


def load_config(path: str | Path) -> dict:
    """Read a YAML config and fill in defaults."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("data_dir", "data/processed")
    cfg.setdefault("cache_dir", "cache")
    cfg.setdefault("results_dir", "results")
    cfg.setdefault("device", "auto")
    cfg.setdefault("chunking", {})
    cfg.setdefault("depth", 100)
    return cfg


def _unit_sets(cfg: dict, reports: pd.DataFrame) -> dict[str, Units]:
    """Build only the unit sets (chunks and/or whole reports) that some system needs."""
    modes = {s.get("params", {}).get("doc_mode", "chunk_max") for s in cfg["systems"]
             if s["type"] != "hybrid"}
    sets = {}
    if "chunk_max" in modes:
        ccfg = ChunkConfig(**cfg["chunking"])
        sets["chunk_max"] = Units.from_chunks(load_chunks(ccfg, cfg["data_dir"], reports), ccfg.tag)
    if "whole" in modes:
        sets["whole"] = Units.from_reports(reports)
    return sets


def _query_sets(cfg: dict, reports: pd.DataFrame, only: list[str] | None) -> dict[str, QuerySet]:
    """Build the retrieval tracks named in the config (synopsis, topical)."""
    ev = cfg.get("evaluation", {})
    out = {}
    if "synopsis" in ev and (not only or "synopsis" in only):
        out["synopsis"] = synopsis_track(reports, **ev["synopsis"])
    if "topical" in ev and (not only or "topical" in only):
        t = ev["topical"]
        out["topical"] = topical_track(reports, t["queries"], t.get("human_qrels"), t.get("splits"))
    return out


def _write_run(path: Path, qids: list[str], results, system: str) -> None:
    """Save rankings as a TREC run file, with the matched passage id as a last column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, res in zip(qids, results):
            for rank, (d, s, u) in enumerate(zip(res.doc_ids, res.scores, res.best_units), 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{s:.6f}\t{system}\t{u}\n")


def read_run(path: str | Path) -> pd.DataFrame:
    """Load a run file written by ``_write_run``."""
    return pd.read_csv(path, sep="\t", header=None, dtype={0: str, 2: str},
                       names=["qid", "q0", "acn", "rank", "score", "system", "unit_id"])


def _summarise(per_query: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean of each metric with a 95% bootstrap CI, per system and label source."""
    rows = []
    for (system, source), g in per_query.groupby(["system", "labels"], sort=False):
        row = {"system": system, "labels": source, "n_queries": len(g)}
        for m in metrics:
            if g[m].isna().all():
                continue
            vals = g[m].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals)
            row[m] = np.nanmean(vals)
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def _significance(per_query: pd.DataFrame, metric: str) -> pd.DataFrame:
    """p-value for every pair of systems on one metric (paired randomization test)."""
    rows = []
    for source, g in per_query.groupby("labels", sort=False):
        wide = g.pivot(index="qid", columns="system", values=metric)
        for a, b in itertools.combinations(wide.columns, 2):
            rows.append({"labels": source, "metric": metric, "system_a": a, "system_b": b,
                         "mean_a": wide[a].mean(), "mean_b": wide[b].mean(),
                         "diff": wide[a].mean() - wide[b].mean(),
                         "p_value": paired_randomization_test(wide[a].to_numpy(), wide[b].to_numpy())})
    return pd.DataFrame(rows)


def run_retrieval_track(name: str, qs: QuerySet, systems: dict[str, Retriever], out: Path,
                        depth: int, timings: dict) -> pd.DataFrame:
    """Search every query with every system, score the rankings and save the tables."""
    texts, qids = qs.queries["text"].tolist(), qs.queries["qid"].tolist()
    frames = []
    for sname, system in systems.items():
        t0 = time.time()
        results = system.search(texts, k=depth)
        timings.setdefault(sname, {})[f"query_seconds_{name}"] = round(time.time() - t0, 2)
        _write_run(out / "runs" / name / f"{sname}.tsv", qids, results, sname)
        run = {q: list(r.doc_ids) for q, r in zip(qids, results)}
        for source, qrels in qs.qrels.items():
            # judged@10 only matters for incomplete (human) judgments
            metrics = [m for m in qs.metrics if m != "judged@10" or source.startswith("human")]
            pq = evaluate_run(run, qrels, metrics)
            pq.insert(0, "labels", source)
            pq.insert(0, "system", sname)
            frames.append(pq.rename_axis("qid").reset_index())
        print(f"  [{name}] {sname}: {len(qids)} queries in {timings[sname][f'query_seconds_{name}']}s", flush=True)
    per_query = pd.concat(frames, ignore_index=True)
    if "facet" in qs.queries:
        per_query = per_query.merge(qs.queries[["qid", "facet", "split"]], on="qid", how="left")
    per_query.to_csv(out / f"per_query_{name}.csv", index=False)
    summary = _summarise(per_query, [m for m in qs.metrics if m in per_query])
    summary.to_csv(out / f"metrics_{name}.csv", index=False)
    _significance(per_query, PRIMARY_METRIC[name]).to_csv(out / f"significance_{name}.csv", index=False)
    return summary


CONTRAST_CONDITIONS = {
    # The proposal's test: the changed sentence is a minimal edit of the anchor, so word
    # overlap favours it; passing requires meaning to outweigh surface similarity.
    "original_anchor": ("original", "paraphrase", "changed"),
    # Controlled test: original and changed are equally far (lexically) from the anchor,
    # so only the meaning-bearing edit separates them.
    "paraphrase_anchor": ("paraphrase", "original", "changed"),
}


def run_contrast(path: str, systems: dict[str, Retriever], out: Path) -> pd.DataFrame:
    """Meaning-change test: an item passes if the same-meaning sentence scores higher."""
    items = load_contrast(path)
    rows, frames = [], []
    for sname, system in systems.items():
        if not system.supports_pairs:
            continue
        for cond, (anchor, pos, neg) in CONTRAST_CONDITIONS.items():
            s_pos = system.pair_scores(items[anchor].tolist(), items[pos].tolist())
            s_neg = system.pair_scores(items[anchor].tolist(), items[neg].tolist())
            # 1 = pass, 0 = fail, 0.5 = tie.
            passed = np.where(s_pos > s_neg, 1.0, np.where(np.isclose(s_pos, s_neg), 0.5, 0.0))
            frames.append(items[["id", "category"]].assign(
                system=sname, condition=cond, score_same_meaning=s_pos, score_changed=s_neg,
                margin=s_pos - s_neg, passed=passed))
            for cat, g in [("all", None), *items.groupby("category")]:
                p = passed if g is None else passed[g.index]
                lo, hi = wilson_ci(p.sum(), len(p))
                rows.append({"system": sname, "condition": cond, "category": cat, "n": len(p),
                             "accuracy": p.mean(), "acc_lo": lo, "acc_hi": hi,
                             "ties": int((p == 0.5).sum())})
    pd.concat(frames).to_csv(out / "contrast_items.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "metrics_contrast.csv", index=False)
    return summary


def _fmt_table(df: pd.DataFrame, metrics: list[str], ci: bool = True) -> str:
    """Markdown table; adds the CI as [lo, hi] when available."""
    cols = [c for c in ("system", "labels", "n_queries") if c in df.columns]
    head = "| " + " | ".join(cols + metrics) + " |"
    sep = "|" + "---|" * (len(cols) + len(metrics))
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = [str(r[c]) for c in cols]
        for m in metrics:
            v = r[m]
            if ci and f"{m}_lo" in r and not np.isnan(r[f"{m}_lo"]):
                cells.append(f"{v:.3f} [{r[f'{m}_lo']:.3f}, {r[f'{m}_hi']:.3f}]")
            else:
                cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(out: Path, run_name: str, tables: dict, meta: dict) -> None:
    """Write summary.md, the readable tables of one run."""
    md = [f"# Benchmark run `{run_name}`", "",
          f"Corpus: {meta['n_reports']:,} reports, {meta['n_units']} retrieval units. "
          f"Device: {meta['device']}. Means with 95% bootstrap CIs.", ""]
    if "synopsis" in tables:
        md += ["## Synopsis known-item track", "",
               "Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).", "",
               _fmt_table(tables["synopsis"], ["MRR@10", "R@1", "R@10", "R@100"]), ""]
    if "topical" in tables:
        md += ["## Topical track", "",
               "P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.", "",
               _fmt_table(tables["topical"], ["nDCG@10", "P@10", "P@10_lenient", "MRR@10"]), ""]
    if "contrast" in tables:
        md += ["## Meaning-change (contrast) test", "",
               "Accuracy = share of items where the anchor is more similar to the same-meaning "
               "sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.",
               "`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.",
               "`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.", ""]
        for cond, c in tables["contrast"].groupby("condition", sort=False):
            wide = c.pivot(index="system", columns="category", values="accuracy")
            wide = wide.reindex(c["system"].unique())
            wide = wide[["all"] + [k for k in wide.columns if k != "all"]].reset_index()
            md += [f"### {cond}", "", _fmt_table(wide, [k for k in wide.columns if k != "system"], ci=False), ""]
    md += ["## Timings (seconds)", "", "```", json.dumps(meta["timings"], indent=1), "```", ""]
    (out / "summary.md").write_text("\n".join(md))


def run_benchmark(config_path: str, run_name: str | None = None, tracks: list[str] | None = None,
                  only_systems: list[str] | None = None) -> Path:
    """``asrsbench run``: index every system, run every track, write results/<run>/."""
    cfg = load_config(config_path)
    run_name = run_name or cfg.get("name") or Path(config_path).stem
    out = Path(cfg["results_dir"]) / run_name
    out.mkdir(parents=True, exist_ok=True)
    if only_systems:
        cfg["systems"] = [s for s in cfg["systems"] if s["name"] in only_systems]
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Load the corpus and the units (chunks and/or whole reports) the systems index.
    reports = load_reports(cfg["data_dir"])
    unit_sets = _unit_sets(cfg, reports)

    timings: dict = {}
    t0 = time.time()
    systems = {}
    for spec in cfg["systems"]:  # build one at a time to time each
        t = time.time()
        systems = build_systems([spec], unit_sets, cfg["device"], cfg["cache_dir"], systems)
        timings[spec["name"]] = {"index_seconds": round(time.time() - t, 2)}
        print(f"indexed {spec['name']} in {timings[spec['name']]['index_seconds']}s", flush=True)
    timings["_total_index_seconds"] = round(time.time() - t0, 2)

    # Run the retrieval tracks, then the meaning-change test.
    tables = {}
    for name, qs in _query_sets(cfg, reports, tracks).items():
        tables[name] = run_retrieval_track(name, qs, systems, out, cfg["depth"], timings)
    contrast = cfg.get("evaluation", {}).get("contrast")
    if contrast and (not tracks or "contrast" in tracks):
        tables["contrast"] = run_contrast(contrast["path"], systems, out)

    # Record the setup and timings, then write the readable summary.
    meta = {
        "asrsbench_version": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "n_reports": len(reports),
        "n_units": {k: len(v) for k, v in unit_sets.items()},
        "device": next((s.device for s in systems.values() if hasattr(s, "device")), "cpu"),
        "systems": {n: s.describe() for n, s in systems.items()}, "timings": timings,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    write_summary(out, run_name, tables, meta)
    return out


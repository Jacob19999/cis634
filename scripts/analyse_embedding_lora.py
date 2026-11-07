"""Compare every completed LoRA epoch with frozen 8B on development retrieval.

Tables and qualitative examples are generated from saved rankings. The primary checkpoint
selection criterion is ASRS held-out MRR@10, fixed before training; all regressions remain
visible. P-values after checkpoint selection are descriptive, not confirmatory tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from asrsbench.metrics import bootstrap_ci, paired_randomization_test


def main() -> None:
    """Generate the comparison report directly from machine-written evaluation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_8b.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["results"])
    labels = ["frozen", "epoch1", "epoch2", "epoch3"]
    metas = {label: json.loads((root / label / "metrics.json").read_text()) for label in labels}
    comparisons, lines = [], ["# 8B aviation LoRA: development comparison", "",
        "Three epochs; the frozen model is the unchanged baseline. All queries are development "
        "or validation queries; the official ASRS test and contrast sets are untouched.", "",
        "| Condition | Track | Queries | MRR@10 | nDCG@10 | R@100 |",
        "|---|---|---:|---:|---:|---:|"]
    for label, meta in metas.items():
        for track, values in meta["metrics"].items():
            lines.append(f"| {label} | {track} | {values['n_queries']} | "
                         f"{values['MRR@10']:.4f} | {values['nDCG@10']:.4f} | {values['R@100']:.4f} |")
            if label == "frozen":
                continue
            baseline = pd.read_csv(root / "frozen" / f"{track}_per_query.csv", index_col="qid")
            adapted = pd.read_csv(root / label / f"{track}_per_query.csv", index_col="qid")
            if set(baseline.index) != set(adapted.index):
                raise ValueError("evaluation query ids differ across conditions")
            adapted = adapted.reindex(baseline.index)
            for metric in ("MRR@10", "nDCG@10", "R@100"):
                a, b = adapted[metric].to_numpy(), baseline[metric].to_numpy()
                delta = a - b
                low, high = bootstrap_ci(delta, seed=634)
                comparisons.append({"condition": label, "track": track, "metric": metric,
                                    "mean_delta": float(np.mean(delta)), "delta_ci95": [low, high],
                                    "paired_randomisation_p": paired_randomization_test(a, b, seed=634)})
    best = max(labels[1:], key=lambda label: metas[label]["metrics"]["synopsis_dev"]["MRR@10"])
    beat_frozen = (metas[best]["metrics"]["synopsis_dev"]["MRR@10"] >
                   metas["frozen"]["metrics"]["synopsis_dev"]["MRR@10"])
    selection = {"best_adapted_epoch": best, "primary_track": "synopsis_dev",
                 "primary_metric": "MRR@10", "beats_frozen_primary": beat_frozen,
                 "selected_checkpoint": metas[best]["model"], "automatic_ui_swap": False}
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (root / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    pd.DataFrame([{**r, "delta_ci95": json.dumps(r["delta_ci95"])} for r in comparisons]).to_csv(
        root / "comparison.csv", index=False)
    lines += ["", f"Best adapted checkpoint by the prespecified criterion: **{best}**.",
              f"Beats frozen on the primary metric: **{beat_frozen}**.", "",
              "| Track | Metric | Best-epoch change | Paired bootstrap 95% CI | Descriptive p |",
              "|---|---|---:|---:|---:|"]
    for result in comparisons:
        if result["condition"] == best:
            lo, hi = result["delta_ci95"]
            lines.append(f"| {result['track']} | {result['metric']} | {result['mean_delta']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | {result['paired_randomisation_p']:.4g} |")
    baseline = pd.read_csv(root / "frozen/synopsis_dev_per_query.csv", index_col="qid")
    adapted = pd.read_csv(root / best / "synopsis_dev_per_query.csv", index_col="qid")
    delta = (adapted["MRR@10"] - baseline["MRR@10"]).sort_values()
    qs = {q["qid"]: q for q in json.loads((root / best / "runs/synopsis_dev_queries.json").read_text())}
    frozen_runs = json.loads((root / "frozen/runs/synopsis_dev_runs.json").read_text())
    adapted_runs = json.loads((root / best / "runs/synopsis_dev_runs.json").read_text())
    examples = [{**qs[qid], "delta_MRR@10": float(delta[qid]),
                 "frozen_top10": frozen_runs[qid][:10], "adapted_top10": adapted_runs[qid][:10]}
                for qid in dict.fromkeys(list(delta.head(10).index) + list(delta.tail(10).index))]
    (root / "qualitative_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    lines += ["", "The comparison JSON/CSV includes every epoch and any regression. "
              "Qualitative examples list the ten largest improvements and ten largest declines. "
              "NTSB cause retrieval is a known-item task, not a validated measure of cross-source "
              "analogue quality. Topical labels are silver. No UI embeddings were replaced."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

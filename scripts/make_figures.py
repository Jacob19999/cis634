"""Figures and analysis tables for the milestone report.

usage: python scripts/make_figures.py results/milestone [--out reports/milestone/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (reference palette slots 1-3, validated all-pairs, light mode).
SYSTEM_COLORS = {"bm25": "#2a78d6", "minilm": "#eb6834", "e5-small": "#1baf7a"}
LABELS = {"bm25": "BM25", "minilm": "MiniLM", "e5-small": "E5-small"}
CATEGORIES = [("negation", "Negation"), ("altitude", "Numbers"), ("direction", "Direction"),
              ("event_order", "Event order"), ("all", "All")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def contrast_figure(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "metrics_contrast.csv")
    df = df[df["system"].isin(SYSTEM_COLORS)]
    conds = [("original_anchor", "(a) Anchor = original (proposal)"),
             ("paraphrase_anchor", "(b) Anchor = paraphrase (overlap controlled)")]
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    n_sys, width = len(SYSTEM_COLORS), 0.26
    for ax, (cond, title) in zip(axes, conds):
        d = df[df["condition"] == cond]
        for i, (sys_, color) in enumerate(SYSTEM_COLORS.items()):
            vals = [d[(d["system"] == sys_) & (d["category"] == c)]["accuracy"].mean() for c, _ in CATEGORIES]
            xs = [j + (i - (n_sys - 1) / 2) * width for j in range(len(CATEGORIES))]
            ax.bar(xs, vals, width=width - 0.03, color=color, label=LABELS[sys_], zorder=2)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center",
                        va="bottom", fontsize=5.5, color=INK)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1, label="Chance")
        ax.set_xticks(range(len(CATEGORIES)), [lbl for _, lbl in CATEGORIES])
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.5, ncol=4, handlelength=1.4, columnspacing=1)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(out / "contrast.pdf")
    fig.savefig(out / "contrast.png", dpi=200)
    plt.close(fig)


def facet_table(run: Path, out: Path) -> pd.DataFrame:
    pq = pd.read_csv(run / "per_query_topical.csv")
    pq = pq[pq["labels"] == "silver"]
    t = pq.pivot_table(index="facet", columns="system", values="nDCG@10", aggfunc="mean")
    t["n"] = pq.groupby("facet")["qid"].nunique()
    t.round(3).to_csv(out / "topical_by_facet.csv")
    s = pq.pivot_table(index="split", columns="system", values="nDCG@10", aggfunc="mean")
    s.round(3).to_csv(out / "topical_by_split.csv")
    return t


def contrast_failures(run: Path, out: Path, system: str = "e5-small") -> pd.DataFrame:
    items = pd.read_csv(run / "contrast_items.csv")
    src = pd.read_json("data/contrast/contrast_v1.jsonl", lines=True)
    f = items[(items["system"] == system) & (items["condition"] == "paraphrase_anchor")
              & (items["passed"] == 0)].merge(src, on=["id", "category"])
    f = f.sort_values("margin")[["id", "category", "original", "paraphrase", "changed", "margin"]]
    f.to_csv(out / f"contrast_failures_{system}.csv", index=False)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/milestone/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    contrast_figure(a.run, a.out)
    print(facet_table(a.run, a.out).round(3).to_string())
    print(contrast_failures(a.run, a.out).head(12).to_string())


if __name__ == "__main__":
    main()

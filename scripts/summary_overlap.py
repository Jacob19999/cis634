"""Did the summariser reproduce NASA synopses it could not have read? A memorisation check.

usage: python scripts/summary_overlap.py data/summaries/qwen3-8b_v1.jsonl [--out results/summary_overlap.json]

The summariser saw only the narrative, but the ASRS database is public, so the model may
have memorised analyst synopses in pre-training. If it did, its summary of a report would
share wording with *that report's* synopsis that the narrative does not contain. For each
report we count synopsis word 5-grams that occur in the summary but not in the narrative
("novel" overlap). Shuffled pairs (report i's summary and narrative against report j's
synopsis) measure how much of that is stock phrasing ("the flight crew reported"). A rate
for matched pairs well above the shuffled rate points to memorisation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from asrsbench.data import load_reports, load_summaries

N = 5


def grams(text: str, n: int = N) -> set[tuple[str, ...]]:
    """Lower-cased word n-grams."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def novel(summary: str, narrative: str, synopsis: str) -> int:
    """Synopsis n-grams in the summary that the narrative does not contain."""
    return len((grams(synopsis) & grams(summary)) - grams(narrative))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries")
    ap.add_argument("--out", default="results/summary_overlap.json")
    args = ap.parse_args()
    reports = load_reports()
    summ = load_summaries(args.summaries)
    r = reports[reports["acn"].isin(summ)].reset_index(drop=True)
    s = [summ[a] for a in r["acn"]]
    nar, syn = r["narrative"].tolist(), r["synopsis"].tolist()
    perm = np.random.default_rng(0).permutation(len(r))
    matched = np.array([novel(s[i], nar[i], syn[i]) for i in range(len(r))])
    shuffled = np.array([novel(s[i], nar[i], syn[j]) for i, j in enumerate(perm)])
    copied = np.array([len(grams(syn[i]) & grams(nar[i])) / max(1, len(grams(syn[i])))
                       for i in range(len(r))])
    res = {
        "summaries": args.summaries, "n_reports": int(len(r)), "ngram": N,
        "matched": {"share_with_novel": float((matched > 0).mean()),
                    "mean_novel": float(matched.mean()),
                    "share_with_3plus_novel": float((matched >= 3).mean())},
        "shuffled": {"share_with_novel": float((shuffled > 0).mean()),
                     "mean_novel": float(shuffled.mean()),
                     "share_with_3plus_novel": float((shuffled >= 3).mean())},
        "synopsis_ngrams_found_in_narrative": float(copied.mean()),
        "by_split": {sp: {"share_with_novel": float((matched[r["split"] == sp] > 0).mean())}
                     for sp in sorted(r["split"].unique())},
        "examples": [{"acn": r["acn"][i], "novel": int(matched[i]), "synopsis": syn[i],
                      "summary": s[i]} for i in np.argsort(-matched)[:5]],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()

"""Did the summariser reproduce NASA synopses it could not have read? A memorisation check.

usage: python scripts/summary_overlap.py data/summaries/qwen3-8b_v1.jsonl [--run results/summaries]
                                        [--out results/summary_overlap.json]

The summariser saw only the narrative, but the ASRS database is public, so the model may
have memorised analyst synopses in pre-training. If it did, its summary of a report would
share wording with *that report's* synopsis that the narrative does not contain. For each
report we count synopsis word 5-grams that occur in the summary but not in the narrative
("novel" overlap). Two baselines pair report i's summary and narrative with another report
j's synopsis: a random j (shuffled), and the j whose narrative is most similar to i's
(TF-IDF cosine, other incidents only). The second measures stock phrasing for the same kind
of event ("declared an emergency and diverted"), which a model writes without having seen
any synopsis. Matched-pair overlap well above the similar-report baseline points to
memorisation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from asrsbench.data import load_reports, load_summaries

N = 5


def grams(text: str, n: int = N) -> set[tuple[str, ...]]:
    """Lower-cased word n-grams."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def novel(summary: str, narrative: str, synopsis: str) -> int:
    """Synopsis n-grams in the summary that the narrative does not contain."""
    return len((grams(synopsis) & grams(summary)) - grams(narrative))


def nearest_other_incident(narratives: list[str], groups: np.ndarray, block: int = 2000) -> np.ndarray:
    """Index of each report's most similar narrative from a different incident group."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, max_features=200_000,
                        stop_words="english").fit_transform(narratives).tocsr()
    out = np.empty(X.shape[0], dtype=np.int64)
    for s in range(0, X.shape[0], block):
        S = (X[s : s + block] @ X.T).toarray()
        S[groups[s : s + block, None] == groups[None, :]] = -1  # itself and its siblings
        out[s : s + block] = S.argmax(1)
    return out


def summarise(counts: np.ndarray) -> dict:
    """Share of reports with any / 3+ novel synopsis n-grams, and the mean count."""
    return {"share_with_novel": float((counts > 0).mean()), "mean_novel": float(counts.mean()),
            "share_with_3plus_novel": float((counts >= 3).mean())}


def gain_without_overlap(run: Path, reports, summ: dict[str, str], n_perm: int = 10000) -> dict:
    """Synopsis-track MRR@10 gain of each ``<system>-chunk-sum`` over ``<system>``, on all
    queries and on the queries whose summary has no novel overlap with their synopsis.

    If memorisation drove the gain, it would shrink on the overlap-free queries.
    ``run`` is a results folder of a config with both variants (e.g. results/summaries).
    """
    import yaml

    from asrsbench.tracks import synopsis_track

    cfg = yaml.safe_load((run / "config.yaml").read_text())
    qs = synopsis_track(reports, **cfg["evaluation"]["synopsis"]).queries
    r = reports.set_index("acn")
    qs["novel"] = [novel(summ[a], r.loc[a, "narrative"], r.loc[a, "synopsis"]) for a in qs["acn"]]
    pq = pd.read_csv(run / "per_query_synopsis.csv")
    w = pq.pivot_table(index="qid", columns="system", values="MRR@10").join(qs.set_index("qid")["novel"])
    rng = np.random.default_rng(0)
    out = {"n_queries": int(len(w)), "n_with_overlap": int((w["novel"] > 0).sum())}
    for b in [c for c in w.columns if c.endswith("-chunk-sum")]:
        a = b[: -len("-chunk-sum")]
        if a not in w:
            continue
        d = (w[b] - w[a]).to_numpy()
        clean = d[w["novel"].to_numpy() == 0]
        # Paired sign-flip randomisation test on the overlap-free queries.
        flips = np.abs((clean * rng.choice([-1, 1], (n_perm, len(clean)))).mean(1))
        out[b] = {"gain_all": float(d.mean()), "gain_overlap_free": float(clean.mean()),
                  "p_overlap_free": float((flips >= abs(clean.mean())).mean()),
                  "gain_with_overlap": float(d[w["novel"].to_numpy() > 0].mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries")
    ap.add_argument("--out", default="results/summary_overlap.json")
    ap.add_argument("--run", type=Path, default=None,
                    help="results folder with <system> and <system>-chunk-sum (e.g. results/summaries)")
    args = ap.parse_args()
    reports = load_reports()
    summ = load_summaries(args.summaries)
    r = reports[reports["acn"].isin(summ)].reset_index(drop=True)
    s = [summ[a] for a in r["acn"]]
    nar, syn = r["narrative"].tolist(), r["synopsis"].tolist()
    perm = np.random.default_rng(0).permutation(len(r))
    matched = np.array([novel(s[i], nar[i], syn[i]) for i in range(len(r))])
    shuffled = np.array([novel(s[i], nar[i], syn[j]) for i, j in enumerate(perm)])
    near = nearest_other_incident(nar, r["group_id"].to_numpy())
    similar = np.array([novel(s[i], nar[i], syn[j]) for i, j in enumerate(near)])
    copied = np.array([len(grams(syn[i]) & grams(nar[i])) / max(1, len(grams(syn[i])))
                       for i in range(len(r))])
    res = {
        "summaries": args.summaries, "n_reports": int(len(r)), "ngram": N,
        "matched": summarise(matched),
        "similar_report": summarise(similar),
        "shuffled": summarise(shuffled),
        "synopsis_ngrams_found_in_narrative": float(copied.mean()),
        "by_split": {sp: {"share_with_novel": float((matched[r["split"] == sp] > 0).mean())}
                     for sp in sorted(r["split"].unique())},
        "examples": [{"acn": r["acn"][i], "novel": int(matched[i]), "synopsis": syn[i],
                      "summary": s[i]} for i in np.argsort(-matched)[:5]],
    }
    if args.run:
        res["gain_without_overlap"] = {"run": str(args.run),
                                       **gain_without_overlap(args.run, reports, summ)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()

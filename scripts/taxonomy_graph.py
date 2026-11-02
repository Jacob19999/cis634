"""Does a knowledge graph over the ASRS taxonomy help BM25 on the synopsis track?

usage: python scripts/taxonomy_graph.py [--out results/taxonomy_graph.json]

Graph: each indexed report links to its analyst anomaly codes (ASRS metadata). A query
links to code nodes through a classifier, and BM25's top 100 is reranked by how well each
candidate's code set fits the query's predicted codes:

    score = z(BM25) + lam * z(log P(candidate's exact code set | query))

* ``oracle``: the query links to exactly the target report's codes. This is an upper
  bound, since real linking is noisy.
* ``realistic``: TF-IDF + logistic regression trained on train-split synopses, minus 1,000
  held out as tuning queries for lam. The 1,000 benchmark validation queries are scored once.

Synopsis track only. The topical silver labels are derived from these same codes, so a
code-based graph would be graded by its own inputs there.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from asrsbench.data import ChunkConfig, load_chunks, load_reports
from asrsbench.metrics import paired_randomization_test
from asrsbench.retrievers import BM25Retriever, Units
from asrsbench.tracks import synopsis_track

LAMS = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
ORACLE_LAMS = LAMS + [5.0, 10.0, 100.0]  # large lam ~ hard filter on the code set


def z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def mrr10(ranked: list[str], relevant: dict) -> float:
    return next((1 / (r + 1) for r, d in enumerate(ranked[:10]) if d in relevant), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/taxonomy_graph.json")
    args = ap.parse_args()

    reports = load_reports()
    row = {a: i for i, a in enumerate(reports["acn"])}
    codes = reports["anomaly"].map(lambda s: sorted({x.strip() for x in s.split(";") if x.strip()}))
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(codes)
    Y = Y[:, Y.sum(0) >= 20]  # drop very rare codes

    def loglik(P: np.ndarray, rows: list[int]) -> np.ndarray:
        """log P(each candidate's exact code set), treating codes as independent."""
        P = np.clip(P, 1e-4, 1 - 1e-4)
        return (Y[rows] * np.log(P) + (1 - Y[rows]) * np.log(1 - P)).sum(1)

    # Query -> code linker, trained on train-split synopses (not the tuning ones).
    eval_qs = synopsis_track(reports, split="validation", n_queries=1000, seed=13)
    tune_qs = synopsis_track(reports, split="train", n_queries=1000, seed=7)
    train = ((reports["split"] == "train") & ~reports["acn"].isin(tune_qs.queries["acn"])).to_numpy()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200_000)
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=200, solver="liblinear"))
    clf.fit(vec.fit_transform(reports.loc[train, "synopsis"]), Y[train])

    def rerank(qs, results, P, lams):
        """MRR@10 per query for each lam; P holds the query's code probabilities."""
        out = {lam: [] for lam in lams}
        for i, (qid, res) in enumerate(zip(qs.queries["qid"], results)):
            docs, bm = list(res.doc_ids), z(res.scores.astype(float))
            ll = z(loglik(P[i], [row[d] for d in docs]))
            for lam in lams:
                order = np.argsort(-(bm + lam * ll), kind="stable")
                out[lam].append(mrr10([docs[j] for j in order], qs.qrels["auto"][qid]))
        return {lam: np.array(v) for lam, v in out.items()}

    P_eval = clf.predict_proba(vec.transform(eval_qs.queries["text"]))
    P_tune = clf.predict_proba(vec.transform(tune_qs.queries["text"]))
    # Oracle linking: probability 1 for the target's codes, 0 otherwise (clipped in loglik).
    P_oracle = Y[[row[a] for a in eval_qs.queries["acn"]]].astype(float)

    summary: dict = {}
    ccfg = ChunkConfig()
    for mode in ("chunk_max", "whole"):
        units = (Units.from_chunks(load_chunks(ccfg, "data/processed", reports), ccfg.tag)
                 if mode == "chunk_max" else Units.from_reports(reports))
        bm25 = BM25Retriever(stopwords="english", doc_mode=mode)
        bm25.index(units)
        res_eval = bm25.search(eval_qs.queries["text"].tolist(), k=100)
        res_tune = bm25.search(tune_qs.queries["text"].tolist(), k=100)

        tune = rerank(tune_qs, res_tune, P_tune, LAMS)
        lam = max(LAMS, key=lambda l: tune[l].mean())
        real = rerank(eval_qs, res_eval, P_eval, [0, lam])
        oracle = rerank(eval_qs, res_eval, P_oracle, ORACLE_LAMS)
        lam_o = max(ORACLE_LAMS, key=lambda l: oracle[l].mean())  # tuned on eval: a ceiling only
        summary[f"bm25-{mode}"] = {
            "baseline_mrr10": real[0].mean(),
            "realistic": {"lam": lam, "mrr10": real[lam].mean(),
                          "p_value": paired_randomization_test(real[lam], real[0]),
                          "tuning_mrr10_by_lam": {str(l): tune[l].mean() for l in LAMS}},
            "oracle_ceiling": {"lam": lam_o, "mrr10": oracle[lam_o].mean()},
        }

    pred, gold = P_eval >= 0.5, P_oracle.astype(bool)
    summary["linking"] = {
        "n_codes": int(Y.shape[1]),
        "exact_code_set_accuracy": float((pred == gold).all(1).mean()),
        "micro_f1": float(2 * (pred & gold).sum() / (pred.sum() + gold.sum())),
    }
    text = json.dumps(summary, indent=2, default=float)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()

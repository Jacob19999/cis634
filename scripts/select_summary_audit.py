"""Pick a subset of LLM report summaries for a human faithfulness audit.

usage: python scripts/select_summary_audit.py data/summaries/qwen3.5-27b-q6k_v1.jsonl \
           [--data-dir data/processed_v2] [--per-source 40] [--per-flag 8] [--seed 0] \
           [--out data/audit/qwen3.5-27b-q6k_v1]

Two strata, drawn without overlap:

* ``random``: ``--per-source`` reports per source (NTSB, ASRSN, ASRS), each spread evenly
  over narrative-length tertiles. Its error rate estimates the whole run's.
* ``flag:<name>``: up to ``--per-flag`` reports for each automatic check that a summary
  fails (cut off at the token cap, too long, numbers or acronyms the narrative lacks,
  wording copied from the synopsis / NTSB probable cause the model never saw, ...). These
  find failure modes; they are not a rate estimate.

Writes ``<out>/sheet.csv`` for the auditors (audit id, source, narrative, summary and blank
judgement columns, shuffled, with no stratum or flag so that judges are not primed) and
``<out>/key.csv`` (audit id -> doc id, stratum, every flag the report raised), plus
``<out>/flags.json`` with how often each check fired over the whole run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from asrsbench.data import load_reports

MAX_WORDS = 1500  # summarise_reports.MAX_WORDS: longer narratives were cut before summarising
CJK = re.compile(r"[　-鿿가-힯＀-￯]")
# Talk about the task rather than the event. "The reporter states ..." is a normal opening.
META = re.compile(r"<\/?think>|^(here is|here's|summary:|in summary|this narrative|"
                  r"the (following|above) narrative)|\bas an ai\b|\bnarrative (does not|doesn't)",
                  re.I)
# A sentence end: [.!?] then a capital, not after an initial (U.S., N.)
# or ft./Ft./St./No./vs./approx.
SENT_END = re.compile(r"(?<![A-Z])(?<!\b[fF]t)(?<!\bSt)(?<!\bNo)(?<!\bvs)(?<!pprox)[.!?]\s+(?=[A-Z])")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
ACRO = re.compile(r"\b[A-Z][A-Z0-9]{1,}[a-z]?\b")


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Lower-cased word n-grams (as in ``scripts/summary_overlap.py``)."""
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def numbers(text: str) -> set[str]:
    """Numbers with thousands separators removed; FL350 and 35,000 stay distinct on purpose,
    so a converted flight level shows up as a flag for the auditor to clear."""
    return {m.replace(",", "").rstrip(".") for m in NUM.findall(text)}


def flags(summary: str, narrative: str, synopsis: str) -> list[str]:
    """Names of the automatic checks that ``summary`` fails."""
    out = []
    n_words = len(summary.split())
    if not summary:
        return ["empty"]
    if summary[-1] not in '.!?"\')':
        out.append("cut_off")  # likely hit the 96-token cap mid-sentence
    if n_words > 55:
        out.append("over_50_words")
    if len(SENT_END.findall(summary)) + 1 > 2:
        out.append("over_2_sentences")
    if n_words < 12:
        out.append("very_short")
    if META.search(summary):
        out.append("meta_text")
    if CJK.search(summary):
        out.append("non_latin_script")
    if numbers(summary) - numbers(narrative):
        out.append("number_not_in_narrative")
    # Substring test, so "MD" in "MD-11" matches a narrative's "MD11".
    nar_squashed = re.sub(r"[^a-z0-9]", "", narrative.lower())
    if any(a.lower() not in nar_squashed for a in ACRO.findall(summary)):
        out.append("acronym_not_in_narrative")
    novel = (grams(synopsis) & grams(summary)) - grams(narrative) if synopsis else set()
    if len(novel) >= 2:
        out.append("synopsis_wording")  # possible memorisation of the hidden target
    if len(narrative.split()) > MAX_WORDS:
        out.append("narrative_truncated")
    if len(narrative.split()) < 40:
        out.append("short_narrative")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries")
    ap.add_argument("--data-dir", default="data/processed_v2")
    ap.add_argument("--per-source", type=int, default=40)
    ap.add_argument("--per-flag", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = Path(args.summaries)
    out = Path(args.out or f"data/audit/{path.stem}")
    rng = np.random.default_rng(args.seed)

    # Last good record per report wins; failed attempts carry "error" and no summary.
    summ: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("summary"):
            summ[rec["acn"]] = rec["summary"]
    r = load_reports(args.data_dir)
    r = r[r["acn"].isin(summ)].copy()
    r["summary"] = r["acn"].map(summ)
    r["src"] = r["acn"].str.split(":").str[0]
    r["checks"] = [flags(s, n, y or "") for s, n, y in zip(r["summary"], r["narrative"],
                                                           r["synopsis"].fillna(""))]

    # Duplicate summaries across different incidents: boilerplate or a crossed request.
    dup = r.groupby("summary")["group_id"].transform("nunique") > 1
    r.loc[dup, "checks"] = r.loc[dup, "checks"].map(lambda f: f + ["duplicate_summary"])

    picked: dict[str, str] = {}
    for src, g in r.groupby("src"):
        tert = pd.qcut(g["n_words"].rank(method="first"), 3, labels=False)
        k = args.per_source
        for t in range(3):
            pool = g.index[tert == t]
            take = k // 3 + (t < k % 3)
            for i in rng.choice(pool, min(take, len(pool)), replace=False):
                picked[i] = "random"
    counts = pd.Series([f for fs in r["checks"] for f in fs]).value_counts()
    for flag in counts.sort_values().index:  # rarest first, so rare modes are not crowded out
        pool = [i for i in r.index[r["checks"].map(lambda f: flag in f)] if i not in picked]
        for i in rng.permutation(pool)[: args.per_flag]:
            picked[i] = f"flag:{flag}"

    a = r.loc[list(picked)].assign(stratum=[picked[i] for i in picked])
    a = a.iloc[rng.permutation(len(a))].reset_index(drop=True)
    a["audit_id"] = [f"A{j:03d}" for j in range(1, len(a) + 1)]
    out.mkdir(parents=True, exist_ok=True)
    sheet = a[["audit_id", "src", "display_id", "narrative", "summary"]].assign(
        faithful="", hallucinated_detail="", missing_key_fact="", over_length="", notes="")
    sheet.to_csv(out / "sheet.csv", index=False)
    a[["audit_id", "acn", "display_id", "src", "stratum", "n_words"]].assign(
        checks=a["checks"].map(";".join)).to_csv(out / "key.csv", index=False)
    stats = {"summaries": len(r), "by_source": r["src"].value_counts().to_dict(),
             "flag_counts": counts.to_dict(),
             "flag_share": (counts / len(r)).round(4).to_dict(),
             "flag_counts_by_source": {s: pd.Series([f for fs in g["checks"] for f in fs])
                                       .value_counts().to_dict() for s, g in r.groupby("src")},
             "audit_size": len(a), "audit_strata": a["stratum"].value_counts().to_dict()}
    (out / "flags.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

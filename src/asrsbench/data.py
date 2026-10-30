"""Download, clean and chunk the ASRS corpus.

Pipeline (``asrsbench prepare``):

1. Load all three splits of ``elihoole/asrs-aviation-reports`` from the Hugging Face Hub.
2. Keep the narrative(s), the analyst synopsis and the analyst coding fields we use for
   query design and silver relevance labels.
3. Normalise text. The ASRS export replaced every comma with a semicolon
   ("10;000 FT", "IAF; we set"), which corrupts numbers -- one of our contrast categories.
   Editorial placeholders standing in for an empty second narrative are dropped.
4. Group records that describe the same incident (shared accession numbers or identical
   narratives) with union-find, so an incident is never split across train/test.
5. Split each narrative into overlapping, sentence-aligned chunks that fit the encoders'
   context windows (narratives contain no paragraph breaks).

Outputs go to ``data/processed/``: ``reports.parquet`` (one row per report),
``chunks.parquet`` (one row per chunk) and ``stats.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HF_DATASET = "elihoole/asrs-aviation-reports"

# Raw column -> our column name.
COLUMNS = {
    "acn_num_ACN": "acn",
    "Time_Date": "date",
    "Aircraft 1.2_Make Model Name": "aircraft",
    "Aircraft 1.9_Flight Phase": "flight_phase",
    "Person 1.3_Function": "reporter_function",
    "Person 1.7_Human Factors": "human_factors",
    "Person 1.8_Communication Breakdown": "comm_breakdown_1",
    "Person 2.8_Communication Breakdown": "comm_breakdown_2",
    "Person 1.10_ASRS Report Number.Accession Number": "accession_1",
    "Person 2.10_ASRS Report Number.Accession Number": "accession_2",
    "Events_Anomaly": "anomaly",
    "Events.3_Detector": "detector",
    "Events.5_Result": "result",
    "Assessments_Contributing Factors / Situations": "contributing_factors",
    "Assessments.1_Primary Problem": "primary_problem",
    "Report 1_Narrative": "narrative_1",
    "Report 2_Narrative": "narrative_2",
    "Report 1.2_Synopsis": "synopsis",
}

# Anomaly prefixes that define the proposal's focus area (altitude, clearance, communication).
FOCUS_PATTERNS = [
    r"Deviation - Altitude",
    r"Procedural Clearance",
    r"Deviation - Track / Heading",
]


# --------------------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_NUM_SEMI = re.compile(r"(?<=\d);(?=\d{3}\b)")  # 10;000 -> 10,000
_SEMI_SPACE = re.compile(r";(?=\s)")  # "IAF; we" -> "IAF, we"


def normalize_text(text: str | None) -> str:
    """Undo the ASRS comma->semicolon export artefact and collapse whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = _NUM_SEMI.sub(",", text)
    text = _SEMI_SPACE.sub(",", text)
    return _WS.sub(" ", text).strip()


# ASRS editors replace an uninformative second narrative with a bracketed note such as
# "[Report narrative contained no additional information.]". These are boilerplate, not
# reporter text, and dense models match generic queries to them.
_PLACEHOLDER = re.compile(
    r"no (substantive )?(additional|new) (information|info|details)|"
    r"no additional information (was )?(provided|included|contained)",
    re.IGNORECASE,
)


_SELF_REFERENCE = re.compile(r"^\[|narr?ative|report", re.IGNORECASE)


def is_placeholder(text: str, max_words: int = 25) -> bool:
    """Short editorial note about the narrative itself (bracketed or naming the report)."""
    return (bool(text) and len(text.split()) <= max_words and bool(_PLACEHOLDER.search(text))
            and bool(_SELF_REFERENCE.search(text)))


def _clean_id(x) -> str:
    """Accession numbers arrive as '1574727.0', '1574727' or ''."""
    s = str(x).strip()
    if s in ("", "nan", "None"):
        return ""
    return s[:-2] if s.endswith(".0") else s


# --------------------------------------------------------------------------------------
# Incident grouping
# --------------------------------------------------------------------------------------


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic root: smallest id.
            lo, hi = sorted([ra, rb])
            self.parent[hi] = lo


def assign_incident_groups(df: pd.DataFrame) -> pd.Series:
    """Group reports that share an accession number or an identical narrative.

    Returns a Series of group ids (the smallest ACN in the group).
    """
    uf = _UnionFind()
    for acn, a1, a2 in zip(df["acn"], df["accession_1"], df["accession_2"]):
        uf.find(acn)
        for other in (a1, a2):
            if other:
                uf.union(acn, other)
    # Identical narratives (re-submissions) also count as the same incident.
    digests = df["narrative_1"].map(lambda t: hashlib.md5(t.lower().encode()).hexdigest())
    for _, acns in df.groupby(digests)["acn"]:
        acns = list(acns)
        for other in acns[1:]:
            uf.union(acns[0], other)
    return df["acn"].map(uf.find)


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------

# Sentence boundary: terminal punctuation followed by space and an uppercase letter,
# digit or bracket. ASRS uses many abbreviations ("FT.", "approx."), so we are
# conservative: a boundary only counts if the next token looks like a sentence start.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\[\"'(A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


@dataclass(frozen=True)
class ChunkConfig:
    max_words: int = 150
    overlap_sentences: int = 1
    # A leftover tail shorter than this is absorbed into the previous chunk instead of
    # becoming its own fragment (short fragments attract spuriously high dense scores
    # under max-over-chunks aggregation). 0 disables merging (v1 behaviour).
    min_tail_words: int = 40

    @property
    def tag(self) -> str:
        tag = f"w{self.max_words}o{self.overlap_sentences}"
        return tag + (f"m{self.min_tail_words}" if self.min_tail_words else "")


def chunk_text(text: str, cfg: ChunkConfig) -> list[str]:
    """Greedily pack sentences into windows of at most ``cfg.max_words`` words.

    Consecutive windows share ``cfg.overlap_sentences`` sentences so an event that spans a
    boundary is still visible in one chunk. A single sentence longer than the window is
    hard-split on words. If fewer than ``cfg.min_tail_words`` words would remain after a
    window, they are appended to it (so a chunk holds at most max_words + min_tail_words).
    """
    sents: list[list[str]] = []
    for s in split_sentences(text):
        words = s.split()
        for i in range(0, len(words), cfg.max_words):
            sents.append(words[i : i + cfg.max_words])
    if not sents:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(sents):
        end, n = start, 0
        while end < len(sents) and (n + len(sents[end]) <= cfg.max_words or end == start):
            n += len(sents[end])
            end += 1
        if 0 < sum(len(s) for s in sents[end:]) < cfg.min_tail_words:
            end = len(sents)
        chunks.append(" ".join(w for s in sents[start:end] for w in s))
        if end >= len(sents):
            break
        start = max(end - cfg.overlap_sentences, start + 1)
    return chunks


def build_chunks(reports: pd.DataFrame, cfg: ChunkConfig) -> pd.DataFrame:
    """One row per chunk; chunks of a report are contiguous and in reading order."""
    rows = []
    for acn, n1, n2 in zip(reports["acn"], reports["narrative_1"], reports["narrative_2"]):
        for nidx, narr in ((1, n1), (2, n2)):
            for cidx, text in enumerate(chunk_text(narr, cfg)):
                rows.append((f"{acn}_{nidx}_{cidx}", acn, nidx, cidx, text))
    return pd.DataFrame(rows, columns=["chunk_id", "acn", "narrative_idx", "chunk_idx", "text"])


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def load_raw(cache_dir: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, cache_dir=cache_dir)
    frames = []
    for split in ("train", "validation", "test"):
        d = ds[split].to_pandas()
        d["split"] = split
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def clean_reports(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[list(COLUMNS) + ["split"]].rename(columns=COLUMNS).copy()
    df["acn"] = df["acn"].map(_clean_id)
    for c in ("accession_1", "accession_2"):
        df[c] = df[c].map(_clean_id)
    free_text = ("narrative_1", "narrative_2", "synopsis")
    coded = [c for c in df.columns
             if c not in free_text and c not in ("acn", "accession_1", "accession_2", "split")]
    for c in free_text:
        df[c] = df[c].map(normalize_text)
    df["narrative_2_placeholder"] = df["narrative_2"].map(is_placeholder)
    df.loc[df["narrative_2_placeholder"], "narrative_2"] = ""
    # Coded fields use "; " as a list separator, so only tidy whitespace there.
    for c in coded:
        df[c] = df[c].fillna("").astype(str).map(lambda s: _WS.sub(" ", s).strip())

    df["comm_breakdown"] = (df["comm_breakdown_1"] + "; " + df["comm_breakdown_2"]).str.strip("; ")
    df = df.drop(columns=["comm_breakdown_1", "comm_breakdown_2"])

    # The narrative we search: both reporters' accounts.
    df["narrative"] = (df["narrative_1"] + " " + df["narrative_2"]).str.strip()
    df["n_words"] = df["narrative"].str.split().str.len()

    df["group_id"] = assign_incident_groups(df)
    focus_re = re.compile("|".join(FOCUS_PATTERNS))
    df["focus"] = df["anomaly"].str.contains(focus_re) | (df["comm_breakdown"] != "")

    df = df[df["narrative"] != ""].sort_values("acn").reset_index(drop=True)
    return df


def corpus_stats(reports: pd.DataFrame, chunks: pd.DataFrame, cfg: ChunkConfig) -> dict:
    g = reports.groupby("group_id")
    multi = g.filter(lambda x: len(x) > 1)
    crossing = multi.groupby("group_id")["split"].nunique()
    cpd = chunks.groupby("acn").size()
    return {
        "n_reports": int(len(reports)),
        "split_sizes": reports["split"].value_counts().to_dict(),
        "n_incident_groups": int(reports["group_id"].nunique()),
        "n_reports_in_multi_report_groups": int(len(multi)),
        "n_groups_crossing_splits": int((crossing > 1).sum()),
        "share_with_second_narrative": round(float((reports["narrative_2"] != "").mean()), 4),
        "n_placeholder_second_narratives_removed": int(reports["narrative_2_placeholder"].sum()),
        "words_per_report": {
            "mean": round(float(reports["n_words"].mean()), 1),
            "median": float(reports["n_words"].median()),
            "p95": float(reports["n_words"].quantile(0.95)),
            "max": int(reports["n_words"].max()),
            "total": int(reports["n_words"].sum()),
        },
        "synopsis_words_mean": round(float(reports["synopsis"].str.split().str.len().mean()), 1),
        "date_range": [str(reports["date"].min()), str(reports["date"].max())],
        "n_focus_reports": int(reports["focus"].sum()),
        "chunking": {"max_words": cfg.max_words, "overlap_sentences": cfg.overlap_sentences,
                     "min_tail_words": cfg.min_tail_words},
        "n_chunks": int(len(chunks)),
        "chunks_per_report": {
            "mean": round(float(cpd.mean()), 2),
            "median": float(cpd.median()),
            "max": int(cpd.max()),
        },
        "words_per_chunk_mean": round(float(chunks["text"].str.split().str.len().mean()), 1),
        "share_chunks_under_30_words": round(float((chunks["text"].str.split().str.len() < 30).mean()), 4),
    }


def prepare(out_dir: str | Path = "data/processed", cfg: ChunkConfig = ChunkConfig(),
            cache_dir: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = clean_reports(load_raw(cache_dir))
    chunks = build_chunks(reports, cfg)
    reports.to_parquet(out / "reports.parquet", index=False)
    chunks.to_parquet(out / f"chunks_{cfg.tag}.parquet", index=False)
    stats = corpus_stats(reports, chunks, cfg)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_reports(data_dir: str | Path = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "reports.parquet")


def load_chunks(cfg: ChunkConfig, data_dir: str | Path = "data/processed",
                reports: pd.DataFrame | None = None) -> pd.DataFrame:
    path = Path(data_dir) / f"chunks_{cfg.tag}.parquet"
    if not path.exists():
        reports = load_reports(data_dir) if reports is None else reports
        build_chunks(reports, cfg).to_parquet(path, index=False)
    return pd.read_parquet(path)

"""Build FAA reference glossaries in ``data/glossary/`` from two public FAA documents.

usage: python scripts/build_faa_glossary.py [--pdf-dir data/external/faa]

Writes ``faa_v1.tsv`` (every abbreviation the two documents define, a reference list) and
``pcg_v1.jsonl`` (every Pilot/Controller Glossary entry with its definition, the context
that ``scripts/rewrite_queries.py`` gives the LLM). The acronyms the retrievers actually
expand are the curated ``asrs_acronyms_v1.tsv``: matched blindly against ASRS narratives,
``faa_v1.tsv`` gives wrong senses for many frequent ones (AGL = "great lakes region",
SFO = "simulated flameout", CA = "clear above").

Sources (US government works, downloaded 2026-11-05; checksums in ``SOURCES``):

* Pilot/Controller Glossary (PCG), basic with changes 1-3, dated 2026-07-09. Headwords that
  carry an abbreviation, e.g. ``TRAFFIC ALERT AND COLLISION AVOIDANCE SYSTEM (TCAS)−``, and
  abbreviation headwords that point to a term, e.g. ``RA− (See RESOLUTION ADVISORY.)``.
* FAA Order JO 7340.2R Contractions, chapter 2 section 1 (decode), dated 2026-10-29. Only
  usage ``GEN`` and ``ATC``: the weather (NWS, METAR) and ICAO-only senses would expand a
  narrative's "RA" (resolution advisory) as "rain".

The PCG wins when both define an abbreviation. A contraction with two or more GEN/ATC
meanings and no PCG entry is dropped as ambiguous rather than guessed. Abbreviations that are
English stopwords ("AS", "ON", "NO") are dropped because narratives capitalise them for
emphasis. Nothing here reads the ASRS reports or the evaluation queries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PCG = "PCG_Bsc_w_Chg_1_2_and_3_dtd_7-9-26.pdf"
CNT = "7340.2R_CNT_Bsc_dtd_10-29-26_Final.pdf"
SOURCES = {
    PCG: ("https://www.faa.gov/air_traffic/publications/media/PCG_Bsc_w_Chg_1_2_and_3_dtd_7-9-26.pdf",
          "b30eb2d5cac01f51949c13a41c92facf002edcf94ca8fafac5051497d6dd81e5"),
    CNT: ("https://www.faa.gov/documentLibrary/media/Order/7340.2R_CNT_Bsc_dtd_10-29-26_Final.pdf",
          "c4b0a552bdd0441d237367a048d164c9d949c916aefdf5164493aeb4588f2c1f"),
}
CNT_DECODE_PAGES = (19, 57)  # chapter 2 section 1, pages 2-1-1 to 2-1-39
CNT_USAGE = ("GEN", "ATC", "ICAO")  # in order of preference
ABBR = re.compile(r"[A-Z][A-Z0-9]{1,7}")
DASH = "[−–-]"  # the PCG uses U+2212 after headwords


def pdftext(pdf: Path, *args: str) -> str:
    """Text of a PDF via poppler's ``pdftotext -layout``."""
    return subprocess.run(["pdftotext", "-layout", *args, str(pdf), "-"], check=True,
                          capture_output=True, text=True).stdout


def check_sources(pdf_dir: Path) -> None:
    """Fail unless both PDFs are present and match the recorded checksums."""
    for name, (url, sha) in SOURCES.items():
        path = pdf_dir / name
        if not path.exists():
            raise SystemExit(f"missing {path}; download it from {url}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{path} does not match the recorded sha256 (a different edition?)")


def clean_term(s: str) -> str:
    """Lower-case a glossary term and tidy its spacing."""
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s.lower()


def parse_pcg(text: str) -> dict[str, str]:
    """Abbreviation -> term from the Pilot/Controller Glossary."""
    out: dict[str, str] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # TERM (ABBR)− definition
        m = re.match(rf"^([A-Z][A-Z0-9 /,&'.\-]+?) \(([A-Z][A-Z0-9]{{1,7}})\){DASH}", line)
        if m and ABBR.fullmatch(m.group(2)):
            out.setdefault(m.group(2), clean_term(m.group(1)))
            continue
        # ABBR−  followed by "(See TERM.)" on the same or the next line
        m = re.match(rf"^([A-Z][A-Z0-9]{{1,7}}){DASH}\s*(.*)$", line)
        if m:
            rest = m.group(2) or (lines[i + 1] if i + 1 < len(lines) else "")
            see = re.match(r"^\s*\(See ([A-Z][A-Z0-9 /,&'\-]+?)\.?\)", rest)
            if see:
                out.setdefault(m.group(1), clean_term(see.group(1)))
    return out


def pcg_entries(text: str) -> list[dict[str, str]]:
    """Every PCG headword with its definition, page headers and footers removed.

    Pure cross-references ("RA− (See RESOLUTION ADVISORY.)") are dropped: the entry they
    point to is in the list.
    """
    noise = re.compile(r"Pilot/Controller Glossary|^\s*PCG [A-Z]\S*\s*$|^\s*\d+/\d+/\d+\s*$")
    head = re.compile(rf"^([A-Z][A-Z0-9 /,&'.()\[\]“”\-]*[A-Z0-9)\]])\s*{DASH}\s*(.*)$")
    entries: list[list[str]] = []
    started = False
    for line in text.splitlines():
        if noise.search(line) or not line.strip():
            continue
        m = head.match(line)
        if m and m.group(1).upper() == m.group(1):
            started = True
            entries.append([m.group(1).strip(), m.group(2).strip()])
        elif started and entries:
            entries[-1][1] += " " + line.strip()
    out = []
    for term, definition in entries:
        definition = re.sub(r"\s+", " ", definition).strip()
        body = re.sub(r"\(See [^)]*\)\.?", "", definition).strip()
        if body:
            out.append({"term": term, "definition": definition})
    return out


def parse_cnt(pdf: Path) -> dict[str, dict[str, set[str]]]:
    """Abbreviation -> usage -> meanings from the JO 7340.2 decode table (two columns a page)."""
    meanings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    first, last = CNT_DECODE_PAGES
    row = re.compile(r"^(\S+)\s{2,}(\S.*?)\s{2,}([A-Z]{2,6})\s*$")
    cont = re.compile(r"^\s{6,}(\S.*?)\s*$")
    for page in range(first, last + 1):
        for x in (0, 306):  # left and right column of a letter-size page
            text = pdftext(pdf, "-f", str(page), "-l", str(page), "-x", str(x), "-y", "0",
                           "-W", "306", "-H", "792")
            entries: list[list[str]] = []
            for line in text.splitlines():
                m = row.match(line)
                if m and m.group(1) != "Contraction":
                    entries.append([m.group(1), m.group(2), m.group(3)])
                elif entries and (c := cont.match(line)) and not re.search(r"\d+/\d+/\d+", line):
                    entries[-1][1] += " " + c.group(1)  # wrapped definition
            for abbr, meaning, usage in entries:
                if usage in CNT_USAGE and ABBR.fullmatch(abbr):
                    meanings[abbr][usage].add(clean_term(meaning))
    return meanings


def build(pdf_dir: Path) -> list[tuple[str, str, str]]:
    """Merged (abbreviation, expansion, source) rows, sorted by abbreviation."""
    pcg = parse_pcg(pdftext(pdf_dir / PCG))
    cnt = parse_cnt(pdf_dir / CNT)
    rows: dict[str, tuple[str, str]] = {a: (t, "pcg") for a, t in pcg.items()}
    n_ambiguous = 0
    for abbr, by_usage in cnt.items():
        if abbr in rows:
            continue
        # The first usage tier that defines the abbreviation decides; two meanings there
        # make it ambiguous.
        ms = next(by_usage[u] for u in CNT_USAGE if by_usage.get(u))
        if len(ms) == 1:
            rows[abbr] = (next(iter(ms)), "jo7340")
        else:
            n_ambiguous += 1
    drop = {a for a in rows if a.lower() in ENGLISH_STOP_WORDS
            or rows[a][0].lower() == a.lower()}  # a "contraction" that is the word itself
    for a in drop:
        del rows[a]
    print(f"PCG {len(pcg)}, JO 7340.2 GEN/ATC {len(cnt)} ({n_ambiguous} ambiguous, dropped), "
          f"{len(drop)} stopwords / self-expansions dropped -> {len(rows)} entries")
    return sorted((a, t, s) for a, (t, s) in rows.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/external/faa")
    ap.add_argument("--out", default="data/glossary/faa_v1.tsv")
    args = ap.parse_args()
    pdf_dir = Path(args.pdf_dir)
    check_sources(pdf_dir)
    rows = build(pdf_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["abbr", "expansion", "source"])
        w.writerows(rows)
    print(f"wrote {out}")
    entries = pcg_entries(pdftext(pdf_dir / PCG))
    pcg_out = out.parent / "pcg_v1.jsonl"
    with pcg_out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {pcg_out}: {len(entries)} glossary entries")


if __name__ == "__main__":
    main()

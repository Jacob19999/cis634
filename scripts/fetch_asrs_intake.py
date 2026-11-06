"""Monthly ASRS report intake, from NASA's CALLBACK newsletter -> data/asrs/callback_intake_v1.json.

usage: python scripts/fetch_asrs_intake.py [--first 460 --last 561]

Every CALLBACK issue (asrs.arc.nasa.gov/publications/callback/cb_<n>.html) prints the number
of reports ASRS received in one month ("August 2024 Report Intake ... TOTAL 10,917"). ASRS
numbers reports (ACNs) in order of receipt, so cumulative intake dates an ACN;
``scripts/build_corpus_v2.py`` uses it for reports that come without a date.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from pathlib import Path

MONTH = (r"(?:January|February|March|April|May|June|July|August|September|October|November|"
         r"December) 20\d\d")


def intake(issue: int) -> tuple[str, int] | None:
    """(month, total reports) from one CALLBACK issue, or None if it has no intake box."""
    req = urllib.request.Request(
        f"https://asrs.arc.nasa.gov/publications/callback/cb_{issue}.html",
        headers={"User-Agent": "Mozilla/5.0 (asrsbench research script)"})
    t = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    m = re.search(rf"({MONTH}) Report Intake(.*?)TOTAL ([\d,]+)", t)
    return (m.group(1), int(m.group(3).replace(",", ""))) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=460)
    ap.add_argument("--last", type=int, default=561)
    ap.add_argument("--out", default="data/asrs/callback_intake_v1.json")
    args = ap.parse_args()
    out = {}
    for n in range(args.first, args.last + 1):
        r = intake(n)
        if r:
            out[r[0]] = r[1]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=0))
    print(f"{len(out)} months, {min(out, key=lambda k: k)} ...")


if __name__ == "__main__":
    main()

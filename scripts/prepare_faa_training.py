"""Snapshot official FAA/AIM and CFR passages for the approved 5% training stratum.

Downloads are cached with URLs, SHA-256 digests and retrieval dates. The CFR input is
the 2026 annual edition from GovInfo, not an assertion about rules at a historical event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from asrsbench.data import ChunkConfig, chunk_text
from asrsbench.finetuning import bucket, digest, write_jsonl

AIM = "https://www.faa.gov/air_traffic/publications/atpubs/aim_html/"
CFR = ("https://www.govinfo.gov/content/pkg/CFR-2026-title14-vol2/xml/"
       "CFR-2026-title14-vol2-part91.xml")


def snapshot(url: str, path: Path) -> dict:
    """Download once and return a source record for the bytes actually parsed."""
    if not path.exists():
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "asrsbench-research/0.1"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    content = response.read()
                if len(content) < 100:
                    raise ValueError(f"empty source: {url}")
                path.write_bytes(content)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    return {"url": url, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "retrieved_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def aim_sections(html: str) -> list[tuple[str, str, str]]:
    """Extract numbered AIM paragraphs, excluding navigation and repeated headers."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for heading in soup.select("main h4.paragraph-title"):
        parts = []
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) in ("h2", "h3", "h4"):
                break
            text = sibling.get_text(" ", strip=True) if hasattr(sibling, "get_text") else str(sibling)
            parts.append(text)
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if len(text.split()) >= 30:
            out.append((heading.get("id", ""), heading.get_text(" ", strip=True), text))
    return out


def cfr_sections(xml: str) -> list[tuple[str, str, str]]:
    """Extract operational Part 91 sections from the annual official XML edition."""
    root = ET.fromstring(xml)
    out = []
    for sec in root.findall(".//SECTION"):
        number = " ".join(sec.findtext("SECTNO", "").split())
        m = re.search(r"91\.(\d+)", number)
        if not m or not (3 <= int(m[1]) <= 25 or 101 <= int(m[1]) <= 227 or
                         501 <= int(m[1]) <= 535):
            continue
        title = " ".join(sec.findtext("SUBJECT", "").split())
        body = " ".join(" ".join(p.itertext()) for p in sec.findall("P"))
        body = re.sub(r"\s+", " ", body).strip()
        if len(body.split()) >= 30:
            out.append((number.replace("§", "").strip(), title, body))
    return out


def main() -> None:
    """Write source snapshots and section-grouped, versioned narrative-sized passages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/external/finetune_faa_v1")
    args = parser.parse_args()
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [snapshot(AIM + "index.html", out / "aim_index.html")]
    index = BeautifulSoup((out / "aim_index.html").read_text(), "html.parser")
    links = sorted({a["href"].lstrip("./") for a in index.select("a[href]")
                    if re.fullmatch(r"(?:\./)?chap[1-7]_section_\d+\.html", a["href"])})
    sections = []
    for link in links:
        url = urljoin(AIM, link)
        path = out / link
        sources.append(snapshot(url, path))
        sections.extend(("AIM", number, title, text, url, "2026-07-09")
                        for number, title, text in aim_sections(path.read_text()))
    sources.append(snapshot(CFR, out / "part91_2026.xml"))
    sections.extend(("CFR91", number, title, text, CFR, "2026 annual edition")
                    for number, title, text in cfr_sections((out / "part91_2026.xml").read_text()))
    # The existing PCG extraction includes a malformed preamble entry: discard it and
    # very short cross-references rather than teaching extraction noise to the model.
    for line in Path("data/glossary/pcg_v1.jsonl").read_text().splitlines():
        entry = json.loads(line)
        text = entry["definition"]
        if len(text.split()) < 30 or any(s in text for s in ("Terms Modified:", "PCG−1 A")):
            continue
        sections.append(("PCG", entry["term"], entry["term"], text,
                         "https://www.faa.gov/air_traffic/publications/atpubs/pcg_html/",
                         "2026-07-09"))
    passages = []
    for family, number, title, text, url, edition in sections:
        group = f"{family}:{number}"
        split = "dev" if bucket(group) < 100 else "train"
        for i, passage in enumerate(chunk_text(text, ChunkConfig())):
            if len(passage.split()) < 30:
                continue
            passages.append({"task_id": digest(f"{group}:{i}:{passage}")[:24], "family": family,
                             "group_id": group, "doc_id": group, "split": split,
                             "title": title, "positive": passage, "kind": "faa",
                             "source_url": url, "edition": edition})
    if len(passages) < 500 or not all(any(p["family"] == f for p in passages)
                                     for f in ("AIM", "CFR91", "PCG")):
        raise RuntimeError("FAA extraction incomplete; refusing a partial training stratum")
    write_jsonl(out / "passages.jsonl", passages)
    manifest = {"sources": sources, "sections": len(sections), "passages": len(passages),
                "by_family": {f: sum(p["family"] == f for p in passages)
                              for f in ("AIM", "CFR91", "PCG")}}
    (out / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()

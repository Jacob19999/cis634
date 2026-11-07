"""Build the Precedent Graph page's data for corpus v2: ASRS and NTSB reports, 2010–2026.

usage: python scripts/export_precedent_graph_v2.py <triaggent_consulting>/portfolio/asrs-precedent-graph
           [--config configs/corpus_v2.yaml]
           [--graph data/graph/qwen3-8b_v1_corpus_v2.jsonl]
           [--summaries data/summaries/qwen3.5-27b-q6k_v1.jsonl]
           [--without-graph]

Same page model as ``scripts/export_precedent_graph.py`` (whose helpers it reuses), extended
to three sources (``src``): 0 = ASRS with analyst codes (2012–2022), 1 = NTSB (2010–2026),
2 = ASRS narrative only (2021–2025, dates estimated from the ACN, ``de`` = 1).

* Factors: ASRS analyst codes as before; NTSB coded fields (occurrences, findings, phase,
  light and weather, injury and damage, aircraft, FAR part, airport) mapped onto the same
  layers (``src: ntsb``); the newer ASRS reports' primary-problem label; and LLM factors from
  the Qwen3-8B knowledge graph for every report.
* ``nar``: narrative similarity, Qwen3-Embedding-8B (mean of each report's chunk vectors,
  from the cache ``asrsbench`` wrote for ``configs/corpus_v2.yaml``) fused with TF-IDF LSA.
  ``seq``: similarity of the LLM's cause/effect triples (MPNet, as before).
* The map is openTSNE on the Qwen3-8B report vectors; regions are HDBSCAN clusters.
* ``synopsis`` is the ASRS synopsis, the NTSB probable cause, or (newer ASRS) the Qwen3.5-27B
  summary; narrative shards hold ``{"n": narrative, "s": Qwen3.5-27B summary}``.
  ``synopsis_kind`` marks generated fallbacks, including NTSB events without a probable cause.
* ``summary`` is searchable for every source; ``cross`` keeps six other-source narrative
  neighbours by cosine, separately from the calibrated ``nar`` scores. ASRS and ASRSN
  count as one reporting system. Neither kind of similarity establishes incident identity.
* ``--without-graph`` exports before LLM graph extraction, without loading MPNet. Narrative
  embeddings must already exist; this exporter never starts an uncached corpus encoding.

Writes ``data/graph.json`` (meta and factors), ``data/reports/<k>.json`` (reports in shards of
``REPORT_SHARD``, so no file nears GitHub's 100 MB limit) and ``data/narratives/<k>.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from asrsbench.benchmark import _unit_sets, load_config
from asrsbench.data import HF_DATASET, load_reports, load_summaries
from asrsbench.retrievers import DenseRetriever
from asrsbench.retrievers.base import Units

_spec = importlib.util.spec_from_file_location(
    "export_v1", Path(__file__).resolve().parent / "export_precedent_graph.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

REPORT_SHARD = 20_000
SOURCES = [{"id": 0, "label": "ASRS"}, {"id": 1, "label": "NTSB"},
           {"id": 2, "label": "ASRS (narrative only)"}]
HF_PROBLEM = {  # sookiemonster/asrs-narratives label -> ASRS primary-problem wording
    "aircraft": "Aircraft", "airport": "Airport", "airspacestructure": "Airspace Structure",
    "ambiguous": "Ambiguous", "atcequipment/navfacility/buildings": "ATC Equipment / Nav Facility / Buildings",
    "chartorpublication": "Chart Or Publication", "companypolicy": "Company Policy",
    "environment-nonweatherrelated": "Environment - Non Weather Related",
    "equipment/tooling": "Equipment / Tooling", "humanfactors": "Human Factors",
    "incorrect/notinstalled/unavailablepart": "Incorrect / Not Installed / Unavailable Part",
    "logbookentry": "Logbook Entry", "manuals": "Manuals", "mel": "MEL", "procedure": "Procedure",
    "softwareandautomation": "Software and Automation", "staffing": "Staffing", "weather": "Weather"}

# NTSB phase of flight -> the page's ASRS phase labels (prefix match, first wins).
NTSB_PHASE = [("Standing", "Parked"), ("Pushback", "Taxi"), ("Taxi", "Taxi"),
              ("Takeoff", "Takeoff / Launch"), ("Initial climb", "Initial Climb"),
              ("Enroute-climb", "Climb"), ("Enroute-descent", "Descent"),
              ("Emergency descent", "Descent"), ("Uncontrolled descent", "Descent"),
              ("Enroute", "Cruise"), ("Approach-IFR final", "Final Approach"),
              ("Approach-VFR pattern final", "Final Approach"), ("Approach", "Initial Approach"),
              ("Landing", "Landing"), ("Maneuvering", "Maneuvering"), ("Post-impact", "Landing")]
NTSB_LIGHT = {"NITE": "Night", "NDRK": "Night", "NBRT": "Night", "DUSK": "Dusk", "DAWN": "Dawn"}
NTSB_INJURY = {"FATL": "Fatal injuries", "SERS": "Serious injuries", "MINR": "Minor injuries"}
NTSB_DAMAGE = {"DEST": "Aircraft destroyed", "SUBS": "Substantial damage"}
NTSB_WX_WORDS = [(r"\bwind|gust|crosswind|tailwind", "Wind / gusts"), (r"turbulen", "Turbulence"),
                 (r"ic(e|ing)\b|icing", "Icing"), (r"thunderstorm|convect", "Thunderstorm"),
                 (r"fog|visibility|ceiling|cloud|imc", "Low visibility / ceiling"),
                 (r"density altitude|high temp", "Density altitude"), (r"windshear", "Windshear"),
                 (r"glare|sun", "Sun glare"), (r"dark|night", "Dark night")]


def ntsb_phase(name: str) -> str | None:
    """Page phase label for an NTSB phase name."""
    return next((lab for p, lab in NTSB_PHASE if name.startswith(p)), None)


def ntsb_facets(codes: dict, aircraft: str, location: str) -> list[tuple[str, str]]:
    """(layer, label) factors for one NTSB event from its coded fields."""
    out: set[tuple[str, str]] = set()
    for occ in codes.get("occurrences", []):
        phase, _, event = occ.partition(": ")
        if event:
            out.add(("event", event[:1].upper() + event[1:]))
        lab = ntsb_phase(phase)
        if lab:
            out.add(("phase", lab))
    for f in codes.get("findings", []):
        path = [p.strip() for p in re.sub(r" - [CF]$", "", f["finding"]).split("-")]
        top = path[0].lower() if path else ""
        tail = " / ".join(path[2:4]) if len(path) > 3 else " / ".join(path[1:3])
        if top.startswith("personnel"):
            out.add(("human", f"Finding: {tail}"))
        elif top.startswith("aircraft"):
            ata = v1.first_match(v1.ATA, " ".join(path[1:]))
            out.add(("system", ata or f"Finding: {tail}"))
        elif top.startswith("environmental"):
            text = " ".join(path).lower()
            hit = next((lab for rx, lab in NTSB_WX_WORDS if re.search(rx, text)), None)
            out.add(("weather", hit) if hit else ("context", f"Finding: {tail}"))
        elif tail:
            out.add(("context", f"Finding: {tail}"))
    if codes.get("weather") in ("VMC", "IMC"):
        out.add(("weather", codes["weather"]))
    if codes.get("light") in NTSB_LIGHT:
        out.add(("weather", NTSB_LIGHT[codes["light"]]))
    if codes.get("highest_injury") in NTSB_INJURY:
        out.add(("response", NTSB_INJURY[codes["highest_injury"]]))
    if codes.get("damage") in NTSB_DAMAGE:
        out.add(("response", NTSB_DAMAGE[codes["damage"]]))
    fam = v1.first_match(v1.AIRCRAFT, aircraft)
    if fam:
        out.add(("context", f"Aircraft: {fam}"))
    far = str(codes.get("far_part") or "")
    if re.fullmatch(r"\d{3}", far):
        out.add(("context", f"Operating under Part {int(far)}"))
    if location and not location.startswith("ZZZ"):
        out.add(("context", f"Location: {location}"))
    return sorted(out)


def report_vectors(cfg_path: str, reports: pd.DataFrame) -> np.ndarray:
    """Unit-normalised mean Qwen3-8B chunk vector per report, from the embedding cache."""
    cfg = load_config(cfg_path)
    spec = next(s for s in cfg["systems"] if s["name"] == "qwen3-8b")
    params = {k: v for k, v in spec["params"].items() if k != "devices"}
    cfg["systems"] = [spec]
    units = _unit_sets(cfg, reports)["chunk_max"]
    d = DenseRetriever("qwen3-8b", device="cpu", cache_dir=cfg["cache_dir"], **params)
    for start, end in d.segments(units):
        sub = Units(units.texts[start:end], units.unit_ids[start:end],
                    units.unit_ids[start:start + 1], np.zeros(1, dtype=int), units.tag)
        if not d._cache_path(sub).exists():
            raise RuntimeError("Qwen3-8B embeddings are missing; export never encodes the corpus")
    d.index(units)  # loads the cached blocks; encoding here would mean the cache is missing
    if d.encode_seconds is not None:
        raise RuntimeError("Qwen3-8B embeddings were not cached; run configs/corpus_v2.yaml first")
    assert list(units.doc_ids) == reports["acn"].tolist(), "unit order differs from the reports"
    sums = np.add.reduceat(d.E, units.doc_starts, axis=0)
    return (sums / np.linalg.norm(sums, axis=1, keepdims=True)).astype(np.float32)


def cross_source_neighbours(vectors: np.ndarray, sources: np.ndarray,
                            k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Cosine neighbours from the other reporting system (ASRS includes ASRSN).

    Keep a separate shortlist so a dense within-source neighbourhood cannot hide
    cross-source analogues. These are similarities, never incident-identity links.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(vectors).to(device)
    family = torch.from_numpy(np.asarray(sources) == 1).to(device)
    width = min(k, len(vectors))
    indices = np.full((len(vectors), width), -1, dtype=np.int32)
    scores = np.zeros((len(vectors), width), dtype=np.float32)
    for start in range(0, len(vectors), 256):
        end = min(start + 256, len(vectors))
        sim = x[start:end] @ x.T
        sim.masked_fill_(family[start:end, None] == family[None, :], -torch.inf)
        values, ids = torch.topk(sim, width, dim=1)
        valid = torch.isfinite(values) & (values > 0)
        indices[start:end] = torch.where(valid, ids, -1).cpu().numpy()
        scores[start:end] = torch.where(valid, values, 0).cpu().numpy()
    return indices, scores


def synopsis_kind(source: int, synopsis: str) -> str:
    """Identify original text separately from generated fallback summaries."""
    if not synopsis:
        return "generated"
    return "probable_cause" if source == 1 else "analyst"


def tod_bucket(src: int, raw_tod: str, ev_time) -> int | None:
    """Index into v1.TIME_OF_DAY (local-time quarters of the day)."""
    if src == 0:
        return v1.TIME_OF_DAY.index(raw_tod) if raw_tod in v1.TIME_OF_DAY else None
    try:
        hhmm = int(float(ev_time))
    except (TypeError, ValueError):
        return None
    return None if not 0 < hhmm <= 2400 else min(3, (hhmm - 1) // 600)


def fatality_count(value) -> int | None:
    """Non-negative NTSB event fatality count; preserve missing/invalid values as unknown."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return int(n) if np.isfinite(n) and n >= 0 and n.is_integer() else None


def update_fatalities(page: Path, events: pd.DataFrame) -> None:
    """Refresh only fatality metadata in existing v2 shards, retaining report ordering."""
    counts = events.set_index("ev_id")["inj_tot_f"]
    paths = sorted((page / "data" / "reports").glob("*.json"))
    if not paths:
        raise ValueError(f"No existing v2 report shards in {page}")
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for r in rows:
            sid = r.get("id", "").removeprefix("NTSB:")
            n = fatality_count(counts.get(sid)) if r.get("src") == 1 else None
            if n is None:
                r.pop("fatalities", None)
            else:
                r["fatalities"] = n
        path.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    """Export the unified report catalogue and its precomputed similarity links."""
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="the asrs-precedent-graph folder of triaggent_consulting")
    ap.add_argument("--config", default="configs/corpus_v2.yaml")
    ap.add_argument("--graph", default="data/graph/qwen3-8b_v1_corpus_v2.jsonl")
    ap.add_argument("--without-graph", action="store_true",
                    help="export coded factors and narrative links before LLM graphs exist")
    ap.add_argument("--summaries", default="data/summaries/qwen3.5-27b-q6k_v1.jsonl")
    ap.add_argument("--ntsb", default="data/external/ntsb")
    ap.add_argument("--fatalities-only", action="store_true",
                    help="refresh NTSB fatality counts in existing report shards; no encoding")
    args = ap.parse_args()
    out = Path(args.page)
    if args.fatalities_only:
        update_fatalities(out, pd.read_parquet(Path(args.ntsb) / "events.parquet"))
        return

    from datasets import load_dataset
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    cfg = load_config(args.config)
    reports = load_reports(cfg["data_dir"])
    n = len(reports)
    src = np.select([reports["acn"].str.startswith("ASRS:"), reports["acn"].str.startswith("NTSB:")],
                    [0, 1], 2)
    summ = load_summaries(args.summaries)
    missing = (~reports["acn"].isin(summ)).sum()
    if missing:
        raise SystemExit(f"{missing} reports have no Qwen3.5-27B summary yet")
    llm = {}
    if not args.without_graph:
        with open(args.graph, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "graph" in rec:
                    llm[rec["acn"]] = rec["graph"]
    print(f"{n} reports ({Counter(src.tolist())}), {len(llm)} with an LLM graph", flush=True)

    # ASRS v1 raw rows (analyst codes, time of day) and NTSB event times.
    raw = pd.concat([d.to_pandas() for d in load_dataset(HF_DATASET).values()])
    raw = raw.astype({"acn_num_ACN": str}).set_index("acn_num_ACN")
    ntsb_events = pd.read_parquet(Path(args.ntsb) / "events.parquet").set_index("ev_id")
    ev_time = ntsb_events["ev_time"]
    fatalities = ntsb_events["inj_tot_f"]

    # ---- factors
    facet_id: dict[tuple[str, str, str], int] = {}
    rep_f: list[set[int]] = [set() for _ in range(n)]
    fid = lambda layer, label, s: facet_id.setdefault((layer, label, s), len(facet_id))
    for i, (s, sid, codes, ac, loc, lab) in enumerate(zip(src, reports["source_id"], reports["codes"],
                                                          reports["aircraft"], reports["location"],
                                                          reports["label"])):
        if s == 0:
            for layer, label in v1.analyst_facets(raw.loc[sid]):
                rep_f[i].add(fid(layer, label, "asrs"))
        elif s == 1:
            for layer, label in ntsb_facets(json.loads(codes), ac, loc):
                rep_f[i].add(fid(layer, label, "ntsb"))
        elif lab in HF_PROBLEM:
            rep_f[i].add(fid("context", f"Primary problem: {HF_PROBLEM[lab]}", "asrs"))

    model = v1.encoder() if llm else None
    acns = reports["acn"].tolist()
    for field, prefix in {"events": "Event", "causes": "Cause", "equipment": "Equipment"}.items():
        cnt = Counter(p for g in llm.values() for p in {v1.norm(x) for x in g[field]} if p)
        keep = [p for p, c in cnt.items() if c >= v1.LLM_MIN_DF and p not in v1.LLM_STOP]
        canon = v1.merge_phrases(keep, cnt, model) if keep else {}
        for i, acn in enumerate(acns):
            for p in {v1.norm(x) for x in llm.get(acn, {}).get(field, [])}:
                if p in canon:
                    c = canon[p]
                    layer = {"events": "event", "equipment": "system"}.get(field) or v1.cause_layer(c)
                    rep_f[i].add(fid(layer, f"{prefix}: {v1.nice(c)}", "llm"))
        print(f"{field}: {len(keep)} phrases -> {len(set(canon.values()))} LLM factors", flush=True)

    df = Counter(f for fs in rep_f for f in fs)
    keys = sorted(facet_id, key=facet_id.get)
    keep_ids = [k for k in range(len(keys)) if df[k] >= (3 if keys[k][1].startswith(("Location:", "Finding:")) else 1)]
    remap = {old: new for new, old in enumerate(keep_ids)}
    facets = [{"id": remap[o], "layer": keys[o][0], "label": keys[o][1], "df": df[o],
               "idf": round(float(np.log((n + 1) / (df[o] + 1)) + 1), 3), "src": keys[o][2]}
              for o in keep_ids]
    rep_f = [sorted(remap[f] for f in fs if f in remap) for fs in rep_f]

    groups = defaultdict(list)
    for i, g in enumerate(reports["group_id"]):
        groups[g].append(i)
    sib = [set(groups[g]) - {i} for i, g in enumerate(reports["group_id"])]

    # ---- narrative similarity: Qwen3-8B report vectors fused with TF-IDF LSA
    narr = reports["narrative"].tolist()
    E = report_vectors(args.config, reports)
    lsa_path = v1.CACHE / f"lsa_v2_{n}.npy"
    if lsa_path.exists():
        L = np.load(lsa_path)
    else:
        X = TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True, ngram_range=(1, 2),
                            max_features=300_000, stop_words="english").fit_transform(narr)
        L = TruncatedSVD(256, random_state=0).fit_transform(X).astype(np.float32)
        L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-9
        v1.CACHE.mkdir(parents=True, exist_ok=True)
        np.save(lsa_path, L)
    nar_i, nar_v, nar_cal = v1.neighbours([E, L], sib, v1.TOPK)
    cross_i, cross_v = cross_source_neighbours(E, src)
    print("narrative neighbours done", flush=True)

    # ---- sequence similarity: the LLM's cause/effect triples (MPNet, as in v1)
    rel = {"caused": "caused", "contributed_to": "contributed to", "led_to": "led to",
           "detected": "detected", "responded_with": "responded with", "involved": "involved"}
    events = []
    for acn in acns:
        g = llm.get(acn)
        ev = [f"{v1.nice(t['s'].strip())} {rel.get(t['r'], t['r'])} {t['o'].strip()}"
              for t in (g or {}).get("triples", []) if t["s"].strip() and t["o"].strip()]
        events.append(list(dict.fromkeys(ev))[:5] if g else None)
    seq_text = ["; ".join(e) if e else "" for e in events]
    has_seq = np.array([bool(t) for t in seq_text])
    seq_cal = None
    if has_seq.any():
        E_seq = v1.embed(seq_text, "mpnet_sequence_v2", model)
        E_seq[~has_seq] = 0
        seq_i, seq_v, seq_cal = v1.neighbours([E_seq], sib, v1.TOPK)
    print("sequence neighbours done" if has_seq.any() else "sequence links omitted (no graph)",
          flush=True)

    # ---- map and regions
    xy = v1.tsne_xy(E)
    labels = HDBSCAN(min_cluster_size=400, min_samples=40).fit_predict(xy)
    F_layer = [f["layer"] for f in facets]
    clusters, used = [], set()
    for c in sorted(set(labels) - {-1}, key=lambda c: -(labels == c).sum()):
        members = np.flatnonzero(labels == c)
        cnt = Counter(f for i in members for f in rep_f[i]
                      if F_layer[f] not in ("context", "phase") and facets[f]["src"] != "llm")
        scored = sorted(((k * np.log((k / len(members)) / (facets[f]["df"] / n)), f)
                         for f, k in cnt.items() if k >= 0.2 * len(members)), reverse=True)
        top = [facets[f]["label"] for s, f in scored if s > 0][:4]
        if not top:
            labels[members] = -1
            continue
        name = next((f"{top[0]} · {b}" for b in top[1:] if f"{top[0]} · {b}" not in used),
                    " · ".join(top[:2]))
        used.add(name)
        srcs = Counter(src[members].tolist())
        clusters.append({"id": int(c), "n": int(len(members)), "label": name,
                         "src": {str(k): int(v) for k, v in srcs.items()},
                         "x": round(float(np.median(xy[members, 0])), 4),
                         "y": round(float(np.median(xy[members, 1])), 4)})
    print(f"{len(clusters)} regions, {np.mean(labels >= 0):.0%} of reports in one", flush=True)

    # ---- when and where
    raw_tod = raw["Time.1_Local Time Of Day"].astype(str)
    tod = [tod_bucket(s, raw_tod.get(sid, "") if s == 0 else "", ev_time.get(sid) if s == 1 else None)
           for s, sid in zip(src, reports["source_id"])]
    state = [st if st in v1.STATES else None for st in reports["state"].fillna("").str.upper()]
    loc = [l if re.fullmatch(r"[A-Z0-9]{2,5}", l or "") and not l.startswith("ZZZ") else None
           for l in reports["location"].fillna("")]

    # ---- write
    disp = reports["display_id"].tolist()
    syn = [s if s else summ[a] for s, a in zip(reports["synopsis"], acns)]
    rows = []
    for i in range(n):
        d = str(reports["date"][i])
        r = {"acn": disp[i], "id": acns[i], "src": int(src[i]), "date": f"{d[:4]}-{d[4:6]}",
             "aircraft": str(reports["aircraft"][i] or ""), "synopsis": syn[i], "f": rep_f[i],
             "synopsis_kind": synopsis_kind(int(src[i]), reports["synopsis"][i]),
             "summary": summ[acns[i]],
             "cross": [[int(j), round(float(v), 4)]
                       for j, v in zip(cross_i[i], cross_v[i]) if j >= 0],
             "nar": [[int(j), round(float(v), 2)] for j, v in zip(nar_i[i], nar_v[i]) if v > 0],
             "xy": [round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)]}
        if reports["date_estimated"][i]:
            r["de"] = 1
        if has_seq[i]:
            r["seq"] = [[int(j), round(float(v), 2)] for j, v in zip(seq_i[i], seq_v[i])
                        if v > 0 and has_seq[j]]
            r["events"] = events[i]
        if tod[i] is not None:
            r["t"] = tod[i]
        deaths = fatality_count(fatalities.get(reports["source_id"][i])) if src[i] == 1 else None
        if deaths is not None:
            r["fatalities"] = deaths
        if state[i]:
            r["st"] = state[i]
        if loc[i]:
            r["loc"] = loc[i]
        if labels[i] >= 0:
            r["cl"] = int(labels[i])
        if sib[i]:
            r["sib"] = sorted(sib[i])
        rows.append(r)

    def pick(pattern: str, s: int) -> str | None:
        """Display id of the first report of source ``s`` whose synopsis matches."""
        rx = re.compile(pattern, re.I)
        return next((r["acn"] for r in rows if r["src"] == s and rx.search(r["synopsis"])), None)

    examples = [
        {"label": "RNAV / RNP approach, crew communication breakdown", "acn": "1330765"},
        {"label": "Hydraulic system problem, emergency declared", "acn": "1046434"},
        {"label": "Windshear on approach, go-around", "acn": "1703613"},
        {"label": "Unstabilized approach, go-around", "acn": "1317109"},
        {"label": "NTSB: loss of engine power, forced landing", "acn": pick(r"total loss of engine power.*forced landing", 1)},
        {"label": "NTSB: VFR flight into IMC, loss of control", "acn": pick(r"VFR flight into (instrument|IMC).*loss of", 1)},
        {"label": "NTSB: midair collision", "acn": pick(r"failure .* to see and avoid", 1)},
    ]
    states_n = Counter(x for x in state if x)
    meta = {
        "n_reports": n, "n_llm_coded": int(has_seq.sum()),
        "n_summaries": len(set(acns) & summ.keys()),
        "summary_model": "Qwen3.5-27B (Q6_K)",
        "cross_method": "Cosine of mean Qwen3-Embedding-8B narrative chunk vectors; other reporting system only",
        "sources": [{**s, "n": int((src == s["id"]).sum())} for s in SOURCES],
        "layers": [{"id": "event", "label": "Event type"}, {"id": "phase", "label": "Flight phase"},
                   {"id": "human", "label": "Human factors & CRM"},
                   {"id": "approach", "label": "Route, navigation & automation"},
                   {"id": "weather", "label": "Weather & light"}, {"id": "system", "label": "Aircraft system"},
                   {"id": "response", "label": "Crew response & outcome"},
                   {"id": "context", "label": "Operation & aircraft"}],
        "calibration": {"narrative": nar_cal, "sequence": seq_cal},
        "source": ("NASA Aviation Safety Reporting System (ASRS) via elihoole/asrs-aviation-reports "
                   "and sookiemonster/asrs-narratives; NTSB aviation accident database (avall, "
                   "data.ntsb.gov, 2026-11-01)"),
        "scope": f"{n:,} reports: ASRS 2012–2025 and NTSB 2010–2026",
        "llm": ("Qwen3-8B knowledge graph; " if llm else "No LLM graph; ")
               + "Qwen3.5-27B (Q6_K) summaries, run locally on the narrative only",
        "embeddings": "Qwen3-Embedding-8B (fp16) for narratives, fused with TF-IDF LSA"
                      + ("; all-mpnet-base-v2 for cause/effect links" if has_seq.any() else ""),
        "examples": [x for x in examples if x["acn"]],
        "clusters": clusters, "shard": v1.SHARD, "report_shard": REPORT_SHARD,
        "n_report_shards": (n + REPORT_SHARD - 1) // REPORT_SHARD,
        "tod": ["00:01–06:00", "06:01–12:00", "12:01–18:00", "18:01–24:00"],
        "states": [{"id": s, "label": v1.STATES[s], "n": k} for s, k in states_n.most_common()],
    }
    data = out / "data"
    for sub in ("reports", "narratives"):
        (data / sub).mkdir(parents=True, exist_ok=True)
        for old in (data / sub).glob("*.json"):
            old.unlink()
    with open(data / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "facets": facets}, f, ensure_ascii=False, separators=(",", ":"))
    for k in range(meta["n_report_shards"]):
        with open(data / "reports" / f"{k}.json", "w", encoding="utf-8") as f:
            json.dump(rows[k * REPORT_SHARD:(k + 1) * REPORT_SHARD], f, ensure_ascii=False,
                      separators=(",", ":"))
    for k in range(0, n, v1.SHARD):
        with open(data / "narratives" / f"{k // v1.SHARD}.json", "w", encoding="utf-8") as f:
            json.dump({acns[i]: {"n": narr[i], "s": summ[acns[i]]}
                       for i in range(k, min(n, k + v1.SHARD))},
                      f, ensure_ascii=False, separators=(",", ":"))
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted((data / "reports").glob("*.json"))}
    print(f"wrote {n} reports, {len(facets)} factors; report shards (MB): {sizes}; "
          f"{(n + v1.SHARD - 1) // v1.SHARD} narrative shards", flush=True)


if __name__ == "__main__":
    main()

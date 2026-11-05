"""Build the ASRS Precedent Graph web page's data for all 47,723 reports.

usage: python scripts/export_precedent_graph.py <triaggent_consulting>/portfolio/asrs-precedent-graph
                                                [--graph data/graph/qwen3-8b_v1.jsonl]

Writes ``data/graph.json`` and ``data/narratives/<k>.json`` (narratives in shards of
``SHARD`` reports, loaded on demand) into the page folder. The page links two reports
through the factors they share, weighted by rarity, and through two text similarities:

* Analyst factors (``src: asrs``): anomaly, flight phase, human factors and communication
  breakdowns, route and navigation in use, weather and light, components grouped into ATA
  chapters, result, and aircraft family, FAR part, primary problem and airport.
* LLM factors (``src: llm``): frequent event, cause and equipment phrases from
  ``scripts/llm_graph.py`` (Qwen3-8B, narrative only). Near-duplicate phrases are merged by
  embedding similarity. Unlike the earlier 3,000-report build, labels carry no quotes.
* ``nar``: narrative similarity, all-mpnet-base-v2 fused with TF-IDF LSA (z-scored cosines).
* ``seq``: similarity of the LLM's cause/effect triples, also embedded with MPNet.

The map is openTSNE on the narrative embeddings; regions are HDBSCAN clusters on the map,
named after the analyst factors most over-represented in them. Embeddings are cached in
``cache/precedent/`` so a rerun only redoes the cheap steps.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from asrsbench.data import HF_DATASET, load_reports

SHARD = 250   # reports per narrative shard (~0.5 MB each)
TOPK = 12     # neighbours kept per report for each text similarity
LLM_MIN_DF = 50
MERGE_COS = 0.88
CACHE = Path("cache/precedent")

# Readable labels for analyst codes, following the earlier 3,000-report build.
ANOMALY = {
    "Deviation / Discrepancy - Procedural Published Material / Policy": "Deviation from published procedure / policy",
    "ATC Issue All Types": "ATC issue",
    "Deviation / Discrepancy - Procedural Clearance": "Clearance deviation",
    "Aircraft Equipment Problem Less Severe": "Equipment problem (less severe)",
    "Less Severe": "Equipment problem (less severe)",
    "Aircraft Equipment Problem Critical": "Equipment problem (critical)",
    "Critical": "Equipment problem (critical)",
    "Inflight Event / Encounter Weather / Turbulence": "Weather / turbulence encounter",
    "Deviation - Track / Heading All Types": "Track / heading deviation",
    "Deviation / Discrepancy - Procedural FAR": "FAR deviation",
    "Conflict Airborne Conflict": "Airborne conflict",
    "Conflict NMAC": "Near midair collision",
    "Inflight Event / Encounter CFTT / CFIT": "CFIT / CFTT",
    "Deviation - Altitude Excursion From Assigned Altitude": "Altitude excursion",
    "Flight Deck / Cabin / Aircraft Event Smoke / Fire / Fumes / Odor": "Smoke / fire / fumes",
    "Conflict Ground Conflict": "Ground conflict",
    "Airspace Violation All Types": "Airspace violation",
    "Deviation - Speed All Types": "Speed deviation",
    "Deviation / Discrepancy - Procedural Maintenance": "Maintenance procedure deviation",
    "Inflight Event / Encounter Fuel Issue": "Fuel issue",
    "Ground Event / Encounter Fuel Issue": "Fuel issue",
    "Inflight Event / Encounter Unstabilized Approach": "Unstabilized approach",
    "Deviation - Altitude Overshoot": "Altitude overshoot",
    "Deviation - Altitude Undershoot": "Altitude undershoot",
    "Ground Incursion Runway": "Runway incursion",
    "Ground Incursion Taxiway": "Taxiway incursion",
    "Ground Incursion Ramp": "Ramp incursion",
    "Ground Event / Encounter Loss Of Aircraft Control": "Loss of control on ground",
    "Inflight Event / Encounter Loss Of Aircraft Control": "Loss of control in flight",
    "Flight Deck / Cabin / Aircraft Event Illness / Injury": "Illness / injury",
    "Deviation - Altitude Crossing Restriction Not Met": "Crossing restriction not met",
    "Inflight Event / Encounter Wake Vortex Encounter": "Wake vortex encounter",
    "Ground Excursion Runway": "Runway excursion",
    "Ground Excursion Taxiway": "Taxiway excursion",
    "Ground Excursion Ramp": "Ramp excursion",
    "Deviation / Discrepancy - Procedural Hazardous Material Violation": "Hazardous material violation",
    "Deviation / Discrepancy - Procedural MEL / CDL": "MEL / CDL deviation",
    "Deviation / Discrepancy - Procedural Weight And Balance": "Weight and balance issue",
    "Ground Event / Encounter Ground Strike - Aircraft": "Ground strike (tail / wing / pod)",
    "Flight Deck / Cabin / Aircraft Event Passenger Misconduct": "Passenger misconduct",
    "Ground Event / Encounter Object": "Object encounter on ground",
    "Inflight Event / Encounter Object": "Object encounter in flight",
    "Ground Event / Encounter Vehicle": "Vehicle encounter on ground",
    "Inflight Event / Encounter VFR In IMC": "VFR into IMC",
    "Deviation / Discrepancy - Procedural Security": "Security deviation",
    "Inflight Event / Encounter Bird / Animal": "Bird / animal strike",
    "Ground Event / Encounter Person / Animal / Bird": "Bird / animal strike",
    "Ground Event / Encounter Ground Equipment Issue": "Ground equipment issue",
    "Deviation / Discrepancy - Procedural Landing Without Clearance": "Landed without clearance",
    "Ground Event / Encounter Gear Up Landing": "Gear-up landing",
    "Ground Event / Encounter FOD": "FOD encounter",
    "Ground Event / Encounter Jet Blast": "Jet blast",
    "Other Jet Blast": "Jet blast",
    "Flight Deck / Cabin / Aircraft Event Passenger Electronic Device": "Passenger electronic device",
    "Deviation / Discrepancy - Procedural Unauthorized Flight Operations (UAS)": "Unauthorized UAS operation",
    "Inflight Event / Encounter Laser": "Laser strike",
    "Ground Event / Encounter Weather / Turbulence": "Weather / turbulence encounter",
    "Ground Event / Encounter Aircraft": "Ground conflict",
    "Inflight Event / Encounter Aircraft": "Airborne conflict",
}
ANOMALY_SKIP = re.compile(r"Other / Unknown|No Specific Anomaly|Unwanted [Ss]ituation|Scheduling", re.I)

RESULT = {
    "General Maintenance Action": "Maintenance action",
    "Flight Crew Took Evasive Action": "Evasive action",
    "Air Traffic Control Issued New Clearance": "ATC issued new clearance",
    "Flight Crew Became Reoriented": "Crew became reoriented",
    "Air Traffic Control Provided Assistance": "ATC provided assistance",
    "Flight Crew Requested ATC Assistance / Clarification": "Requested ATC assistance",
    "Flight Crew Overcame Equipment Problem": "Crew overcame equipment problem",
    "Flight Crew Landed in Emergency Condition": "Landed in emergency condition",
    "Air Traffic Control Issued Advisory / Alert": "ATC advisory / alert",
    "Flight Crew Returned To Clearance": "Returned to clearance",
    "General Flight Cancelled / Delayed": "Flight cancelled / delayed",
    "Flight Crew Diverted": "Diverted",
    "Flight Crew Returned To Departure Airport": "Returned to departure airport",
    "Aircraft Aircraft Damaged": "Aircraft damaged",
    "General Declared Emergency": "Declared emergency",
    "Flight Crew Landed As Precaution": "Precautionary landing",
    "Flight Crew Executed Go Around / Missed Approach": "Go-around / missed approach",
    "Flight Crew FLC complied w / Automation / Advisory": "Complied with automation / advisory (e.g. TCAS RA)",
    "Flight Crew Regained Aircraft Control": "Regained aircraft control",
    "Air Traffic Control Separated Traffic": "ATC separated traffic",
    "Flight Crew Returned To Gate": "Returned to gate",
    "Flight Crew Rejected Takeoff": "Rejected takeoff",
    "Flight Crew Inflight Shutdown": "In-flight engine shutdown",
    "Aircraft Equipment Problem Dissipated": "Equipment problem dissipated",
    "General Physical Injury / Incapacitation": "Injury / incapacitation",
    "Flight Crew FLC Overrode Automation": "Crew overrode automation",
    "Flight Crew Overrode Automation": "Crew overrode automation",
    "General Evacuated": "Evacuation",
    "General Work Refused": "Work refused",
    "General Release Refused / Aircraft Not Accepted": "Release refused / aircraft not accepted",
    "Flight Crew Exited Penetrated Airspace": "Exited penetrated airspace",
    "General Police / Security Involved": "Police / security involved",
    "Aircraft Automation Overrode Flight Crew": "Automation overrode crew",
}

TIME_OF_DAY = ["0001-0600", "0601-1200", "1201-1800", "1801-2400"]  # ASRS local-time buckets
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "GU": "Guam", "VI": "US Virgin Islands", "AS": "American Samoa",
    "AB": "Canada: Alberta", "BC": "Canada: British Columbia", "MB": "Canada: Manitoba",
    "NF": "Canada: Newfoundland", "NS": "Canada: Nova Scotia", "ON": "Canada: Ontario",
    "PQ": "Canada: Quebec",
    "US": "United States, state withheld", "FO": "Outside the US and Canada",
}

PHASES = {"Parked", "Taxi", "Takeoff / Launch", "Initial Climb", "Climb", "Cruise", "Descent",
          "Initial Approach", "Final Approach", "Landing"}
WEATHER = {"Turbulence", "Rain", "Thunderstorm", "Icing", "Windshear", "Haze / Smoke", "Snow",
           "Fog", "Cloudy", "Hail", "Mist"}
CONDITIONS = {"VMC": "VMC", "IMC": "IMC", "Mixed": "Mixed VMC / IMC", "Marginal": "Marginal"}
PARTY = {"ATC": "ATC", "Flight Crew": "flight crew", "Maintenance": "maintenance",
         "Ground Personnel": "ground crew", "Dispatch": "dispatch", "Flight Attendant": "cabin crew",
         "Other": "other"}

# Component -> ATA chapter, first match wins (order matters: "fuel quantity indication" is fuel).
ATA = [
    (r"pressuri|air condition|pack\b|outflow|cabin alt|bleed air temp", "ATA 21 Air conditioning & pressurisation"),
    (r"autopilot|autoflight|auto ?flight|autothrottle|auto ?throttle|\bmcp\b|flight director|yaw damper", "ATA 22 Auto flight"),
    (r"communicat|radio|vhf|\bhf\b|cpdlc|acars|interphone|satcom|headset|speaker|microphone", "ATA 23 Communications"),
    (r"electric|generator|alternator|batter|\bbus\b|inverter|circuit breaker|\btru\b", "ATA 24 Electrical power"),
    (r"galley|seat|lavator|furnish|slide|cargo|escape|evacuat|cabin equipment|carpet", "ATA 25 Equipment & furnishings"),
    (r"fire|smoke detect|extinguish|overheat detect", "ATA 26 Fire protection"),
    (r"flap|slat|aileron|elevator|rudder|trim|spoiler|speed ?brake|flight control|stabili[sz]er|stick shaker|control column|yoke", "ATA 27 Flight controls"),
    (r"fuel", "ATA 28 Fuel"),
    (r"hydraulic", "ATA 29 Hydraulic power"),
    (r"ice|icing|wiper|rain repell|pitot heat|probe heat", "ATA 30 Ice & rain protection"),
    (r"eicas|ecam|display|\bpfd\b|\bmfd\b|\bnd\b|recorder|clock|warning|caution|indicat|annunciat", "ATA 31 Indicating & recording"),
    (r"gear|brake|tire|tyre|wheel|steering|anti-?skid|strut", "ATA 32 Landing gear & brakes"),
    (r"light", "ATA 33 Lights"),
    (r"fms|fmc|gps|navigat|\bils\b|\bvor\b|altimeter|airspeed|air data|pitot|static|tcas|gpws|taws|radar|\badf\b|\bdme\b|\bins\b|\birs\b|adiru|compass|attitude|transponder|heading|localizer|glideslope", "ATA 34 Navigation"),
    (r"oxygen", "ATA 35 Oxygen"),
    (r"pneumatic|bleed", "ATA 36 Pneumatic"),
    (r"\bapu\b|auxiliary power", "ATA 49 APU"),
    (r"door|hatch", "ATA 52 Doors"),
    (r"window|windshield|windscreen", "ATA 56 Windows"),
    (r"propeller|prop\b|governor", "ATA 61 Propellers"),
    (r"reverser|exhaust", "ATA 78 Exhaust & thrust reverser"),
    (r"lubricat|\boil\b", "ATA 79 Engine oil"),
    (r"throttle|power lever|thrust lever|fadec|engine control", "ATA 76 Engine controls"),
    (r"engine|turbine|nacelle|cowl|compressor|ignition|starter", "ATA 71-72 Engine"),
    (r"efb|electronic fl|tablet|chart|ipad", "Electronic flight bag & charts"),
    (r"structur|fuselage|wing|skin|panel|fitting|fastener", "ATA 51-57 Structures"),
]
ATA = [(re.compile(p, re.I), lab) for p, lab in ATA]

AIRCRAFT = [
    (r"^B737|^B-737|737", "Boeing 737"), (r"^B757|757", "Boeing 757"), (r"^B767|767", "Boeing 767"),
    (r"^B777|777", "Boeing 777"), (r"^B787|787", "Boeing 787"), (r"^B747|747", "Boeing 747"),
    (r"^B717|717", "Boeing 717"), (r"^A3(18|19|20|21)", "Airbus A320 family"), (r"^A330", "Airbus A330"),
    (r"^A350", "Airbus A350"), (r"^A380", "Airbus A380"), (r"^A3(00|10)", "Airbus A300 / A310"),
    (r"^MD-?11", "MD-11"), (r"^MD-?8|^MD-?9|^DC-9", "MD-80 / DC-9 family"), (r"^DC-?10", "DC-10"),
    (r"ERJ 1[34]5|ERJ-1[34]5", "Embraer ERJ 135 / 145"), (r"ERJ 1[79]0|ERJ 175|ERJ 195|E1[79]", "Embraer E-Jet"),
    (r"CRJ|Regional Jet", "Bombardier CRJ"), (r"Dash 8|DHC-8|Q400", "Dash 8"), (r"ATR", "ATR 42 / 72"),
    (r"Saab", "Saab 340"), (r"Skyhawk 172|Cessna 172|C172", "Cessna 172"), (r"Skylane 182|C182", "Cessna 182"),
    (r"Citation|C5[0-9]{2}|C680|C750", "Cessna Citation"), (r"PA-28", "Piper PA-28"),
    (r"PA-(3[0-4]|44|46)|Seneca|Malibu|Seminole", "Piper twin / Malibu"), (r"King Air|BE20|B200|C90", "Beech King Air"),
    (r"Bonanza|BE3[56]", "Beech Bonanza"), (r"SR2[02]|Cirrus", "Cirrus SR20 / SR22"),
    (r"Learjet|Lear", "Learjet"), (r"Challenger|CL-?6", "Bombardier Challenger"), (r"Gulfstream|G-?IV|G-?V|G[2-7]50", "Gulfstream"),
    (r"Falcon", "Dassault Falcon"), (r"Global Express|BD-700", "Bombardier Global"), (r"Phenom", "Embraer Phenom"),
    (r"PC-12|Pilatus", "Pilatus PC-12"), (r"Caravan|C208", "Cessna Caravan"), (r"Helicopter|Rotorcraft|R44|R22|EC1|AS350|Bell", "Helicopter"),
    (r"UAV|UAS|Drone", "Drone / UAS"),
]
AIRCRAFT = [(re.compile(p, re.I), lab) for p, lab in AIRCRAFT]

# LLM phrases too generic to link reports.
LLM_STOP = {"atc", "runway", "aircraft", "tower", "frequency", "airplane", "taxiway", "ground",
            "controller", "pilot", "crew", "captain", "first officer", "airport", "ramp", "gate",
            "approach", "departure", "radio", "checklist", "procedure", "engine"}
ACRONYMS = {"tcas", "ra", "ta", "fms", "fmc", "atc", "apu", "ils", "gps", "qrh", "acars", "eicas",
            "ecam", "mcp", "egpws", "gpws", "taws", "vnav", "lnav", "rnav", "rnp", "mel", "tracon",
            "artcc", "vfr", "ifr", "imc", "vmc", "cfit", "nmac", "faa", "far", "sid", "star", "dme",
            "vor", "ndb", "adf", "adiru", "irs", "ins", "efb", "cdu", "pfd", "nd", "mfd", "hud",
            "rvr", "atis", "cpdlc", "uas", "lahso", "afm", "fo", "pf", "pm", "pnf", "ctaf", "notam",
            "arff", "egt", "idg", "ecs", "cvr", "fdr", "mcdu", "adc", "ahrs", "ptu", "rat", "ewd",
            "ifsd", "fadec", "eec", "isfd", "ecl", "cmc"}
# Whole words only: a bare "rain" would also match "training" and "terrain".
WEATHER_WORDS = re.compile(r"\b(weather|turbulen\w*|wind\w*|icing|ice|\w*storms?|visibility|fog|rain|"
                           r"snow|cloud\w*|wake|haze|smoke|glare|sun\w*|darkness|night|imc|gusts?|"
                           r"gusty|convect\w*|lightning|crosswind|tailwind|headwind)\b", re.I)
# Causes that name hardware go to the aircraft-system layer, not human factors.
MECHANICAL_WORDS = re.compile(r"\b(fail\w*|malfunction\w*|leak\w*|system|engine|hydraulic|electrical|"
                              r"fuel|gear|brakes?|autopilot|fms|fmc|software|equipment|mechanical|"
                              r"component|sensor|indication|valve|pump|generator|battery|tire|"
                              r"flaps?|door|bird strike|defect\w*|wear|corrosion)\b", re.I)


def cause_layer(phrase: str) -> str:
    """Page layer for an LLM cause phrase: weather, aircraft system or human factors."""
    if WEATHER_WORDS.search(phrase):
        return "weather"
    if MECHANICAL_WORDS.search(phrase) and not re.search(r"communicat|training|awareness", phrase):
        return "system"
    return "human"


def split(v) -> list[str]:
    """Split an ASRS ';'-separated coded field into stripped values."""
    if not isinstance(v, str):
        return []
    return [x.strip() for x in v.split(";") if x.strip()]


def first_match(rules, text: str) -> str | None:
    """Label of the first (regex, label) rule matching ``text``."""
    for rx, lab in rules:
        if rx.search(text):
            return lab
    return None


def comm_pairs(v) -> set[str]:
    """'Party1 ATC; Party2 Flight Crew' -> {'Communication breakdown: ATC ↔ flight crew'}."""
    p1 = [PARTY[x[7:]] for x in split(v) if x.startswith("Party1 ") and x[7:] in PARTY]
    p2 = [PARTY[x[7:]] for x in split(v) if x.startswith("Party2 ") and x[7:] in PARTY]
    out = set()
    for a in p1:
        for b in p2:
            a2, b2 = sorted([a, b], key=str.lower)
            out.add(f"Communication breakdown: {a2} ↔ {b2}")
    return out


def analyst_facets(row) -> list[tuple[str, str]]:
    """(layer, label) analyst factors for one raw ASRS row."""
    out: set[tuple[str, str]] = set()
    for a in split(row["Events_Anomaly"]):
        if ANOMALY_SKIP.search(a):
            continue
        lab = ANOMALY.get(a) or re.sub(r"^Other ", "", a).capitalize()
        out.add(("event", lab))
    for ph in split(row["Aircraft 1.9_Flight Phase"]):
        if ph in PHASES:
            out.add(("phase", ph))
    for col in ("Person 1.7_Human Factors", "Person 2.7_Human Factors"):
        for h in split(row[col]):
            if h != "Other / Unknown":
                out.add(("human", h))
    for col in ("Person 1.8_Communication Breakdown", "Person 2.8_Communication Breakdown"):
        out |= {("human", p) for p in comm_pairs(row[col])}
    for r in split(row["Aircraft 1.10_Route In Use"]):
        lab = {"Visual Approach": "Visual approach (ATC route)", "Vectors": "Radar vectors",
               "Direct": "Direct routing", "Oceanic": "Oceanic route", "VFR Route": "VFR route",
               "Other Instrument Approach": "Instrument approach (ATC route)", "Other ILS": "Instrument approach (ATC route)",
               "Other Traffic Pattern": "Traffic pattern"}.get(r)
        if lab is None and r.startswith("STAR"):
            lab = "On a STAR"
        elif lab is None and r.startswith("SID"):
            lab = "On a SID"
        if lab:
            out.add(("approach", lab))
    for n in split(row["Aircraft 1.8_Nav In Use"]):
        lab = ("FMS / FMC navigation" if n == "FMS Or FMC" else "GPS navigation" if n == "GPS"
               else "INS navigation" if n == "INS" else "ILS / localizer in use" if n.startswith("Localizer")
               else "VOR / VORTAC in use" if n.startswith("VOR") else "NDB in use" if n.startswith("NDB") else None)
        if lab:
            out.add(("approach", lab))
    for c in split(row["Environment_Flight Conditions"]):
        if c in CONDITIONS:
            out.add(("weather", CONDITIONS[c]))
    for w in split(row["Environment.1_Weather Elements / Visibility"]):
        if w in WEATHER:
            out.add(("weather", w))
    for l in split(row["Environment.3_Light"]):
        if l in {"Night", "Dusk", "Dawn"}:
            out.add(("weather", l))
    for p in split(row["Component.3_Problem"]):
        out.add(("system", f"Equipment {p.lower()}"))
    for comp in split(row["Component_Aircraft Component"]):
        ata = first_match(ATA, comp)
        if ata:
            out.add(("system", ata))
        out.add(("system", f"Component: {comp}"))
    for r in split(row["Events.5_Result"]):
        if r in RESULT:
            out.add(("response", RESULT[r]))
    fam = first_match(AIRCRAFT, str(row["Aircraft 1.2_Make Model Name"]))
    if fam:
        out.add(("context", f"Aircraft: {fam}"))
    for p in split(row["Aircraft 1.5_Operating Under FAR Part"]):
        if p.startswith("Part "):
            out.add(("context", f"Operating under {p}"))
    for p in split(row["Assessments.1_Primary Problem"]):
        out.add(("context", f"Primary problem: {p}"))
    loc = str(row["Place_Locale Reference"] or "")
    m = re.match(r"^([A-Z0-9]{3,4})\.(\w+)", loc)
    if m and not m.group(1).startswith("ZZZ"):
        out.add(("context", f"Location: {m.group(1)}" + ("" if m.group(2) == "Airport" else f" {m.group(2)}")))
    return sorted(out)


def nice(phrase: str) -> str:
    """Sentence-case an LLM phrase, keeping aviation acronyms upper case."""
    words = [w.upper() if w in ACRONYMS else w for w in phrase.split()]
    s = " ".join(words)
    return s[:1].upper() + s[1:]


def norm(s: str) -> str:
    """Normalise an LLM phrase for counting: lower case, single spaces, hyphens as spaces."""
    s = re.sub(r"[-_/]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip(" .;:,")
    return re.sub(r"^(the|a|an) ", "", s)


def encoder():
    """all-mpnet-base-v2 on the GPU when there is one."""
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                            device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = 384
    return m


def embed(texts: list[str], name: str, model) -> np.ndarray:
    """Unit-normalised MPNet embeddings, cached by name and corpus size."""
    path = CACHE / f"{name}_{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    E = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float32)
    np.save(path, E)
    return E


def merge_phrases(phrases: list[str], counts: Counter, model) -> dict[str, str]:
    """Map each phrase to the most frequent phrase within MERGE_COS cosine of it."""
    E = model.encode(phrases, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    order = sorted(range(len(phrases)), key=lambda i: -counts[phrases[i]])
    canon, heads = {}, []
    for i in order:
        best = next((h for h in heads if float(E[i] @ E[h]) >= MERGE_COS), None)
        if best is None:
            heads.append(i)
            canon[phrases[i]] = phrases[i]
        else:
            canon[phrases[i]] = phrases[best]
    return canon


def neighbours(blocks: list[np.ndarray], exclude: list[set[int]], k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Top-k neighbours by fused similarity: the mean of per-space z-scored cosines.

    Scores are calibrated to [0, 1]: 0 at the median random pair, 1 at the 95th percentile
    of each report's best match, so the page's link strengths spread out instead of all
    neighbours saturating at 1.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    n = blocks[0].shape[0]
    a, b = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    stats = []
    for X in blocks:
        s = (X[a] * X[b]).sum(1)
        stats.append((float(s.mean()), float(s.std())))
    T = [torch.from_numpy(X).to(dev) for X in blocks]
    rand = np.mean([((X[a] * X[b]).sum(1) - mu) / sd for X, (mu, sd) in zip(blocks, stats)], 0)
    idx = np.zeros((n, k), np.int32)
    val = np.zeros((n, k), np.float32)
    step = 1024
    for s0 in range(0, n, step):
        s1 = min(n, s0 + step)
        S = sum((t[s0:s1] @ t.T - mu) / sd for t, (mu, sd) in zip(T, stats)) / len(T)
        rows = torch.arange(s0, s1, device=dev)
        S[rows - s0, rows] = -1e9
        for r in range(s0, s1):
            if exclude[r]:
                S[r - s0, list(exclude[r])] = -1e9
        v, i = torch.topk(S, k, dim=1)
        idx[s0:s1], val[s0:s1] = i.cpu().numpy(), v.cpu().numpy()
    lo, hi = float(np.median(rand)), float(np.quantile(val[:, 0], 0.95))
    cal = np.clip((val - lo) / (hi - lo), 0, 1)
    return idx, cal, {"median": lo, "top1_p95": hi}


def tsne_xy(E: np.ndarray) -> np.ndarray:
    """2-D openTSNE layout of the embeddings, scaled into [-1, 1]."""
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    path = CACHE / f"tsne_{len(E)}.npy"
    if path.exists():
        return np.load(path)
    P = PCA(50, random_state=0).fit_transform(E)
    Y = np.asarray(TSNE(perplexity=50, n_jobs=-1, random_state=0, verbose=True).fit(P))
    Y = (Y - np.median(Y, 0)) / np.quantile(np.abs(Y - np.median(Y, 0)), 0.999)
    Y = np.clip(Y, -1, 1) * 0.98
    np.save(path, Y)
    return Y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="the asrs-precedent-graph folder of triaggent_consulting")
    ap.add_argument("--graph", default="data/graph/qwen3-8b_v1.jsonl")
    args = ap.parse_args()
    out = Path(args.page)

    from datasets import load_dataset
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    reports = load_reports()
    ds = load_dataset(HF_DATASET)
    raw = pd.concat([ds[s].to_pandas() for s in ds]).astype({"acn_num_ACN": str})
    raw = raw.set_index("acn_num_ACN").loc[reports["acn"]]
    llm = {}
    with open(args.graph, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "graph" in rec:
                llm[rec["acn"]] = rec["graph"]
    n = len(reports)
    print(f"{n} reports, {len(llm)} with an LLM graph", flush=True)

    # ---- analyst factors
    facet_id: dict[tuple[str, str, str], int] = {}
    rep_f: list[set[int]] = [set() for _ in range(n)]

    def fid(layer: str, label: str, src: str) -> int:
        return facet_id.setdefault((layer, label, src), len(facet_id))

    for i, (_, row) in enumerate(raw.iterrows()):
        for layer, label in analyst_facets(row):
            rep_f[i].add(fid(layer, label, "asrs"))

    # ---- LLM factors: frequent phrases, near-duplicates merged
    model = encoder()
    fields = {"events": "Event", "causes": "Cause", "equipment": "Equipment"}
    for field, prefix in fields.items():
        cnt = Counter(p for g in llm.values() for p in {norm(x) for x in g[field]} if p)
        keep = [p for p, c in cnt.items() if c >= LLM_MIN_DF and p not in LLM_STOP]
        canon = merge_phrases(keep, cnt, model)
        for i, acn in enumerate(reports["acn"]):
            for p in {norm(x) for x in llm.get(acn, {}).get(field, [])}:
                if p in canon:
                    c = canon[p]
                    layer = {"events": "event", "equipment": "system"}.get(field) or cause_layer(c)
                    rep_f[i].add(fid(layer, f"{prefix}: {nice(c)}", "llm"))
        print(f"{field}: {len(keep)} phrases -> {len(set(canon.values()))} LLM factors", flush=True)

    # Drop location facets seen fewer than 3 times; keep everything else.
    df = Counter(f for fs in rep_f for f in fs)
    keys = sorted(facet_id, key=facet_id.get)
    keep_ids = [k for k in range(len(keys))
                if df[k] >= (3 if keys[k][1].startswith("Location:") else 1)]
    remap = {old: new for new, old in enumerate(keep_ids)}
    facets = []
    for old in keep_ids:
        layer, label, src = keys[old]
        facets.append({"id": remap[old], "layer": layer, "label": label, "df": df[old],
                       "idf": round(float(np.log((n + 1) / (df[old] + 1)) + 1), 3), "src": src})
    rep_f = [sorted(remap[f] for f in fs if f in remap) for fs in rep_f]

    # ---- siblings: other reports of the same incident
    groups = defaultdict(list)
    for i, g in enumerate(reports["group_id"]):
        groups[g].append(i)
    sib = [set(groups[g]) - {i} for i, g in enumerate(reports["group_id"])]

    # ---- narrative similarity: MPNet fused with TF-IDF LSA
    narr = reports["narrative"].tolist()
    E_nar = embed(narr, "mpnet_narrative", model)
    lsa_path = CACHE / f"lsa_{n}.npy"
    if lsa_path.exists():
        L = np.load(lsa_path)
    else:
        X = TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True, ngram_range=(1, 2),
                            max_features=300_000, stop_words="english").fit_transform(narr)
        L = TruncatedSVD(256, random_state=0).fit_transform(X).astype(np.float32)
        L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-9
        np.save(lsa_path, L)
    nar_i, nar_v, nar_cal = neighbours([E_nar, L], sib, TOPK)
    print("narrative neighbours done", flush=True)

    # ---- sequence similarity: the LLM's cause/effect triples
    rel_words = {"caused": "caused", "contributed_to": "contributed to", "led_to": "led to",
                 "detected": "detected", "responded_with": "responded with", "involved": "involved"}
    events = []
    for acn in reports["acn"]:
        g = llm.get(acn)
        ev = [f"{nice(t['s'].strip())} {rel_words.get(t['r'], t['r'])} {t['o'].strip()}"
              for t in (g or {}).get("triples", []) if t["s"].strip() and t["o"].strip()]
        events.append(list(dict.fromkeys(ev))[:5] if g else None)
    seq_text = ["; ".join(e) if e else "" for e in events]
    has_seq = np.array([bool(t) for t in seq_text])
    E_seq = embed(seq_text, "mpnet_sequence", model)
    E_seq[~has_seq] = 0  # uncoded reports score ~0 and are dropped from the lists below
    seq_i, seq_v, seq_cal = neighbours([E_seq], sib, TOPK)
    print("sequence neighbours done", flush=True)

    # ---- map and regions
    xy = tsne_xy(E_nar)
    labels = HDBSCAN(min_cluster_size=250, min_samples=25).fit_predict(xy)
    F_layer = [f["layer"] for f in facets]
    clusters, used = [], set()
    for c in sorted(set(labels) - {-1}, key=lambda c: -(labels == c).sum()):
        members = np.flatnonzero(labels == c)
        cnt = Counter(f for i in members for f in rep_f[i]
                      if F_layer[f] not in ("context", "phase") and facets[f]["src"] == "asrs")
        scored = sorted(((k * np.log((k / len(members)) / (facets[f]["df"] / n)), f)
                         for f, k in cnt.items() if k >= 0.2 * len(members)), reverse=True)
        top = [facets[f]["label"] for s, f in scored if s > 0][:4]
        if not top:
            labels[members] = -1
            continue
        # Bigger regions pick first; a smaller one with the same top pair takes its next factor.
        name = next((f"{top[0]} · {b}" for b in top[1:] if f"{top[0]} · {b}" not in used),
                    " · ".join(top[:2]))
        used.add(name)
        clusters.append({"id": int(c), "n": int(len(members)), "label": name,
                         "x": round(float(np.median(xy[members, 0])), 4),
                         "y": round(float(np.median(xy[members, 1])), 4)})
    print(f"{len(clusters)} regions, {np.mean(labels >= 0):.0%} of reports in one", flush=True)

    # ---- when and where, for the page's date / time / place filters
    tod = [TIME_OF_DAY.index(v) if v in TIME_OF_DAY else None
           for v in raw["Time.1_Local Time Of Day"].astype(str)]
    state = [s if s in STATES else None
             for s in raw["Place.1_State Reference"].fillna("").astype(str).str.strip().str.upper()]
    loc = []
    for v in raw["Place_Locale Reference"].fillna("").astype(str):
        code = v.split(".")[0].strip().upper()
        # ZZZ / ZZZZ are ASRS's de-identified places.
        loc.append(code if re.fullmatch(r"[A-Z0-9]{2,5}", code) and not code.startswith("ZZZ") else None)

    # ---- write
    acns = reports["acn"].tolist()
    rows = []
    for i in range(n):
        r = {"acn": acns[i], "date": f"{str(reports['date'][i])[:4]}-{str(reports['date'][i])[4:6]}",
             "aircraft": str(reports["aircraft"][i]), "synopsis": reports["synopsis"][i], "f": rep_f[i],
             "nar": [[int(j), round(float(v), 2)] for j, v in zip(nar_i[i], nar_v[i]) if v > 0],
             "xy": [round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)]}
        if has_seq[i]:
            r["seq"] = [[int(j), round(float(v), 2)] for j, v in zip(seq_i[i], seq_v[i])
                        if v > 0 and has_seq[j]]
            r["events"] = events[i]
        if tod[i] is not None:
            r["t"] = tod[i]
        if state[i]:
            r["st"] = state[i]
        if loc[i]:
            r["loc"] = loc[i]
        if labels[i] >= 0:
            r["cl"] = int(labels[i])
        if sib[i]:
            r["sib"] = sorted(sib[i])
        rows.append(r)
    meta = {
        "n_reports": n, "n_llm_coded": int(has_seq.sum()),
        "layers": [{"id": "event", "label": "Event type"}, {"id": "phase", "label": "Flight phase"},
                   {"id": "human", "label": "Human factors & CRM"},
                   {"id": "approach", "label": "Route, navigation & automation"},
                   {"id": "weather", "label": "Weather & light"}, {"id": "system", "label": "Aircraft system"},
                   {"id": "response", "label": "Crew response & outcome"},
                   {"id": "context", "label": "Operation & aircraft"}],
        "calibration": {"narrative": nar_cal, "sequence": seq_cal},
        "source": "NASA Aviation Safety Reporting System (ASRS) via the elihoole/asrs-aviation-reports dataset (Apache-2.0)",
        "scope": f"All {n:,} reports in the dataset, every flight phase",
        "llm": "Qwen3-8B (4-bit, Ollama), run locally on the narrative only",
        "embeddings": "all-mpnet-base-v2, run locally, for the narratives (fused with TF-IDF LSA) and for the LLM's cause/effect links",
        "examples": [
            {"label": "RNAV / RNP approach, crew communication breakdown", "acn": "1330765"},
            {"label": "Hydraulic system problem, emergency declared", "acn": "1046434"},
            {"label": "Windshear on approach, go-around", "acn": "1703613"},
            {"label": "Unstabilized approach, go-around", "acn": "1317109"},
            {"label": "CFIT / terrain warning in IMC", "acn": "1256598"},
            {"label": "Landing gear problem, critical", "acn": "1839641"},
        ],
        "clusters": clusters, "shard": SHARD,
        "tod": ["00:01–06:00", "06:01–12:00", "12:01–18:00", "18:01–24:00"],
        "states": [{"id": s, "label": STATES[s], "n": k} for s, k in Counter(x for x in state if x).most_common()],
    }
    meta["examples"] = [x for x in meta["examples"] if x["acn"] in set(acns)]
    (out / "data").mkdir(parents=True, exist_ok=True)
    with open(out / "data" / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "facets": facets, "reports": rows}, f, ensure_ascii=False,
                  separators=(",", ":"))
    shard_dir = out / "data" / "narratives"
    shard_dir.mkdir(exist_ok=True)
    for old in shard_dir.glob("*.json"):
        old.unlink()
    for k in range(0, n, SHARD):
        with open(shard_dir / f"{k // SHARD}.json", "w", encoding="utf-8") as f:
            json.dump({acns[i]: narr[i] for i in range(k, min(n, k + SHARD))}, f,
                      ensure_ascii=False, separators=(",", ":"))
    old_single = out / "data" / "narratives.json"
    if old_single.exists():
        old_single.unlink()
    size = (out / "data" / "graph.json").stat().st_size / 1e6
    print(f"wrote {n} reports, {len(facets)} factors, graph.json {size:.1f} MB, "
          f"{(n + SHARD - 1) // SHARD} narrative shards", flush=True)


if __name__ == "__main__":
    main()

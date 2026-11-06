"""Extract a knowledge graph from every ASRS narrative with a local LLM (Ollama).

usage: python scripts/llm_graph.py [--model qwen3:8b] [--workers 4] [--limit N]
                                   [--host URL[,URL...]] [--data-dir data/processed]
                                   [--out data/graph/<model>_v1.jsonl]

Each report's narrative goes to a local model served by Ollama, which returns JSON
constrained to ``SCHEMA``: entity lists (aircraft, flight phases, events, causes, actors,
equipment, locations, outcomes) plus a few typed (subject, relation, object) triples. One
JSON line per report is appended to the output file, so an interrupted run resumes where it
stopped; failed reports are retried on the next run. Several Ollama servers (e.g. one per
GPU) can share the work: ``--host`` takes a comma-separated list and ``--workers`` is per host.
With ``--data-dir data/processed_v2`` (corpus v2) reports are keyed by document id.

Only the narrative is sent. The analyst synopsis is the query on the synopsis track and the
analyst codes define the topical silver labels, so showing either to the model would let
the graph leak the evaluation targets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from asrsbench.data import load_reports

PROMPT_VERSION = "v1"
MAX_WORDS = 1500  # ~2k tokens; 99%+ of narratives fit, longer ones are cut at the end

PHASES = ["Parked", "Taxi", "Takeoff", "Initial Climb", "Climb", "Cruise", "Descent",
          "Initial Approach", "Final Approach", "Landing", "Go-Around", "Unknown"]
ACTORS = ["Captain", "First Officer", "Pilot (single)", "Flight Crew", "ATC", "Ground Crew",
          "Maintenance", "Dispatch", "Cabin Crew", "Other Aircraft", "Other"]
RELATIONS = ["caused", "contributed_to", "led_to", "detected", "responded_with", "involved"]

# Length caps live in the schema so the sampling grammar enforces them: without them the
# model occasionally repeats one string until num_predict and returns truncated JSON.
_str = {"type": "string", "maxLength": 60}
_strs = {"type": "array", "items": _str, "maxItems": 6}
SCHEMA = {
    "type": "object",
    "properties": {
        "aircraft": _strs,
        "flight_phases": {"type": "array", "items": {"type": "string", "enum": PHASES},
                          "maxItems": 4},
        "events": _strs,
        "causes": _strs,
        "actors": {"type": "array", "items": {"type": "string", "enum": ACTORS},
                   "maxItems": 6},
        "equipment": _strs,
        "locations": _strs,
        "outcomes": _strs,
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"s": _str,
                               "r": {"type": "string", "enum": RELATIONS},
                               "o": _str},
                "required": ["s", "r", "o"],
            },
            "maxItems": 6,
        },
    },
    "required": ["aircraft", "flight_phases", "events", "causes", "actors", "equipment",
                 "locations", "outcomes", "triples"],
}

SYSTEM = f"""You extract a knowledge graph from NASA ASRS aviation safety report narratives.
Return JSON only, matching the schema. Use short canonical noun phrases (1-4 words, lower
case except proper names and codes), e.g. "altitude deviation", "runway incursion",
"tcas ra", "fatigue", "frequency congestion", "autopilot", "fms".
- aircraft: aircraft types or models mentioned (e.g. "B737-800", "Cessna 172"); [] if none.
- flight_phases: phases during which the events happened.
- events: what went wrong (anomalies, deviations, conflicts, malfunctions).
- causes: contributing factors stated or clearly implied by the reporter.
- actors: people or parties involved.
- equipment: aircraft systems, components or ATC/airport equipment involved.
- locations: airports, runways, taxiways, fixes, airspace, as written.
- outcomes: how the event ended or was resolved.
- triples: up to 6 (subject, relation, object) links between the phrases above,
  relation one of {RELATIONS}.
Do not invent facts that the narrative does not support. Keep each list to at most 6 items."""


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract(narrative: str, model: str, host: str, timeout: float = 300) -> dict:
    """Ask the model for one report's graph; returns the parsed JSON and token counts."""
    words = narrative.split()
    text = " ".join(words[:MAX_WORDS])
    resp = _post(f"{host}/api/chat", {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Narrative:\n{text}"}],
        "format": SCHEMA,
        "think": False,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 768},
    }, timeout)
    return {
        "graph": json.loads(resp["message"]["content"]),
        "truncated": len(words) > MAX_WORDS,
        "prompt_tokens": resp.get("prompt_eval_count"),
        "output_tokens": resp.get("eval_count"),
    }


def done_acns(path: Path) -> set[str]:
    """ACNs already extracted successfully in an earlier (possibly interrupted) run."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # half-written last line from a killed run
                continue
            if "graph" in rec:
                out.add(rec["acn"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--host", default="http://localhost:11434", help="comma-separated Ollama servers")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests per Ollama server")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pending reports")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"data/graph/{args.model.replace(':', '-')}_{PROMPT_VERSION}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    reports = load_reports(args.data_dir)
    hosts = args.host.split(",")
    done = done_acns(out)
    todo = reports[~reports["acn"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(reports)} reports, {len(done)} done, {len(todo)} to extract -> {out}",
          flush=True)

    t0, n_ok, n_fail, toks = time.time(), 0, 0, 0
    with out.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(args.workers * len(hosts)) as pool:
        futs = {pool.submit(extract, nar, args.model, hosts[k % len(hosts)]): acn
                for k, (acn, nar) in enumerate(zip(todo["acn"], todo["narrative"]))}
        for i, fut in enumerate(as_completed(futs), 1):
            acn = futs[fut]
            meta = {"acn": acn, "model": args.model, "prompt_version": PROMPT_VERSION}
            try:
                rec = {**meta, **fut.result()}
                n_ok += 1
                toks += rec["output_tokens"] or 0
            except Exception as e:  # logged and retried on the next run
                rec = {**meta, "error": f"{type(e).__name__}: {e}"[:300]}
                n_fail += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(futs):
                f.flush()
                dt = time.time() - t0
                eta = (len(futs) - i) * dt / i / 3600
                print(f"{i}/{len(futs)} ok={n_ok} fail={n_fail} {i / dt:.2f} rep/s "
                      f"{toks / dt:.0f} out-tok/s eta {eta:.1f} h", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

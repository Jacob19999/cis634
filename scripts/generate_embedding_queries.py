"""Generate evidence-grounded retrieval queries on local llama.cpp servers, resumably.

The request contains only a raw narrative or an official reference passage. Synopses,
probable causes, analyst codes and evaluation queries are never sent to the generator.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from asrsbench.finetuning import SEED, read_jsonl, validate_generated

PROMPT_VERSION = "aviation_retrieval_queries_v4"
GENERATOR = "Qwen3.5-27B-Q6_K"
SYSTEM = (
    "Create retrieval training examples from the supplied aviation text. Return JSON with "
    "an items array containing exactly three objects, each with query and evidence strings. "
    "Each query must be a different natural search request of 8 to 30 words, grounded only "
    "in the text. Select operationally informative aspects, not generic boilerplate. Do not mention "
    "the passage, section number, case identifier or the generation task. Do not invent "
    "facts, numbers, causation or regulatory violations. Evidence must be an exact, "
    "contiguous quotation of 8 to 35 words from the supplied text supporting that query. "
    "Use different supported aspects of the text. Output JSON only, without thinking."
)
INCIDENT = (
    " This input is an INCIDENT NARRATIVE. All three queries must start with 'Reports of' "
    "or 'Incidents involving'. Describe a concrete operational scenario, event sequence or "
    "equipment problem to search across OTHER reports. Do not ask what happened in this "
    "particular report. Do not include report-specific identifiers, runway numbers, taxiway "
    "letters or names. Avoid vague requests about 'the specific sensation', 'equipment "
    "problem status' or 'locate information'."
)
REFERENCE = (
    " This input is an OFFICIAL REFERENCE PASSAGE. All three queries must be practical "
    "questions beginning with What, How, When, Which, Where, Why, Can, Must, Does, Do or Is. "
    "The passage must directly answer each question. Do not write 'Reports of' or 'Incidents "
    "involving' searches and do not invent an incident that violated this reference."
)
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "minItems": 3,
          "maxItems": 3, "items": {"type": "object", "properties": {
              "query": {"type": "string"}, "evidence": {"type": "string"}},
              "required": ["query", "evidence"], "additionalProperties": False}}},
          "required": ["items"], "additionalProperties": False}


def generate(task: dict, host: str) -> dict:
    """Generate and automatically validate one task, with bounded transient retries."""
    last_error, quality_failure = "", False
    count = requested_items(task)
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["items"].update(minItems=count, maxItems=count)
    for attempt in range(3):
        try:
            kind = "Reference passage" if task["kind"] == "faa" else "Incident narrative"
            body = {"model": GENERATOR, "temperature": 0, "seed": SEED + attempt,
                    "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_object", "schema": schema},
                    "messages": [{"role": "system", "content": (SYSTEM +
                                  (REFERENCE if task["kind"] == "faa" else INCIDENT)).replace(
                                      "exactly three", f"exactly {count}").replace("All three", "All")},
                                 {"role": "user", "content": kind + ":\n" + task["positive"]}]}
            req = urllib.request.Request(host + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.loads(response.read())
            raw = result["choices"][0]["message"]["content"]
            obj = json.loads(raw)
            valid, rejected = [], []
            for item in obj.get("items", []):
                ok, reason = validate_generated(item, task["positive"])
                if task["kind"] == "synthetic" and not item.get("query", "").startswith(
                        ("Reports of ", "Incidents involving ")):
                    ok, reason = False, "incident_query_style"
                if task["kind"] == "faa" and not item.get("query", "").startswith(
                        ("What ", "How ", "When ", "Which ", "Where ", "Why ", "Can ", "Must ",
                         "Does ", "Do ", "Is ")):
                    ok, reason = False, "reference_query_style"
                if ok and item["query"] not in {p["query"] for p in valid}:
                    valid.append(item)
                else:
                    rejected.append({"item": item, "reason": reason})
            if len(valid) < min(2, count) or result["choices"][0].get("finish_reason") == "length":
                raise ValueError(f"only {len(valid)} supported queries, or token cap reached")
            return {"task_id": task["task_id"], "items": valid, "rejected": rejected,
                    "generator": GENERATOR, "prompt_version": PROMPT_VERSION,
                    "requested_items": count,
                    "request_seed": SEED + attempt, "usage": result.get("usage"),
                    "server_model": result.get("model"), "raw": raw}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            quality_failure = isinstance(exc, (ValueError, KeyError, TypeError))
            time.sleep(1 + attempt)
    return {"task_id": task["task_id"], "error": last_error, "rejected_task": quality_failure,
            "generator": GENERATOR,
            "prompt_version": PROMPT_VERSION}


def requested_items(task: dict) -> int:
    """Ask short definitions for fewer questions instead of forcing redundant paraphrases."""
    if task["kind"] != "faa":
        return 3
    n = len(task["positive"].split())
    return 1 if n < 60 else 2 if n < 100 else 3


def main() -> None:
    """Resume pending tasks, recording both accepted items and failed attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/finetuning/aviation_embedding_v1")
    parser.add_argument("--hosts", default="http://127.0.0.1:8080,http://127.0.0.1:8081,http://127.0.0.1:8082")
    parser.add_argument("--per-host", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    data = Path(args.data)
    tasks = read_jsonl(data / "generation_tasks.jsonl")
    task_lookup = {p["task_id"]: p for p in tasks}
    path = data / "generated.jsonl"
    done = set()
    if path.exists():
        # Repair only a partial final line in this script's own resumable output.
        lines = path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            path.write_text("".join(lines[:-1]))
        done = {p["task_id"] for p in read_jsonl(path)
                if (len(p.get("items", [])) >= min(2, requested_items(task_lookup[p["task_id"]]))
                    or p.get("rejected_task"))
                and p.get("prompt_version") == PROMPT_VERSION}
    pending = [p for p in tasks if p["task_id"] not in done]
    # A limited pilot spans all three incident sources and each FAA reference family.
    strata = defaultdict(list)
    for task in pending:
        strata[task.get("family", task["doc_id"].split(":")[0])].append(task)
    pending = []
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                pending.append(strata[key].pop())
    if args.limit:
        pending = pending[:args.limit]
    hosts = args.hosts.split(",")
    print(f"{len(done)} finished; {len(pending)} pending this invocation", flush=True)
    start, errors, rejected = time.time(), 0, 0
    with path.open("a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=len(hosts) * args.per_host) as pool:
        futures = {pool.submit(generate, task, hosts[i % len(hosts)]): task
                   for i, task in enumerate(pending)}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            errors += bool(result.get("error")) and not result.get("rejected_task", False)
            rejected += bool(result.get("rejected_task"))
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0 or i == len(pending):
                elapsed = time.time() - start
                print(f"{i}/{len(pending)} tasks; {i / elapsed:.2f} tasks/s; "
                      f"eta {(len(pending) - i) * elapsed / i / 3600:.2f} h; "
                      f"errors={errors}; quality_rejected={rejected}",
                      flush=True)
    if errors:
        raise SystemExit(f"{errors} failed tasks: rerun to retry before sealing")


if __name__ == "__main__":
    main()

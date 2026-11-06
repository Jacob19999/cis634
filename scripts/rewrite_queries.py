"""Rewrite a config's queries into reporter vocabulary with a local LLM and FAA glossary context.

usage: python scripts/rewrite_queries.py CONFIG [--model Qwen/Qwen3-8B]
                                                [--out data/queries/rewrites/<model>_pcg_v1.jsonl]

For every query of the config's synopsis and topical tracks, the five Pilot/Controller
Glossary entries that BM25 finds closest to the query (``data/glossary/pcg_v1.jsonl``) go
into the prompt, and the model writes a short passage describing the same event the way a
pilot or controller would in an ASRS narrative (query2doc-style; Wang et al., 2023). A
retriever uses the file through ``query_rewrites`` (see ``Retriever.rewrite``).

Only query text reaches the model: no narratives, synopses of other reports, or labels.
The prompt is fixed as ``PROMPT_VERSION``; it was written before any results were seen,
and a changed prompt needs a new version and output file. One JSON line per query is
appended, keyed by the query text, so a rerun skips queries already done and a later run
with another config adds its queries to the same file.

Runs with Hugging Face ``transformers`` on one CUDA GPU in float16 (Volta GPUs have no
bfloat16; recent vLLM releases do not support them).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from asrsbench.benchmark import _query_sets, load_config
from asrsbench.data import load_reports
from asrsbench.retrievers import BM25Retriever, Units

PROMPT_VERSION = "pcg_v1"
N_CONTEXT = 5
CONTEXT_WORDS = 50  # per glossary definition
SYSTEM = (
    "You help search NASA ASRS aviation safety reports. Reports are first-person narratives "
    "by pilots, air traffic controllers, mechanics and cabin crew. Rewrite the user's search "
    "query as a short passage (2-3 sentences, at most 60 words) describing the same event in "
    "the words a reporter would use in a narrative: the operational terms, acronyms and "
    "phrases pilots and controllers actually say. Keep every specific detail of the query "
    "(flight phase, numbers, who did what, negations). Do not add details the query does "
    "not imply. Use the glossary entries only where they fit. Output only the passage."
)


def glossary_index(path: str = "data/glossary/pcg_v1.jsonl") -> tuple[BM25Retriever, list[str]]:
    """BM25 over glossary entries (term + definition); returns the index and entry texts."""
    entries = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [f"{e['term']}: {' '.join(e['definition'].split()[:CONTEXT_WORDS])}" for e in entries]
    ids = np.array([str(i) for i in range(len(texts))])
    bm = BM25Retriever("pcg", stopwords="english")
    bm.index(Units(texts, ids, ids, np.arange(len(texts)), "pcg"))
    return bm, texts


def prompts(queries: list[str], tok, bm: BM25Retriever, entries: list[str]):
    """Chat-formatted prompts and the glossary terms each one was given."""
    out, terms = [], []
    for q, res in zip(queries, bm.search(queries, k=N_CONTEXT)):
        ctx = [entries[int(i)] for i in res.doc_ids]
        terms.append([c.split(":", 1)[0] for c in ctx])
        user = ("Glossary entries (FAA Pilot/Controller Glossary):\n"
                + "\n".join(f"- {c}" for c in ctx) + f"\n\nQuery: {q}")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False))
    return out, terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    slug = args.model.split("/")[-1].lower()
    out = Path(args.out or f"data/queries/rewrites/{slug}_{PROMPT_VERSION}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("rewrite"):
                done.add(rec["text"])

    cfg = load_config(args.config)
    qsets = _query_sets(cfg, load_reports(cfg["data_dir"]), None)
    queries = list(dict.fromkeys(t for qs in qsets.values() for t in qs.queries["text"]))
    todo = [q for q in queries if q not in done]
    print(f"{len(queries)} queries in {args.config}, {len(todo)} to rewrite -> {out}", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                 device_map="cuda:0").eval()
    bm, entries = glossary_index()
    t0, n_bad = time.time(), 0
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            texts, terms = prompts(batch, tok, bm, entries)
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for q, o, t in zip(batch, outs, terms):
                o = " ".join(o.replace("<think>", "").replace("</think>", "").split())
                n_bad += not o
                f.write(json.dumps({"text": q, "rewrite": o, "model": args.model,
                                    "prompt_version": PROMPT_VERSION, "context": t},
                                   ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            print(f"{i + len(batch)}/{len(todo)} {dt / (i + len(batch)):.2f} s/query "
                  f"empty={n_bad}", flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()

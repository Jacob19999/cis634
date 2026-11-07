"""Summarise every ASRS narrative with a local LLM, for indexing next to the chunks.

usage: python scripts/summarise_reports.py [--model Qwen/Qwen3-8B] [--shard 0 --n-shards 3]
       python scripts/summarise_reports.py --backend ollama --model qwen3.5-27b-q6k \
           --hosts http://127.0.0.1:11434,http://127.0.0.1:11435 --data-dir data/processed_v2 \
           --order NTSB:,ASRSN:,ASRS:

Writes ``data/summaries/<model>_v1.jsonl`` (one JSON line per report: acn, summary), or one
``...v1.<shard>.jsonl`` file per shard so that several GPUs can run at once; the loader
(``asrsbench.data.load_summaries``) reads all shards. An interrupted run resumes.

Only the narrative is sent, cut at ``MAX_WORDS``. The analyst synopsis is the query on the
synopsis track and the analyst codes define the topical silver labels, so the model never
sees either, and the prompt does not ask for the synopsis's house style. Because the ASRS
database is public, the model may still have memorised synopses during pre-training:
``scripts/summary_overlap.py`` measures that. The prompt is fixed as ``PROMPT_VERSION``;
a changed prompt needs a new version and output file.

Two backends: Hugging Face ``transformers`` in float16 on one CUDA GPU per process
(``--backend hf``), or one or more Ollama servers, e.g. one per GPU serving a GGUF quant
(``--backend ollama``) or llama.cpp ``llama-server`` instances (``--backend llamacpp``; on
Volta GPUs, where Ollama's bundled CUDA 13 build has no kernels), with ``--per-host`` requests
in flight on each. Corpus v2
(``--data-dir data/processed_v2``) keys reports by document id (``NTSB:<ev_id>`` etc.), and
``--order`` sets which sources go first.

``--redo FILE`` (one doc id per line) summarises those reports again even if they have a
summary, e.g. with a larger ``--max-new-tokens`` for summaries cut off at the cap. The new
records are appended and, being last, replace the old ones in ``load_summaries``.
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
MAX_WORDS = 1500  # as in scripts/llm_graph.py; covers > 99% of narratives
SYSTEM = (
    "You summarise aviation safety report narratives written by pilots, air traffic "
    "controllers, mechanics and cabin crew. Write a summary of at most two sentences "
    "(50 words) saying who was involved, the aircraft or phase of flight, what happened, "
    "and the cause or outcome as the reporter describes it. Use plain operational language "
    "and keep specific details such as altitudes, equipment and clearances. Do not add "
    "anything the narrative does not say. Output only the summary."
)


def ollama_summary(host: str, model: str, narrative: str, max_new_tokens: int) -> str:
    """One summary from an Ollama server (thinking off, greedy decoding)."""
    body = {"model": model, "stream": False, "think": False, "keep_alive": "60m",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user",
                          "content": "Narrative:\n" + " ".join(narrative.split()[:MAX_WORDS])}],
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": max_new_tokens}}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def llamacpp_summary(host: str, model: str, narrative: str, max_new_tokens: int) -> str:
    """One summary from a llama.cpp ``llama-server`` (OpenAI-style API, thinking off, greedy)."""
    body = {"model": model, "temperature": 0, "max_tokens": max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user",
                          "content": "Narrative:\n" + " ".join(narrative.split()[:MAX_WORDS])}]}
    req = urllib.request.Request(f"{host}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def run_ollama(todo, out: Path, args) -> int:
    """Summarise ``todo`` on the servers in ``args.hosts`` (Ollama or llama-server); returns
    the exit code."""
    hosts = args.hosts.split(",")
    call = llamacpp_summary if args.backend == "llamacpp" else ollama_summary
    acns, nars = todo["acn"].tolist(), todo["narrative"].tolist()
    t0, n_ok, n_bad = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(len(hosts) * args.per_host) as pool:
        futs = {pool.submit(call, hosts[i % len(hosts)], args.model, n,
                            args.max_new_tokens): i for i, n in enumerate(nars)}
        for k, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            rec = {"acn": acns[i], "model": args.model, "backend": args.backend,
                   "prompt_version": PROMPT_VERSION, "max_new_tokens": args.max_new_tokens,
                   "truncated": len(nars[i].split()) > MAX_WORDS}
            try:
                rec["summary"] = " ".join(fut.result().split())
                n_ok += bool(rec["summary"])
                n_bad += not rec["summary"]
            except Exception as e:  # logged; the report is retried on the next run
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                n_bad += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 200 == 0 or k == len(nars):
                f.flush()
                dt = time.time() - t0
                print(f"{k}/{len(nars)} {k / dt:.2f} reports/s, eta {(len(nars) - k) * dt / k / 3600:.1f} h "
                      f"ok={n_ok} failed={n_bad}", flush=True)
    return 1 if n_bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--word-budget", type=int, default=8000,
                    help="max narrative words per batch (GPU memory)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pending reports")
    ap.add_argument("--backend", choices=["hf", "ollama", "llamacpp"], default="hf")
    ap.add_argument("--hosts", default="http://127.0.0.1:11434",
                    help="comma-separated Ollama servers (--backend ollama)")
    ap.add_argument("--per-host", type=int, default=8, help="requests in flight per Ollama server")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--order", default="", help="comma-separated id prefixes to summarise first")
    ap.add_argument("--redo", default=None,
                    help="file of doc ids (one per line) to summarise again even if done")
    args = ap.parse_args()

    slug = args.model.split("/")[-1].lower()
    suffix = f".{args.shard}" if args.n_shards > 1 else ""
    out = Path(f"data/summaries/{slug}_{PROMPT_VERSION}{suffix}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # half-written last line from a killed run
                continue
            if rec.get("summary"):
                done.add(rec["acn"])

    reports = load_reports(args.data_dir)
    reports = reports.iloc[args.shard :: args.n_shards]
    redo = set(Path(args.redo).read_text().split()) if args.redo else set()
    if redo - set(reports["acn"]):
        sys.exit(f"--redo: {len(redo - set(reports['acn']))} ids are not in {args.data_dir}")
    todo = reports[reports["acn"].isin(redo) if redo else ~reports["acn"].isin(done)]
    if args.order:
        prefixes = args.order.split(",")
        rank = todo["acn"].map(lambda a: next((i for i, p in enumerate(prefixes) if a.startswith(p)),
                                              len(prefixes)))
        todo = todo.assign(_r=rank).sort_values(["_r", "acn"], kind="stable")
    elif args.backend == "hf":
        # Similar lengths in a batch waste less padding; longest first surfaces memory errors early.
        todo = todo.assign(_n=todo["n_words"]).sort_values("_n", ascending=False)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"shard {args.shard}/{args.n_shards}: {len(reports)} reports, {len(done)} done, "
          f"{len(todo)} to summarise -> {out}", flush=True)
    if args.backend in ("ollama", "llamacpp"):
        sys.exit(run_ollama(todo, out, args))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                 device_map="cuda:0").eval()
    def generate(narratives: list[str]) -> list[str]:
        """Summaries for one batch; on CUDA out-of-memory, halve the batch and retry."""
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Narrative:\n" + " ".join(n.split()[:MAX_WORDS])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for n in narratives]
        try:
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            if len(narratives) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(narratives) // 2
            return generate(narratives[:half]) + generate(narratives[half:])
        return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # Batches hold at most ``batch_size`` reports and ``word_budget`` narrative words: Volta
    # GPUs have no memory-efficient attention, so 24 long narratives at once run out of memory.
    acns, nars = todo["acn"].tolist(), todo["narrative"].tolist()
    batches, cur, words = [], [], 0
    for j, n in enumerate(nars):
        w = min(len(n.split()), MAX_WORDS)
        if cur and (len(cur) == args.batch_size or words + w > args.word_budget):
            batches.append(cur)
            cur, words = [], 0
        cur.append(j)
        words += w
    if cur:
        batches.append(cur)

    t0, n_bad, k = time.time(), 0, 0
    with out.open("a", encoding="utf-8") as f:
        for b, idx in enumerate(batches):
            outs = generate([nars[j] for j in idx])
            for j, o in zip(idx, outs):
                o = " ".join(o.replace("<think>", "").replace("</think>", "").split())
                n_bad += not o
                f.write(json.dumps({"acn": acns[j], "summary": o, "model": args.model,
                                    "prompt_version": PROMPT_VERSION,
                                    "truncated": len(nars[j].split()) > MAX_WORDS},
                                   ensure_ascii=False) + "\n")
            f.flush()
            k += len(idx)
            if b % 20 == 0 or k == len(acns):
                dt = time.time() - t0
                print(f"{k}/{len(acns)} {k / dt:.2f} reports/s, eta {(len(acns) - k) * dt / k / 3600:.1f} h "
                      f"empty={n_bad}", flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()

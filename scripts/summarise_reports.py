"""Summarise every ASRS narrative with a local LLM, for indexing next to the chunks.

usage: python scripts/summarise_reports.py [--model Qwen/Qwen3-8B] [--shard 0 --n-shards 3]

Writes ``data/summaries/<model>_v1.jsonl`` (one JSON line per report: acn, summary), or one
``...v1.<shard>.jsonl`` file per shard so that several GPUs can run at once; the loader
(``asrsbench.data.load_summaries``) reads all shards. An interrupted run resumes.

Only the narrative is sent, cut at ``MAX_WORDS``. The analyst synopsis is the query on the
synopsis track and the analyst codes define the topical silver labels, so the model never
sees either, and the prompt does not ask for the synopsis's house style. Because the ASRS
database is public, the model may still have memorised synopses during pre-training:
``scripts/summary_overlap.py`` measures that. The prompt is fixed as ``PROMPT_VERSION``;
a changed prompt needs a new version and output file.

Runs with Hugging Face ``transformers`` in float16 on one CUDA GPU per process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    reports = load_reports()
    reports = reports.iloc[args.shard :: args.n_shards]
    todo = reports[~reports["acn"].isin(done)]
    # Similar lengths in a batch waste less padding; longest first surfaces memory errors early.
    todo = todo.assign(_n=todo["n_words"]).sort_values("_n", ascending=False)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"shard {args.shard}/{args.n_shards}: {len(reports)} reports, {len(done)} done, "
          f"{len(todo)} to summarise -> {out}", flush=True)

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

# Handover: corpus v2 (ASRS + NTSB) → embeddings → Qwen3.5-27B summaries → Precedent Graph page

Last updated 2026-11-06 05:15 UTC by Claude (Opus 5.5). Read `AGENTS.md` first; this file
covers only the work in progress. Project owner: Jacob Tang (with Aarambh Shrestha).

## Goal

The ASRS Precedent Graph page shows **ASRS and NTSB reports for the same period, 2010–2026**.
- The live page is `portfolio/asrs-precedent-graph/` in `Jacob19999/triaggent_consulting`
  (Vercel, https://triaggent-consulting.vercel.app).
- Its data comes from this repo, `Jacob19999/cis634`, the source of truth.

The approved plan (full text: `~/.claude/plans/pure-kindling-kurzweil.md`) has two phases.

**Phase A (approved, in progress):**

| Step | What | Status |
|---|---|---|
| A1 | Build unified corpus v2 | **done** |
| A2 | Embed with Qwen3-Embedding-8B | **done** |
| A3 | Summarise every report with Qwen3.5-27B Q6_K | **running**, ~19–22 h, started 05:05 UTC 2026-11-06 |
| A4 | Qwen3-8B knowledge graph for all reports (`scripts/llm_graph.py`) | not started; GPUs are full until A3 ends |
| A5 | Export page data, update the page, preview locally | code written, not run |
| A6 | Commit, push, PRs; Vercel preview | not done; **merging to the live site needs the user's OK** |

**Phase B (later, needs its own go-ahead):** fine-tune Qwen3-Embedding-8B with LoRA on a mix:
- ASRS synopsis pairs and second narratives,
- synthetic issue queries from Qwen3.5-27B,
- NTSB probable cause → narrative,
- FAR/AIM pairs (about 5%).

Train splits only, with 10% of ASRS train `group_id`s held out. The user allowed non-frozen
embeddings for the UI if they beat frozen on dev; the frozen results stay the research
baseline.

## Server

- **Machine:** JupyterHub `https://cis-neuron.mnsu.edu/user/15835731/`, user `15835731`. Needs the
  **MNSU VPN**: off-campus connections time out (HTTP 000).
- **Hardware:** 3 × Quadro GV100 32 GB (Volta, compute capability 7.0: fp16 only, no bf16, no
  FlashAttention-2), 56 cores, 125 GB RAM, `/dev/shm` 2 GB.
- **Persistent:** only `~/work` and `~/.cache`. Everything else in `$HOME` is temporary.
- **Repo:** `~/work/cis634`, branch `llm-context` at `54ac435`, plus uploaded uncommitted files
  (see "Code state"). Venv `.venv`, created with `--system-site-packages` on conda Python 3.13 /
  torch 2.10+cu126. Always `. .venv/bin/activate`: it sets
  `LD_LIBRARY_PATH=~/.conda/lib`, which conda's SciPy needs.
- **Access:** a JupyterHub API token in `$JUPYTERHUB_TOKEN`, set by the user in `~/.zshrc` on the
  Mac. **Never print it, paste it, or ask for it in chat.** The user revokes it at
  `/hub/token` when done.
- **Helpers in `agents/tools/`** (they need `uv`; on the Mac they use the macOS trust store via
  `truststore`):
  - `jr 'shell cmd'`: runs a command on the server through a temporary kernel and prints its
    output. Set `JRUN_TIMEOUT=<s>` for long commands (default 120).
  - `jfetch.py <local_repo> <path>...`: copies files or folders from `~/work/cis634` to the Mac,
    skipping `runs/`.
  - Uploading a file uses the contents API: `PUT /user/15835731/api/contents/cis634/<path>`
    with `{"type":"file","format":"text","content":...}`. The contents API's root is `~/work`.
- **User's own view:** a JupyterLab terminal at `/user/15835731/lab/terminals/1`; `sh ~/work/status.sh`
  prints progress.

### Server gotchas (all hit in practice)

- **Start long jobs with `setsid nohup … < /dev/null &`.** Plain `nohup` jobs launched through a
  temporary kernel were killed when that kernel closed.
- **`pkill -f PATTERN` can kill the `jr` shell itself** if the pattern appears in the command
  line. Kill by PID, or use the `[x]yz` regex trick, and keep the pattern string out of the rest
  of the command.
- **`cd dir && cmd > log &` backgrounds the whole chain,** so the `cd` doesn't apply to later
  commands. Use absolute paths.
- **Ollama v0.40.15 can't use the GPUs:** its CUDA 13 build skips sm_70 and its CUDA 12 GPU
  discovery times out. **Use llama.cpp's `llama-server` from the Ollama bundle with the CUDA 12
  backend:** `~/work/llamacpp_v12/` holds symlinks to `~/work/ollama/lib/ollama/*` plus
  `cuda_v12/*`, run with `LD_LIBRARY_PATH=$PWD`.
- **Multi-GPU encoding:** sentence-transformers' own pool shares the model through `/dev/shm` and
  fails. `DenseRetriever` now starts one process per device instead (`devices:` option).

## Current jobs (as of 05:15 UTC)

**Three llama-servers**, one per GPU:
- Ports 8080, 8081 and 8082 on 127.0.0.1, about 27.7 GB per GPU.
- Started in `~/work/llamacpp_v12` with:
  ```
  CUDA_VISIBLE_DEVICES=$g setsid nohup ./llama-server -m ~/work/models/Qwen3.5-27B-Q6_K.gguf \
    -ngl 999 -c 65536 -np 16 -fa on --jinja --host 127.0.0.1 --port 808$g > srv$g.log 2>&1 < /dev/null &
  ```
- Health check: `curl 127.0.0.1:8080/health`.

**The summariser** (PID 25053 at launch):
```
python scripts/summarise_reports.py --backend llamacpp --model qwen3.5-27b-q6k \
  --hosts http://127.0.0.1:8080,http://127.0.0.1:8081,http://127.0.0.1:8082 --per-host 16 \
  --data-dir data/processed_v2 --order NTSB:,ASRSN:,ASRS:
```
- Log: `~/work/cis634/logs/summ27b.log`.
- Output: `~/work/cis634/data/summaries/qwen3.5-27b-q6k_v1.jsonl`, keyed by doc id.
- Resumable: rerun the same command and it skips finished reports and retries failed ones.
- Order: NTSB first, then the narrative-only ASRS reports, then ASRS v1.
- Rate: 1.2–1.3 reports/s. Expect it to finish around 02:00–03:00 UTC on 2026-11-07.

**Pilot results:** 99 reports, about 45 words each, faithful to the narratives, no thinking
text. 16 slots with flash attention gave 0.48 reports/s per GPU, against 0.42 with 8 slots.
Q4_K_M would be about 25–30% faster; the user chose Q6_K.

## What exists (data)

**`~/work/cis634/data/processed_v2/`** (git-ignored), built by `scripts/build_corpus_v2.py`:

| Source | Reports | Years | Id prefix | Notes |
|---|---|---|---|---|
| ASRS v1 | 47,723 | Jan 2012 – Mar 2022 | `ASRS:` | |
| NTSB (`avall.mdb`) | 24,557 | 2010–2026 | `NTSB:` | 21,347 have a probable cause |
| ASRS narrative only (`sookiemonster/asrs-narratives`) | 19,547 | ≈ Jun 2021 – Dec 2025 | `ASRSN:` | No license stated; text not republished without the user's decision |

- **Size:** 233,852 chunks.
- **Ids:** `acn` = doc id; `display_id` = the ACN or the NTSB case number (e.g. ERA10LA103).
- **Estimated dates:** the narrative-only reports' months come from ACN ↔ cumulative ASRS
  monthly intake (`data/asrs/callback_intake_v1.json`, from 102 CALLBACK issues). Fit
  ACN ≈ c + 1.004 × intake; held-out error 0.33 months. Flagged `date_estimated`.
- **NTSB text repair:** typographic characters came out as control bytes (`\x19` = ’);
  `fix_ntsb_text` repairs them.

**Raw inputs** in `~/work/cis634/data/external/` (git-ignored):
- `ntsb/avall.zip` (sha256 `c1a44783…bbc`, rebuilt 2026-11-01), `avall.mdb`, plus parquet
  exports of `events`, `aircraft`, `narratives`, `Findings`, `Events_Sequence`, `injury` and
  `ct_seqevt`,
- `asrs_hf_narratives.parquet`,
- `faa/` (the FAA PDFs).

**Embeddings** (`~/work/cis634/cache/embeddings/`): Qwen3-Embedding-8B, fp16, 512 tokens, one
cache file per source block. The v1 ASRS block is the old v1 cache; the NTSB and narrative-only
blocks were written at 04:52. `configs/corpus_v2.yaml` loads them, and
`asrsbench search configs/corpus_v2.yaml qwen3-8b "…"` returns mixed ASRS and NTSB results.

**Models** in `~/work/models/` (`Qwen3.5-27B-Q6_K.gguf`, 22.5 GB) and `~/work/ollama/` (bundle;
`qwen3.5-27b-q6k` registered, but Ollama itself doesn't work on these GPUs).

## Code state

**cis634, Mac `~/Documents/cis634`, branch `llm-context`:** all corpus-v2 work is **uncommitted**.
- New:
  - `scripts/build_corpus_v2.py`, `scripts/fetch_asrs_intake.py`,
    `scripts/export_precedent_graph_v2.py`
  - `configs/corpus_v2.yaml`, `data/asrs/callback_intake_v1.json`
  - `tests/test_corpus_v2.py`, `tests/test_export_v2.py`
  - `agents/` (this file and the tools)
- Changed:
  - `retrievers/dense.py` (per-source cache blocks, `devices:`)
  - `scripts/summarise_reports.py` (Ollama / llama.cpp backends, `--data-dir`, `--order`)
  - `scripts/llm_graph.py` (comma-separated `--host`, `--data-dir`)
  - `tests/test_retrievers.py`, `AGENTS.md`, `.gitignore`
- Tests: `.venv/bin/pytest -q` → 39 passed.
- The same files are uploaded to the server checkout. Plan from A6: **commit on a new branch
  `corpus-v2` stacked on `llm-context`.** Commits carry **no Claude co-author or attribution
  line**: the user's standing preference, and it covers PR descriptions too.

**Open PRs, both unmerged:**
- [Jacob19999/cis634#3](https://github.com/Jacob19999/cis634/pull/3), tutorial → main.
- [Jacob19999/cis634#4](https://github.com/Jacob19999/cis634/pull/4), `llm-context`, stacked on #3:
  FAA acronym expansion and LLM query rewrites (both negative on dev), summaries, Qwen3 GPU runs.
  It confirms chunks + summary: synopsis MRR@10 +0.019 to +0.034 (p ≤ 0.008), topical neutral,
  robust to the memorisation check.

**Page repo:** fresh clone at `~/Documents/triaggent_consulting_jacob`, branch
`asrs-ntsb-corpus-v2`. **Do not use `~/Documents/BData/triaggent_consulting`:** its remote is
`eatingcereal/…` and it doesn't have the page.
- `portfolio/asrs-precedent-graph/index.html` is patched (uncommitted) by
  `agents/tools/patch_page.py`, 32 exact edits:
  - loads `data/graph.json` and then `data/reports/<k>.json` shards,
  - `src` field, source filter and "colour by source" map mode,
  - `idLabel` (ACN vs NTSB case number) and `dateLabel` (≈ for estimated dates),
  - "NTSB probable cause" wording,
  - narrative shards holding `{n, s}`, the narrative plus the 27B summary,
  - 2010–2026 year bands.
- Its script passes `node --check`. **Not yet viewed in a browser:** that needs the exported
  data.

## Remaining steps

1. **A3:** watch `logs/summ27b.log` or `sh ~/work/status.sh`. When it ends, check that all
   91,827 doc ids have a non-empty summary; rerun the command to fill any gaps. Then:
   - stop the llama-servers, by PID;
   - embed the summaries with Qwen3-8B, by adding a `doc_mode: summary` system with
     `summaries: data/summaries/qwen3.5-27b-q6k_v1.jsonl` to a config;
   - run `scripts/summary_overlap.py data/summaries/qwen3.5-27b-q6k_v1.jsonl` (memorisation
     check: ASRS synopses and NTSB causes). It may need `--data-dir` support first: it currently
     reads `load_reports()` (v1).
2. **A4:** a Qwen3-8B knowledge graph for all 91,827 reports.
   - `llm_graph.py` talks to Ollama, which doesn't work on these GPUs. Either add a
     llama.cpp/OpenAI backend (llama-server supports `response_format` with a JSON schema), or
     run Ollama CPU-only (too slow).
   - The original graph used `qwen3:8b` (4-bit). A Qwen3-8B GGUF (e.g. Q4_K_M) is needed; that
     download needs the user's OK.
   - Output: `data/graph/qwen3-8b_v1_corpus_v2.jsonl`, keyed by doc id. Estimate 4–6 h on
     3 GPUs.
3. **A5:** on the server, run
   `python scripts/export_precedent_graph_v2.py <page dir>` (it needs the GPU for MPNet and
   the neighbour search).
   - Copy `data/graph.json`, `data/reports/` and `data/narratives/` into the page clone.
   - Preview with `python3 -m http.server` in the page folder, using the in-app browser pane.
   - Check: quick load, the source and date filters, ASRS ↔ NTSB neighbours, no console errors.
   - Check the report shard sizes stay well under 100 MB.
4. **A6:** commit cis634 (`corpus-v2`) and the page repo (`asrs-ntsb-corpus-v2`), push, and
   open PRs. Check the Vercel preview. **Ask the user before merging to the live site.**
   Then list any README or report numbers that changed (none expected: v1 results are
   untouched).

## Research-integrity reminders (from AGENTS.md)

- Benchmark results and v1 files stay untouched.
- Design choices use dev splits only.
- Never hand-edit `results/`.
- Summaries and the knowledge graph see the narrative only, never synopses, probable causes or
  analyst codes.
- Ask before deleting anything under `cache/` or `data/processed/`.

# 8B aviation embedding fine-tuning

The user approved going directly to Qwen3-Embedding-8B LoRA for three epochs, including
FAR/AIM at 5%, on 7 November 2026. The 0.6B pilot is intentionally skipped. The actual
8B hardware smoke test is separate from the three-epoch experiment.

## Live run

The detached driver was launched on `cis-neuron` on 7 November 2026, at 07:20 UTC.
Last verified driver PID: **207096**. It is in **generation**, with 13,590 tasks pending
after resuming the inspected pilot. Always read `logs/finetune_v1/status.json` for current
state rather than treating this snapshot as live. Generators belong to this run: PIDs
205930, 205931 and 206807, also recorded in `generator_pids.json`.

The final generator prompt is `aviation_retrieval_queries_v4`; older pilot attempts stay
in the append-only generation log but are excluded when sealing. The inspected v4 pilot
has 24 tasks, four per source/reference family, and 61 supported query pairs with no errors.
Short reference definitions request fewer questions to avoid redundant paraphrases. This
was an agent inspection, not a human annotation study. The review record is at
`cache/finetune_pilot_review.json` on both machines.

Single-GPU and three-GPU 8B smoke runs passed real optimiser updates, adapter save/reload
and finite embedding checks. The chosen microbatch is 8 inside a contrastive batch of 32
per GPU. The distributed smoke used about 15.57 GiB peak allocation, took 30.28 seconds
for two updates, and reloaded within 0.000040 per embedding component. There are 15,335,424
trainable LoRA parameters. PEFT was upgraded **inside the project venv** to 0.20.0 because
Transformers 5.22.1 requires it; the system installation remains unchanged.

The base model is pinned to revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
The full three-epoch fit has not started at this snapshot. It starts automatically once
generation passes and the corpus is sealed. The driver then evaluates frozen and all three
epochs and writes the analysis. It records a failed stage and stops on unexpected failures.
Generation transport errors are retried. Quality-rejected tasks are excluded and counted;
sealing fails if more than 10% of tasks are rejected or a required pool is too small.

The frozen dev preflight succeeded over all 91,827 reports / 233,852 chunks. Its outputs
have been fetched to `results/finetune_8b_v1/frozen/`. The driver will regenerate that
condition before epoch comparisons. These are new dev results, not replacements for any
reported frozen benchmark. Local offline validation: **50 tests passed**.

Approved sampling weights: 35% ASRS synopsis, 30% synthetic incident queries, 25% NTSB
probable cause, 5% paired reporter narratives, 5% FAA reference questions. Each sampled
epoch has 100,000 pairs; oversampling is explicit and distinct from unique pair counts.

## Data and isolation

- Corpus: `data/finetuning/aviation_embedding_v1/` (ignored, regenerable).
- Preparation merges existing incident groups, exact narrative-component duplicates and
  conservative near duplicates before splitting. Components touching official validation
  or test data are protected; 10% of remaining incident components become dev.
- The completed audit found 48 near-duplicate links and excluded 30 nominal training
  reports through protected components. It does not establish every cross-source incident link.
- Eligible incident reports after the new dev holdout: 34,750 original ASRS, 14,041 newer
  ASRS, 17,733 NTSB. The support-filtered supervised training pools have 32,187 synopsis,
  15,066 cause and 7,184 second-narrative pairs.
- A second narrative is queried against narrative 1 alone, never a concatenation containing
  the query itself. Human queries must overlap the selected source passage; this lexical
  screen is not a human relevance judgement.
- Synthetic generation sees a raw narrative passage or official reference text only.
  Generated questions need a supporting exact quotation and pass automatic numeric and
  overlap checks. A source-stratified pilot is inspected before the full generator starts.
- FAA source snapshots and URL/checksum manifest: `data/external/finetune_faa_v1/`.
  Sources: FAA AIM effective 9 July 2026; GovInfo's official 2026 annual Part 91 XML;
  the existing versioned FAA Pilot/Controller Glossary extraction. Source sections are
  partitioned before question generation. No rules are asserted to apply retrospectively.
- FAA extraction: 1,085 sections/entries and 3,107 passages (2,102 AIM, 339 CFR91, 666 PCG).
- Test query text and contrast items do not enter preparation or checkpoint selection.

## Running and checking

From any server terminal, run `bash ~/work/cis634/scripts/finetune_status.sh` for a read-only
snapshot. Use `watch -n 10 bash ~/work/cis634/scripts/finetune_status.sh` to refresh every
10 seconds. The script reports the stage, driver health, generation progress, training
epoch/loss, saved checkpoints, recent log output and GPU use, without scanning the corpus.

Use `.venv` on `cis-neuron`, always activating it to obtain the conda library path.
All persistent outputs live under `~/work/cis634`. No model or dataset is uploaded to HF.

```sh
python scripts/prepare_faa_training.py
python scripts/prepare_embedding_training.py --stage prepare
python scripts/generate_embedding_queries.py --limit 60
python scripts/train_embedding_lora.py --smoke
# After inspecting the pilot and verifying smoke checkpoint reload:
python -u scripts/run_embedding_finetune.py
```

The generator requires the owned local llama.cpp Qwen3.5-27B Q6_K servers on ports
8080–8082. Their owned PIDs are recorded in `logs/finetune_v1/generator_pids.json`.
The driver stops only those PIDs after sealing the corpus, then uses torchrun on all
three GV100s. Launch long jobs in a detached session, using an absolute working directory
or `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL)`.

- Configuration: `configs/finetune_8b.yaml`.
- Status: `logs/finetune_v1/status.json`; separate generation, training and evaluation logs.
  `generation.log` gives throughput, remaining ETA, transport errors and quality exclusions.
- Adapters and resumable epoch checkpoints: `models/finetuning/qwen3-8b-aviation-v1/`.
- Pipeline-written analysis: `results/finetune_8b_v1/summary.md`, `comparison.json`,
  `qualitative_examples.json`, `selection.json`, and per-condition query metrics/rankings.
- Run metadata records library versions, parameter count, corpus hashes and the config.

Retrieval evaluation uses all 91,827 incident reports and the existing chunk MaxP rule.
It compares 1,000 newly held-out ASRS synopses, 1,000 NTSB validation causes and topical
dev queries. Topical retrieval remains scoped to v1 ASRS, where silver labels exist.
The fixed primary checkpoint criterion is ASRS held-out MRR@10. All epoch regressions and
paired uncertainty are reported. Post-selection p-values are descriptive. NTSB known-item
retrieval does not establish quality of ASRS–NTSB analogues. There is no automatic UI swap.

Existing benchmark results, evaluation files, submitted PDF and frozen embedding caches
remain unchanged. New checkpoint caches are content-addressed under `cache/finetune_v1/`.

The driver sets `LD_LIBRARY_PATH=~/.conda/lib` for children even when its launcher did not
activate the venv, so the later SciPy imports remain valid. Resume checks require the same
sealed corpus hashes and config; completed Trainer checkpoints can be resumed by rerunning
the driver. Any proposed corpus or hyperparameter change after launch needs a new run version.

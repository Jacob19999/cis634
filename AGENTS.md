# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs are in
[README.md](README.md) and the step-by-step [TUTORIAL.md](TUTORIAL.md).

## Project

CIS 634 (NLP) class project by Jacob Tang and Aarambh Shrestha. `asrsbench` benchmarks
lexical (BM25), dense (frozen sentence-transformers) and hybrid retrieval on ~47.7k NASA
ASRS aviation safety reports, on three tracks: `synopsis` (known-item), `topical` (40
graded queries) and `contrast` (100 meaning-change items). It is a research codebase. The
committed results and the milestone report are deliverables, so their correctness and
reproducibility matter more than features.

## Setup and commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -q                                # 22 tests, ~2 s, no data/models/network needed
.venv/bin/asrsbench prepare                        # HF download + clean + chunk -> data/processed/
.venv/bin/asrsbench run configs/smoke.yaml         # quick end-to-end check
.venv/bin/asrsbench run configs/milestone.yaml --systems bm25 --tracks topical   # subset
.venv/bin/asrsbench search configs/milestone.yaml e5-small "query text"
.venv/bin/python scripts/analysis.py results/milestone      # -> results/milestone/analysis.json
.venv/bin/python scripts/make_figures.py results/milestone  # -> reports/milestone/figures/
cd reports/milestone && latexmk -pdf main.tex               # pdflatex + bibtex
```

Use the `.venv` interpreter; the system Python doesn't have the package installed.

**Cost.** `prepare` downloads the dataset. Encoding the corpus (136,842 chunks) on an
M1 Max (MPS) took ~2.5 min for MiniLM and ~5 min for E5-small. Base-size models take much
longer and need `batch_size: 32` to avoid swapping on 32 GB. Cached embeddings make reruns
take seconds. Prefer `smoke.yaml` or `--systems` / `--tracks` for checks, and ask before
starting an uncached `milestone.yaml` or `extended.yaml` run. Device (CUDA / MPS / CPU)
is auto-detected.

## Layout

| Path | Contents |
|---|---|
| `src/asrsbench/data.py` | download, comma repair, placeholder removal, incident grouping (union-find), chunking |
| `src/asrsbench/retrievers/` | `base.py` interface + MaxP search; `bm25.py`, `dense.py`, `hybrid.py`; `__init__.py` registry (`TYPES`, `build_systems`) |
| `src/asrsbench/tracks.py` | query sets and qrels for the three tracks; silver-label rules |
| `src/asrsbench/metrics.py` | IR metrics, bootstrap CIs, paired randomization test, Wilson CI |
| `src/asrsbench/benchmark.py` | `asrsbench run`: builds systems, runs tracks, writes `results/<run>/` |
| `src/asrsbench/labeling.py` | blind judging pools and label import (Cohen's kappa) |
| `src/asrsbench/cli.py` | argparse CLI; heavy imports stay inside each command branch |
| `configs/*.yaml` | one file per benchmark run; the run name sets the `results/` folder |
| `data/queries/`, `data/contrast/`, `data/qrels/` | committed, versioned evaluation data |
| `results/<run>/` | committed metric tables, `summary.md`, `meta.json` |
| `reports/milestone/` | IEEE LaTeX report (`main.tex`), figures, submitted PDF |
| `Research/` | proposal and the course's project/report guidelines (PDFs) |

`scripts/llm_graph.py` extracts a per-report knowledge graph from the narratives with a local
Ollama model (narrative only: never the synopsis or analyst codes, which are evaluation
targets) into `data/graph/<model>_v1.jsonl`. It resumes an interrupted run.
`scripts/export_precedent_graph.py <triaggent_consulting>/portfolio/asrs-precedent-graph` turns
that graph plus the raw analyst codes into the ASRS Precedent Graph page's `data/` (all 47,723
reports; MPNet embeddings and the t-SNE map are cached in `cache/precedent/`).

Git-ignored and regenerable: `data/raw/`, `data/processed/`, `data/graph/` (LLM graph), `cache/` (embeddings),
`results/**/runs/` (TREC run files), LaTeX build artefacts. `scripts/analysis.py` and
`asrsbench pool` need the ignored run files, so they only work after a local `run`.

## Research-integrity rules

These protect the validity of the reported results. Don't break them silently; raise the
issue with the user instead.

- **Dev/test split.** Each topical query has `split: dev` or `split: test`. Only dev
  queries may inform design choices (preprocessing, prefixes, fusion settings, silver
  rules). Don't tune against test-split numbers.
- **Versioned evaluation data.** `topical_v1.yaml`, `contrast_v1.jsonl` and
  `data/qrels/pool_v1/` have already been used in reported results, and the pools went out
  to annotators. Don't edit them in place; create a `_v2` file and point a config at it.
- **Blind judging.** Pool CSVs must not reveal which system retrieved a report.
- **Incident groups.** Reports of the same incident share a `group_id`, and 13 groups
  cross the official splits. Any training or fine-tuning split must split by `group_id`,
  and retrieving a sibling report counts as relevant.
- **Frozen historical run.** `results/milestone_v1_rawchunks/` is the preprocessing-v1
  "before" condition (see its `NOTE.md`). Don't regenerate or overwrite it.
- **Results come from the pipeline.** Never hand-edit files in `results/`; rerun the
  config. The numbers in `README.md` and `reports/milestone/main.tex` are copied by hand
  from `results/*/summary.md` and `analysis.json`. If a change shifts any metric, list the
  affected numbers for the user rather than leaving the docs stale. `TUTORIAL.md` also
  quotes outputs, from `smoke.yaml`, its `my_run.yaml` example and two `search` calls.
- **Human labels pending.** `data/qrels/human_v1.tsv` doesn't exist yet. When it appears,
  configs pick it up automatically as a `human:human_v1` label source next to `silver`.

## Architecture notes

- Every retriever scores *units*: chunks (`doc_mode: chunk_max`) or whole narratives
  (`doc_mode: whole`). `Units` keeps each report's units contiguous, so
  `Retriever.search` can take the per-report max (MaxP) with `np.maximum.reduceat`.
- A new retriever type subclasses `Retriever`, implements `build` and `score_units`
  (plus `pair_scores` for the contrast test, or sets `supports_pairs = False`), and is
  registered in `TYPES` in `retrievers/__init__.py`.
- A new *model* needs only a config entry (see README "Adding a model"), no code.
- A hybrid's `components` must name systems defined earlier in the same config. Hybrids
  fuse report-level rankings and are skipped by the contrast test.
- The dense embedding cache (`cache/embeddings/`) is keyed on model, passage prefix,
  `max_seq_length`, the unit-set tag (e.g. `w150o1m40`) and the size/first/last text of
  the unit set. Changing chunking or preprocessing invalidates it automatically. Changing
  *how* embeddings are computed (normalisation, pooling, dtype) does not, so delete the
  affected `.npy` files.
- Text normalisation repairs the ASRS export's comma-to-semicolon artefact in free text
  only; coded analyst fields keep `;` as their list separator.

## Conventions

- Python ≥ 3.10, `from __future__ import annotations`, type hints, a docstring on every
  public function and a short module docstring explaining its role. Comments explain
  *why*, at the density of the surrounding code. No formatter or linter is configured;
  lines stay around 100 characters.
- Add or update a `tests/` case for any change to normalisation, chunking, grouping,
  metrics, retriever scoring or label import. Tests use small synthetic data. Keep them
  offline and fast, without model downloads.
- Prose (README, report, docstrings) uses British spelling (normalise, summarise,
  artefact) and plain, specific wording with numbers to back claims.
- Commit messages: short imperative sentence-case subject, no type prefix (e.g.
  "Match silver-label patterns at word starts; rerun benchmarks").
- The submitted milestone PDF is `reports/milestone/phase 2 jacob tang Aarambh.pdf`.
  `latexmk` writes `main.pdf`; copy it over the submitted file only when the user asks.

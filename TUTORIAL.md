# asrsbench tutorial

This tutorial takes you from a fresh clone to a benchmark of a model you choose. Allow
about 30 minutes; most of that is the one-off download and corpus encoding. For the
research question and the results so far, see the [README](README.md).

By the end you will have:

1. installed `asrsbench` and checked that it works,
2. built the cleaned corpus of 47,723 NASA ASRS reports,
3. searched it with a keyword system (BM25) and an embedding model (MiniLM),
4. run a small benchmark and read its results,
5. written your own config and benchmarked a different model.

The outputs below are real, from an M1 Max laptop (Apple MPS). Your timings will differ.
BM25 numbers should match exactly. Encoder numbers can differ slightly on other hardware,
and the confidence intervals and p-values use fixed random seeds.

## Before you start

- **Python 3.10 or newer** (we use 3.12), and [uv](https://docs.astral.sh/uv/) or plain `pip`.
- **Internet access** the first time you download the dataset and each model.
- **Disk space.** The dataset download is about 150 MB and the processed corpus 195 MB.
  Each small model's cached embeddings take 210 MB; 1024-dimension models take 560 MB.
  PyTorch and the model weights need more on top.
- **A GPU is optional.** The device is detected automatically (CUDA, then Apple MPS, then
  CPU). We have not timed CPU-only encoding, but expect it to be much slower than the MPS
  times quoted here.

Five terms used throughout:

| Term | Meaning |
|---|---|
| report | One ASRS incident report, identified by its ACN (accession number). |
| chunk | A window of up to 150 words from a report's narrative. Systems score chunks; a report's score is its best chunk's score ("MaxP"). |
| system | One retrieval method with its settings: `bm25`, `dense` (an embedding model) or `hybrid` (a fusion of other systems). |
| track | A set of test queries with relevance labels: `synopsis`, `topical` or `contrast`. |
| config | A YAML file in `configs/` that lists the systems and tracks to run. Its `name` sets the results folder. |

## 1. Install

```bash
git clone https://github.com/Jacob19999/cis634.git
cd cis634
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

Without uv, use `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`. On Google
Colab, run `!pip install -e .` inside the clone.

Check the install:

```bash
pytest -q          # 22 passed in 2.22s
asrsbench --help   # lists: prepare, run, search, pool, labels
```

The tests use small synthetic data, so they need no dataset, model or network.

Run every later command from the repository root with the virtual environment active.
Configs refer to `data/` and `configs/` by relative path.

## 2. Build the corpus

```bash
asrsbench prepare
```

This downloads [`elihoole/asrs-aviation-reports`](https://huggingface.co/datasets/elihoole/asrs-aviation-reports)
from Hugging Face and then:

- restores the commas that the ASRS export turned into semicolons (`10;000 FT` → `10,000 FT`),
- drops 2,906 placeholder second narratives such as "[Report narrative contained no additional information.]",
- groups reports that describe the same incident, so a sibling report counts as a correct hit,
- splits each narrative into sentence-aligned chunks of at most 150 words.

It writes `data/processed/reports.parquet`, `chunks_w150o1m40.parquet` and `stats.json`,
then prints the statistics. Check these two lines:

```
  "n_reports": 47723,
  "n_chunks": 136842,
```

You only need to do this once. `data/processed/` is git-ignored.

## 3. Search the corpus

`asrsbench search` shows one system's top reports for a query you type. It takes a
config, the name of a system in that config, and the query:

```bash
asrsbench search configs/milestone.yaml bm25 "crew descended to the altitude meant for another aircraft" -k 3
```

```
#1  ACN 1127687  score=14.0672
  synopsis: Light twin pilot reported an NMAC with another light aircraft on a visual approach to MYEF.
  passage:  All of a sudden, I saw the [other aircraft] overtake us from above, and descend through our altitude, ...

#2  ACN 1118664  score=13.4948
  synopsis: Air carrier flight crew reports accepting a descent clearance meant for another flight along with a
            frequency change. At the new altitude a TCAS RA is generated and the crew returns to FL360 ...
  passage:  Crew had accepted descent to FL350 meant for another aircraft with similar sounding call sign. ...

#3  ACN 1244848  score=12.8011
  synopsis: A319 Captain experiences FMGC anomalies during descent on the SKNRD3 arrival to IAH. ...
```

Each hit shows the report's ACN, its score, NASA's analyst synopsis (a summary of the
report) and the matched passage, which is the chunk that scored highest.

Now try the same query with MiniLM, an embedding model:

```bash
asrsbench search configs/milestone.yaml minilm "crew descended to the altitude meant for another aircraft" -k 3
```

The first time you use an embedding model, it encodes all 136,842 chunks (about 2.5 min
for MiniLM on an M1 Max) and saves the vectors to `cache/embeddings/`. Later calls load
them in seconds.

```
#1  ACN 1155676  score=0.7059
  synopsis: B747 Captain describes an approach into OAIX where the aircraft is allowed to go below profile due to
            the First Officer setting a lower altitude in the MCP window ...
#2  ACN 998731  score=0.7042
  synopsis: A CE560XL departed on the TEB RUUDY 4 SID and failed to level at 1,500 FT ...
#3  ACN 1016979  score=0.6846
  synopsis: LR60 Captain reported and uncontrolled 800 FT climb from FL290 because of updraft.
```

Points to notice:

- **Scores don't compare across systems.** BM25 scores are unbounded sums of term
  weights; dense scores are cosine similarities, at most 1.
- **The systems fail differently.** BM25's #2 is exactly this issue, a descent clearance
  meant for another flight. Its #1 only shares words ("descend through our altitude") and
  is about a near mid-air collision. MiniLM's top three are all altitude deviations, but
  none involves another aircraft's clearance. One query proves nothing either way;
  measuring this over many queries is what the benchmark is for.
- **Hybrids work too**, for example `asrsbench search configs/smoke.yaml hybrid "..."`.
  The command builds the components first.
- **`search` is for spot checks.** It rebuilds the BM25 index on every call (about 7 s).

## 4. Run a benchmark

```bash
asrsbench run configs/smoke.yaml
```

`configs/smoke.yaml` runs BM25, MiniLM and their fusion on all three tracks, using only
50 synopsis queries. Once the MiniLM embeddings are cached, it takes about 30 s:

```
indexed bm25 in 7.68s
indexed minilm in 5.58s
indexed hybrid in 0.0s
  [synopsis] bm25: 50 queries in 0.16s
  [synopsis] minilm: 50 queries in 0.63s
  ...
# Benchmark run `smoke`
...
```

The command then prints the summary tables and writes everything to `results/smoke/`.

You may see `Exception ignored in: <function ResourceTracker.__del__ ...>` and a warning
about unauthenticated Hugging Face requests at the end. Both are harmless, and the results
are already written by then.

## 5. Read the results

| File in `results/<run>/` | Contents |
|---|---|
| `summary.md` | The readable tables printed at the end of the run. |
| `metrics_<track>.csv` | Mean of each metric per system, with 95% CI bounds in `<metric>_lo` and `<metric>_hi`. |
| `per_query_<track>.csv` | One row per query, system and label source. Use it for error analysis. |
| `significance_<track>.csv` | A paired randomization test for every pair of systems on the track's primary metric. |
| `metrics_contrast.csv`, `contrast_items.csv` | Contrast accuracy per category; scores for every item. |
| `meta.json`, `config.yaml` | System settings, timings, device and versions; the config as run, with defaults filled in. |
| `runs/<track>/<system>.tsv` | Rankings in TREC run format, with the matched chunk id in the last column (git-ignored). |

### The three tracks and their metrics

**Synopsis** (known-item search). The query is NASA's analyst synopsis of a report, and
the target is that report (plus any report of the same incident). Analysts write the
synopsis in their own, shorter words, so the query paraphrases the target.

```
| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
| bm25   | auto   | 50 | 0.418 [0.300, 0.541] | 0.340 [0.220, 0.480] | 0.580 [0.440, 0.720] | 0.640 [0.500, 0.780] |
| minilm | auto   | 50 | 0.236 [0.137, 0.348] | 0.180 [0.080, 0.300] | 0.360 [0.240, 0.500] | 0.480 [0.340, 0.620] |
| hybrid | auto   | 50 | 0.392 [0.277, 0.515] | 0.320 [0.200, 0.460] | 0.580 [0.440, 0.720] | 0.700 [0.580, 0.820] |
```

- `MRR@10` is the mean of 1 / (rank of the target), counting 0 if it isn't in the top 10.
- `R@k` is the share of queries whose target is in the top k.
- The brackets give a 95% bootstrap confidence interval. With only 50 queries they are
  wide (BM25's MRR@10 spans 0.300 to 0.541), so don't report smoke numbers.
  `configs/milestone.yaml` uses 1,000 synopsis queries.

**Topical** (issue search). There are 40 analyst-style descriptions of an operational
issue in `data/queries/topical_v1.yaml`. Each report gets grade 2 (same issue), 1 (related)
or 0. The `silver` labels come from rules over NASA's analyst codes. Once human judgments
exist, a second label source, `human:human_v1`, appears next to them.

- `nDCG@10` rewards grade-2 reports near the top. It is the track's primary metric.
- `P@10` is the share of the top 10 graded 2; `P@10_lenient` counts grade 1 as well.

**Contrast** (meaning-change test). There are 100 items, 25 in each of four categories:
negation, numbers, direction and event order. Each item has an original sentence, a
paraphrase with the same meaning, and a "changed" sentence whose meaning differs by one
edit. A system passes an item if it rates the same-meaning pair as more similar than the
changed pair. Chance is 0.5.

- `original_anchor`: the anchor is the original. The changed sentence shares almost all of
  its words, so word overlap pulls systems towards the wrong answer.
- `paraphrase_anchor`: the anchor is the paraphrase, which overlaps the original and the
  changed sentence equally. This is the controlled condition.
- Hybrids fuse rankings and have no pairwise score, so they don't appear in this table.

### Is a difference real?

`significance_topical.csv` from the smoke run (columns trimmed):

```
system_a,system_b,mean_a,diff,p_value
bm25,hybrid,0.630,-0.065,0.0058
bm25,minilm,0.630,0.039,0.2934
hybrid,minilm,0.696,0.105,0.0003
```

The hybrid beats BM25 on topical nDCG@10 (p = 0.006). BM25 and MiniLM are not
significantly different (p = 0.29).

### Where does a system lose?

The per-query files load straight into pandas:

```python
import pandas as pd

pq = pd.read_csv("results/smoke/per_query_topical.csv")
wide = pq.pivot(index="qid", columns="system", values="nDCG@10")
print((wide["minilm"] - wide["bm25"]).sort_values().head(3))
```

```
T29   -0.761
T03   -0.609
T18   -0.370
```

MiniLM loses most on T29 ("dispatched with a deferred maintenance item") and T03 ("did
not make a published crossing restriction"). Look up a query's text in
`data/queries/topical_v1.yaml`, then use `asrsbench search` to see what each system
returned for it.

## 6. Benchmark your own model

Any [sentence-transformers](https://www.sbert.net/) checkpoint works without code
changes. Create `configs/my_run.yaml`:

```yaml
name: my_run                      # results go to results/my_run/
systems:
  - {name: bm25, type: bm25}
  - name: bge-small
    type: dense
    params: {model: BAAI/bge-small-en-v1.5, max_seq_length: 512,
             query_prefix: "Represent this sentence for searching relevant passages: ", pair_prefix: ""}
  - name: hybrid-bm25-bge
    type: hybrid
    params: {components: [bm25, bge-small], method: convex, weights: [0.3, 0.7]}
evaluation:
  topical: {queries: data/queries/topical_v1.yaml, splits: [dev]}
  contrast: {path: data/contrast/contrast_v1.jsonl}
```

```bash
asrsbench run configs/my_run.yaml
```

```
| system          | labels | n_queries | nDCG@10              | P@10                 | P@10_lenient         | MRR@10               |
| bm25            | silver | 20        | 0.713 [0.609, 0.809] | 0.650 [0.545, 0.755] | 0.770 [0.670, 0.860] | 0.902 [0.788, 1.000] |
| bge-small       | silver | 20        | 0.670 [0.584, 0.753] | 0.540 [0.420, 0.655] | 0.795 [0.730, 0.860] | 0.929 [0.825, 1.000] |
| hybrid-bm25-bge | silver | 20        | 0.698 [0.619, 0.777] | 0.590 [0.480, 0.695] | 0.815 [0.740, 0.885] | 0.950 [0.875, 1.000] |
```

How this config works:

- **Defaults fill the gaps.** Keys you leave out take the values in the reference below,
  so this short file uses the same corpus, chunking and cache as the committed runs.
- **Prefixes come from the model card.** Many models expect fixed text before each input.
  E5 wants `"query: "` and `"passage: "`; BGE wants an instruction on the query only.
  `pair_prefix` is the prefix used for the sentence-vs-sentence contrast test; by default
  it is the same as `query_prefix`. A wrong prefix silently lowers scores, so check the card.
- **`splits: [dev]` runs only the 20 dev queries.** Use dev queries while you try out
  settings, and keep the 20 test queries for the final comparison (see section 9).
- **Encoding cost grows with model size.** A small (33M-parameter) model takes 2.5 to
  5 min on an M1 Max. In `configs/large.yaml`, `bge-large` took about 45 min and
  Qwen3-Embedding-0.6B about 105 min. If the machine starts swapping, set `batch_size: 32`.

### Config reference

Top-level keys (all optional except `systems`):

| Key | Default | Meaning |
|---|---|---|
| `name` | config file name | Results folder under `results_dir`. |
| `data_dir` | `data/processed` | Output of `asrsbench prepare`. |
| `cache_dir` | `cache` | Where embeddings are cached. |
| `results_dir` | `results` | Parent folder for run results. |
| `device` | `auto` | `cuda`, `mps`, `cpu` or `auto`. |
| `depth` | `100` | Reports retrieved per query. |
| `chunking` | `{max_words: 150, overlap_sentences: 1, min_tail_words: 40}` | How narratives are split. A new setting builds its chunk file on first use; you don't need to rerun `prepare`. |

System parameters, under `params:`:

| Type | Parameters (defaults) |
|---|---|
| all except `hybrid` | `doc_mode` (`chunk_max`): `chunk_max` scores chunks; `whole` scores each narrative as one unit, truncated at the encoder's `max_seq_length`. |
| `bm25` | `k1` (1.2), `b` (0.75), `stopwords` (`english`, or `english_keep_negation`, `none`), `stem` (true). |
| `dense` | `model` (required), `query_prefix`, `passage_prefix`, `pair_prefix`, `max_seq_length`, `batch_size` (128), `dtype` (e.g. `float16`), `trust_remote_code` (false). |
| `hybrid` | `components` (required; systems defined earlier in the same file), `method` (`rrf` or `convex`), `k` (60, RRF constant), `weights` (equal), `depth` (1000 reports taken from each component). |

Evaluation tracks, under `evaluation:` (leave one out to skip it):

| Track | Keys |
|---|---|
| `synopsis` | `split` (`validation`), `n_queries` (1000), `seed` (13) |
| `topical` | `queries` (required), `human_qrels` (list of qrels files, used if they exist), `splits` (e.g. `[dev]`) |
| `contrast` | `path` (required) |

### Run part of a config

```bash
asrsbench run configs/milestone.yaml --systems bm25 minilm hybrid-bm25-minilm --tracks topical --name try_topical
```

- **List a hybrid's components too.** `--systems hybrid-bm25-minilm` on its own fails with
  `KeyError: 'bm25'`.
- **Pass `--name` for a partial run of a committed config.** Without it, the run writes to
  `results/milestone/` and replaces the committed `summary.md` and `meta.json` with partial
  versions. If that happens, `git restore results/milestone` undoes it.

## 7. Bigger runs

| Config | What it runs | Cost on an M1 Max |
|---|---|---|
| `smoke.yaml` | BM25, MiniLM, hybrid; 50 synopsis queries | ~30 s with cached embeddings |
| `milestone.yaml` | BM25, MiniLM, E5-small, two hybrids; 1,000 synopsis queries | Encoding: MiniLM ~2.5 min, E5-small ~5 min |
| `extended.yaml` | Adds BGE-small, MPNet, E5-base, whole-report and keep-negation variants | Base-size models are much slower; they use `batch_size: 32` |
| `large.yaml` | BGE and E5 up to 335M parameters, Qwen3-Embedding-0.6B | bge-large ~45 min, Qwen3-0.6B ~105 min |
| `large_gpu.yaml` | Qwen3-Embedding 4B and 8B, whole-report Qwen3 | Needs a CUDA GPU, e.g. a Colab A100 or L4 |

Run long jobs in the background and follow the log:

```bash
nohup asrsbench run configs/extended.yaml --name extended_try > extended_try.log 2>&1 &
tail -f extended_try.log
```

The scripts in `scripts/` produce the report's analyses and figures:

```bash
python scripts/analysis.py results/milestone       # error analyses -> results/milestone/analysis.json
python scripts/make_figures.py results/milestone   # report figures -> reports/milestone/figures/
python scripts/taxonomy_graph.py                   # taxonomy-graph rerank -> results/taxonomy_graph.json
```

`analysis.py` reads the run files in `results/<run>/runs/`, which are git-ignored, so run
the config locally before using it. The other two scripts need only the committed results
and `data/processed/`.

## 8. Human relevance judging

The silver labels are a proxy. Gold labels come from two annotators who grade pooled
results without seeing which system found each report.

1. **Build pools** from a run's topical results:

   ```bash
   asrsbench pool results/milestone --out data/qrels/pool_v2
   ```

   Always pass a new `--out`. The default, `data/qrels/pool_v1/`, holds the pools already
   sent to annotators, and the command would overwrite them.

2. **Grade.** Each annotator fills the `grade` column of their CSV with 2, 1 or 0,
   following the guide in the pool folder's `README.txt`. A quarter of the queries go to
   both annotators so that agreement can be measured.

3. **Import the labels:**

   ```bash
   asrsbench labels data/qrels/pool_v1/pool_A.csv data/qrels/pool_v1/pool_B.csv
   ```

   This prints Cohen's kappa and writes `data/qrels/human_v1.tsv`. Disagreements take the
   lower grade unless you fill in a `grade_final` column. Every config that lists the file
   under `human_qrels` (all except `smoke.yaml`) picks it up automatically, and later runs
   report a `human:human_v1` row next to `silver`.

## 9. Rules for results you plan to report

These keep the reported results valid. If one gets in your way, raise it with the team
rather than working around it.

- **Tune only on dev queries.** Choose settings (prefixes, fusion weights, chunking, silver
  rules) using `split: dev` topical queries. Report test-split numbers; never tune on them.
- **Don't edit evaluation data in place.** `topical_v1.yaml`, `contrast_v1.jsonl` and
  `data/qrels/pool_v1/` have been used in reported results. Create a `_v2` file and point a
  config at it.
- **Results come from the pipeline.** Never hand-edit files in `results/`; rerun the
  config. Leave `results/milestone_v1_rawchunks/` alone, because it is a frozen "before"
  condition.
- **Split training data by incident.** If you fine-tune anything, split by `group_id`,
  because 13 incident groups cross the official dataset splits.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `command not found: asrsbench` or `ModuleNotFoundError: No module named 'asrsbench'` | The virtual environment isn't active. Run `source .venv/bin/activate`, or call `.venv/bin/asrsbench`. |
| `FileNotFoundError: .../reports.parquet` | Run `asrsbench prepare` first, and run commands from the repository root. |
| `KeyError: '<system name>'` | A hybrid was run without one of its components: add the components to `--systems`. |
| A failed run leaves `results/<name>/` holding only `config.yaml` | The folder is created before the data loads. Delete it, fix the error and rerun. |
| `ValueError: unknown system type` | A typo in `type:`. It must be `bm25`, `dense` or `hybrid`. |
| Encoding slows to a crawl and the machine swaps | Lower `batch_size` for that system (32 works for base-size models on 32 GB). |
| You changed how embeddings are computed (pooling, normalisation, dtype) and nothing changed | The cache key covers the model, passage prefix, `max_seq_length` and chunk set, not code. Delete that model's `.npy` files in `cache/embeddings/`. |
| `ResourceTracker` traceback or Hugging Face token warning at exit | Harmless. The results are already written. |

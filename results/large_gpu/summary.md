# Benchmark run `large_gpu`

Corpus: 47,723 reports, {'chunk_max': 136842, 'whole': 47723} retrieval units. Device: cuda. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.425 [0.397, 0.453] | 0.362 [0.333, 0.393] | 0.555 [0.524, 0.586] | 0.753 [0.727, 0.780] |
| bm25-whole | auto | 1000 | 0.458 [0.429, 0.486] | 0.395 [0.364, 0.425] | 0.598 [0.567, 0.628] | 0.782 [0.757, 0.808] |
| qwen3-0.6b-generic | auto | 1000 | 0.372 [0.346, 0.399] | 0.305 [0.276, 0.334] | 0.531 [0.500, 0.562] | 0.749 [0.722, 0.775] |
| qwen3-0.6b-whole | auto | 1000 | 0.384 [0.356, 0.410] | 0.315 [0.286, 0.344] | 0.560 [0.529, 0.590] | 0.773 [0.747, 0.799] |
| qwen3-4b | auto | 1000 | 0.429 [0.401, 0.456] | 0.349 [0.320, 0.379] | 0.618 [0.587, 0.647] | 0.836 [0.812, 0.858] |
| qwen3-8b | auto | 1000 | 0.450 [0.423, 0.478] | 0.369 [0.339, 0.399] | 0.631 [0.601, 0.661] | 0.848 [0.825, 0.870] |
| hybrid-bm25-qwen3-generic | auto | 1000 | 0.445 [0.417, 0.472] | 0.372 [0.342, 0.401] | 0.616 [0.585, 0.645] | 0.805 [0.781, 0.830] |
| hybrid-bm25-qwen3-4b | auto | 1000 | 0.478 [0.450, 0.506] | 0.398 [0.367, 0.428] | 0.662 [0.633, 0.692] | 0.844 [0.821, 0.866] |
| hybrid-bm25-qwen3-8b | auto | 1000 | 0.493 [0.465, 0.520] | 0.414 [0.384, 0.445] | 0.662 [0.632, 0.691] | 0.860 [0.838, 0.881] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 40 | 0.630 [0.554, 0.708] | 0.537 [0.450, 0.625] | 0.703 [0.628, 0.775] | 0.838 [0.746, 0.922] |
| bm25-whole | silver | 40 | 0.621 [0.543, 0.699] | 0.528 [0.438, 0.620] | 0.700 [0.622, 0.775] | 0.850 [0.764, 0.928] |
| qwen3-0.6b-generic | silver | 40 | 0.748 [0.674, 0.818] | 0.645 [0.547, 0.740] | 0.840 [0.775, 0.900] | 0.924 [0.854, 0.983] |
| qwen3-0.6b-whole | silver | 40 | 0.771 [0.705, 0.833] | 0.660 [0.568, 0.747] | 0.852 [0.800, 0.900] | 0.923 [0.852, 0.981] |
| qwen3-4b | silver | 40 | 0.777 [0.706, 0.841] | 0.680 [0.585, 0.770] | 0.868 [0.805, 0.918] | 0.938 [0.863, 0.988] |
| qwen3-8b | silver | 40 | 0.787 [0.721, 0.848] | 0.703 [0.613, 0.785] | 0.890 [0.835, 0.935] | 0.928 [0.863, 0.978] |
| hybrid-bm25-qwen3-generic | silver | 40 | 0.748 [0.680, 0.813] | 0.662 [0.578, 0.747] | 0.832 [0.780, 0.885] | 0.909 [0.837, 0.971] |
| hybrid-bm25-qwen3-4b | silver | 40 | 0.757 [0.688, 0.822] | 0.665 [0.577, 0.750] | 0.867 [0.812, 0.912] | 0.887 [0.812, 0.954] |
| hybrid-bm25-qwen3-8b | silver | 40 | 0.762 [0.699, 0.824] | 0.672 [0.593, 0.750] | 0.853 [0.798, 0.900] | 0.908 [0.842, 0.963] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-whole | 0.060 | 0.040 | 0.160 | 0.040 | 0.000 |
| qwen3-0.6b-generic | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-0.6b-whole | 0.360 | 0.240 | 0.280 | 0.120 | 0.800 |
| qwen3-4b | 0.720 | 0.720 | 0.840 | 0.360 | 0.960 |
| qwen3-8b | 0.740 | 0.680 | 0.880 | 0.400 | 1.000 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-whole | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| qwen3-0.6b-generic | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-0.6b-whole | 0.870 | 1.000 | 0.960 | 0.520 | 1.000 |
| qwen3-4b | 0.880 | 1.000 | 0.960 | 0.560 | 1.000 |
| qwen3-8b | 0.890 | 1.000 | 0.960 | 0.600 | 1.000 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 28.42,
  "query_seconds_synopsis": 8.09,
  "query_seconds_topical": 0.3
 },
 "bm25-whole": {
  "index_seconds": 24.75,
  "query_seconds_synopsis": 3.43,
  "query_seconds_topical": 0.12
 },
 "qwen3-0.6b-generic": {
  "index_seconds": 14.24,
  "query_seconds_synopsis": 9.86,
  "query_seconds_topical": 0.46
 },
 "qwen3-0.6b-whole": {
  "index_seconds": 3.74,
  "query_seconds_synopsis": 11.17,
  "query_seconds_topical": 0.46
 },
 "qwen3-4b": {
  "index_seconds": 8.43,
  "query_seconds_synopsis": 18.81,
  "query_seconds_topical": 0.69
 },
 "qwen3-8b": {
  "index_seconds": 14.91,
  "query_seconds_synopsis": 27.26,
  "query_seconds_topical": 0.89
 },
 "hybrid-bm25-qwen3-generic": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 34.42,
  "query_seconds_topical": 1.31
 },
 "hybrid-bm25-qwen3-4b": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 43.6,
  "query_seconds_topical": 1.61
 },
 "hybrid-bm25-qwen3-8b": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 52.46,
  "query_seconds_topical": 1.9
 },
 "_total_index_seconds": 94.5
}
```

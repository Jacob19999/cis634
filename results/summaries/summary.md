# Benchmark run `summaries`

Corpus: 47,723 reports, {'chunk_max': 136842, 'summary': 47723, 'chunk_summary': 184565} retrieval units. Device: cuda. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.425 [0.397, 0.453] | 0.362 [0.333, 0.393] | 0.555 [0.524, 0.586] | 0.753 [0.727, 0.780] |
| bm25-chunk-sum | auto | 1000 | 0.459 [0.431, 0.487] | 0.398 [0.367, 0.428] | 0.597 [0.566, 0.627] | 0.787 [0.762, 0.812] |
| qwen3-0.6b | auto | 1000 | 0.372 [0.346, 0.399] | 0.305 [0.276, 0.334] | 0.531 [0.500, 0.562] | 0.749 [0.722, 0.775] |
| qwen3-0.6b-sum | auto | 1000 | 0.311 [0.285, 0.336] | 0.243 [0.217, 0.270] | 0.466 [0.435, 0.497] | 0.700 [0.671, 0.728] |
| qwen3-0.6b-chunk-sum | auto | 1000 | 0.383 [0.357, 0.411] | 0.312 [0.283, 0.340] | 0.551 [0.520, 0.581] | 0.780 [0.754, 0.805] |
| qwen3-8b | auto | 1000 | 0.450 [0.423, 0.478] | 0.369 [0.339, 0.399] | 0.631 [0.601, 0.661] | 0.848 [0.825, 0.870] |
| qwen3-8b-sum | auto | 1000 | 0.381 [0.353, 0.408] | 0.310 [0.282, 0.339] | 0.541 [0.510, 0.572] | 0.772 [0.746, 0.798] |
| qwen3-8b-chunk-sum | auto | 1000 | 0.470 [0.442, 0.497] | 0.390 [0.360, 0.421] | 0.640 [0.611, 0.670] | 0.871 [0.850, 0.891] |
| hybrid-bm25-qwen3-8b | auto | 1000 | 0.493 [0.465, 0.520] | 0.414 [0.384, 0.445] | 0.662 [0.632, 0.691] | 0.860 [0.838, 0.881] |
| hybrid-bm25-qwen3-8b-chunk-sum | auto | 1000 | 0.512 [0.485, 0.540] | 0.435 [0.406, 0.466] | 0.690 [0.661, 0.718] | 0.882 [0.863, 0.902] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 40 | 0.630 [0.554, 0.708] | 0.537 [0.450, 0.625] | 0.703 [0.628, 0.775] | 0.838 [0.746, 0.922] |
| bm25-chunk-sum | silver | 40 | 0.644 [0.565, 0.723] | 0.545 [0.453, 0.637] | 0.718 [0.637, 0.792] | 0.859 [0.765, 0.942] |
| qwen3-0.6b | silver | 40 | 0.748 [0.674, 0.818] | 0.645 [0.547, 0.740] | 0.840 [0.775, 0.900] | 0.924 [0.854, 0.983] |
| qwen3-0.6b-sum | silver | 40 | 0.741 [0.668, 0.810] | 0.617 [0.517, 0.715] | 0.848 [0.790, 0.900] | 0.938 [0.887, 0.988] |
| qwen3-0.6b-chunk-sum | silver | 40 | 0.745 [0.672, 0.814] | 0.637 [0.540, 0.730] | 0.847 [0.790, 0.900] | 0.925 [0.863, 0.975] |
| qwen3-8b | silver | 40 | 0.787 [0.721, 0.848] | 0.703 [0.613, 0.785] | 0.890 [0.835, 0.935] | 0.928 [0.863, 0.978] |
| qwen3-8b-sum | silver | 40 | 0.791 [0.730, 0.850] | 0.693 [0.605, 0.780] | 0.885 [0.838, 0.927] | 0.958 [0.908, 1.000] |
| qwen3-8b-chunk-sum | silver | 40 | 0.786 [0.718, 0.849] | 0.682 [0.595, 0.770] | 0.887 [0.830, 0.935] | 0.928 [0.859, 0.988] |
| hybrid-bm25-qwen3-8b | silver | 40 | 0.762 [0.699, 0.824] | 0.672 [0.593, 0.750] | 0.853 [0.798, 0.900] | 0.908 [0.842, 0.963] |
| hybrid-bm25-qwen3-8b-chunk-sum | silver | 40 | 0.787 [0.720, 0.847] | 0.685 [0.600, 0.767] | 0.872 [0.815, 0.922] | 0.954 [0.896, 1.000] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-chunk-sum | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| qwen3-0.6b | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-0.6b-sum | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-8b | 0.740 | 0.680 | 0.880 | 0.400 | 1.000 |
| qwen3-8b-sum | 0.740 | 0.680 | 0.880 | 0.400 | 1.000 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-chunk-sum | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| qwen3-0.6b | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-0.6b-sum | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-8b | 0.890 | 1.000 | 0.960 | 0.600 | 1.000 |
| qwen3-8b-sum | 0.890 | 1.000 | 0.960 | 0.600 | 1.000 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 28.58,
  "query_seconds_synopsis": 8.05,
  "query_seconds_topical": 0.3
 },
 "bm25-chunk-sum": {
  "index_seconds": 32.64,
  "query_seconds_synopsis": 10.14,
  "query_seconds_topical": 0.34
 },
 "qwen3-0.6b": {
  "index_seconds": 19.11,
  "query_seconds_synopsis": 9.74,
  "query_seconds_topical": 0.45
 },
 "qwen3-0.6b-sum": {
  "index_seconds": 0.09,
  "query_seconds_synopsis": 6.07,
  "query_seconds_topical": 0.22
 },
 "qwen3-0.6b-chunk-sum": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 31.56,
  "query_seconds_topical": 1.21
 },
 "qwen3-8b": {
  "index_seconds": 14.05,
  "query_seconds_synopsis": 26.66,
  "query_seconds_topical": 0.78
 },
 "qwen3-8b-sum": {
  "index_seconds": 0.34,
  "query_seconds_synopsis": 22.97,
  "query_seconds_topical": 0.66
 },
 "qwen3-8b-chunk-sum": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 66.04,
  "query_seconds_topical": 2.21
 },
 "hybrid-bm25-qwen3-8b": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 51.04,
  "query_seconds_topical": 1.8
 },
 "hybrid-bm25-qwen3-8b-chunk-sum": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 85.59,
  "query_seconds_topical": 2.89
 },
 "_total_index_seconds": 94.81
}
```

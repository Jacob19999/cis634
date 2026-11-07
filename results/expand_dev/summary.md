# Benchmark run `expand_dev`

Corpus: 47,723 reports, {'chunk_max': 136842} retrieval units. Device: cuda. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.423 [0.395, 0.451] | 0.359 [0.329, 0.389] | 0.565 [0.534, 0.595] | 0.776 [0.750, 0.802] |
| bm25-acro | auto | 1000 | 0.415 [0.387, 0.443] | 0.347 [0.318, 0.377] | 0.550 [0.519, 0.581] | 0.774 [0.749, 0.800] |
| minilm | auto | 1000 | 0.235 [0.211, 0.259] | 0.187 [0.163, 0.212] | 0.363 [0.333, 0.392] | 0.613 [0.583, 0.642] |
| minilm-acro | auto | 1000 | 0.231 [0.208, 0.256] | 0.183 [0.159, 0.207] | 0.345 [0.316, 0.375] | 0.618 [0.587, 0.647] |
| e5-small | auto | 1000 | 0.225 [0.203, 0.248] | 0.170 [0.147, 0.193] | 0.368 [0.338, 0.398] | 0.590 [0.558, 0.619] |
| e5-small-acro | auto | 1000 | 0.235 [0.212, 0.258] | 0.180 [0.157, 0.203] | 0.370 [0.340, 0.400] | 0.582 [0.551, 0.611] |
| qwen3-0.6b | auto | 1000 | 0.339 [0.312, 0.365] | 0.275 [0.247, 0.302] | 0.500 [0.469, 0.531] | 0.739 [0.711, 0.765] |
| qwen3-0.6b-acro | auto | 1000 | 0.332 [0.305, 0.358] | 0.273 [0.245, 0.300] | 0.488 [0.457, 0.519] | 0.724 [0.695, 0.750] |
| hybrid-bm25-e5 | auto | 1000 | 0.375 [0.348, 0.402] | 0.306 [0.277, 0.334] | 0.542 [0.511, 0.572] | 0.775 [0.750, 0.800] |
| hybrid-bm25-e5-acro | auto | 1000 | 0.370 [0.342, 0.397] | 0.307 [0.278, 0.335] | 0.534 [0.503, 0.564] | 0.772 [0.747, 0.798] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 20 | 0.713 [0.609, 0.809] | 0.650 [0.545, 0.755] | 0.770 [0.670, 0.860] | 0.902 [0.788, 1.000] |
| bm25-acro | silver | 20 | 0.694 [0.595, 0.785] | 0.615 [0.510, 0.715] | 0.730 [0.635, 0.820] | 0.933 [0.825, 1.000] |
| minilm | silver | 20 | 0.639 [0.526, 0.748] | 0.535 [0.405, 0.660] | 0.750 [0.645, 0.845] | 0.835 [0.698, 0.960] |
| minilm-acro | silver | 20 | 0.644 [0.533, 0.749] | 0.555 [0.420, 0.680] | 0.760 [0.665, 0.845] | 0.820 [0.675, 0.960] |
| e5-small | silver | 20 | 0.493 [0.378, 0.606] | 0.400 [0.285, 0.515] | 0.545 [0.425, 0.665] | 0.826 [0.686, 0.950] |
| e5-small-acro | silver | 20 | 0.505 [0.394, 0.611] | 0.395 [0.285, 0.505] | 0.565 [0.450, 0.675] | 0.838 [0.713, 0.950] |
| qwen3-0.6b | silver | 20 | 0.824 [0.746, 0.896] | 0.735 [0.610, 0.845] | 0.900 [0.845, 0.950] | 0.975 [0.925, 1.000] |
| qwen3-0.6b-acro | silver | 20 | 0.806 [0.722, 0.883] | 0.710 [0.580, 0.825] | 0.875 [0.810, 0.935] | 0.975 [0.925, 1.000] |
| hybrid-bm25-e5 | silver | 20 | 0.669 [0.583, 0.754] | 0.580 [0.475, 0.685] | 0.750 [0.665, 0.830] | 0.900 [0.800, 0.975] |
| hybrid-bm25-e5-acro | silver | 20 | 0.670 [0.584, 0.755] | 0.580 [0.480, 0.685] | 0.750 [0.670, 0.830] | 0.892 [0.783, 0.975] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-acro | 0.060 | 0.040 | 0.160 | 0.040 | 0.000 |
| minilm | 0.070 | 0.080 | 0.000 | 0.000 | 0.200 |
| minilm-acro | 0.070 | 0.080 | 0.000 | 0.000 | 0.200 |
| e5-small | 0.170 | 0.160 | 0.080 | 0.000 | 0.440 |
| e5-small-acro | 0.180 | 0.160 | 0.080 | 0.000 | 0.480 |
| qwen3-0.6b | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-0.6b-acro | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-acro | 0.520 | 0.600 | 0.780 | 0.500 | 0.200 |
| minilm | 0.700 | 0.960 | 0.760 | 0.400 | 0.680 |
| minilm-acro | 0.700 | 0.920 | 0.760 | 0.440 | 0.680 |
| e5-small | 0.850 | 0.960 | 0.920 | 0.600 | 0.920 |
| e5-small-acro | 0.850 | 0.960 | 0.920 | 0.600 | 0.920 |
| qwen3-0.6b | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-0.6b-acro | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 28.9,
  "query_seconds_synopsis": 8.13,
  "query_seconds_topical": 0.15
 },
 "bm25-acro": {
  "index_seconds": 54.06,
  "query_seconds_synopsis": 8.53,
  "query_seconds_topical": 0.14
 },
 "minilm": {
  "index_seconds": 11.43,
  "query_seconds_synopsis": 6.03,
  "query_seconds_topical": 0.14
 },
 "minilm-acro": {
  "index_seconds": 207.75,
  "query_seconds_synopsis": 6.11,
  "query_seconds_topical": 0.12
 },
 "e5-small": {
  "index_seconds": 278.6,
  "query_seconds_synopsis": 6.25,
  "query_seconds_topical": 0.13
 },
 "e5-small-acro": {
  "index_seconds": 284.03,
  "query_seconds_synopsis": 6.43,
  "query_seconds_topical": 0.12
 },
 "qwen3-0.6b": {
  "index_seconds": 2.59,
  "query_seconds_synopsis": 9.3,
  "query_seconds_topical": 0.2
 },
 "qwen3-0.6b-acro": {
  "index_seconds": 1026.47,
  "query_seconds_synopsis": 9.52,
  "query_seconds_topical": 0.2
 },
 "hybrid-bm25-e5": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 31.69,
  "query_seconds_topical": 0.59
 },
 "hybrid-bm25-e5-acro": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 32.19,
  "query_seconds_topical": 0.6
 },
 "_total_index_seconds": 1893.82
}
```

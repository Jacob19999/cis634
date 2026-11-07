# Benchmark run `summary_dev`

Corpus: 47,723 reports, {'chunk_max': 136842, 'summary': 47723, 'chunk_summary': 184565} retrieval units. Device: cuda. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.423 [0.395, 0.451] | 0.359 [0.329, 0.389] | 0.565 [0.534, 0.595] | 0.776 [0.750, 0.802] |
| bm25-sum | auto | 1000 | 0.374 [0.347, 0.401] | 0.309 [0.281, 0.338] | 0.526 [0.495, 0.558] | 0.738 [0.711, 0.765] |
| bm25-chunk-sum | auto | 1000 | 0.451 [0.423, 0.479] | 0.382 [0.352, 0.412] | 0.605 [0.574, 0.634] | 0.814 [0.789, 0.837] |
| qwen3-0.6b | auto | 1000 | 0.339 [0.312, 0.365] | 0.275 [0.247, 0.302] | 0.500 [0.469, 0.531] | 0.739 [0.711, 0.765] |
| qwen3-0.6b-sum | auto | 1000 | 0.299 [0.273, 0.325] | 0.234 [0.208, 0.260] | 0.448 [0.417, 0.479] | 0.690 [0.660, 0.718] |
| qwen3-0.6b-chunk-sum | auto | 1000 | 0.373 [0.346, 0.400] | 0.309 [0.280, 0.337] | 0.527 [0.496, 0.558] | 0.771 [0.745, 0.797] |
| qwen3-8b | auto | 1000 | 0.437 [0.410, 0.464] | 0.357 [0.328, 0.387] | 0.610 [0.579, 0.640] | 0.828 [0.803, 0.851] |
| qwen3-8b-sum | auto | 1000 | 0.390 [0.363, 0.417] | 0.314 [0.285, 0.343] | 0.560 [0.530, 0.591] | 0.790 [0.765, 0.815] |
| qwen3-8b-chunk-sum | auto | 1000 | 0.469 [0.442, 0.497] | 0.390 [0.360, 0.420] | 0.639 [0.609, 0.669] | 0.847 [0.824, 0.869] |
| hybrid-bm25-qwen3-8b | auto | 1000 | 0.477 [0.449, 0.504] | 0.395 [0.365, 0.425] | 0.667 [0.637, 0.695] | 0.863 [0.841, 0.883] |
| hybrid-bm25-qwen3-8b-chunk-sum | auto | 1000 | 0.504 [0.477, 0.532] | 0.416 [0.385, 0.447] | 0.692 [0.663, 0.721] | 0.881 [0.861, 0.901] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 20 | 0.713 [0.609, 0.809] | 0.650 [0.545, 0.755] | 0.770 [0.670, 0.860] | 0.902 [0.788, 1.000] |
| bm25-sum | silver | 20 | 0.636 [0.535, 0.729] | 0.565 [0.460, 0.665] | 0.700 [0.595, 0.795] | 0.866 [0.731, 0.975] |
| bm25-chunk-sum | silver | 20 | 0.691 [0.583, 0.787] | 0.620 [0.505, 0.735] | 0.745 [0.635, 0.840] | 0.897 [0.772, 1.000] |
| qwen3-0.6b | silver | 20 | 0.824 [0.746, 0.896] | 0.735 [0.610, 0.845] | 0.900 [0.845, 0.950] | 0.975 [0.925, 1.000] |
| qwen3-0.6b-sum | silver | 20 | 0.769 [0.679, 0.853] | 0.665 [0.535, 0.780] | 0.850 [0.780, 0.915] | 0.925 [0.850, 1.000] |
| qwen3-0.6b-chunk-sum | silver | 20 | 0.792 [0.709, 0.869] | 0.700 [0.580, 0.805] | 0.880 [0.825, 0.935] | 0.925 [0.850, 1.000] |
| qwen3-8b | silver | 20 | 0.843 [0.776, 0.905] | 0.790 [0.695, 0.875] | 0.910 [0.865, 0.950] | 0.975 [0.925, 1.000] |
| qwen3-8b-sum | silver | 20 | 0.823 [0.749, 0.893] | 0.730 [0.615, 0.835] | 0.895 [0.830, 0.950] | 0.950 [0.875, 1.000] |
| qwen3-8b-chunk-sum | silver | 20 | 0.826 [0.751, 0.897] | 0.745 [0.640, 0.845] | 0.895 [0.830, 0.950] | 0.950 [0.875, 1.000] |
| hybrid-bm25-qwen3-8b | silver | 20 | 0.811 [0.737, 0.879] | 0.760 [0.685, 0.835] | 0.870 [0.805, 0.930] | 0.917 [0.825, 1.000] |
| hybrid-bm25-qwen3-8b-chunk-sum | silver | 20 | 0.797 [0.709, 0.874] | 0.725 [0.625, 0.820] | 0.860 [0.770, 0.935] | 0.950 [0.875, 1.000] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-sum | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-chunk-sum | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| qwen3-0.6b | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-0.6b-sum | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-8b | 0.740 | 0.680 | 0.880 | 0.400 | 1.000 |
| qwen3-8b-sum | 0.740 | 0.680 | 0.880 | 0.400 | 1.000 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-sum | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-chunk-sum | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| qwen3-0.6b | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-0.6b-sum | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-8b | 0.890 | 1.000 | 0.960 | 0.600 | 1.000 |
| qwen3-8b-sum | 0.890 | 1.000 | 0.960 | 0.600 | 1.000 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 28.95,
  "query_seconds_synopsis": 8.15,
  "query_seconds_topical": 0.15
 },
 "bm25-sum": {
  "index_seconds": 4.44,
  "query_seconds_synopsis": 3.09,
  "query_seconds_topical": 0.06
 },
 "bm25-chunk-sum": {
  "index_seconds": 32.95,
  "query_seconds_synopsis": 10.21,
  "query_seconds_topical": 0.18
 },
 "qwen3-0.6b": {
  "index_seconds": 19.09,
  "query_seconds_synopsis": 8.8,
  "query_seconds_topical": 0.23
 },
 "qwen3-0.6b-sum": {
  "index_seconds": 140.77,
  "query_seconds_synopsis": 5.34,
  "query_seconds_topical": 0.11
 },
 "qwen3-0.6b-chunk-sum": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 30.45,
  "query_seconds_topical": 0.57
 },
 "qwen3-8b": {
  "index_seconds": 9.01,
  "query_seconds_synopsis": 25.67,
  "query_seconds_topical": 0.45
 },
 "qwen3-8b-sum": {
  "index_seconds": 968.33,
  "query_seconds_synopsis": 21.7,
  "query_seconds_topical": 0.33
 },
 "qwen3-8b-chunk-sum": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 64.65,
  "query_seconds_topical": 1.03
 },
 "hybrid-bm25-qwen3-8b": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 51.17,
  "query_seconds_topical": 0.88
 },
 "hybrid-bm25-qwen3-8b-chunk-sum": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 87.12,
  "query_seconds_topical": 1.48
 },
 "_total_index_seconds": 1203.54
}
```

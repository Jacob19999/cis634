# Benchmark run `extended`

Corpus: 47,723 reports, {'chunk_max': 136842, 'whole': 47723} retrieval units. Device: mps. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.425 [0.397, 0.453] | 0.362 [0.333, 0.393] | 0.555 [0.524, 0.586] | 0.753 [0.727, 0.780] |
| bm25-keepneg | auto | 1000 | 0.428 [0.400, 0.455] | 0.366 [0.337, 0.396] | 0.557 [0.526, 0.588] | 0.756 [0.730, 0.783] |
| bm25-whole | auto | 1000 | 0.458 [0.429, 0.486] | 0.395 [0.364, 0.425] | 0.598 [0.567, 0.628] | 0.782 [0.757, 0.808] |
| minilm | auto | 1000 | 0.268 [0.243, 0.292] | 0.212 [0.188, 0.238] | 0.398 [0.367, 0.427] | 0.648 [0.619, 0.677] |
| minilm-whole | auto | 1000 | 0.271 [0.246, 0.296] | 0.213 [0.189, 0.239] | 0.402 [0.371, 0.431] | 0.623 [0.592, 0.653] |
| e5-small | auto | 1000 | 0.241 [0.217, 0.265] | 0.186 [0.163, 0.210] | 0.358 [0.328, 0.389] | 0.607 [0.576, 0.637] |
| e5-small-whole | auto | 1000 | 0.255 [0.230, 0.279] | 0.200 [0.175, 0.225] | 0.389 [0.358, 0.418] | 0.595 [0.564, 0.625] |
| bge-small | auto | 1000 | 0.287 [0.261, 0.313] | 0.232 [0.206, 0.258] | 0.418 [0.389, 0.448] | 0.640 [0.611, 0.670] |
| mpnet | auto | 1000 | 0.257 [0.234, 0.282] | 0.196 [0.172, 0.221] | 0.412 [0.382, 0.443] | 0.665 [0.636, 0.694] |
| e5-base | auto | 1000 | 0.280 [0.255, 0.305] | 0.218 [0.193, 0.244] | 0.408 [0.378, 0.439] | 0.632 [0.603, 0.662] |
| hybrid-bm25-e5 | auto | 1000 | 0.378 [0.351, 0.404] | 0.300 [0.272, 0.329] | 0.556 [0.525, 0.587] | 0.759 [0.733, 0.786] |
| hybrid-bm25-bge | auto | 1000 | 0.393 [0.366, 0.420] | 0.324 [0.295, 0.352] | 0.546 [0.516, 0.577] | 0.766 [0.740, 0.793] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 40 | 0.630 [0.554, 0.708] | 0.537 [0.450, 0.625] | 0.703 [0.628, 0.775] | 0.838 [0.746, 0.922] |
| bm25-keepneg | silver | 40 | 0.626 [0.548, 0.704] | 0.528 [0.443, 0.615] | 0.700 [0.627, 0.772] | 0.843 [0.752, 0.925] |
| bm25-whole | silver | 40 | 0.621 [0.543, 0.699] | 0.528 [0.438, 0.620] | 0.700 [0.622, 0.775] | 0.850 [0.764, 0.928] |
| minilm | silver | 40 | 0.591 [0.511, 0.672] | 0.463 [0.365, 0.563] | 0.740 [0.663, 0.812] | 0.786 [0.689, 0.879] |
| minilm-whole | silver | 40 | 0.615 [0.542, 0.687] | 0.478 [0.388, 0.570] | 0.720 [0.640, 0.793] | 0.887 [0.812, 0.954] |
| e5-small | silver | 40 | 0.438 [0.349, 0.528] | 0.352 [0.260, 0.448] | 0.505 [0.417, 0.593] | 0.713 [0.597, 0.823] |
| e5-small-whole | silver | 40 | 0.410 [0.324, 0.497] | 0.340 [0.253, 0.428] | 0.483 [0.388, 0.575] | 0.640 [0.525, 0.751] |
| bge-small | silver | 40 | 0.610 [0.531, 0.689] | 0.470 [0.370, 0.573] | 0.733 [0.655, 0.805] | 0.895 [0.807, 0.969] |
| mpnet | silver | 40 | 0.622 [0.545, 0.699] | 0.478 [0.378, 0.580] | 0.742 [0.677, 0.805] | 0.900 [0.825, 0.963] |
| e5-base | silver | 40 | 0.451 [0.365, 0.541] | 0.360 [0.273, 0.453] | 0.557 [0.473, 0.642] | 0.661 [0.560, 0.762] |
| hybrid-bm25-e5 | silver | 40 | 0.633 [0.560, 0.707] | 0.537 [0.453, 0.625] | 0.707 [0.635, 0.778] | 0.883 [0.812, 0.950] |
| hybrid-bm25-bge | silver | 40 | 0.658 [0.581, 0.734] | 0.543 [0.450, 0.635] | 0.765 [0.690, 0.832] | 0.850 [0.762, 0.929] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-keepneg | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-whole | 0.060 | 0.040 | 0.160 | 0.040 | 0.000 |
| minilm | 0.070 | 0.080 | 0.000 | 0.000 | 0.200 |
| minilm-whole | 0.070 | 0.080 | 0.000 | 0.000 | 0.200 |
| e5-small | 0.170 | 0.160 | 0.080 | 0.000 | 0.440 |
| e5-small-whole | 0.170 | 0.160 | 0.080 | 0.000 | 0.440 |
| bge-small | 0.180 | 0.120 | 0.160 | 0.000 | 0.440 |
| mpnet | 0.240 | 0.160 | 0.120 | 0.080 | 0.600 |
| e5-base | 0.150 | 0.080 | 0.120 | 0.000 | 0.400 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-keepneg | 0.490 | 0.600 | 0.780 | 0.500 | 0.080 |
| bm25-whole | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| minilm | 0.700 | 0.960 | 0.760 | 0.400 | 0.680 |
| minilm-whole | 0.700 | 0.960 | 0.760 | 0.400 | 0.680 |
| e5-small | 0.850 | 0.960 | 0.920 | 0.600 | 0.920 |
| e5-small-whole | 0.850 | 0.960 | 0.920 | 0.600 | 0.920 |
| bge-small | 0.850 | 0.960 | 0.960 | 0.600 | 0.880 |
| mpnet | 0.850 | 0.920 | 0.840 | 0.720 | 0.920 |
| e5-base | 0.820 | 0.920 | 0.920 | 0.520 | 0.920 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 6.91,
  "query_seconds_synopsis": 2.75,
  "query_seconds_topical": 0.08
 },
 "bm25-keepneg": {
  "index_seconds": 6.45,
  "query_seconds_synopsis": 2.8,
  "query_seconds_topical": 0.09
 },
 "bm25-whole": {
  "index_seconds": 5.24,
  "query_seconds_synopsis": 1.48,
  "query_seconds_topical": 0.05
 },
 "minilm": {
  "index_seconds": 3.4,
  "query_seconds_synopsis": 2.36,
  "query_seconds_topical": 0.13
 },
 "minilm-whole": {
  "index_seconds": 0.02,
  "query_seconds_synopsis": 1.61,
  "query_seconds_topical": 0.06
 },
 "e5-small": {
  "index_seconds": 0.27,
  "query_seconds_synopsis": 2.67,
  "query_seconds_topical": 0.09
 },
 "e5-small-whole": {
  "index_seconds": 0.01,
  "query_seconds_synopsis": 1.98,
  "query_seconds_topical": 0.07
 },
 "bge-small": {
  "index_seconds": 0.36,
  "query_seconds_synopsis": 3.24,
  "query_seconds_topical": 0.1
 },
 "mpnet": {
  "index_seconds": 0.55,
  "query_seconds_synopsis": 5.11,
  "query_seconds_topical": 0.21
 },
 "e5-base": {
  "index_seconds": 0.45,
  "query_seconds_synopsis": 4.72,
  "query_seconds_topical": 0.14
 },
 "hybrid-bm25-e5": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 7.79,
  "query_seconds_topical": 0.28
 },
 "hybrid-bm25-bge": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 7.87,
  "query_seconds_topical": 0.28
 },
 "_total_index_seconds": 23.64
}
```

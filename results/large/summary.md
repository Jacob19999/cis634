# Benchmark run `large`

Corpus: 47,723 reports, {'chunk_max': 136842, 'whole': 47723} retrieval units. Device: mps. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.425 [0.397, 0.453] | 0.362 [0.333, 0.393] | 0.555 [0.524, 0.586] | 0.753 [0.727, 0.780] |
| bm25-whole | auto | 1000 | 0.458 [0.429, 0.486] | 0.395 [0.364, 0.425] | 0.598 [0.567, 0.628] | 0.782 [0.757, 0.808] |
| bge-small | auto | 1000 | 0.287 [0.261, 0.313] | 0.232 [0.206, 0.258] | 0.418 [0.389, 0.448] | 0.640 [0.611, 0.670] |
| bge-base | auto | 1000 | 0.308 [0.282, 0.334] | 0.251 [0.224, 0.278] | 0.448 [0.416, 0.478] | 0.667 [0.638, 0.697] |
| bge-large | auto | 1000 | 0.336 [0.310, 0.362] | 0.275 [0.247, 0.301] | 0.476 [0.446, 0.507] | 0.711 [0.682, 0.738] |
| e5-small | auto | 1000 | 0.241 [0.217, 0.265] | 0.186 [0.163, 0.210] | 0.358 [0.328, 0.389] | 0.607 [0.576, 0.637] |
| e5-base | auto | 1000 | 0.280 [0.255, 0.305] | 0.218 [0.193, 0.244] | 0.408 [0.378, 0.439] | 0.632 [0.603, 0.662] |
| e5-large | auto | 1000 | 0.299 [0.274, 0.324] | 0.233 [0.207, 0.259] | 0.441 [0.410, 0.471] | 0.675 [0.646, 0.704] |
| qwen3-0.6b | auto | 1000 | 0.303 [0.278, 0.328] | 0.230 [0.204, 0.256] | 0.479 [0.448, 0.509] | 0.727 [0.699, 0.754] |
| qwen3-0.6b-generic | auto | 1000 | 0.371 [0.345, 0.398] | 0.303 [0.274, 0.332] | 0.531 [0.500, 0.562] | 0.750 [0.723, 0.776] |
| hybrid-bm25-bge | auto | 1000 | 0.393 [0.366, 0.420] | 0.324 [0.295, 0.352] | 0.546 [0.516, 0.577] | 0.766 [0.740, 0.793] |
| hybrid-bm25-bge-large | auto | 1000 | 0.419 [0.392, 0.446] | 0.346 [0.318, 0.376] | 0.584 [0.553, 0.614] | 0.793 [0.768, 0.819] |
| hybrid-bm25-qwen3 | auto | 1000 | 0.416 [0.389, 0.443] | 0.332 [0.303, 0.361] | 0.602 [0.571, 0.631] | 0.806 [0.782, 0.831] |
| hybrid-bm25-qwen3-generic | auto | 1000 | 0.446 [0.418, 0.472] | 0.373 [0.343, 0.402] | 0.615 [0.584, 0.644] | 0.805 [0.780, 0.830] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 40 | 0.630 [0.554, 0.708] | 0.537 [0.450, 0.625] | 0.703 [0.628, 0.775] | 0.838 [0.746, 0.922] |
| bm25-whole | silver | 40 | 0.621 [0.543, 0.699] | 0.528 [0.438, 0.620] | 0.700 [0.622, 0.775] | 0.850 [0.764, 0.928] |
| bge-small | silver | 40 | 0.610 [0.531, 0.689] | 0.470 [0.370, 0.573] | 0.733 [0.655, 0.805] | 0.895 [0.807, 0.969] |
| bge-base | silver | 40 | 0.583 [0.505, 0.663] | 0.460 [0.365, 0.557] | 0.710 [0.640, 0.778] | 0.820 [0.730, 0.903] |
| bge-large | silver | 40 | 0.623 [0.547, 0.699] | 0.515 [0.420, 0.610] | 0.725 [0.655, 0.793] | 0.867 [0.783, 0.942] |
| e5-small | silver | 40 | 0.438 [0.349, 0.528] | 0.352 [0.260, 0.448] | 0.505 [0.417, 0.593] | 0.713 [0.597, 0.823] |
| e5-base | silver | 40 | 0.451 [0.365, 0.541] | 0.360 [0.273, 0.453] | 0.557 [0.473, 0.642] | 0.661 [0.560, 0.762] |
| e5-large | silver | 40 | 0.556 [0.473, 0.638] | 0.432 [0.342, 0.525] | 0.657 [0.578, 0.733] | 0.799 [0.699, 0.891] |
| qwen3-0.6b | silver | 40 | 0.728 [0.649, 0.800] | 0.618 [0.527, 0.705] | 0.818 [0.745, 0.880] | 0.892 [0.808, 0.963] |
| qwen3-0.6b-generic | silver | 40 | 0.746 [0.672, 0.816] | 0.645 [0.547, 0.738] | 0.838 [0.770, 0.897] | 0.924 [0.854, 0.983] |
| hybrid-bm25-bge | silver | 40 | 0.658 [0.581, 0.734] | 0.543 [0.450, 0.635] | 0.765 [0.690, 0.832] | 0.850 [0.762, 0.929] |
| hybrid-bm25-bge-large | silver | 40 | 0.680 [0.605, 0.754] | 0.575 [0.478, 0.670] | 0.765 [0.695, 0.832] | 0.887 [0.825, 0.950] |
| hybrid-bm25-qwen3 | silver | 40 | 0.751 [0.682, 0.816] | 0.647 [0.560, 0.733] | 0.843 [0.785, 0.895] | 0.912 [0.842, 0.975] |
| hybrid-bm25-qwen3-generic | silver | 40 | 0.746 [0.678, 0.813] | 0.665 [0.580, 0.750] | 0.830 [0.775, 0.882] | 0.901 [0.830, 0.963] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| bm25-whole | 0.060 | 0.040 | 0.160 | 0.040 | 0.000 |
| bge-small | 0.180 | 0.120 | 0.160 | 0.000 | 0.440 |
| bge-base | 0.190 | 0.240 | 0.080 | 0.000 | 0.440 |
| bge-large | 0.350 | 0.280 | 0.320 | 0.120 | 0.680 |
| e5-small | 0.170 | 0.160 | 0.080 | 0.000 | 0.440 |
| e5-base | 0.150 | 0.080 | 0.120 | 0.000 | 0.400 |
| e5-large | 0.230 | 0.200 | 0.040 | 0.080 | 0.600 |
| qwen3-0.6b | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |
| qwen3-0.6b-generic | 0.350 | 0.240 | 0.240 | 0.120 | 0.800 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bm25-whole | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| bge-small | 0.850 | 0.960 | 0.960 | 0.600 | 0.880 |
| bge-base | 0.820 | 0.960 | 0.920 | 0.560 | 0.840 |
| bge-large | 0.880 | 0.960 | 0.960 | 0.640 | 0.960 |
| e5-small | 0.850 | 0.960 | 0.920 | 0.600 | 0.920 |
| e5-base | 0.820 | 0.920 | 0.920 | 0.520 | 0.920 |
| e5-large | 0.890 | 1.000 | 0.960 | 0.640 | 0.960 |
| qwen3-0.6b | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |
| qwen3-0.6b-generic | 0.860 | 1.000 | 0.960 | 0.480 | 1.000 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 5.95,
  "query_seconds_synopsis": 2.83,
  "query_seconds_topical": 0.08
 },
 "bm25-whole": {
  "index_seconds": 4.98,
  "query_seconds_synopsis": 1.48,
  "query_seconds_topical": 0.05
 },
 "bge-small": {
  "index_seconds": 4.31,
  "query_seconds_synopsis": 3.31,
  "query_seconds_topical": 0.14
 },
 "bge-base": {
  "index_seconds": 0.77,
  "query_seconds_synopsis": 4.23,
  "query_seconds_topical": 0.15
 },
 "bge-large": {
  "index_seconds": 1.38,
  "query_seconds_synopsis": 9.99,
  "query_seconds_topical": 0.27
 },
 "e5-small": {
  "index_seconds": 0.78,
  "query_seconds_synopsis": 2.62,
  "query_seconds_topical": 0.1
 },
 "e5-base": {
  "index_seconds": 0.94,
  "query_seconds_synopsis": 4.14,
  "query_seconds_topical": 0.14
 },
 "e5-large": {
  "index_seconds": 1.77,
  "query_seconds_synopsis": 9.26,
  "query_seconds_topical": 0.28
 },
 "qwen3-0.6b": {
  "index_seconds": 2.28,
  "query_seconds_synopsis": 22.75,
  "query_seconds_topical": 0.58
 },
 "qwen3-0.6b-generic": {
  "index_seconds": 0.08,
  "query_seconds_synopsis": 20.85,
  "query_seconds_topical": 0.54
 },
 "hybrid-bm25-bge": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 7.98,
  "query_seconds_topical": 0.28
 },
 "hybrid-bm25-bge-large": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 15.03,
  "query_seconds_topical": 0.45
 },
 "hybrid-bm25-qwen3": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 27.94,
  "query_seconds_topical": 0.76
 },
 "hybrid-bm25-qwen3-generic": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 25.94,
  "query_seconds_topical": 0.73
 },
 "_total_index_seconds": 23.24
}
```

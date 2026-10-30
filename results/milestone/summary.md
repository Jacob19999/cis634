# Benchmark run `milestone`

Corpus: 47,723 reports, {'chunk_max': 136842} retrieval units. Device: mps. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.425 [0.397, 0.453] | 0.362 [0.333, 0.393] | 0.555 [0.524, 0.586] | 0.753 [0.727, 0.780] |
| minilm | auto | 1000 | 0.268 [0.243, 0.292] | 0.212 [0.188, 0.238] | 0.398 [0.367, 0.427] | 0.648 [0.619, 0.677] |
| e5-small | auto | 1000 | 0.241 [0.217, 0.265] | 0.186 [0.163, 0.210] | 0.358 [0.328, 0.389] | 0.607 [0.576, 0.637] |
| hybrid-bm25-e5 | auto | 1000 | 0.378 [0.351, 0.404] | 0.300 [0.272, 0.329] | 0.556 [0.525, 0.587] | 0.759 [0.733, 0.786] |
| hybrid-bm25-minilm | auto | 1000 | 0.380 [0.353, 0.406] | 0.304 [0.277, 0.333] | 0.559 [0.528, 0.590] | 0.786 [0.761, 0.811] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 40 | 0.632 [0.555, 0.710] | 0.537 [0.450, 0.625] | 0.705 [0.630, 0.777] | 0.838 [0.746, 0.922] |
| minilm | silver | 40 | 0.593 [0.513, 0.674] | 0.468 [0.370, 0.568] | 0.740 [0.663, 0.812] | 0.786 [0.689, 0.879] |
| e5-small | silver | 40 | 0.438 [0.349, 0.528] | 0.352 [0.260, 0.448] | 0.505 [0.417, 0.593] | 0.713 [0.597, 0.823] |
| hybrid-bm25-e5 | silver | 40 | 0.633 [0.560, 0.707] | 0.537 [0.453, 0.625] | 0.707 [0.635, 0.778] | 0.883 [0.812, 0.950] |
| hybrid-bm25-minilm | silver | 40 | 0.706 [0.639, 0.773] | 0.585 [0.495, 0.675] | 0.798 [0.733, 0.860] | 0.950 [0.900, 0.988] |

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| minilm | 0.070 | 0.080 | 0.000 | 0.000 | 0.200 |
| e5-small | 0.170 | 0.160 | 0.080 | 0.000 | 0.440 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| minilm | 0.700 | 0.960 | 0.760 | 0.400 | 0.680 |
| e5-small | 0.850 | 0.960 | 0.920 | 0.600 | 0.920 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 6.28,
  "query_seconds_synopsis": 2.74,
  "query_seconds_topical": 0.09
 },
 "minilm": {
  "index_seconds": 191.71,
  "query_seconds_synopsis": 2.32,
  "query_seconds_topical": 0.11
 },
 "e5-small": {
  "index_seconds": 317.52,
  "query_seconds_synopsis": 2.58,
  "query_seconds_topical": 0.11
 },
 "hybrid-bm25-e5": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 7.93,
  "query_seconds_topical": 0.3
 },
 "hybrid-bm25-minilm": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 7.58,
  "query_seconds_topical": 0.27
 },
 "_total_index_seconds": 515.51
}
```

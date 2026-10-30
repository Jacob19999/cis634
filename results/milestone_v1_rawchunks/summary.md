# Benchmark run `milestone`

Corpus: 47,723 reports, {'chunk_max': 149959} retrieval units. Device: mps. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.428 [0.400, 0.456] | 0.370 [0.341, 0.400] | 0.556 [0.525, 0.587] | 0.751 [0.725, 0.778] |
| minilm | auto | 1000 | 0.266 [0.242, 0.291] | 0.210 [0.186, 0.235] | 0.405 [0.374, 0.434] | 0.646 [0.617, 0.675] |
| e5-small | auto | 1000 | 0.242 [0.218, 0.266] | 0.187 [0.163, 0.211] | 0.363 [0.334, 0.394] | 0.607 [0.576, 0.636] |
| hybrid-bm25-e5 | auto | 1000 | 0.380 [0.354, 0.407] | 0.304 [0.276, 0.333] | 0.555 [0.525, 0.586] | 0.763 [0.738, 0.789] |
| hybrid-bm25-minilm | auto | 1000 | 0.381 [0.354, 0.407] | 0.304 [0.276, 0.333] | 0.557 [0.526, 0.588] | 0.785 [0.760, 0.810] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 40 | 0.638 [0.561, 0.715] | 0.540 [0.453, 0.628] | 0.705 [0.630, 0.777] | 0.863 [0.772, 0.946] |
| minilm | silver | 40 | 0.576 [0.492, 0.661] | 0.448 [0.348, 0.550] | 0.725 [0.643, 0.802] | 0.777 [0.674, 0.874] |
| e5-small | silver | 40 | 0.444 [0.358, 0.533] | 0.338 [0.250, 0.430] | 0.510 [0.425, 0.597] | 0.734 [0.613, 0.847] |
| hybrid-bm25-e5 | silver | 40 | 0.633 [0.560, 0.708] | 0.537 [0.455, 0.622] | 0.710 [0.635, 0.782] | 0.844 [0.756, 0.925] |
| hybrid-bm25-minilm | silver | 40 | 0.704 [0.638, 0.770] | 0.588 [0.500, 0.675] | 0.802 [0.738, 0.863] | 0.933 [0.875, 0.988] |

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
  "index_seconds": 6.36,
  "query_seconds_synopsis": 2.86,
  "query_seconds_topical": 0.09
 },
 "minilm": {
  "index_seconds": 3.16,
  "query_seconds_synopsis": 2.32,
  "query_seconds_topical": 0.14
 },
 "e5-small": {
  "index_seconds": 398.33,
  "query_seconds_synopsis": 2.65,
  "query_seconds_topical": 0.14
 },
 "hybrid-bm25-e5": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 8.11,
  "query_seconds_topical": 0.28
 },
 "hybrid-bm25-minilm": {
  "index_seconds": 0.0,
  "query_seconds_synopsis": 7.68,
  "query_seconds_topical": 0.28
 },
 "_total_index_seconds": 407.85
}
```

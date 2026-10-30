# Benchmark run `smoke`

Corpus: 47,723 reports, {'chunk_max': 149959} retrieval units. Device: mps. Means with 95% bootstrap CIs.

## Meaning-change (contrast) test

Accuracy = share of items where the anchor is more similar to the same-meaning sentence than to the meaning-changed one. Chance = 0.5; ties count 0.5.
`original_anchor` (proposal): anchor = original; the changed version is a one-word edit of it.
`paraphrase_anchor` (controlled): anchor = paraphrase; original and changed overlap it equally.

### original_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.050 | 0.040 | 0.120 | 0.040 | 0.000 |
| minilm | 0.070 | 0.080 | 0.000 | 0.000 | 0.200 |

### paraphrase_anchor

| system | all | altitude | direction | event_order | negation |
|---|---|---|---|---|---|
| bm25 | 0.525 | 0.600 | 0.780 | 0.500 | 0.220 |
| minilm | 0.700 | 0.960 | 0.760 | 0.400 | 0.680 |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 6.35
 },
 "minilm": {
  "index_seconds": 3.85
 },
 "hybrid": {
  "index_seconds": 0.0
 },
 "_total_index_seconds": 10.2
}
```

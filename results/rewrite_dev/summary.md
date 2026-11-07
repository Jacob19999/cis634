# Benchmark run `rewrite_dev`

Corpus: 47,723 reports, {'chunk_max': 136842} retrieval units. Device: cuda. Means with 95% bootstrap CIs.

## Synopsis known-item track

Query = NASA analyst synopsis; relevant = the report it summarises (and reports of the same incident).

| system | labels | n_queries | MRR@10 | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| bm25 | auto | 1000 | 0.423 [0.395, 0.451] | 0.359 [0.329, 0.389] | 0.565 [0.534, 0.595] | 0.776 [0.750, 0.802] |
| bm25-rw-append | auto | 1000 | 0.404 [0.375, 0.433] | 0.348 [0.318, 0.378] | 0.528 [0.497, 0.559] | 0.760 [0.734, 0.786] |
| bm25-rw-replace | auto | 1000 | 0.334 [0.307, 0.361] | 0.277 [0.249, 0.305] | 0.465 [0.434, 0.496] | 0.677 [0.648, 0.706] |
| e5-small | auto | 1000 | 0.225 [0.203, 0.248] | 0.170 [0.147, 0.193] | 0.368 [0.338, 0.398] | 0.590 [0.558, 0.619] |
| e5-small-rw-append | auto | 1000 | 0.218 [0.194, 0.242] | 0.173 [0.150, 0.197] | 0.330 [0.300, 0.358] | 0.574 [0.543, 0.603] |
| e5-small-rw-replace | auto | 1000 | 0.180 [0.159, 0.202] | 0.134 [0.113, 0.155] | 0.288 [0.260, 0.316] | 0.515 [0.484, 0.545] |
| qwen3-0.6b | auto | 1000 | 0.339 [0.312, 0.365] | 0.275 [0.247, 0.302] | 0.500 [0.469, 0.531] | 0.739 [0.711, 0.765] |
| qwen3-0.6b-rw-append | auto | 1000 | 0.335 [0.309, 0.362] | 0.267 [0.239, 0.295] | 0.508 [0.477, 0.538] | 0.736 [0.707, 0.763] |
| qwen3-0.6b-rw-replace | auto | 1000 | 0.306 [0.279, 0.331] | 0.245 [0.219, 0.271] | 0.454 [0.423, 0.485] | 0.699 [0.671, 0.727] |
| qwen3-8b | auto | 1000 | 0.437 [0.410, 0.464] | 0.357 [0.328, 0.387] | 0.610 [0.579, 0.640] | 0.828 [0.803, 0.851] |
| qwen3-8b-rw-append | auto | 1000 | 0.420 [0.393, 0.448] | 0.343 [0.314, 0.372] | 0.596 [0.566, 0.626] | 0.824 [0.799, 0.847] |
| qwen3-8b-rw-replace | auto | 1000 | 0.397 [0.370, 0.424] | 0.319 [0.290, 0.347] | 0.570 [0.540, 0.601] | 0.807 [0.782, 0.831] |

## Topical track

P@10 counts grade-2 (same issue); P@10_lenient counts grade >= 1. `silver` labels come from analyst fields.

| system | labels | n_queries | nDCG@10 | P@10 | P@10_lenient | MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | silver | 20 | 0.713 [0.609, 0.809] | 0.650 [0.545, 0.755] | 0.770 [0.670, 0.860] | 0.902 [0.788, 1.000] |
| bm25-rw-append | silver | 20 | 0.707 [0.600, 0.806] | 0.640 [0.525, 0.750] | 0.780 [0.670, 0.875] | 0.887 [0.775, 0.975] |
| bm25-rw-replace | silver | 20 | 0.628 [0.501, 0.749] | 0.550 [0.420, 0.680] | 0.735 [0.605, 0.850] | 0.800 [0.658, 0.929] |
| e5-small | silver | 20 | 0.493 [0.378, 0.606] | 0.400 [0.285, 0.515] | 0.545 [0.425, 0.665] | 0.826 [0.686, 0.950] |
| e5-small-rw-append | silver | 20 | 0.598 [0.480, 0.706] | 0.495 [0.365, 0.615] | 0.685 [0.565, 0.795] | 0.823 [0.671, 0.950] |
| e5-small-rw-replace | silver | 20 | 0.525 [0.408, 0.637] | 0.415 [0.290, 0.530] | 0.620 [0.495, 0.735] | 0.768 [0.621, 0.900] |
| qwen3-0.6b | silver | 20 | 0.824 [0.746, 0.896] | 0.735 [0.610, 0.845] | 0.900 [0.845, 0.950] | 0.975 [0.925, 1.000] |
| qwen3-0.6b-rw-append | silver | 20 | 0.743 [0.645, 0.837] | 0.650 [0.525, 0.770] | 0.845 [0.760, 0.920] | 0.904 [0.792, 1.000] |
| qwen3-0.6b-rw-replace | silver | 20 | 0.695 [0.577, 0.803] | 0.590 [0.445, 0.725] | 0.810 [0.695, 0.905] | 0.833 [0.692, 0.950] |
| qwen3-8b | silver | 20 | 0.843 [0.776, 0.905] | 0.790 [0.695, 0.875] | 0.910 [0.865, 0.950] | 0.975 [0.925, 1.000] |
| qwen3-8b-rw-append | silver | 20 | 0.783 [0.701, 0.860] | 0.695 [0.580, 0.805] | 0.885 [0.825, 0.940] | 0.950 [0.875, 1.000] |
| qwen3-8b-rw-replace | silver | 20 | 0.780 [0.694, 0.862] | 0.700 [0.580, 0.815] | 0.885 [0.820, 0.945] | 0.925 [0.850, 1.000] |

## Timings (seconds)

```
{
 "bm25": {
  "index_seconds": 29.08,
  "query_seconds_synopsis": 8.09,
  "query_seconds_topical": 0.14
 },
 "bm25-rw-append": {
  "index_seconds": 28.27,
  "query_seconds_synopsis": 9.4,
  "query_seconds_topical": 0.18
 },
 "bm25-rw-replace": {
  "index_seconds": 28.33,
  "query_seconds_synopsis": 8.98,
  "query_seconds_topical": 0.17
 },
 "e5-small": {
  "index_seconds": 11.73,
  "query_seconds_synopsis": 6.84,
  "query_seconds_topical": 0.14
 },
 "e5-small-rw-append": {
  "index_seconds": 0.09,
  "query_seconds_synopsis": 7.09,
  "query_seconds_topical": 0.13
 },
 "e5-small-rw-replace": {
  "index_seconds": 0.09,
  "query_seconds_synopsis": 6.45,
  "query_seconds_topical": 0.12
 },
 "qwen3-0.6b": {
  "index_seconds": 2.96,
  "query_seconds_synopsis": 9.68,
  "query_seconds_topical": 0.19
 },
 "qwen3-0.6b-rw-append": {
  "index_seconds": 0.24,
  "query_seconds_synopsis": 11.68,
  "query_seconds_topical": 0.23
 },
 "qwen3-0.6b-rw-replace": {
  "index_seconds": 0.24,
  "query_seconds_synopsis": 9.84,
  "query_seconds_topical": 0.22
 },
 "qwen3-8b": {
  "index_seconds": 11.76,
  "query_seconds_synopsis": 26.69,
  "query_seconds_topical": 0.52
 },
 "qwen3-8b-rw-append": {
  "index_seconds": 0.99,
  "query_seconds_synopsis": 43.57,
  "query_seconds_topical": 0.8
 },
 "qwen3-8b-rw-replace": {
  "index_seconds": 0.99,
  "query_seconds_synopsis": 32.52,
  "query_seconds_topical": 0.71
 },
 "_total_index_seconds": 114.78
}
```

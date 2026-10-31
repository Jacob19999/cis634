# Provenance

First milestone run, produced at commit `ce01873` with preprocessing v1:
150-word chunks, 1-sentence overlap, **no** tail merging, and ASRS placeholder second
narratives ("[Report narrative contained no additional information.]") still indexed.
That gave 149,959 chunks, 3.7% of them under 30 words.

Error analysis of this run showed E5-small's top-10 matched passages were 18.5% short
fragments (< 30 words), mostly placeholders. This motivated preprocessing v2
(placeholder removal + tail merging). Kept as the "before" condition of that ablation.

The topical metrics in this folder use the silver labels as they were then, before rule
patterns were matched at word starts. `scripts/analysis.py --v1` re-scores the saved run
files with the current labels (E5-small nDCG@10 stays 0.444).

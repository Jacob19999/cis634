# Provenance

First milestone run, produced at commit `ce01873` with preprocessing v1:
150-word chunks, 1-sentence overlap, **no** tail merging, and ASRS placeholder second
narratives ("[Report narrative contained no additional information.]") still indexed.
That gave 149,959 chunks, 3.7% of them under 30 words.

Error analysis of this run showed E5-small's top-10 matched passages were 18.5% short
fragments (< 30 words), mostly placeholders. This motivated preprocessing v2
(placeholder removal + tail merging). Kept as the "before" condition of that ablation.

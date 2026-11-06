"""Acronym expansion: write out an aviation acronym's meaning after its first use.

``TCAS issued an RA`` becomes ``TCAS (traffic alert and collision avoidance system) issued
an RA (resolution advisory)``, the way a careful writer defines an acronym once. BM25 then
matches queries that spell the term out, and encoders see the words they were trained on.

A retriever applies it to its units and its queries when its config sets
``expand: <glossary.tsv>`` (see ``Retriever.prepare``). Glossaries live in ``data/glossary/``
and are versioned like the evaluation data; ``asrs_acronyms_v1.tsv`` says how its entries
were chosen.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path


class AcronymExpander:
    """Appends ``(expansion)`` after the first occurrence of each known acronym in a text.

    Matching is case-sensitive and whole-word, so "RA" matches but "Ra" and "RAIN" do not;
    a plural "RAs" counts as a use of "RA". An acronym whose expansion already appears in
    the text (e.g. "resolution advisory (RA)") is left alone.
    """

    def __init__(self, glossary: dict[str, str], name: str = "custom"):
        self.glossary = dict(glossary)
        self.name = name
        # Longest first, so a longer acronym wins over a prefix of it in the alternation.
        keys = sorted(self.glossary, key=len, reverse=True)
        self._pattern = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")s?\b")

    @classmethod
    def load(cls, path: str | Path) -> "AcronymExpander":
        """Read a ``abbr<TAB>expansion<TAB>source`` glossary; its file stem names the expander."""
        return _load(str(path))

    def __call__(self, text: str) -> str:
        """``text`` with each known acronym's first use followed by its expansion."""
        lower = text.lower()
        seen: set[str] = set()

        def repl(m: re.Match) -> str:
            abbr = m.group(1)
            if abbr in seen:
                return m.group(0)
            seen.add(abbr)
            exp = self.glossary[abbr]
            return m.group(0) if exp in lower else f"{m.group(0)} ({exp})"

        return self._pattern.sub(repl, text)

    def apply(self, texts: list[str]) -> list[str]:
        """Expand every text in a list."""
        return [self(t) for t in texts]


@lru_cache(maxsize=None)
def _load(path: str) -> AcronymExpander:
    # Systems in one config usually share a glossary: parse it once.
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return AcronymExpander({r["abbr"]: r["expansion"].lower() for r in rows}, Path(path).stem)

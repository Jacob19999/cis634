"""Build retrieval systems from config entries.

A config entry looks like::

    - name: e5-small
      type: dense            # bm25 | dense | hybrid
      params: {model: intfloat/e5-small-v2, query_prefix: "query: ", passage_prefix: "passage: "}

Hybrid systems name their components, which must be defined earlier in the list.
"""

from __future__ import annotations

from .base import DOC_MODES, Retriever, SearchResult, Units
from .bm25 import BM25Retriever
from .dense import DenseRetriever, resolve_device
from .hybrid import HybridRetriever

# Config "type" -> retriever class.
TYPES = {"bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever}


def build_systems(specs: list[dict], unit_sets: dict[str, Units], device: str = "auto",
                  cache_dir: str = "cache",
                  existing: dict[str, Retriever] | None = None) -> dict[str, Retriever]:
    """Instantiate and index every system in ``specs``.

    ``unit_sets`` maps a doc_mode ("chunk_max" / "whole") to the units to index.
    ``existing`` holds already-built systems that hybrids may reference; the returned dict
    contains them plus the new ones.
    """
    systems: dict[str, Retriever] = dict(existing or {})
    for spec in specs:
        kind, name = spec["type"], spec["name"]
        params = dict(spec.get("params", {}))
        if name in systems:
            raise ValueError(f"duplicate system name {name!r}")
        # A hybrid fuses systems built earlier in the list, so it has no index of its own.
        if kind == "hybrid":
            comps = [systems[c] for c in params.pop("components")]
            system = HybridRetriever(name, comps, **params)
            system.index(comps[0].units)
        elif kind in TYPES:
            # Dense models also need the device and the embedding cache folder.
            if kind == "dense":
                params.setdefault("device", device)
                params.setdefault("cache_dir", cache_dir)
            system = TYPES[kind](name=name, **params)
            system.index(unit_sets[system.doc_mode])
        else:
            raise ValueError(f"unknown system type {kind!r}; choose from {sorted(TYPES)}")
        systems[name] = system
    return systems


__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "Retriever", "SearchResult",
           "Units", "DOC_MODES", "TYPES", "build_systems", "resolve_device"]

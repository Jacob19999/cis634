"""Frozen bi-encoder retrieval with sentence-transformers.

Any sentence-transformers checkpoint works; models that expect instruction prefixes
(E5: "query: " / "passage: ", BGE: a query instruction) get them from the config.
Unit embeddings are cached under ``cache/embeddings`` so re-running a benchmark only
encodes the queries.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .base import Retriever, Units


# Systems that use the same checkpoint (e.g. chunk vs whole-report variants) share one
# loaded model instead of each holding a copy in (GPU) memory.
_MODEL_CACHE: dict[tuple, object] = {}


def _free_accelerator_memory(device: str) -> None:
    """Release GPU memory after encoding the corpus."""
    import gc

    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def resolve_device(device: str = "auto") -> str:
    """With "auto", use CUDA if available, else Apple MPS, else CPU."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseRetriever(Retriever):
    """Embed every unit once; a query's score for a unit is the cosine similarity."""

    def __init__(self, name: str, model: str, query_prefix: str = "", passage_prefix: str = "",
                 pair_prefix: str | None = None, max_seq_length: int | None = None,
                 batch_size: int = 128, device: str = "auto", cache_dir: str = "cache",
                 **kw):
        super().__init__(name, **kw)
        self.model_name = model
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        # Symmetric sentence-vs-sentence scoring (contrast test). E5's card recommends the
        # "query: " prefix on both sides for similarity tasks.
        self.pair_prefix = query_prefix if pair_prefix is None else pair_prefix
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.cache_dir = Path(cache_dir) / "embeddings"
        self._model = None
        self.encode_seconds: float | None = None

    @property
    def model(self):
        """Load the model on first use, shared by systems with the same checkpoint."""
        if self._model is None:
            key = (self.model_name, self.device, self.max_seq_length)
            if key not in _MODEL_CACHE:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.model_name, device=self.device)
                if self.max_seq_length:
                    model.max_seq_length = self.max_seq_length
                _MODEL_CACHE[key] = model
            self._model = _MODEL_CACHE[key]
        return self._model

    def encode(self, texts: list[str], prefix: str = "", show_progress: bool = False) -> np.ndarray:
        """Embed texts (with an optional prefix) as unit-length vectors."""
        return self.model.encode(
            [prefix + t for t in texts], batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=show_progress,
        ).astype(np.float32)

    def _cache_path(self, units: Units) -> Path:
        """Cache file named after the model, prefix, max length and the unit set."""
        key = json.dumps([self.model_name, self.passage_prefix, self.model.max_seq_length,
                          units.tag, len(units), units.texts[0], units.texts[-1]])
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_name).strip("-")
        return self.cache_dir / f"{slug}__{units.tag}__{digest}.npy"

    def build(self, units: Units) -> None:
        # Reuse the saved embeddings if this exact unit set was encoded before.
        path = self._cache_path(units)
        if path.exists():
            self.E = np.load(path)
            self.encode_seconds = None
            return
        t0 = time.time()
        self.E = self.encode(units.texts, self.passage_prefix, show_progress=True)
        self.encode_seconds = time.time() - t0
        _free_accelerator_memory(self.device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.E)

    def score_units(self, queries: list[str]) -> np.ndarray:
        # Vectors are unit length, so the dot product equals cosine similarity.
        q = self.encode(queries, self.query_prefix)
        return q @ self.E.T

    def pair_scores(self, queries: list[str], texts: list[str]) -> np.ndarray:
        """Cosine similarity of each (query, text) pair (meaning-change test)."""
        a = self.encode(queries, self.pair_prefix)
        b = self.encode(texts, self.pair_prefix)
        return (a * b).sum(axis=1)

    def describe(self) -> dict:
        return {**super().describe(), "model": self.model_name,
                "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "max_seq_length": self.max_seq_length or "model default", "device": self.device,
                "encode_seconds": self.encode_seconds}

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


def _encode_slice(model_name: str, dtype: str | None, max_seq_length: int | None,
                  trust_remote_code: bool, device: str, texts: list[str], batch_size: int,
                  out_path: str, show_progress: bool) -> None:
    """Worker for multi-device encoding: embed ``texts`` on ``device`` and save them."""
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"dtype": getattr(torch, dtype)} if dtype else {}
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs,
                                trust_remote_code=trust_remote_code)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    E = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=show_progress)
    np.save(out_path, E.astype(np.float32))


class DenseRetriever(Retriever):
    """Embed every unit once; a query's score for a unit is the cosine similarity."""

    def __init__(self, name: str, model: str, query_prefix: str = "", passage_prefix: str = "",
                 pair_prefix: str | None = None, max_seq_length: int | None = None,
                 batch_size: int = 128, device: str = "auto", cache_dir: str = "cache",
                 dtype: str | None = None, trust_remote_code: bool = False,
                 devices: list[str] | None = None, **kw):
        super().__init__(name, **kw)
        # Encode the corpus on several GPUs at once (one worker per device); queries still
        # use ``device``. Only takes effect when some units are not cached yet.
        self.devices = list(devices) if devices else None
        self.model_name = model
        # Weight precision, e.g. "float16". Half precision is 1.5x faster for Qwen3-0.6B on
        # Apple MPS; embeddings keep cosine >= 0.998 with float32.
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
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
            self._model = self._load(self.device)
        return self._model

    def _load(self, device: str):
        """The checkpoint on ``device``, from the shared model cache."""
        key = (self.model_name, device, self.max_seq_length, self.dtype)
        if key not in _MODEL_CACHE:
            import torch
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"dtype": getattr(torch, self.dtype)} if self.dtype else {}
            model = SentenceTransformer(self.model_name, device=device, model_kwargs=model_kwargs,
                                        trust_remote_code=self.trust_remote_code)
            if self.max_seq_length:
                model.max_seq_length = self.max_seq_length
            _MODEL_CACHE[key] = model
        return _MODEL_CACHE[key]

    def encode(self, texts: list[str], prefix: str = "", show_progress: bool = False) -> np.ndarray:
        """Embed texts (with an optional prefix) as unit-length vectors."""
        return self.model.encode(
            [prefix + t for t in texts], batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=show_progress,
        ).astype(np.float32)

    def _cache_path(self, units: Units) -> Path:
        """Cache file named after the model, prefix, max length and the unit set."""
        # The configured max length equals the loaded model's, so the model is only loaded
        # here when no max length is configured.
        max_len = self.max_seq_length or self.model.max_seq_length
        parts = [self.model_name, self.passage_prefix, max_len,
                 units.tag, len(units), units.texts[0], units.texts[-1]]
        if self.dtype:  # only when set, so float32 caches written before this option stay valid
            parts.append(self.dtype)
        key = json.dumps(parts)
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_name).strip("-")
        return self.cache_dir / f"{slug}__{units.tag}__{digest}.npy"

    @staticmethod
    def segments(units: Units) -> list[tuple[int, int]]:
        """Contiguous blocks of units from one source, each cached on its own.

        Corpus v2 ids carry a source prefix (``ASRS:``, ``NTSB:``, ``ASRSN:``), and sorting
        keeps each source contiguous, so the ASRS block of v2 has exactly the v1 units and
        reuses their cache. Ids without a prefix (corpus v1) form a single block.
        """
        ids = [str(u) for u in units.unit_ids]
        if not ids or ":" not in ids[0]:
            return [(0, len(units))]
        src = [u.split(":", 1)[0] for u in ids]
        cuts = [0] + [i for i in range(1, len(src)) if src[i] != src[i - 1]] + [len(src)]
        return list(zip(cuts[:-1], cuts[1:]))

    def build(self, units: Units) -> None:
        # Reuse the saved embeddings of every block that was encoded before.
        blocks = []
        for s, e in self.segments(units):
            sub = Units(units.texts[s:e], units.unit_ids[s:e], units.unit_ids[s:s + 1],
                        np.zeros(1, dtype=int), units.tag)
            path = self._cache_path(sub)
            blocks.append((sub, path, np.load(path) if path.exists() else None))
        todo = [b for b in blocks if b[2] is None]
        self.encode_seconds = None
        if todo:
            t0 = time.time()
            texts = [t for sub, _, _ in todo for t in sub.texts]
            E = self._encode_corpus(texts)
            self.encode_seconds = time.time() - t0
            k = 0
            for i, (sub, path, _) in enumerate(blocks):
                if blocks[i][2] is None:
                    blocks[i] = (sub, path, E[k : k + len(sub)])
                    k += len(sub)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, blocks[i][2])
        self.E = np.concatenate([b[2] for b in blocks]) if len(blocks) > 1 else blocks[0][2]

    def _encode_corpus(self, texts: list[str]) -> np.ndarray:
        """Passage embeddings, on all of ``devices`` when set."""
        if not self.devices:
            E = self.encode(texts, self.passage_prefix, show_progress=True)
            _free_accelerator_memory(self.device)
            return E
        # One process per device, each loading the checkpoint itself and returning its slice
        # through a file: sentence-transformers' own pool shares the parent's model through
        # /dev/shm, which containers often cap below the size of a multi-billion-parameter model.
        import multiprocessing as mp
        import tempfile
        from concurrent.futures import ProcessPoolExecutor

        texts = [self.passage_prefix + t for t in texts]
        bounds = np.linspace(0, len(texts), len(self.devices) + 1).astype(int)
        with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(
                len(self.devices), mp_context=mp.get_context("spawn")) as pool:
            paths = [str(Path(tmp) / f"part{i}.npy") for i in range(len(self.devices))]
            futures = [pool.submit(_encode_slice, self.model_name, self.dtype, self.max_seq_length,
                                   self.trust_remote_code, dev, texts[a:b], self.batch_size, p, i == 0)
                       for i, (dev, a, b, p) in enumerate(zip(self.devices, bounds[:-1],
                                                              bounds[1:], paths))]
            for f in futures:
                f.result()  # re-raises a worker's error
            return np.concatenate([np.load(p) for p in paths])

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
                "dtype": self.dtype or "float32", "devices": self.devices,
                "encode_seconds": self.encode_seconds}

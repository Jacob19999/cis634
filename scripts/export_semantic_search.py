"""Export a compact, static MiniLM summary index for the Vercel-hosted explorer.

Run after exporting the report catalogue. This separate UI index never changes benchmark
embeddings, evaluation data or metrics. Query encoding runs in the browser with the ONNX
conversion of the same checkpoint; no inference server or paid API is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BROWSER_MODEL = "Xenova/all-MiniLM-L6-v2"


def catalogue(page: Path) -> list[dict]:
    """Read report shards in exactly the order used by the explorer."""
    data = json.loads((page / "data/graph.json").read_text())
    if "reports" in data:
        return data["reports"]
    return [r for k in range(data["meta"]["n_report_shards"])
            for r in json.loads((page / f"data/reports/{k}.json").read_text())]


def identity(report: dict) -> str:
    """Return the source-qualified identity used by UnifiedReports.identity."""
    source = {0: "ASRS", 1: "NTSB", 2: "ASRSN"}[report.get("src", 0)]
    return report.get("id") or f"{source}:{report['acn']}"


def quantise(vectors: np.ndarray) -> np.ndarray:
    """Quantise each vector symmetrically; its scale cancels during cosine scoring."""
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite matrix of report embeddings")
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("A report has a zero embedding")
    return np.rint(vectors / scale * 127).clip(-127, 127).astype(np.int8)


def write_index(page: Path, reports: list[dict], vectors: np.ndarray,
                browser_revision: str, rows_per_shard: int = 20_000) -> dict:
    """Write content-addressed int8 shards and publish their manifest last."""
    if len(reports) != len(vectors) or not reports:
        raise ValueError("Embeddings must match the non-empty report catalogue")
    ids = [identity(r) for r in reports]
    if len(set(ids)) != len(ids):
        raise ValueError("Report identities must be unique")
    packed = quantise(vectors)
    out = page / "data/semantic"
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    for start in range(0, len(reports), rows_per_shard):
        body = packed[start:start + rows_per_shard].tobytes()
        digest = hashlib.sha256(body).hexdigest()
        name = f"vectors-{start}-{digest[:16]}.bin"
        (out / name).write_bytes(body)
        shards.append({"file": name, "start": start,
                       "rows": min(rows_per_shard, len(reports) - start), "sha256": digest})
    manifest = {"version": 1, "model": BROWSER_MODEL, "revision": browser_revision,
                "dimensions": packed.shape[1], "reports": len(reports),
                "max_length": 256, "dtype": "int8", "text": "AI summary (synopsis fallback)",
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "shards": shards}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Encode exported summaries locally and export the independent browser index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path)
    parser.add_argument("--browser-revision", required=True, help="Pinned ONNX repository commit")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    from asrsbench.retrievers.dense import resolve_device

    reports = catalogue(args.page)
    texts = [r.get("summary") or r.get("synopsis") or "" for r in reports]
    if not all(t.strip() for t in texts):
        raise ValueError("Every report needs a summary or synopsis")
    # Keep this cache separate from all benchmark model and unit-set caches.
    fingerprint = hashlib.sha256(json.dumps([MODEL, MODEL_REVISION, texts], ensure_ascii=False).encode()).hexdigest()
    cache = Path("cache/browser-search") / f"{fingerprint}.npy"
    if cache.exists():
        vectors = np.load(cache)
    else:
        model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device=resolve_device(args.device))
        model.max_seq_length = 256
        vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    manifest = write_index(args.page, reports, vectors, args.browser_revision)
    print(json.dumps({"reports": manifest["reports"], "dimensions": manifest["dimensions"],
                      "bytes": vectors.shape[0] * vectors.shape[1]}, indent=2))


if __name__ == "__main__":
    main()

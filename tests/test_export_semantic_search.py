"""Offline checks for the browser-only semantic index export."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "export_semantic", Path(__file__).resolve().parents[1] / "scripts/export_semantic_search.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_static_index_preserves_order_and_writes_bounded_shards(tmp_path):
    reports = [{"src": 0, "acn": "42"}, {"src": 1, "acn": "42"},
               {"src": 2, "acn": "43", "id": "ASRSN:43"}]
    vectors = np.array([[1., 0., 0.], [0., -1., 0.], [0.3, 0.4, 0.]])
    manifest = export.write_index(tmp_path, reports, vectors, "pinned", rows_per_shard=2)
    assert manifest["reports"] == 3
    assert [s["start"] for s in manifest["shards"]] == [0, 2]
    assert [s["rows"] for s in manifest["shards"]] == [2, 1]
    assert manifest == json.loads((tmp_path / "data/semantic/manifest.json").read_text())
    body = b"".join((tmp_path / "data/semantic" / s["file"]).read_bytes()
                    for s in manifest["shards"])
    quantised = np.frombuffer(body, dtype=np.int8).reshape(3, 3)
    reconstructed = quantised / np.linalg.norm(quantised.astype(float), axis=1, keepdims=True)
    expected = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert np.all(np.sum(reconstructed * expected, axis=1) > 0.9999)
    with pytest.raises(ValueError, match="unique"):
        export.write_index(tmp_path, [reports[0]] * 3, vectors, "pinned")


def test_static_index_rejects_invalid_embeddings():
    for vectors in [np.zeros((1, 3)), np.array([[np.nan]]), np.array([[np.inf]]), np.ones(3)]:
        with pytest.raises(ValueError):
            export.quantise(vectors)

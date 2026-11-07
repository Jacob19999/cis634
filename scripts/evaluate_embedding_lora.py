"""Evaluate frozen 8B or one LoRA checkpoint on fixed, held-out development queries.

Every incident query searches all 91,827 reports using the existing chunk MaxP rule.
Topical queries are dev-only and search v1 ASRS, where their silver labels are defined.
New checkpoint caches are content-addressed and never replace frozen embeddings.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from asrsbench.data import ChunkConfig, load_chunks, load_reports
from asrsbench.finetuning import QUERY_PREFIX, digest, read_jsonl
from asrsbench.metrics import evaluate_run
from asrsbench.retrievers.base import Units
from asrsbench.retrievers.dense import DenseRetriever
from asrsbench.tracks import topical_track


def queries_and_labels(cfg: dict, reports: pd.DataFrame) -> dict:
    """Define stable queries from held-out incident groups; never read test query text."""
    partitions = pd.DataFrame(read_jsonl(Path(cfg["data"]) / "partitions.jsonl"))
    r = reports.merge(partitions[["acn", "ft_split", "ft_group"]], on="acn", validate="one_to_one")
    group_docs = r.groupby("ft_group").acn.apply(list).to_dict()
    tracks = {}
    for name, prefix, split, limit in (
            ("synopsis_dev", "ASRS:", "dev", cfg["evaluation"]["synopsis_queries"]),
            ("ntsb_validation", "NTSB:", "protected", cfg["evaluation"]["ntsb_queries"])):
        pool = r[r.acn.str.startswith(prefix) & (r.ft_split == split) &
                 (r.synopsis.str.split().str.len() >= 8)]
        if prefix == "NTSB:":
            pool = pool[pool.split == "validation"]
        pool = pool.sort_values("acn").drop_duplicates("ft_group")
        order = sorted(pool.index, key=lambda i: digest(f"634:{pool.at[i, 'acn']}"))[:limit]
        sample = pool.loc[order]
        queries = [{"qid": row.acn, "text": row.synopsis, "target": row.acn}
                   for row in sample.itertuples()]
        labels = {row.acn: {doc: 2 for doc in group_docs[row.ft_group]}
                  for row in sample.itertuples()}
        tracks[name] = (queries, labels, None)
    topical = topical_track(load_reports(), "data/queries/topical_v1.yaml", splits=["dev"])
    queries = [{"qid": row.qid, "text": row.text} for row in topical.queries.itertuples()]
    labels = {q: {"ASRS:" + str(doc): grade for doc, grade in rel.items()}
              for q, rel in topical.qrels["silver"].items()}
    tracks["topical_dev_silver"] = (queries, labels, "ASRS:")
    return tracks


def main() -> None:
    """Encode one content-addressed checkpoint on three GPUs and score fixed dev tracks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_8b.yaml")
    parser.add_argument("--checkpoint", default=None, help="omit to use frozen 8B and existing caches")
    parser.add_argument("--label", default="frozen")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    import torch
    import torch.distributed as dist
    from sentence_transformers import SentenceTransformer

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=4))
    torch.set_num_threads(6)
    units = Units.from_chunks(load_chunks(ChunkConfig(), "data/processed_v2"), ChunkConfig().tag)
    reports = load_reports("data/processed_v2")
    model_path = args.checkpoint or cfg["model"]
    content = [model_path, cfg.get("model_revision"), cfg["max_seq_length"],
               len(units), units.texts[0], units.texts[-1]]
    if args.checkpoint:
        files = sorted(Path(args.checkpoint).rglob("*.safetensors"))
        if not files:
            raise ValueError("checkpoint has no model/adapter weights")
        content.extend(digest(path.read_bytes()) for path in files)
    fingerprint = digest(json.dumps(content))[:20]
    cache = Path("cache/finetune_v1") / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    bounds = np.linspace(0, len(units), world + 1).astype(int)
    model = None
    blocks = []
    if args.checkpoint:
        a, b = map(int, bounds[rank:rank + 2])
        path = cache / f"part{rank}.npy"
        if not path.exists():
            model = SentenceTransformer(model_path, device=f"cuda:{rank}",
                                        local_files_only=True, model_kwargs={"dtype": torch.float16})
            model.max_seq_length = cfg["max_seq_length"]
            tmp = path.with_suffix(".tmp.npy")
            embeddings = model.encode(units.texts[a:b], batch_size=8, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=True)
            if not np.isfinite(embeddings).all():
                raise ValueError("non-finite checkpoint embeddings")
            np.save(tmp, embeddings.astype(np.float32))
            tmp.replace(path)
            del embeddings
        blocks = [(int(bounds[i]), int(bounds[i + 1]), cache / f"part{i}.npy") for i in range(world)]
    else:
        retriever = DenseRetriever("frozen", model=cfg["model"], dtype="float16",
                                  max_seq_length=cfg["max_seq_length"], device="cpu")
        for a, b in DenseRetriever.segments(units):
            sub = Units(units.texts[a:b], units.unit_ids[a:b], units.unit_ids[a:a + 1],
                        np.zeros(1, dtype=int), units.tag)
            path = retriever._cache_path(sub)
            if not path.exists():
                raise FileNotFoundError(f"required frozen cache: {path}")
            blocks.append((a, b, path))
    if world > 1:
        dist.barrier()
    if rank:
        # Release model copies before the scoring worker allocates GPU scratch tensors.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        dist.destroy_process_group()
        return
    if model is None:
        model = SentenceTransformer(model_path, device="cuda:0", local_files_only=True,
                                     revision=cfg.get("model_revision") if not args.checkpoint else None,
                                     model_kwargs={"dtype": torch.float16})
        model.max_seq_length = cfg["max_seq_length"]
    tracks = queries_and_labels(cfg, reports)
    all_queries = [q for qs, _, _ in tracks.values() for q in qs]
    Q = model.encode([QUERY_PREFIX + q["text"] for q in all_queries], batch_size=8,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Score in blocks with FP32 arithmetic. The full embedding index stays memory-mapped
    # on CPU; the score matrix (about 2 GB) is temporary and does not retain activations.
    score_path = Path(tempfile.gettempdir()) / f"asrsbench-ft-{os.getpid()}-{fingerprint}.npy"
    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype="float32",
                                      shape=(len(Q), len(units)))
    q_gpu = torch.as_tensor(Q, device="cuda:0", dtype=torch.float32)
    for a, b, path in blocks:
        E = np.load(path, mmap_mode="r")
        if len(E) != b - a:
            raise ValueError("embedding block length mismatch")
        for j in range(0, len(E), 8192):
            e_gpu = torch.tensor(np.asarray(E[j:j + 8192]), device="cuda:0", dtype=torch.float32)
            scores[:, a + j:a + j + len(e_gpu)] = (q_gpu @ e_gpu.T).cpu().numpy()
            del e_gpu
    scores.flush()
    out = Path(cfg["results"]) / args.label
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "runs"
    raw.mkdir(parents=True, exist_ok=True)
    summaries, offset = {}, 0
    for track, (queries, qrels, scope) in tracks.items():
        run = {}
        allowed = np.flatnonzero(np.array([scope is None or str(d).startswith(scope) for d in units.doc_ids]))
        for i, query in enumerate(queries):
            report_scores = np.maximum.reduceat(np.asarray(scores[offset + i]), units.doc_starts)
            # Stable tie breaking is shared by frozen and adapted conditions.
            top = allowed[np.argsort(-report_scores[allowed], kind="stable")[:100]]
            run[query["qid"]] = units.doc_ids[top].tolist()
        metrics = ["MRR@10", "R@1", "R@10", "R@100", "nDCG@10"]
        if scope:
            metrics += ["P@10", "P@10_lenient"]
        frame = evaluate_run(run, qrels, metrics)
        frame.index.name = "qid"
        frame.to_csv(out / f"{track}_per_query.csv")
        summaries[track] = {"n_queries": len(frame), **frame.mean().to_dict()}
        (raw / f"{track}_runs.json").write_text(json.dumps(run) + "\n")
        (raw / f"{track}_queries.json").write_text(json.dumps(queries) + "\n")
        offset += len(queries)
    meta = {"label": args.label, "model": model_path, "fingerprint": fingerprint,
            "n_reports": len(units.doc_ids), "n_chunks": len(units),
            "source_sha256": digest(Path(__file__).read_bytes()),
            "selection_uses_test": False, "metrics": summaries}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    # This file belongs solely to this evaluation and can be regenerated from the caches.
    del scores
    score_path.unlink()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

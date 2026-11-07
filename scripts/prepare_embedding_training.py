"""Prepare and seal the approved five-stratum embedding fine-tuning dataset.

Run ``--stage prepare`` before synthetic question generation, then ``--stage seal``.
Outputs are new, regenerable files; corpus v1/v2 and evaluation files are read-only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from asrsbench.data import ChunkConfig, chunk_text
from asrsbench.finetuning import (SEED, WEIGHTS, balanced_epoch, bucket, digest, read_jsonl,
                                  supervised_pairs, training_partitions, write_jsonl)


def prepare(args) -> None:
    """Partition reports before selecting passages and before generating questions."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports_path = Path(args.data_dir) / "reports.parquet"
    r, audit = training_partitions(pd.read_parquet(reports_path))
    write_jsonl(out / "partitions.jsonl", r[["acn", "split", "ft_group", "ft_split"]].to_dict("records"))
    pairs, rejects = supervised_pairs(r)
    write_jsonl(out / "supervised.jsonl", pairs)
    audit.update({"supervised_pairs": dict(Counter((p["kind"] + "/" + p["split"]) for p in pairs)),
                  "supervised_rejected": rejects,
                  "reports_sha256": digest(reports_path.read_bytes())})
    tasks = []
    # One representative per incident component; questions are supported by the selected
    # raw passage, not by a synopsis, cause, generated summary or analyst code.
    for split, total in (("train", args.synthetic_reports), ("dev", 300)):
        for prefix in ("ASRS", "ASRSN", "NTSB"):
            pool = r[(r.ft_split == split) & r.acn.str.startswith(prefix + ":")]
            pool = pool.sort_values("acn").drop_duplicates("ft_group")
            ids = sorted(pool.index, key=lambda i: digest(f"{SEED}:{r.at[i, 'acn']}"))
            for i in ids[:(total + 2) // 3]:
                row = r.loc[i]
                chunks = chunk_text(row.narrative, ChunkConfig())
                candidates = [p for p in chunks if len(p.split()) >= 60]
                if not candidates:
                    continue
                passage = candidates[bucket(row.acn, len(candidates))]
                tasks.append({"task_id": digest(f"synthetic:{row.acn}:{passage}")[:24],
                              "kind": "synthetic", "doc_id": row.acn,
                              "group_id": row.ft_group, "split": split, "positive": passage})
    faa = read_jsonl(Path(args.faa_dir) / "passages.jsonl")
    for split, limit in (("train", args.faa_passages), ("dev", 150)):
        for family, fraction in (("AIM", .45), ("CFR91", .35), ("PCG", .20)):
            pool = [p for p in faa if p["split"] == split and p["family"] == family]
            pool.sort(key=lambda p: digest(f"{SEED}:{p['task_id']}"))
            tasks.extend(pool[:max(1, int(limit * fraction))])
    write_jsonl(out / "generation_tasks.jsonl", tasks)
    audit["generation_tasks"] = dict(Counter(p["kind"] + "/" + p["split"] for p in tasks))
    audit["faa_sources"] = json.loads((Path(args.faa_dir) / "sources.json").read_text())
    (out / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "faa_sources"}, indent=2), flush=True)


def seal(args) -> None:
    """Verify completion, deduplicate questions and write a fixed weighted epoch."""
    out = Path(args.out)
    tasks = {p["task_id"]: p for p in read_jsonl(out / "generation_tasks.jsonl")}
    generated = read_jsonl(out / "generated.jsonl")
    from generate_embedding_queries import PROMPT_VERSION
    by_task = {p["task_id"]: p for p in generated if (p.get("items") or p.get("rejected_task"))
               and p.get("prompt_version") == PROMPT_VERSION}
    missing = set(tasks) - set(by_task)
    if missing:
        raise ValueError(f"{len(missing)} generation tasks unfinished; rerun generation")
    pairs = read_jsonl(out / "supervised.jsonl")
    from asrsbench.finetuning import tokens, validate_generated
    rejects = Counter()
    rejected_tasks = sum(bool(p.get("rejected_task")) for p in by_task.values())
    if rejected_tasks / max(1, len(tasks)) > .10:
        raise ValueError("more than 10% of generation tasks failed quality checks; inspect before training")
    for tid, result in by_task.items():
        if tid not in tasks:
            raise ValueError("unknown generation task")
        task = tasks[tid]
        if result.get("rejected_task"):
            rejects["generation_task_quality"] += 1
        for item in result.get("items", []):
            valid, reason = validate_generated(item, task["positive"])
            if not valid:
                rejects[reason] += 1
                continue
            pairs.append({**task, "query": item["query"].strip(), "evidence": item["evidence"],
                          "generator": result["generator"], "prompt_version": result["prompt_version"]})
    # Exact duplicate query/passage strings cannot straddle splits. Keep development
    # examples and discard colliding training pairs, never moving held-out content.
    dev_keys = {digest(" ".join(tokens(p[f]))) for p in pairs if p["split"] == "dev"
                for f in ("query", "positive")}
    seen, unique = set(), []
    for p in pairs:
        if p["split"] == "train" and any(digest(" ".join(tokens(p[f]))) in dev_keys
                                          for f in ("query", "positive")):
            rejects["dev_text_collision"] += 1
            continue
        key = (p["split"], " ".join(tokens(p["query"])), " ".join(tokens(p["positive"])))
        if key in seen:
            rejects["duplicate_pair"] += 1
            continue
        seen.add(key)
        unique.append(p)
    groups = {s: {p["group_id"] for p in unique if p["split"] == s} for s in ("train", "dev")}
    if groups["train"] & groups["dev"]:
        raise ValueError("incident/section group leakage")
    pools = defaultdict(list)
    for p in unique:
        if p["split"] == "train":
            pools[p["kind"]].append(p)
    # Do not fill a failed synthetic stratum by silently repeating a tiny remainder.
    if len(pools["synthetic"]) < args.synthetic_reports * 2 or len(pools["faa"]) < 1500:
        raise ValueError("too few validated synthetic or FAA pairs; inspect generation quality")
    train = balanced_epoch(pools, args.epoch_size)
    dev = [p for p in unique if p["split"] == "dev"]
    write_jsonl(out / "pairs_unique.jsonl", unique)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    manifest = {"version": "aviation_embedding_v1", "seed": SEED, "weights": WEIGHTS,
                "epoch_size": len(train), "unique_training_pairs": {k: len(v) for k, v in pools.items()},
                "sampled_pairs_per_epoch": dict(Counter(p["kind"] for p in train)),
                "dev_pairs": dict(Counter(p["kind"] for p in dev)), "rejected": dict(rejects),
                "sha256": {f: digest((out / f).read_bytes())
                           for f in ("train.jsonl", "dev.jsonl", "partitions.jsonl",
                                     "generation_tasks.jsonl", "generated.jsonl")},
                "preparation": json.loads((out / "preparation.json").read_text())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "preparation"}, indent=2), flush=True)


def main() -> None:
    """Run either the deterministic preparation or final corpus-sealing step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "seal"], required=True)
    parser.add_argument("--data-dir", default="data/processed_v2")
    parser.add_argument("--faa-dir", default="data/external/finetune_faa_v1")
    parser.add_argument("--out", default="data/finetuning/aviation_embedding_v1")
    parser.add_argument("--synthetic-reports", type=int, default=12000)
    parser.add_argument("--faa-passages", type=int, default=2400)
    parser.add_argument("--epoch-size", type=int, default=100000)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else seal)(args)


if __name__ == "__main__":
    main()

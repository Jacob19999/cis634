"""Command-line entry point: ``asrsbench <command>`` (or ``python -m asrsbench``)."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="asrsbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare", help="download, clean and chunk the ASRS corpus")
    pr.add_argument("--out", default="data/processed")
    pr.add_argument("--max-words", type=int, default=150)
    pr.add_argument("--overlap-sentences", type=int, default=1)

    rn = sub.add_parser("run", help="benchmark the systems in a config on its tracks")
    rn.add_argument("config")
    rn.add_argument("--name", help="results sub-folder (default: config name)")
    rn.add_argument("--tracks", nargs="+", choices=["synopsis", "topical", "contrast"])
    rn.add_argument("--systems", nargs="+", help="only run these system names")

    sh = sub.add_parser("search", help="interactive look at one system's results")
    sh.add_argument("config")
    sh.add_argument("system")
    sh.add_argument("query")
    sh.add_argument("-k", type=int, default=5)

    po = sub.add_parser("pool", help="build blind judging pools from a run's topical results")
    po.add_argument("run_dir")
    po.add_argument("--queries", default="data/queries/topical_v1.yaml")
    po.add_argument("--out", default="data/qrels/pool_v1")
    po.add_argument("--depth", type=int, default=10)
    po.add_argument("--overlap", type=float, default=0.25)

    lb = sub.add_parser("labels", help="import judged pools -> qrels + agreement stats")
    lb.add_argument("pools", nargs="+")
    lb.add_argument("--out", default="data/qrels/human_v1.tsv")

    args = p.parse_args(argv)

    if args.cmd == "prepare":
        from .data import ChunkConfig, prepare

        stats = prepare(args.out, ChunkConfig(args.max_words, args.overlap_sentences))
        print(json.dumps(stats, indent=2))
    elif args.cmd == "run":
        from .benchmark import run_benchmark

        out = run_benchmark(args.config, args.name, args.tracks, args.systems)
        print((out / "summary.md").read_text())
    elif args.cmd == "search":
        _search(args)
    elif args.cmd == "pool":
        from .labeling import build_pool

        for who, path in build_pool(args.run_dir, args.queries, args.out, args.depth, args.overlap).items():
            print(f"annotator {who}: {path}")
    elif args.cmd == "labels":
        from .labeling import import_labels

        print(json.dumps(import_labels(args.pools, args.out), indent=2))


def _search(args) -> None:
    from .benchmark import _unit_sets, load_config
    from .data import load_reports
    from .retrievers import build_systems

    cfg = load_config(args.config)
    specs = {s["name"]: s for s in cfg["systems"]}
    needed = [args.system]
    if specs[args.system]["type"] == "hybrid":
        needed = specs[args.system]["params"]["components"] + needed
    cfg["systems"] = [specs[n] for n in needed]
    reports = load_reports(cfg["data_dir"])
    units = _unit_sets(cfg, reports)
    systems = build_systems(cfg["systems"], units, cfg["device"], cfg["cache_dir"])
    sys_ = systems[args.system]
    [res] = sys_.search([args.query], k=args.k)
    text = dict(zip(sys_.units.unit_ids, sys_.units.texts)) if sys_.units else {}
    syn = reports.set_index("acn")["synopsis"]
    for rank, (d, s, u) in enumerate(zip(res.doc_ids, res.scores, res.best_units), 1):
        print(f"\n#{rank}  ACN {d}  score={s:.4f}\n  synopsis: {syn[d]}\n  passage:  {text.get(u, '')[:400]}")


if __name__ == "__main__":
    main()

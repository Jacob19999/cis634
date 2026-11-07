"""Run the approved resumable generation -> seal -> 3-epoch train -> dev analysis job.

This driver owns only the llama.cpp process ids recorded by this run. It records failures
and stops, rather than training on an incomplete corpus or silently changing hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    """Execute dependent stages in order and keep durable status and separate logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_8b.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    logs = Path("logs/finetune_v1")
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / "status.json"
    python = sys.executable
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               LD_LIBRARY_PATH=str(Path.home() / ".conda/lib") + os.pathsep +
               os.environ.get("LD_LIBRARY_PATH", ""))
    current = "starting"

    def status(stage, **extra):
        status_path.write_text(json.dumps({"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
                                           "driver_pid": os.getpid(), **extra}, indent=2) + "\n")
        print(stage, extra, flush=True)

    def run(stage, command):
        nonlocal current
        current = stage
        status(stage, command=command)
        with (logs / f"{stage}.log").open("a") as out:
            subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)

    def distributed(script, *extra):
        return [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=3", script, *extra]

    try:
        # A successful pilot and smoke run are deliberately required before this driver.
        smoke_root = cfg["output"] + "-" + cfg.get("smoke_tag", "smoke")
        smoke = Path(smoke_root) / "TRAINING_COMPLETE"
        if not smoke.exists() or not (smoke.parent / "SMOKE_RELOAD_OK").exists():
            raise ValueError("actual 8B backward/save/reload smoke must pass before full launch")
        for attempt in range(3):
            try:
                run("generation", [python, "-u", "scripts/generate_embedding_queries.py"])
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(5)
        run("seal", [python, "scripts/prepare_embedding_training.py", "--stage", "seal"])
        current = "release_generator_gpus"
        status(current)
        pids = json.loads((logs / "generator_pids.json").read_text())
        for pid in pids:
            path = Path(f"/proc/{pid}/cmdline")
            if path.exists():
                cmdline = path.read_bytes().decode(errors="replace")
                if "llama-server" not in cmdline or "Qwen3.5-27B-Q6_K.gguf" not in cmdline:
                    raise ValueError(f"recorded generator pid {pid} no longer belongs to this run")
                os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            active = [pid for pid in pids if Path(f"/proc/{pid}/cmdline").exists() and
                      b"llama-server" in Path(f"/proc/{pid}/cmdline").read_bytes()]
            if not active:
                break
            time.sleep(1)
        else:
            raise RuntimeError("owned generators have not released GPUs")
        distributed_check = Path(smoke_root + "-ddp") / "SMOKE_RELOAD_OK"
        if not distributed_check.exists():
            run("distributed_smoke", distributed("scripts/train_embedding_lora.py", args.config, "--smoke"))
        if not distributed_check.exists():
            raise ValueError("distributed backward/save/reload check failed")
        training_root = Path(cfg["output"])
        if (training_root / "run_meta.json").exists():
            previous = json.loads((training_root / "run_meta.json").read_text())
            sealed = json.loads((Path(cfg["data"]) / "manifest.json").read_text())
            if previous["config"] != cfg or previous["corpus_manifest"]["sha256"] != sealed["sha256"]:
                raise ValueError("cannot resume with a changed config or sealed corpus")
        if not (training_root / "TRAINING_COMPLETE").exists():
            command = distributed("scripts/train_embedding_lora.py", args.config)
            existing = sorted(training_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if existing:
                command += ["--resume", str(existing[-1])]
            run("training", command)
        run("evaluate_frozen", distributed("scripts/evaluate_embedding_lora.py", "--config", args.config))
        checkpoints = []
        for path in Path(cfg["output"]).glob("checkpoint-*"):
            state = json.loads((path / "trainer_state.json").read_text())
            checkpoints.append((float(state["epoch"]), path))
        checkpoints.sort()
        if len(checkpoints) != 3 or any(abs(epoch - i) > .01 for i, (epoch, _) in enumerate(checkpoints, 1)):
            raise ValueError(f"expected exactly three epoch checkpoints: {checkpoints}")
        for epoch, checkpoint in checkpoints:
            label = f"epoch{round(epoch)}"
            run("evaluate_" + label, distributed("scripts/evaluate_embedding_lora.py", "--config", args.config,
                                                  "--checkpoint", str(checkpoint), "--label", label))
        run("analysis", [python, "scripts/analyse_embedding_lora.py", "--config", args.config])
        status("complete", summary=str(Path(cfg["results"]) / "summary.md"))
    except Exception as exc:
        status("failed", failed_stage=current, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

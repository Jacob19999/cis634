"""Print a read-only snapshot of the aviation 8B fine-tuning pipeline for a terminal."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs/finetune_v1"
MODEL = ROOT / "models/finetuning/qwen3-8b-aviation-v1"
DATA = ROOT / "data/finetuning/aviation_embedding_v1"
RESULTS = ROOT / "results/finetune_8b_v1"


def read_json(path: Path) -> dict:
    """Tolerate a missing or momentarily incomplete status file."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path: Path, size: int = 16000) -> str:
    """Read only the end of a potentially large log, without scanning training data."""
    try:
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - size))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def main() -> None:
    """Show stage, process health, progress, checkpoints, recent output and GPU use."""
    now = datetime.now(timezone.utc)
    print("8B AVIATION FINE-TUNING  |  " + now.strftime("%Y-%m-%d %H:%M:%S UTC"))
    status = read_json(LOGS / "status.json")
    stage = status.get("stage", "status unavailable; retry shortly")
    print(f"Stage: {stage}")
    if status.get("updated_utc"):
        print("Stage recorded: " + status["updated_utc"])
    pid = status.get("driver_pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            proc = Path(f"/proc/{pid}/cmdline")
            running = not proc.exists() or b"run_embedding_finetune.py" in proc.read_bytes()
        except (OSError, ValueError):
            running = False
        print(f"Driver: {'running' if running else 'stopped'} (PID {pid})")
        if not running and stage not in ("complete", "failed"):
            print("ATTENTION: recorded stage is unfinished, but its driver is not running.")
    if stage == "failed":
        print("FAILED AT: " + status.get("failed_stage", "unknown"))
        print("ERROR: " + status.get("error", "see driver.log"))

    generation = tail(LOGS / "generation.log", size=64000)
    starts = list(re.finditer(r"(\d+) finished; (\d+) pending this invocation", generation))
    if starts:
        start = starts[-1]
        done, pending = map(int, start.groups())
        progress = list(re.finditer(r"(\d+)/(\d+) tasks;[^\n\r]*", generation[start.end():]))
        processed = int(progress[-1][1]) if progress else 0
        total = done + pending
        resolved = min(total, done + processed)
        fraction = resolved / total if total else 1.
        bar = "#" * int(fraction * 30) + "-" * (30 - int(fraction * 30))
        print(f"\nGeneration: [{bar}] {resolved:,}/{total:,} tasks ({fraction:.1%})")
        print("Counts include quality exclusions; each accepted task supplies 1–3 query pairs.")
        if progress:
            print("Last generator report: " + progress[-1][0])

    manifest = read_json(DATA / "manifest.json")
    if manifest:
        print(f"\nCorpus sealed: {manifest.get('epoch_size', 0):,} sampled pairs per epoch")
        for kind, count in manifest.get("unique_training_pairs", {}).items():
            print(f"  {kind}: {count:,} unique training pairs")

    journal = tail(MODEL / "training_log.jsonl")
    entries = []
    for line in journal.splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            pass
    training = [p for p in entries if "loss" in p]
    if training:
        last = training[-1]
        epochs = read_json(MODEL / "run_meta.json").get("config", {}).get("epochs", 3)
        print(f"\nTraining: epoch {float(last.get('epoch', 0)):.3f}/{epochs}, "
              f"step {last.get('step', '?')}, loss {last.get('loss', '?')}")
    checkpoints = sorted(MODEL.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    saved = [(p.name, read_json(p / "trainer_state.json").get("epoch")) for p in checkpoints]
    if saved:
        print("Saved checkpoints: " + ", ".join(f"{name} (epoch {epoch})" for name, epoch in saved))
    if (MODEL / "TRAINING_COMPLETE").exists():
        print("Three-epoch training: complete")
    evaluated = [p.name for p in RESULTS.glob("*") if p.is_dir() and (p / "metrics.json").exists()]
    if evaluated:
        print("Evaluated conditions: " + ", ".join(sorted(evaluated)))
    if stage == "complete":
        print("Analysis: " + str(RESULTS / "summary.md"))

    log_stage = status.get("failed_stage", "driver") if stage == "failed" else stage
    log = LOGS / f"{log_stage}.log"
    if not log.exists():
        log = LOGS / "driver.log"
    if log.exists():
        stamp = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc)
        print(f"\nLatest log: {log}")
        print(f"Last log write: {stamp:%H:%M:%S UTC} ({int((now - stamp).total_seconds())} seconds ago)")
        text = re.sub(r"\x1b\[[0-9;]*m", "", tail(log))
        for line in [s for s in text.splitlines() if s.strip()][-4:]:
            print("  " + line[:240])
    print("\nGPUs (index, utilisation %, used MiB, total MiB):")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        print(result.stdout.strip() if result.returncode == 0 else "GPU status unavailable")
    except (OSError, subprocess.TimeoutExpired):
        print("GPU status unavailable")


if __name__ == "__main__":
    main()

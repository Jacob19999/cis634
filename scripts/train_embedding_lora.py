"""Fine-tune Qwen3-Embedding-8B with LoRA and a memory-cached contrastive loss.

Run with torchrun on the three GV100 GPUs. The no-duplicates sampler sees incident
group ids; the collator removes those ids before tokenisation. Negative gathering across
devices is disabled so sibling groups on other workers never become extra negatives.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import yaml

from asrsbench.finetuning import QUERY_PREFIX, digest, read_jsonl


def main() -> None:
    """Check the sealed corpus, train three epochs and save resumable epoch checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/finetune_8b.yaml")
    parser.add_argument("--smoke", action="store_true", help="two actual updates; separate disposable output")
    parser.add_argument("--resume", default=None, help="explicit Trainer checkpoint directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback, set_seed

    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    data = Path(cfg["data"])
    if args.smoke:
        source = data / "supervised.jsonl"
        manifest = {"smoke_only": True, "supervised_sha256": digest(source.read_bytes())}
        train = sorted(read_jsonl(source), key=lambda p: digest(p["query"]))[:128]
    else:
        manifest = json.loads((data / "manifest.json").read_text())
        for name in ("train.jsonl", "dev.jsonl"):
            if digest((data / name).read_bytes()) != manifest["sha256"][name]:
                raise ValueError(f"sealed training input changed: {name}")
        train = read_jsonl(data / "train.jsonl")
    smoke_suffix = "-" + cfg.get("smoke_tag", "smoke")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        smoke_suffix += "-ddp"
    output = Path(cfg["output"] + (smoke_suffix if args.smoke else ""))
    if output.exists() and list(output.glob("checkpoint-*")) and not args.resume:
        raise ValueError("output already has checkpoints; use --resume or a new run version")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = SentenceTransformer(cfg["model"], device=f"cuda:{local_rank}",
                                local_files_only=True, revision=cfg.get("model_revision"),
                                model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"})
    model.max_seq_length = cfg["max_seq_length"]
    model[0].auto_model.config.use_cache = False
    lora = cfg["lora"]
    model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora["r"],
                                revision=cfg.get("model_revision"),
                                lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                target_modules=lora["target_modules"], bias="none"))
    # FP32 trainable adapters permit GradScaler unscaling while the frozen 8B stays FP16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model[0].auto_model.enable_input_require_grads()
    dataset = Dataset.from_list([{"anchor": QUERY_PREFIX + p["query"],
                                  "positive": p["positive"], "group_id": p["group_id"]}
                                 for p in train])
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_batch_size"],
                                               gather_across_devices=False)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output), num_train_epochs=cfg["epochs"], max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"], fp16=True, bf16=False, tf32=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        batch_sampler=BatchSamplers.NO_DUPLICATES, dataloader_num_workers=0,
        dataloader_drop_last=True, remove_unused_columns=False, ddp_find_unused_parameters=False,
        save_strategy="steps" if args.smoke else "epoch", save_steps=2,
        save_total_limit=None, logging_steps=1 if args.smoke else 10, logging_first_step=True,
        eval_strategy="no", report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
        optim="adamw_torch", disable_tqdm=True, max_grad_norm=1.0)

    class Journal(TrainerCallback):
        """Persist machine-readable logs independently of terminal progress bars."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                with (output / "training_log.jsonl").open("a") as f:
                    f.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                                        **(logs or {})}) + "\n")

    trainer = SentenceTransformerTrainer(model=model, args=training_args,
                                         train_dataset=dataset, loss=loss, callbacks=[Journal()])
    original_collator = trainer.data_collator

    class GroupSafeCollator:
        """Preserve the library collator's sampling attributes while removing group ids."""

        def __call__(self, features):
            return original_collator([{k: v for k, v in row.items() if k != "group_id"}
                                      for row in features])

        def __getattr__(self, name):
            return getattr(original_collator, name)

    trainer.data_collator = GroupSafeCollator()
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        meta = {"config": cfg, "corpus_manifest": manifest,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "base_model_config": model[0].auto_model.config.to_dict(),
                "source_sha256": {str(path): digest(path.read_bytes()) for path in
                    [Path(args.config), Path(__file__), Path("src/asrsbench/finetuning.py")]},
                "versions": {p: importlib.metadata.version(p) for p in
                             ("torch", "transformers", "sentence-transformers", "peft", "accelerate")},
                "smoke": args.smoke, "negative_policy": "within-worker, incident-unique batches"}
        (output / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({k: v for k, v in meta.items() if k != "corpus_manifest"}), flush=True)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / "final"))
    if trainer.is_world_process_zero():
        if args.smoke:
            import numpy as np

            probes = [QUERY_PREFIX + "engine vibration after takeoff", "The crew returned to land."]
            before = model.encode(probes, normalize_embeddings=True, convert_to_numpy=True)
            b_norm = sum(p.detach().float().norm().item() for n, p in model.named_parameters()
                         if "lora_B" in n)
            if not np.isfinite(before).all() or b_norm <= 0:
                raise ValueError("smoke did not produce finite embeddings and updated LoRA weights")
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            reloaded = SentenceTransformer(str(output / "final"), device=f"cuda:{local_rank}",
                                           model_kwargs={"dtype": torch.float16})
            reloaded.max_seq_length = cfg["max_seq_length"]
            after = reloaded.encode(probes, normalize_embeddings=True, convert_to_numpy=True)
            if not np.allclose(before, after, atol=2e-4, rtol=2e-3):
                raise ValueError("saved adapter reload changes embeddings")
            (output / "SMOKE_RELOAD_OK").write_text(json.dumps({"lora_B_norm": b_norm,
                "max_embedding_difference": float(np.max(np.abs(before - after))),
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}) + "\n")
        # Adapter files, optimiser state and exact config remain local, never pushed to HF.
        if not args.smoke:
            shutil.copy2(data / "manifest.json", output / "corpus_manifest.json")
        (output / "TRAINING_COMPLETE").write_text("three epochs completed\n" if not args.smoke
                                                  else "smoke completed\n")


if __name__ == "__main__":
    main()

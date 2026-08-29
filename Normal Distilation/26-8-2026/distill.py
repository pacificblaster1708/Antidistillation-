# -*- coding: utf-8 -*-
"""
Stage: distil the attacker's student from the teacher's traces (SFT, LoRA).

Identical in both modes -- the *only* difference between a NORMAL run and an ADS
run is which traces this reads. That is the point of the experiment: if ADS
works, the same training recipe on ADS traces yields a worse student.

The student has its own tokenizer and may have a different architecture; it only
ever sees the decoded text of the teacher's completion, re-templated into its own
chat format. Loss is taken on the completion only, masked by a recorded prompt
token count rather than by searching for a template marker.

Standalone:
    python distill.py --ads=false --normal=true
"""

from __future__ import annotations

import json
import os

import torch
from accelerate import PartialState
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from ads import (
    CompletionOnlyCollator,
    align_vocab,
    banner,
    build_on_main,
    describe,
    init_runtime,
    load_causal_lm,
    load_tokenizer,
    resolve_attn,
    resolve_dtype,
)
from config import Config
from data import chat_messages

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _training_arguments(**kwargs) -> TrainingArguments:
    """`evaluation_strategy` was renamed `eval_strategy` in transformers 4.46."""
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
            return TrainingArguments(**kwargs)
        raise


def _lora_targets(model) -> object:
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    hit = [t for t in LORA_TARGETS if t in present]
    return hit if hit else "all-linear"


def _build_dataset(cfg: Config, state, tokenizer, path: str, column: str, desc: str):
    traces = load_from_disk(path)
    if column not in traces.column_names:
        column = "completion"

    def tokenize(examples):
        convs = [chat_messages(p) for p in examples["problem"]]
        prompts = tokenizer.apply_chat_template(convs, add_generation_prompt=True)
        if isinstance(prompts[0], int):
            prompts = [prompts]
        input_ids, prompt_len = [], []
        for prompt, completion in zip(prompts, examples[column]):
            tail = tokenizer.encode(completion.strip(), add_special_tokens=False)
            tail = tail + [tokenizer.eos_token_id]
            ids = (list(prompt) + tail)[: cfg.train_max_length]
            input_ids.append(ids)
            prompt_len.append(min(len(prompt), len(ids)))
        return {"input_ids": input_ids, "prompt_len": prompt_len}

    def build():
        workers = (
            cfg.map_workers if len(traces) >= 1024 and cfg.map_workers > 1 else None
        )
        out = traces.map(
            tokenize,
            batched=True,
            batch_size=512,
            num_proc=workers,
            remove_columns=traces.column_names,
            desc=desc,
        )
        return out.filter(
            lambda x: len(x["input_ids"]) > x["prompt_len"],
            desc="Dropping empty completions",
        )

    return build_on_main(
        state,
        os.path.join(cfg.exp_dir, ".cache"),
        f"distill|{path}|{cfg.student_tokenizer}|{column}|{cfg.train_max_length}",
        build,
    )


def distill(cfg: Config) -> dict:
    state = PartialState()
    main = state.is_main_process
    init_runtime(cfg.seed)

    if not os.path.exists(cfg.train_traces):
        raise SystemExit(f"training traces not found at {cfg.train_traces}")

    if main:
        banner(
            f"DISTILLATION [{cfg.mode}]",
            f"student : {cfg.student}\n"
            f"traces  : {cfg.train_traces}\n"
            f"output  : {cfg.student_final}",
        )

    tokenizer = load_tokenizer(
        cfg.student_tokenizer or cfg.student, padding_side="right"
    )
    dtype = resolve_dtype(cfg.dtype)

    model = load_causal_lm(
        cfg.student, dtype, resolve_attn(cfg.attn_impl), use_cache=False
    )
    align_vocab(tokenizer, model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    if cfg.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=_lora_targets(model),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        if main:
            model.print_trainable_parameters()

    column = "completion_af" if cfg.train_on_answer_forced else "completion"
    train_ds = _build_dataset(
        cfg, state, tokenizer, cfg.train_traces, column, "Tokenizing train traces"
    )
    eval_ds = None
    if cfg.do_eval and os.path.exists(cfg.holdout_traces):
        eval_ds = _build_dataset(
            cfg,
            state,
            tokenizer,
            cfg.holdout_traces,
            "completion",
            "Tokenizing holdout traces",
        )

    if main:
        lengths = [len(x) for x in train_ds["input_ids"]]
        print(f"[distill] train sequences: {json.dumps(describe(lengths), indent=2)}")
        print(
            f"[distill] example:\n{tokenizer.decode(train_ds[0]['input_ids'])[:1500]}"
        )

    world = max(1, state.num_processes)
    accum = max(1, cfg.train_batch_size // (cfg.per_device_batch_size * world))
    effective = cfg.per_device_batch_size * world * accum
    if main and effective != cfg.train_batch_size:
        print(
            f"[distill] WARNING: train_batch_size={cfg.train_batch_size} is not reachable with "
            f"{world} process(es) x per_device_batch_size={cfg.per_device_batch_size}; "
            f"the effective global batch size is {effective}."
        )
    use_cuda = torch.cuda.is_available()

    args = _training_arguments(
        output_dir=cfg.model_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=accum,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        bf16=use_cuda and dtype is torch.bfloat16,
        fp16=use_cuda and dtype is torch.float16,
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        disable_tqdm=not main,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id),
    )

    train_out = trainer.train()
    metrics = dict(train_out.metrics)
    if eval_ds is not None:
        metrics.update(trainer.evaluate())

    # ------------------------------------------------------------------ save
    state.wait_for_everyone()
    summary: dict = {}
    if main:
        final = trainer.model
        if cfg.lora:
            final = final.merge_and_unload()
        final.config.use_cache = True
        os.makedirs(cfg.student_final, exist_ok=True)
        final.save_pretrained(cfg.student_final, safe_serialization=True)
        tokenizer.save_pretrained(cfg.student_final)

        summary = {
            "student": cfg.student,
            "train_traces": cfg.train_traces,
            "num_train": len(train_ds),
            "num_eval": len(eval_ds) if eval_ds is not None else 0,
            "final_model": cfg.student_final,
            "metrics": {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            },
        }
        with open(cfg.model_path + ".json", "w") as fh:
            json.dump({"summary": summary, "config": cfg.to_dict()}, fh, indent=2)
        banner("DISTILLATION DONE", json.dumps(summary, indent=2))

    state.wait_for_everyone()
    return summary


if __name__ == "__main__":
    distill(Config.load())

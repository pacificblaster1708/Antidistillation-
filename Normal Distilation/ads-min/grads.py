# -*- coding: utf-8 -*-
"""
Stage: compute the proxy student's gradient on the clean holdout traces.

This is the only place the defender's "guess at the attacker" enters. We ask:
if a student were fine-tuned on the teacher's own traces, which direction in its
parameter space *increases* its loss? That direction g is saved once and reused
at sampling time, where the teacher steers toward tokens that move a student
along +g (see ADSLogitsProcessor).

What is saved is the gradient of the mean per-token cross-entropy over the whole
holdout set:

    g = d/dtheta  [ (1/T) * sum_over_all_completion_tokens  -log p_theta(token) ]

Accumulating token *sums* and dividing by the global token count at the end makes
the result independent of batch size and of how the data happens to shard, which
a per-batch mean is not.

Only ADS runs need this stage. Standalone:
    python grads.py --ads=true --normal=false
"""

from __future__ import annotations

import json
import os
from typing import Dict

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ads import (CompletionOnlyCollator, align_vocab, banner, build_on_main,
                 count_label_tokens, init_runtime, load_causal_lm, load_tokenizer,
                 normalize_param_name, resolve_attn, save_grads)
from config import Config
from data import chat_messages


def compute_grads(cfg: Config) -> dict:
    accelerator = Accelerator()
    main = accelerator.is_main_process
    init_runtime(cfg.seed)

    if not os.path.exists(cfg.holdout_traces):
        raise SystemExit(f"holdout traces not found at {cfg.holdout_traces}; run the holdout "
                         f"generation stage first")

    if main:
        banner("PROXY STUDENT GRADIENTS",
               f"proxy student : {cfg.proxy_student}\n"
               f"holdout traces: {cfg.holdout_traces}\n"
               f"output        : {cfg.grad_path}")

    # The proxy student shares the teacher's tokenizer -- that is what lets its
    # logits be added to the teacher's logits later on.
    tokenizer = load_tokenizer(cfg.teacher, padding_side="right")

    # fp32: eps is small and the perturbation has to survive the round trip.
    model = load_causal_lm(cfg.proxy_student, torch.float32, resolve_attn(cfg.attn_impl),
                           use_cache=False)
    vocab = align_vocab(tokenizer, model)
    if main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[grads] {cfg.proxy_student}: {n_params / 1e6:.1f}M params, vocab {vocab}")

    # ------------------------------------------------------------------ data
    traces = load_from_disk(cfg.holdout_traces)
    column = "completion_af" if (cfg.train_on_answer_forced and "completion_af" in traces.column_names) \
        else "completion"

    def tokenize(examples):
        convs = [chat_messages(p) for p in examples["problem"]]
        prompts = tokenizer.apply_chat_template(convs, add_generation_prompt=True)
        if isinstance(prompts[0], int):
            prompts = [prompts]
        input_ids, prompt_len = [], []
        for prompt, completion in zip(prompts, examples[column]):
            tail = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
            ids = (list(prompt) + tail)[:cfg.train_max_length]
            input_ids.append(ids)
            prompt_len.append(min(len(prompt), len(ids)))
        return {"input_ids": input_ids, "prompt_len": prompt_len}

    def build():
        out = traces.map(tokenize, batched=True, batch_size=512,
                         num_proc=1 if len(traces) < 1024 else cfg.map_workers,
                         remove_columns=traces.column_names, desc="Tokenizing holdout traces")
        return out.filter(lambda x: len(x["input_ids"]) > x["prompt_len"],
                          desc="Dropping empty completions")

    tokenized = build_on_main(accelerator, os.path.join(cfg.exp_dir, ".cache"),
                              f"grads|{cfg.holdout_traces}|{cfg.teacher}|{column}|"
                              f"{cfg.train_max_length}", build)
    if len(tokenized) == 0:
        raise SystemExit("every holdout trace has an empty completion; nothing to differentiate")

    loader = DataLoader(
        tokenized.with_format("python"),
        batch_size=cfg.grad_batch_size, shuffle=False,
        collate_fn=CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id),
    )

    model, loader = accelerator.prepare(model, loader)

    # ----------------------------------------------------------- accumulate
    grads: Dict[str, torch.Tensor] = {
        normalize_param_name(name): torch.zeros_like(param, dtype=torch.float32)
        for name, param in model.named_parameters() if param.requires_grad
    }

    local_tokens = 0
    model.train()
    model.zero_grad(set_to_none=True)
    for batch in tqdm(loader, desc="accumulating grads", disable=not main):
        n_tokens = count_label_tokens(batch["labels"])
        if n_tokens == 0:
            continue
        out = model(**batch)
        # model loss is a mean over the batch's label tokens; rescale to a sum so
        # the running total is a plain sum over the dataset.
        accelerator.backward(out.loss * n_tokens)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[normalize_param_name(name)].add_(param.grad.detach().float())
        model.zero_grad(set_to_none=True)
        local_tokens += n_tokens

    # ------------------------------------------------------------ reduce
    token_tensor = torch.tensor([float(local_tokens)], device=accelerator.device)
    accelerator.wait_for_everyone()
    total_tokens = float(accelerator.reduce(token_tensor, reduction="sum").item())
    if total_tokens == 0:
        raise SystemExit("no supervised tokens found in the holdout traces")

    for name in list(grads):
        if accelerator.num_processes > 1:
            grads[name] = accelerator.reduce(grads[name], reduction="sum")
        grads[name] = grads[name] / total_tokens

    summary: dict = {}
    if main:
        norm = sum(float(torch.sum(g.double() ** 2)) for g in grads.values()) ** 0.5
        save_grads(cfg.grad_path, grads, meta={
            "proxy_student": cfg.proxy_student,
            "tokenizer": cfg.teacher,
            "vocab_size": vocab,
            "holdout_traces": cfg.holdout_traces,
            "num_examples": len(tokenized),
            "num_tokens": total_tokens,
            "grad_norm": norm,
        })
        summary = {"grad_path": cfg.grad_path, "grad_norm": norm,
                   "num_tensors": len(grads), "num_tokens": total_tokens,
                   "num_examples": len(tokenized)}
        banner("GRADIENTS SAVED", json.dumps(summary, indent=2))

    accelerator.wait_for_everyone()
    return summary


if __name__ == "__main__":
    compute_grads(Config.load())

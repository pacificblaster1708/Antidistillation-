# -*- coding: utf-8 -*-
"""Build the optional finite-horizon EMARC influence direction.

For holdout loss gradient ``g`` and Hessian ``H`` the recurrence

    v_0 = 0
    v_{j+1} = v_j + eta * (g - (H + damping I) v_j)

produces ``eta * sum_{k=0}^{m-1}(I-eta(H+damping I))^k g``.  The final
direction is norm-matched to ``g`` so ``m=1`` is exactly the original ADS
direction and ``eps`` keeps the same interpretation across horizons.

The ``m>1`` stage intentionally runs in one process with eager attention:
PyTorch DDP does not support ``autograd.grad`` Hessian-vector products, and
some fused attention kernels do not implement second derivatives.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ads import (
    CompletionOnlyCollator,
    align_vocab,
    banner,
    count_label_tokens,
    init_runtime,
    load_causal_lm,
    load_grads,
    load_tokenizer,
    normalize_param_name,
    save_grads,
)
from config import Config
from data import chat_messages

TensorMap = Dict[str, torch.Tensor]


def finite_horizon_recurrence(
    gradient: TensorMap,
    hvp: Callable[[TensorMap], TensorMap],
    *,
    steps: int,
    lr: float,
    damping: float,
) -> TensorMap:
    """Pure recurrence used by the stage and unit tests."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if lr <= 0 or damping < 0:
        raise ValueError("lr must be > 0 and damping must be >= 0")
    direction = {name: torch.zeros_like(value) for name, value in gradient.items()}
    for step in range(steps):
        curvature = (
            {name: torch.zeros_like(value) for name, value in direction.items()}
            if step == 0
            else hvp(direction)
        )
        if set(curvature) != set(direction):
            missing = sorted(set(direction) - set(curvature))
            extra = sorted(set(curvature) - set(direction))
            raise ValueError(
                f"HVP keys differ from gradient (missing={missing[:3]}, "
                f"extra={extra[:3]})"
            )
        direction = {
            name: value + lr * (gradient[name] - curvature[name] - damping * value)
            for name, value in direction.items()
        }
    return direction


def _norm(values: TensorMap) -> float:
    return (
        sum(
            float(torch.sum(value.detach().double().square()))
            for value in values.values()
        )
        ** 0.5
    )


def _tokenize_holdout(cfg: Config, tokenizer):
    if not os.path.exists(cfg.holdout_traces):
        raise SystemExit(f"holdout traces not found at {cfg.holdout_traces}")
    traces = load_from_disk(cfg.holdout_traces)
    column = (
        "completion_af"
        if cfg.train_on_answer_forced and "completion_af" in traces.column_names
        else "completion"
    )

    def tokenize(examples):
        conversations = [chat_messages(problem) for problem in examples["problem"]]
        prompts = tokenizer.apply_chat_template(
            conversations, add_generation_prompt=True
        )
        if isinstance(prompts[0], int):
            prompts = [prompts]
        input_ids, prompt_lengths = [], []
        for prompt, completion in zip(prompts, examples[column]):
            tail = tokenizer.encode(completion, add_special_tokens=False)
            tail.append(tokenizer.eos_token_id)
            ids = (list(prompt) + tail)[: cfg.train_max_length]
            input_ids.append(ids)
            prompt_lengths.append(min(len(prompt), len(ids)))
        return {"input_ids": input_ids, "prompt_len": prompt_lengths}

    workers = cfg.map_workers if len(traces) >= 1024 and cfg.map_workers > 1 else None
    tokenized = traces.map(
        tokenize,
        batched=True,
        batch_size=512,
        num_proc=workers,
        remove_columns=traces.column_names,
        desc="Tokenizing horizon holdout traces",
    )
    tokenized = tokenized.filter(
        lambda row: len(row["input_ids"]) > row["prompt_len"],
        desc="Dropping empty completions",
    )
    if len(tokenized) == 0:
        raise SystemExit("every holdout trace has an empty completion")
    return tokenized


def _model_hvp(
    model,
    loader,
    parameter_by_name: Dict[str, torch.nn.Parameter],
    vector: TensorMap,
    device: torch.device,
    max_batches: int,
) -> TensorMap:
    """Exact token-mean Hessian-vector product on the selected holdout batches."""
    names = list(parameter_by_name)
    parameters = [parameter_by_name[name] for name in names]
    vector_on_device = [
        vector[name].to(device=device, dtype=torch.float32) for name in names
    ]
    accum = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in parameter_by_name.items()
    }
    total_tokens = 0

    model.eval()
    for batch_index, batch in enumerate(tqdm(loader, desc="Hessian-vector batches")):
        if max_batches and batch_index >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        n_tokens = count_label_tokens(batch["labels"])
        if n_tokens == 0:
            continue
        output = model(**batch)
        first = torch.autograd.grad(
            output.loss * n_tokens, parameters, create_graph=True, allow_unused=True
        )
        products = [
            (grad * vec).sum()
            for grad, vec in zip(first, vector_on_device)
            if grad is not None
        ]
        if not products:
            raise RuntimeError(
                "the holdout loss is disconnected from every proxy parameter"
            )
        directional = torch.stack(products).sum()
        second = torch.autograd.grad(directional, parameters, allow_unused=True)
        for name, value in zip(names, second):
            if value is not None:
                accum[name].add_(value.detach().float())
        total_tokens += n_tokens
        del output, first, second, directional, products

    if total_tokens == 0:
        raise RuntimeError(
            "no supervised tokens were available for the Hessian-vector product"
        )
    return {name: value / float(total_tokens) for name, value in accum.items()}


def compute_horizon_direction(cfg: Config) -> dict:
    if not cfg.emarc:
        raise SystemExit("horizon.py is an EMARC-only stage (set --emarc=true)")
    init_runtime(cfg.seed)
    if not os.path.exists(cfg.grad_path):
        raise SystemExit(
            f"base gradient not found at {cfg.grad_path}; run grads.py first"
        )

    gradient_cpu, gradient_meta = load_grads(cfg.grad_path)
    saved_proxy = gradient_meta.get("proxy_student")
    if saved_proxy not in (None, cfg.proxy_student):
        raise RuntimeError(
            f"gradient was built for proxy {saved_proxy!r}, not {cfg.proxy_student!r}"
        )
    gradient_cpu = {
        normalize_param_name(name): value.float()
        for name, value in gradient_cpu.items()
    }
    gradient_norm = _norm(gradient_cpu)
    if not math_is_positive_finite(gradient_norm):
        raise RuntimeError(
            f"base gradient norm must be positive and finite, got {gradient_norm}"
        )

    banner(
        "EMARC FINITE-HORIZON DIRECTION",
        f"steps       : {cfg.emarc_horizon_steps}\n"
        f"horizon lr  : {cfg.emarc_horizon_lr}\n"
        f"damping     : {cfg.emarc_horizon_damping}\n"
        f"output      : {cfg.direction_path}",
    )

    if cfg.emarc_horizon_steps == 1:
        # Recurrence gives lr*g; norm matching makes the saved tensor exactly g.
        direction_cpu = {name: value.clone() for name, value in gradient_cpu.items()}
        raw_norm = cfg.emarc_horizon_lr * gradient_norm
        batches_used = 0
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = load_tokenizer(cfg.teacher, padding_side="right")
        model = load_causal_lm(
            cfg.proxy_student, torch.float32, "eager", device=device, use_cache=False
        )
        vocab = align_vocab(tokenizer, model)
        expected_vocab = gradient_meta.get("vocab_size")
        if expected_vocab not in (None, vocab):
            raise RuntimeError(
                f"gradient vocab_size={expected_vocab}, current vocab_size={vocab}"
            )

        parameter_by_name = {
            normalize_param_name(name): parameter
            for name, parameter in model.named_parameters()
            if normalize_param_name(name) in gradient_cpu
        }
        missing = sorted(set(gradient_cpu) - set(parameter_by_name))
        if missing:
            raise RuntimeError(
                f"{len(missing)} gradient tensors do not match the proxy model; "
                f"examples: {missing[:3]}"
            )
        gradient = {name: gradient_cpu[name].to(device) for name in parameter_by_name}
        tokenized = _tokenize_holdout(cfg, tokenizer)
        loader = DataLoader(
            tokenized.with_format("python"),
            batch_size=cfg.grad_batch_size,
            shuffle=False,
            collate_fn=CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id),
        )

        def hvp(vector: TensorMap) -> TensorMap:
            return _model_hvp(
                model,
                loader,
                parameter_by_name,
                vector,
                device,
                cfg.emarc_horizon_max_batches,
            )

        direction = finite_horizon_recurrence(
            gradient,
            hvp,
            steps=cfg.emarc_horizon_steps,
            lr=cfg.emarc_horizon_lr,
            damping=cfg.emarc_horizon_damping,
        )
        raw_norm = _norm(direction)
        if not math_is_positive_finite(raw_norm):
            raise RuntimeError(f"finite-horizon direction norm is invalid: {raw_norm}")
        scale = gradient_norm / raw_norm
        direction_cpu = {
            name: value.detach().cpu().float() * scale
            for name, value in direction.items()
        }
        batches_used = min(len(loader), cfg.emarc_horizon_max_batches or len(loader))

    final_norm = _norm(direction_cpu)
    cosine_numerator = sum(
        float((direction_cpu[name].double() * gradient_cpu[name].double()).sum())
        for name in gradient_cpu
    )
    cosine = cosine_numerator / max(1e-30, final_norm * gradient_norm)
    meta = dict(gradient_meta)
    meta.update(
        {
            "direction": "finite_horizon",
            "horizon_steps": cfg.emarc_horizon_steps,
            "horizon_lr": cfg.emarc_horizon_lr,
            "horizon_damping": cfg.emarc_horizon_damping,
            "horizon_max_batches": cfg.emarc_horizon_max_batches,
            "horizon_batches_used": batches_used,
            "raw_direction_norm": raw_norm,
            "direction_norm": final_norm,
            "base_grad_norm": gradient_norm,
            "cosine_with_base_gradient": cosine,
        }
    )
    save_grads(cfg.direction_path, direction_cpu, meta)
    summary = {
        "direction_path": cfg.direction_path,
        "direction_norm": final_norm,
        "cosine_with_base_gradient": cosine,
        "horizon_steps": cfg.emarc_horizon_steps,
        "horizon_batches_used": batches_used,
    }
    banner("EMARC DIRECTION SAVED", json.dumps(summary, indent=2))
    return summary


def math_is_positive_finite(value: float) -> bool:
    """Tiny helper kept separate so failures report before allocating a model."""
    return value > 0 and math.isfinite(value)


if __name__ == "__main__":
    compute_horizon_direction(Config.load())

# -*- coding: utf-8 -*-
"""
Stage: generate reasoning traces with a model.

Used four times by run.py, with different flags each time:

  holdout traces   teacher, ADS off   -> input to the gradient stage + SFT eval set
  train traces     teacher, ADS on/off-> what the student is distilled from
  student eval     student, ADS off   -> did distillation work?
  teacher eval     teacher, ADS on/off-> did the defence cost the teacher anything?

Standalone:
    python generate.py --ads=true --normal=false --gen_split=train --gen_out=... [...]
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LogitsProcessorList

from ads import (ADSLogitsProcessor, IncrementalLM, align_vocab, apply_perturbation,
                 banner, build_on_main, describe, init_runtime, load_causal_lm,
                 load_grads, load_tokenizer, resolve_attn, resolve_dtype, strip_terminals)
from config import Config, to_bool
from data import ANSWER_FORCE_STRING, chat_messages, is_correct, load_split


# --------------------------------------------------------------------------- #
def _resolve_stage(cfg: Config) -> dict:
    model = cfg.gen_model or cfg.teacher
    tokenizer = cfg.gen_tokenizer or model
    split = cfg.gen_split
    if cfg.gen_use_ads == "auto":
        use_ads = bool(cfg.ads) and split == "train"
    else:
        use_ads = to_bool(cfg.gen_use_ads)
    if cfg.gen_max_samples >= 0:
        max_samples: Optional[int] = cfg.gen_max_samples
    else:
        max_samples = {"train": cfg.max_train_samples,
                       "holdout": cfg.max_holdout_samples,
                       "test": cfg.max_test_samples}[split]
    out = cfg.gen_out or {"train": cfg.train_traces,
                          "holdout": cfg.holdout_traces,
                          "test": os.path.join(cfg.traces_dir, "test")}[split]
    return {"model": model, "tokenizer": tokenizer, "split": split, "use_ads": use_ads,
            "max_samples": max_samples, "out": out,
            "label": cfg.gen_label or f"{split}/{'ads' if use_ads else 'plain'}"}


def _left_pad(batch: List[dict], pad_id: int) -> dict:
    width = max(len(f["input_ids"]) for f in batch)
    ids, mask = [], []
    for f in batch:
        seq = list(f["input_ids"])
        pad = width - len(seq)
        ids.append([pad_id] * pad + seq)
        mask.append([0] * pad + [1] * len(seq))
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long)}


# --------------------------------------------------------------------------- #
def generate_traces(cfg: Config) -> dict:
    stage = _resolve_stage(cfg)
    accelerator = Accelerator()
    main = accelerator.is_main_process
    init_runtime(cfg.seed)

    if main:
        banner(f"GENERATE [{stage['label']}]",
               json.dumps({**{k: v for k, v in stage.items()},
                           "tau": cfg.tau, "lam_min": cfg.lam_min, "lam_max": cfg.lam_max,
                           "eps": cfg.eps}, indent=2))

    dtype = resolve_dtype(cfg.dtype)
    attn = resolve_attn(cfg.attn_impl)
    tokenizer = load_tokenizer(stage["tokenizer"], padding_side="left")

    # ---------------------------------------------------------------- models
    model = load_causal_lm(stage["model"], dtype, attn, device=accelerator.device)
    model.eval()

    plus = minus = None
    if stage["use_ads"]:
        if not os.path.exists(cfg.grad_path):
            raise SystemExit(f"ADS needs proxy-student gradients; {cfg.grad_path} does not exist. "
                             f"Run the grads stage first.")
        grads, meta = load_grads(cfg.grad_path)
        plus_model = load_causal_lm(cfg.proxy_student, dtype, attn, device=accelerator.device)
        minus_model = load_causal_lm(cfg.proxy_student, dtype, attn, device=accelerator.device)
        vocab = align_vocab(tokenizer, model, plus_model, minus_model)
        if meta.get("vocab_size") not in (None, vocab):
            raise SystemExit(f"gradients were computed with vocab_size={meta['vocab_size']} but "
                             f"this tokenizer has {vocab} tokens")
        stats_p = apply_perturbation(plus_model, grads, +cfg.eps)
        apply_perturbation(minus_model, grads, -cfg.eps)
        del grads
        plus_model.eval(); minus_model.eval()
        plus, minus = IncrementalLM(plus_model), IncrementalLM(minus_model)
        if main:
            step = cfg.eps * stats_p["grad_rms"]
            print(f"[ads] param RMS {stats_p['param_rms']:.3e} | grad RMS {stats_p['grad_rms']:.3e} "
                  f"| eps*grad RMS {step:.3e} over {stats_p['num_params']:,} params")
            if dtype in (torch.float16, torch.bfloat16):
                resolution = 1e-3 if dtype is torch.float16 else 8e-3
                if step < stats_p["param_rms"] * resolution:
                    print(f"[ads] WARNING: eps*grad is small next to {dtype} resolution; the two "
                          f"perturbed students may be numerically identical. Raise --eps or set "
                          f"--dtype=float32 for the proxy students.")
    else:
        align_vocab(tokenizer, model)

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------ data
    def build():
        raw = load_split(cfg.dataset, stage["split"], stage["max_samples"])
        raw = raw.add_column("idx", list(range(len(raw))))

        def tokenize(examples):
            convs = [chat_messages(p) for p in examples["problem"]]
            ids = tokenizer.apply_chat_template(convs, add_generation_prompt=True)
            if isinstance(ids[0], int):                 # single-conversation edge case
                ids = [ids]
            return {"input_ids": ids, "prompt_len": [len(x) for x in ids]}

        out = raw.map(tokenize, batched=True, batch_size=512,
                      num_proc=1 if len(raw) < 1024 else cfg.map_workers,
                      desc="Tokenizing prompts")
        before = len(out)
        out = out.filter(lambda x: x["prompt_len"] <= cfg.max_prompt_length,
                         desc="Dropping over-long prompts")
        if len(out) < before:
            print(f"[gen] dropped {before - len(out)} prompt(s) longer than "
                  f"{cfg.max_prompt_length} tokens")
        return out

    key = f"gen|{cfg.dataset}|{stage['split']}|{stage['tokenizer']}|" \
          f"{cfg.max_prompt_length}|{stage['max_samples']}"
    proc = build_on_main(accelerator, os.path.join(cfg.exp_dir, ".cache"), key, build)
    if len(proc) == 0:
        raise SystemExit("every prompt was filtered out; raise --max_prompt_length")
    if main:
        print(f"[gen] {len(proc)} prompt(s); example:\n{tokenizer.decode(proc[0]['input_ids'])}")

    shard = proc.shard(num_shards=accelerator.num_processes, index=accelerator.process_index) \
        if accelerator.num_processes > 1 else proc

    loader = DataLoader(
        shard.select_columns(["input_ids"]).with_format("python"),
        batch_size=cfg.gen_batch_size, shuffle=False,
        collate_fn=lambda b: _left_pad(b, tokenizer.pad_token_id),
    )

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.tau > 0,
        temperature=cfg.tau if cfg.tau > 0 else None,
        top_p=cfg.top_p if cfg.tau > 0 else None,
        top_k=None,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ------------------------------------------------------------ generation
    chk_path = stage["out"] + "_checkpoint.jsonl"
    saved_completions: Dict[int, str] = {}
    if os.path.exists(chk_path):
        try:
            with open(chk_path, "r", encoding="utf-8") as fh:
                for l in fh:
                    if l.strip():
                        data = json.loads(l)
                        saved_completions[int(data["idx"])] = data["completion"]
            if main and len(saved_completions) > 0:
                print(f"[gen] Resuming from partial checkpoint: {len(saved_completions)} completion(s) already saved.")
        except Exception as e:
            if main:
                print(f"[gen] Warning: could not load partial checkpoint ({e}). Starting fresh.")
            saved_completions = {}

    completions: List[str] = []
    shard_indices = list(shard["idx"])
    current_offset = 0
    chk_file = open(chk_path, "a", encoding="utf-8") if main else None

    for batch in tqdm(loader, total=len(loader), desc=f"generate[{stage['label']}]",
                      disable=not main):
        bsz = batch["input_ids"].shape[0]
        batch_indices = shard_indices[current_offset : current_offset + bsz]
        current_offset += bsz

        if all(idx in saved_completions for idx in batch_indices):
            for idx in batch_indices:
                completions.append(saved_completions[idx])
            continue

        batch_device = {k: v.to(accelerator.device) for k, v in batch.items()}
        prompt_width = batch_device["input_ids"].shape[1]

        processors = None
        if stage["use_ads"]:
            # A fresh processor and a fresh KV cache per batch: the cache is keyed
            # to this batch's prompts and its batch size.
            plus.reset(); minus.reset()
            processors = LogitsProcessorList(
                [ADSLogitsProcessor(plus, minus, cfg.lam_min, cfg.eps,
                                    batch_device["attention_mask"],
                                    lam_max=cfg.lam_max, beta=cfg.beta,
                                    gamma=cfg.gamma, sigma2_prior=cfg.sigma2_prior,
                                    warmup_steps=cfg.warmup_steps,
                                    warmup_val=cfg.warmup_val)]
            )

        with torch.inference_mode():
            out = model.generate(
                **batch_device, **gen_kwargs,
                logits_processor=processors,
                renormalize_logits=bool(stage["use_ads"]),
            )
        batch_comps = []
        for row in out[:, prompt_width:]:
            comp = strip_terminals(tokenizer.decode(row, skip_special_tokens=False),
                                   tokenizer).rstrip()
            batch_comps.append(comp)
            completions.append(comp)

        if main and chk_file is not None:
            for idx, comp in zip(batch_indices, batch_comps):
                saved_completions[idx] = comp
                chk_file.write(json.dumps({"idx": idx, "completion": comp}) + "\n")
            chk_file.flush()

    if chk_file is not None:
        chk_file.close()

    shard = shard.add_column("completion", completions)

    # The two proxy students are only needed while sampling; free them before
    # the answer-forcing pass so it gets the whole device to itself.
    if stage["use_ads"]:
        plus.reset(); minus.reset()
        del plus, minus, plus_model, minus_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------- answer forcing
    if cfg.answer_force:
        forced = _answer_force(cfg, accelerator, tokenizer, model, shard, main)
        shard = shard.add_column("completion_af", forced)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------- scoring
    def score(example):
        out = {"is_correct": is_correct(example["completion"], example["solution"])}
        if cfg.answer_force:
            out["is_correct_af"] = is_correct(example["completion_af"], example["solution"])
        return out

    shard = shard.map(score, desc="Scoring")

    # ------------------------------------------------------- gather and save
    tmp = tempfile.mkdtemp(prefix="ads_shard_")
    shard_path = os.path.join(tmp, f"rank{accelerator.process_index:05d}")
    shard.save_to_disk(shard_path)
    accelerator.wait_for_everyone()
    all_paths = sorted(gather_object([shard_path]))

    summary: dict = {}
    if main:
        merged = concatenate_datasets([load_from_disk(p) for p in all_paths]).sort("idx")
        os.makedirs(os.path.dirname(os.path.abspath(stage["out"])), exist_ok=True)
        if os.path.exists(stage["out"]):
            shutil.rmtree(stage["out"])
        merged.save_to_disk(stage["out"])
        merged.to_parquet(stage["out"] + ".parquet")
        if os.path.exists(chk_path):
            os.remove(chk_path)

        lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in merged["completion"]]
        summary = {
            "stage": stage["label"],
            "model": stage["model"],
            "split": stage["split"],
            "use_ads": stage["use_ads"],
            "n": len(merged),
            "tau": cfg.tau, "lam_min": cfg.lam_min, "lam_max": cfg.lam_max, "eps": cfg.eps,
            "accuracy": float(sum(merged["is_correct"])) / len(merged),
            "completion_tokens": describe(lengths),
            "path": stage["out"],
        }
        if cfg.answer_force:
            summary["accuracy_af"] = float(sum(merged["is_correct_af"])) / len(merged)
        with open(stage["out"] + ".json", "w") as fh:
            json.dump({"summary": summary, "config": cfg.to_dict()}, fh, indent=2)

        example = merged[0]
        banner("EXAMPLE",
               f"PROBLEM\n{example['problem'][:600]}\n\n"
               f"COMPLETION\n{example['completion'][:1200]}\n\n"
               f"GOLD\n{example['solution'][:300]}")
        banner("SUMMARY", json.dumps(summary, indent=2))

    accelerator.wait_for_everyone()          # nobody deletes a shard main is still reading
    shutil.rmtree(tmp, ignore_errors=True)
    return summary


# --------------------------------------------------------------------------- #
def _answer_force(cfg: Config, accelerator, tokenizer, model, shard, main) -> List[str]:
    """
    Re-feed <prompt + reasoning + "**Final Answer** \\boxed{"> and greedily finish it.

    Greedy and ADS-free on purpose: this is a read-out of the answer the trace
    already implies, not part of the sampled trace.
    """
    items = []
    for row in shard:
        completion = row["completion"]
        suffix = ""
        if "<think>" in completion and "</think>" not in completion:
            suffix = "\n</think>"
        suffix += ANSWER_FORCE_STRING
        tail = tokenizer.encode(completion + suffix, add_special_tokens=False)
        ids = list(row["input_ids"]) + tail
        budget = cfg.max_prompt_length + cfg.max_new_tokens + 64
        if len(ids) > budget:                       # keep the prompt, trim the middle
            keep = row["prompt_len"]
            ids = ids[:keep] + ids[len(ids) - (budget - keep):]
        items.append({"input_ids": ids, "suffix": suffix, "completion": completion})

    batch_size = max(1, cfg.gen_batch_size // 2)
    results: List[str] = []
    for start in tqdm(range(0, len(items), batch_size), desc="answer-force", disable=not main):
        chunk = items[start:start + batch_size]
        batch = _left_pad(chunk, tokenizer.pad_token_id)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        width = batch["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **batch, do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=cfg.answer_force_tokens, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        for item, row in zip(chunk, out[:, width:]):
            finish = strip_terminals(tokenizer.decode(row, skip_special_tokens=False), tokenizer)
            results.append(item["completion"] + item["suffix"] + finish.rstrip())
    return results


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    generate_traces(Config.load())

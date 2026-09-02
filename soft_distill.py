#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
# soft_distill.py -- Soft (top-K logit) knowledge distillation, 7B teacher -> 3B student
# ==============================================================================
#
# WHAT THIS DOES
# --------------
# The repo's `distill.py` does *hard* distillation: the student is trained with
# plain cross-entropy on the teacher's sampled text. This script does *soft*
# distillation: at every completion token the teacher's next-token distribution
# is truncated to its top-K entries and the student is trained to match it.
#
#     loss = alpha * T^2 * KL_topK(teacher || student)  +  (1 - alpha) * CE(hard labels)
#
# T^2 is the usual Hinton correction so the soft-gradient magnitude does not
# shrink as the temperature grows.
#
# TWO AXES OF CONFIGURATION (both requested)
# ------------------------------------------
# 1. --vocab_mode
#      shared : teacher and student share a tokenizer (e.g. teacher
#               deepseek-ai/DeepSeek-R1-Distill-Qwen-7B + student Qwen/Qwen2.5-3B).
#               Top-K KL is exact. This is the recommended setting.
#               Both models are then driven by the *teacher's* tokenizer, which
#               is what the traces were generated with. The two embedding
#               matrices are different sizes (152064 vs 151936), so the KL is
#               taken over their common prefix -- the `topK cov` number in the
#               log tells you how much teacher probability mass survives that.
#      cross  : teacher and student use different tokenizers (e.g. teacher Qwen +
#               student meta-llama/Llama-3.2-3B). The script aligns the two
#               tokenizations of the *same completion string* by character
#               offsets, and maps teacher token ids to student token ids by
#               surface form. Positions/ids that cannot be aligned are dropped
#               from the KL term (they still get the CE term). This is an
#               approximation -- see the coverage numbers the script prints.
#
# 2. --teacher_logits
#      online : the frozen teacher runs one no-grad forward pass per batch.
#      cached : `--mode precompute` writes top-K teacher logits to disk once;
#               training then never loads the teacher.
#
# USAGE
# -----
#   # (A) online, shared vocab -- the simple case
#   accelerate launch --config_file acc_config.yaml soft_distill.py \
#       --mode train --teacher_logits online --vocab_mode shared \
#       --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#       --student Qwen/Qwen2.5-3B \
#       --train_traces ./experiments/traces/tau1.0_lam0.0e+00_eps0.0 \
#       --top_k 50 --temperature 2.0 --alpha 0.9 \
#       --output_dir ./experiments/models/soft_k50
#
#   # (B) cached teacher logits (two steps)
#   accelerate launch --config_file acc_config.yaml soft_distill.py \
#       --mode precompute --vocab_mode shared --top_k 50 \
#       --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --student Qwen/Qwen2.5-3B \
#       --train_traces <traces> --cache_dir ./experiments/topk_cache_k50
#   accelerate launch --config_file acc_config.yaml soft_distill.py \
#       --mode train --teacher_logits cached --cache_dir ./experiments/topk_cache_k50 \
#       --vocab_mode shared --student Qwen/Qwen2.5-3B --top_k 50 \
#       --output_dir ./experiments/models/soft_k50
#
#   # (C) cross-tokenizer (keeps the repo's Llama student)
#   accelerate launch --config_file acc_config.yaml soft_distill.py \
#       --mode train --teacher_logits online --vocab_mode cross \
#       --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#       --student meta-llama/Llama-3.2-3B \
#       --student_tokenizer meta-llama/Llama-3.2-3B-Instruct \
#       --train_traces <traces> --top_k 50 --output_dir <out>
#
# Only torch / transformers / datasets / peft / accelerate are required.
# ==============================================================================

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("soft_distill")

# The repo's system prompt. Imported when available so this file stays in sync
# with utils.py; falls back to a copy so the script also runs standalone.
try:
    from utils import SYSTEM_PROMPT  # type: ignore
except Exception:  # pragma: no cover
    SYSTEM_PROMPT = (
        "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
        "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units "
        "within the brackets and fully evaluate arithmetic expressions.\n"
    )

# Markers used by gentraces.py when it writes DeepSeek-R1 traces.
DEEPSEEK_ASSISTANT_MARKER = "<｜Assistant｜>"
DEEPSEEK_EOS_MARKER = "<｜end▁of▁sentence｜>"

IGNORE_INDEX = -100
CACHE_SCHEMA_VERSION = 2
CHECKPOINT_SUCCESS = "_SUCCESS"


# ==============================================================================
# ARGUMENTS
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Soft top-K logit distillation (7B teacher -> 3B student).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- what to run -----------------------------------------------------
    p.add_argument("--mode", choices=["train", "precompute", "validate_cache"], default="train",
                   help=(
                       "'precompute' writes the top-K teacher cache; 'validate_cache' verifies an "
                       "existing cache against the current data/model/tokenizer contract and exits."
                   ))
    p.add_argument("--teacher_logits", choices=["online", "cached"], default="online",
                   help="Get teacher top-K on the fly, or from --cache_dir.")
    p.add_argument("--vocab_mode", choices=["shared", "cross"], default="shared",
                   help="'shared': one tokenizer for both models. 'cross': align two tokenizers.")

    # --- models ----------------------------------------------------------
    p.add_argument("--teacher", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--student", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--teacher_revision", type=str, default=None,
                   help="Optional immutable teacher revision/commit. Recommended for published runs.")
    p.add_argument("--student_revision", type=str, default=None,
                   help="Optional immutable student revision/commit. Recommended for published runs.")
    p.add_argument("--teacher_tokenizer", type=str, default=None,
                   help="Defaults to --teacher.")
    p.add_argument("--student_tokenizer", type=str, default=None,
                   help="Defaults to --teacher_tokenizer in 'shared' mode, else --student.")
    p.add_argument("--model_dtype", choices=["auto", "bf16", "fp32"], default="auto",
                   help="Student dtype. auto = bf16 with LoRA, fp32 otherwise (accelerate autocasts).")
    p.add_argument("--teacher_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   help="Falls back to 'sdpa' automatically if unavailable.")

    # --- distillation hyper-parameters -----------------------------------
    p.add_argument("--top_k", type=int, required=True,
                   help="K: how many teacher logits to keep per token. This is the K you enter.")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.9,
                   help="Weight on the soft KL term; (1-alpha) goes to hard-label CE.")
    p.add_argument(
        "--kl_variant",
        choices=["tail_bucket", "student_full", "student_renorm"],
        default="tail_bucket",
        help=(
            "tail_bucket: proper KL over K explicit tokens plus one aggregated tail bucket "
            "(recommended, shared vocabulary only). student_full: truncated teacher target against "
            "the full student softmax. student_renorm: conditional K-way KL."
        ),
    )
    p.add_argument("--kl_chunk_size", type=int, default=1024,
                   help="Positions per chunk when computing the KL. Lower = less peak memory.")

    # --- data ------------------------------------------------------------
    p.add_argument("--train_traces", type=str, default=None,
                   help="datasets.load_from_disk path (repo traces) or a .jsonl file.")
    p.add_argument("--eval_traces", type=str, default=None, help="Optional held-out set.")
    p.add_argument("--input_format", choices=["repo", "jsonl"], default="repo",
                   help="'repo': gentraces.py output. 'jsonl': {'prompt','completion'} per line.")
    p.add_argument("--trace_colname", type=str, default="auto",
                   help="Column holding the teacher trace. 'auto' recognizes trace/completion variants.")
    p.add_argument("--problem_colname", type=str, default="auto",
                   help="Problem column. 'auto' recognizes problem/prompt/question variants.")
    p.add_argument("--dataset_split", type=str, default=None,
                   help="Split to select if load_from_disk returns a DatasetDict.")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Where the top-K teacher cache lives (precompute writes it, cached-train reads it).")
    p.add_argument("--overwrite_cache", action=argparse.BooleanOptionalAction, default=False,
                   help="Replace an existing cache. Off by default to prevent accidental data loss.")
    p.add_argument("--teacher_topk_chunk_size", type=int, default=256,
                   help="Teacher prediction sites processed together during top-K extraction.")

    # --- LoRA ------------------------------------------------------------
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--merge_lora_on_save", action=argparse.BooleanOptionalAction, default=True)

    # --- optimisation ----------------------------------------------------
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", choices=["cosine", "linear", "constant"], default="cosine")
    p.add_argument("--num_epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=16, help="Global (effective) batch size.")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--precompute_batch_size", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=0, help="0 = only at the end of training.")
    p.add_argument("--save_steps", type=int, default=100,
                   help="Optimizer steps between resumable checkpoints; 0 disables checkpoints.")
    p.add_argument("--save_total_limit", type=int, default=3,
                   help="Maximum resumable checkpoints retained in --output_dir; 0 keeps all.")
    p.add_argument("--resume_from_checkpoint", type=str, default="auto",
                   help="'auto', 'none', or a checkpoint-N directory.")
    p.add_argument("--stop_after_steps", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="./soft_distill_out")

    return p


# ==============================================================================
# TOKENIZER SETUP
# ==============================================================================
def load_tokenizer(name: str, revision: Optional[str] = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        name, revision=revision, use_fast=True, trust_remote_code=True
    )
    if not tok.is_fast:
        raise RuntimeError(
            f"{name} did not load a *fast* tokenizer. This script needs fast tokenizers "
            "for character offsets (cross-vocab alignment) and for speed."
        )
    # Training uses right padding; padded positions are masked out of the loss
    # by the attention mask and by labels == -100, so the pad id is cosmetic.
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def tokenizer_fingerprint(tok) -> str:
    """Stable hash of the token-id contract and generation-relevant specials."""
    vocab = tok.get_vocab()
    ordered = sorted(((int(idx), token) for token, idx in vocab.items()), key=lambda x: (x[0], x[1]))
    payload = {
        "vocab": ordered,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "unk_token_id": tok.unk_token_id,
        "additional_special_tokens_ids": list(getattr(tok, "additional_special_tokens_ids", []) or []),
        "chat_template": getattr(tok, "chat_template", None),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_shared_tokenizers(t_tok, s_tok, t_name: str, s_name: str) -> None:
    """Prove that every common token id denotes the same token on both sides."""
    if t_tok is s_tok:
        log.info("Shared-vocabulary contract: one tokenizer object is used by both models.")
        return

    t_vocab = t_tok.get_vocab()
    s_vocab = s_tok.get_vocab()
    t_by_id = {int(idx): token for token, idx in t_vocab.items()}
    s_by_id = {int(idx): token for token, idx in s_vocab.items()}
    common_limit = min(max(t_by_id, default=-1), max(s_by_id, default=-1)) + 1
    mismatches = []
    for idx in range(common_limit):
        if t_by_id.get(idx) != s_by_id.get(idx):
            mismatches.append((idx, t_by_id.get(idx), s_by_id.get(idx)))
            if len(mismatches) == 5:
                break
    if mismatches:
        details = "; ".join(f"id {i}: {a!r} != {b!r}" for i, a, b in mismatches)
        raise ValueError(
            "--vocab_mode shared requires identical token-id meanings across the common vocabulary. "
            f"teacher tokenizer={t_name!r}, student tokenizer={s_name!r}; first mismatches: {details}. "
            "Use the teacher tokenizer for both Qwen models or select --vocab_mode cross."
        )
    log.info(
        "Shared-vocabulary contract verified exhaustively for %d common token ids (%s vs %s).",
        common_limit,
        t_name,
        s_name,
    )


def render_prompt_ids(tok, problem: str) -> List[int]:
    """System+user prompt, ending right where the assistant should start."""
    problem = problem.strip()
    if getattr(tok, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict):  # some templates return a BatchEncoding
            ids = ids["input_ids"]
        if len(ids) and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    # Fallback for base checkpoints that ship no chat template.
    text = f"{SYSTEM_PROMPT}\n### Problem:\n{problem}\n\n### Solution:\n"
    return tok(text, add_special_tokens=True)["input_ids"]


# ==============================================================================
# RAW DATA LOADING
# ==============================================================================
TRACE_COLUMN_CANDIDATES = ("trace", "completion_af", "completion", "response", "text")
PROBLEM_COLUMN_CANDIDATES = ("problem", "prompt", "question", "query")


def _choose_column(column_names: Sequence[str], requested: str, candidates: Sequence[str], kind: str) -> str:
    if requested != "auto":
        if requested not in column_names:
            raise KeyError(
                f"Column '{requested}' not found. Available: {list(column_names)}. "
                f"Pass --{kind}_colname or use --{kind}_colname auto."
            )
        return requested
    found = [name for name in candidates if name in column_names]
    if not found:
        raise KeyError(
            f"Could not auto-detect the {kind} column. Available: {list(column_names)}. "
            f"Tried: {list(candidates)}. Pass --{kind}_colname explicitly."
        )
    if len(found) > 1:
        log.warning("Several %s columns exist (%s); selecting '%s'.", kind, found, found[0])
    else:
        log.info("Auto-detected %s column: '%s'.", kind, found[0])
    return found[0]


def source_fingerprint(path: str, input_format: str) -> str:
    """Content hash used to bind a cache to the exact source traces."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Trace source does not exist: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    digest.update(f"format={input_format}\n".encode())
    for file_path in files:
        relative = file_path.name if root.is_file() else str(file_path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_pairs(path: str, args, limit: Optional[int]) -> List[Dict[str, str]]:
    """Returns [{'prompt': <problem text>, 'completion': <teacher response text>}, ...]."""
    pairs: List[Dict[str, str]] = []

    if args.input_format == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pairs.append({"prompt": row["prompt"], "completion": row["completion"]})
    else:
        import datasets as hf_datasets

        ds = hf_datasets.load_from_disk(path)
        if isinstance(ds, hf_datasets.DatasetDict):
            requested_split = getattr(args, "dataset_split", None)
            if requested_split:
                if requested_split not in ds:
                    raise KeyError(
                        f"Dataset split '{requested_split}' not found. Available: {list(ds.keys())}."
                    )
                ds = ds[requested_split]
            elif len(ds) == 1:
                only = next(iter(ds.keys()))
                log.info("DatasetDict has one split; selecting '%s'.", only)
                ds = ds[only]
            elif "train" in ds:
                log.warning("DatasetDict has several splits; selecting 'train'. Pass --dataset_split to override.")
                ds = ds["train"]
            else:
                raise ValueError(
                    f"DatasetDict has several splits {list(ds.keys())}; pass --dataset_split."
                )
        trace_col = _choose_column(
            ds.column_names, args.trace_colname, TRACE_COLUMN_CANDIDATES, "trace"
        )
        problem_col = _choose_column(
            ds.column_names, args.problem_colname, PROBLEM_COLUMN_CANDIDATES, "problem"
        )
        for row in ds:
            trace = row[trace_col]
            if trace is None:
                continue
            # gentraces.py stores the full prompt+response string.
            if DEEPSEEK_ASSISTANT_MARKER in trace:
                completion = trace.split(DEEPSEEK_ASSISTANT_MARKER, 1)[1]
            else:
                completion = trace
            completion = completion.replace(DEEPSEEK_EOS_MARKER, "").strip()
            if not completion:
                continue
            pairs.append({"prompt": row[problem_col], "completion": completion})

    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError(f"No usable examples found in {path}.")
    return pairs


# ==============================================================================
# ENCODING  (this is where the two vocab modes differ)
# ==============================================================================
@dataclass
class Encoded:
    """One training example, already tokenised."""
    student_input_ids: np.ndarray      # [Ls]
    student_labels: np.ndarray         # [Ls], IGNORE_INDEX on the prompt
    teacher_input_ids: np.ndarray      # [Lt] (== student_input_ids in shared mode)
    kl_student_pos: np.ndarray         # [M] index into logits[:-1] that predicts a KL token
    kl_teacher_pos: np.ndarray         # [M] the matching teacher position


def _truncate_by_offsets(ids: List[int], offsets: List[Tuple[int, int]], budget: int):
    """Keep at most `budget` tokens; return (ids, offsets)."""
    if budget <= 0:
        return [], []
    return ids[:budget], offsets[:budget]


def encode_shared(problem: str, completion: str, tok, max_length: int) -> Optional[Encoded]:
    prompt_ids = render_prompt_ids(tok, problem)
    enc = tok(completion, add_special_tokens=False, return_offsets_mapping=False)
    comp_ids = list(enc["input_ids"])
    if tok.eos_token_id is not None:
        comp_ids = comp_ids + [tok.eos_token_id]

    # Prompt keeps its tail (the part that actually conditions the answer).
    max_prompt = max(1, max_length // 2)
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    room = max_length - len(prompt_ids)
    if room < 2:
        return None
    comp_ids = comp_ids[:room]
    if len(comp_ids) < 1:
        return None

    input_ids = np.asarray(prompt_ids + comp_ids, dtype=np.int64)
    labels = np.full_like(input_ids, IGNORE_INDEX)
    labels[len(prompt_ids):] = input_ids[len(prompt_ids):]

    # logits[p] predicts token p+1, so completion token i is predicted at
    # p = len(prompt) + i - 1.  i ranges over all completion tokens.
    pos = np.arange(len(prompt_ids) - 1, len(input_ids) - 1, dtype=np.int64)
    return Encoded(
        student_input_ids=input_ids,
        student_labels=labels,
        teacher_input_ids=input_ids,
        kl_student_pos=pos,
        kl_teacher_pos=pos,
    )


def encode_cross(problem: str, completion: str, s_tok, t_tok, max_length: int) -> Optional[Encoded]:
    """
    Different tokenizers. The two models see the *same completion string* but
    chop it differently, so we align by character offset: student completion
    token i and teacher completion token j describe the same prediction point
    iff they start at the same character.
    """
    s_prompt = render_prompt_ids(s_tok, problem)
    t_prompt = render_prompt_ids(t_tok, problem)

    s_enc = s_tok(completion, add_special_tokens=False, return_offsets_mapping=True)
    t_enc = t_tok(completion, add_special_tokens=False, return_offsets_mapping=True)
    s_ids, s_off = list(s_enc["input_ids"]), [tuple(o) for o in s_enc["offset_mapping"]]
    t_ids, t_off = list(t_enc["input_ids"]), [tuple(o) for o in t_enc["offset_mapping"]]
    if not s_ids or not t_ids:
        return None

    max_prompt = max(1, max_length // 2)
    if len(s_prompt) > max_prompt:
        s_prompt = s_prompt[-max_prompt:]
    if len(t_prompt) > max_prompt:
        t_prompt = t_prompt[-max_prompt:]

    # Reserve one slot for EOS on each side.
    s_room = max_length - len(s_prompt) - 1
    t_room = max_length - len(t_prompt) - 1
    if s_room < 1 or t_room < 1:
        return None

    # Truncate both to the *same character span* so the alignment stays valid.
    s_ids, s_off = _truncate_by_offsets(s_ids, s_off, s_room)
    t_ids, t_off = _truncate_by_offsets(t_ids, t_off, t_room)
    cut_char = min(s_off[-1][1], t_off[-1][1])
    s_keep = [k for k, o in enumerate(s_off) if o[1] <= cut_char]
    t_keep = [k for k, o in enumerate(t_off) if o[1] <= cut_char]
    if not s_keep or not t_keep:
        return None
    s_ids, s_off = s_ids[: s_keep[-1] + 1], s_off[: s_keep[-1] + 1]
    t_ids, t_off = t_ids[: t_keep[-1] + 1], t_off[: t_keep[-1] + 1]

    # Character-start -> teacher completion index (first occurrence wins).
    t_start_to_idx: Dict[int, int] = {}
    for j, (st, _) in enumerate(t_off):
        t_start_to_idx.setdefault(st, j)

    kl_s, kl_t = [], []
    for i, (st, _) in enumerate(s_off):
        j = t_start_to_idx.get(st)
        if j is None:
            continue
        kl_s.append(len(s_prompt) + i - 1)
        kl_t.append(len(t_prompt) + j - 1)

    # EOS is appended after the aligned span; it gets CE but not KL, because the
    # two tokenizers use different end-of-text ids.
    s_full = s_prompt + s_ids + ([s_tok.eos_token_id] if s_tok.eos_token_id is not None else [])
    t_full = t_prompt + t_ids + ([t_tok.eos_token_id] if t_tok.eos_token_id is not None else [])

    input_ids = np.asarray(s_full, dtype=np.int64)
    labels = np.full_like(input_ids, IGNORE_INDEX)
    labels[len(s_prompt):] = input_ids[len(s_prompt):]

    if not kl_s:
        return None
    return Encoded(
        student_input_ids=input_ids,
        student_labels=labels,
        teacher_input_ids=np.asarray(t_full, dtype=np.int64),
        kl_student_pos=np.asarray(kl_s, dtype=np.int64),
        kl_teacher_pos=np.asarray(kl_t, dtype=np.int64),
    )


def encode_all(pairs, args, s_tok, t_tok) -> List[Encoded]:
    out: List[Encoded] = []
    dropped = 0
    n_target, n_aligned = 0, 0
    for row in pairs:
        if args.vocab_mode == "shared":
            e = encode_shared(row["prompt"], row["completion"], s_tok, args.max_length)
        else:
            e = encode_cross(row["prompt"], row["completion"], s_tok, t_tok, args.max_length)
        if e is None:
            dropped += 1
            continue
        n_target += int((e.student_labels != IGNORE_INDEX).sum())
        n_aligned += len(e.kl_student_pos)
        out.append(e)
    if dropped:
        log.warning("Dropped %d/%d examples (empty or no room after truncation).", dropped, len(pairs))
    if not out:
        raise ValueError("Every example was dropped -- check --max_length and the trace columns.")
    log.info("Encoded %d examples | %d supervised tokens | %d (%.1f%%) carry a teacher top-K target.",
             len(out), n_target, n_aligned, 100.0 * n_aligned / max(1, n_target))
    if args.vocab_mode == "cross" and n_aligned < 0.5 * n_target:
        log.warning("Fewer than half the tokens could be aligned across the two tokenizers. "
                    "--vocab_mode shared (Qwen student) gives a much stronger signal.")
    return out


def validate_student_token_ids(examples: Sequence[Encoded], student_vocab_size: int) -> None:
    """Fail before GPU work if the chosen tokenizer can emit out-of-range ids."""
    bad_example = None
    bad_id = -1
    for index, example in enumerate(examples):
        if example.student_input_ids.size:
            current = int(example.student_input_ids.max())
            if current >= student_vocab_size:
                bad_example, bad_id = index, current
                break
    if bad_example is not None:
        raise ValueError(
            f"Student tokenizer emitted token id {bad_id}, but the student output/embedding vocabulary "
            f"has size {student_vocab_size} (example {bad_example}). Use a tokenizer compatible with "
            "the student checkpoint; do not force an expanded teacher tokenizer onto this model."
        )


# ==============================================================================
# CROSS-TOKENIZER VOCABULARY MAP
# ==============================================================================
def build_vocab_map(t_tok, s_tok, teacher_logit_dim: int, student_logit_dim: int) -> torch.Tensor:
    """
    teacher id -> student id, by exact surface form; -1 where no single student
    token has the same surface form. Both Qwen and Llama-3 use byte-level BPE,
    so the token strings ('Ġthe') are directly comparable.

    The table is sized to the teacher's *logit dimension* (which can exceed the
    tokenizer size when the embedding matrix is padded, as it is for
    DeepSeek-R1-Distill-Qwen-7B: 152064 logits vs ~151.6k real tokens), so
    indexing it with any argmax/top-k id is always safe.
    """
    s_vocab = s_tok.get_vocab()
    t_vocab = t_tok.get_vocab()
    size = max(teacher_logit_dim, max(t_vocab.values()) + 1, len(t_tok))
    mapping = torch.full((size,), -1, dtype=torch.long)

    leftover_tokens, leftover_ids = [], []
    hits = 0
    for token, t_id in t_vocab.items():
        s_id = s_vocab.get(token)
        if s_id is not None and s_id < student_logit_dim:
            mapping[t_id] = s_id
            hits += 1
        else:
            leftover_tokens.append(token)
            leftover_ids.append(t_id)

    # Second pass: match by decoded surface text, batched so this stays fast on
    # a 150k-entry vocabulary.
    if leftover_tokens:
        texts = [t_tok.convert_tokens_to_string([tk]) for tk in leftover_tokens]
        keep = [i for i, tx in enumerate(texts) if tx]
        if keep:
            enc = s_tok([texts[i] for i in keep], add_special_tokens=False)["input_ids"]
            for i, cand in zip(keep, enc):
                if len(cand) == 1 and cand[0] < student_logit_dim:
                    mapping[leftover_ids[i]] = cand[0]
                    hits += 1

    log.info("Cross-vocab map: %d/%d teacher tokens map to a single student token (%.1f%%).",
             hits, len(t_vocab), 100.0 * hits / max(1, len(t_vocab)))
    if hits < 0.2 * len(t_vocab):
        log.warning("Very low vocabulary overlap -- cross-tokenizer distillation will be weak here. "
                    "Consider --vocab_mode shared with a Qwen student.")
    return mapping


# ==============================================================================
# DATASET / COLLATOR
# ==============================================================================
class OnlineDataset(Dataset):
    def __init__(self, examples: List[Encoded]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return {
            "student_input_ids": e.student_input_ids,
            "student_labels": e.student_labels,
            "teacher_input_ids": e.teacher_input_ids,
            "kl_student_pos": e.kl_student_pos,
            "kl_teacher_pos": e.kl_teacher_pos,
        }


class CachedDataset(Dataset):
    """Reads the precomputed top-K cache; the teacher is not needed at all."""

    def __init__(self, hf_ds, top_k: int):
        self.ds = hf_ds
        self.k = top_k

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        n = len(row["kl_student_pos"])
        k = self.k
        ids = np.asarray(row["topk_ids"], dtype=np.int64).reshape(n, k) if n else np.zeros((0, k), np.int64)
        val = np.asarray(row["topk_logits"], dtype=np.float32).reshape(n, k) if n else np.zeros((0, k), np.float32)
        coverage = np.asarray(row["coverage"], dtype=np.float32).reshape(n) if n else np.zeros(0, np.float32)
        full_lse = np.asarray(row["teacher_logsumexp"], dtype=np.float32).reshape(n) if n else np.zeros(0, np.float32)
        return {
            "student_input_ids": np.asarray(row["student_input_ids"], dtype=np.int64),
            "student_labels": np.asarray(row["student_labels"], dtype=np.int64),
            "kl_student_pos": np.asarray(row["kl_student_pos"], dtype=np.int64),
            "topk_ids": ids,
            "topk_logits": val,
            "coverage": coverage,
            "teacher_logsumexp": full_lse,
        }


def _pad_stack(seqs: Sequence[np.ndarray], pad_value: int) -> torch.Tensor:
    maxlen = max(len(s) for s in seqs)
    out = np.full((len(seqs), maxlen), pad_value, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return torch.from_numpy(out)


class Collator:
    """
    Produces, per batch:
      student_input_ids / student_attention_mask / student_labels
      kl_batch_idx, kl_student_pos              (flat lists of KL sites)
      + teacher_input_ids / teacher_attention_mask / kl_teacher_pos   (online)
      + topk_ids / topk_logits                                        (cached)
    """

    def __init__(self, student_pad_id: int, teacher_pad_id: int, cached: bool):
        self.s_pad = student_pad_id
        self.t_pad = teacher_pad_id
        self.cached = cached

    def __call__(self, features):
        s_ids = _pad_stack([f["student_input_ids"] for f in features], self.s_pad)
        s_lab = _pad_stack([f["student_labels"] for f in features], IGNORE_INDEX)
        s_mask = torch.zeros_like(s_ids)
        for i, f in enumerate(features):
            s_mask[i, : len(f["student_input_ids"])] = 1

        batch = {
            "student_input_ids": s_ids,
            "student_attention_mask": s_mask,
            "student_labels": s_lab,
        }

        kl_b, kl_s = [], []
        for i, f in enumerate(features):
            p = f["kl_student_pos"]
            kl_b.append(np.full(len(p), i, dtype=np.int64))
            kl_s.append(p)
        batch["kl_batch_idx"] = torch.from_numpy(np.concatenate(kl_b)) if kl_b else torch.zeros(0, dtype=torch.long)
        batch["kl_student_pos"] = torch.from_numpy(np.concatenate(kl_s)) if kl_s else torch.zeros(0, dtype=torch.long)

        if self.cached:
            ids = [f["topk_ids"] for f in features]
            val = [f["topk_logits"] for f in features]
            cov = [f["coverage"] for f in features]
            full_lse = [f["teacher_logsumexp"] for f in features]
            batch["topk_ids"] = torch.from_numpy(np.concatenate(ids, axis=0)) if ids else torch.zeros(0, 0, dtype=torch.long)
            batch["topk_logits"] = torch.from_numpy(np.concatenate(val, axis=0)) if val else torch.zeros(0, 0)
            batch["coverage"] = torch.from_numpy(np.concatenate(cov, axis=0)) if cov else torch.zeros(0)
            batch["teacher_logsumexp"] = (
                torch.from_numpy(np.concatenate(full_lse, axis=0)) if full_lse else torch.zeros(0)
            )
        else:
            t_ids = _pad_stack([f["teacher_input_ids"] for f in features], self.t_pad)
            t_mask = torch.zeros_like(t_ids)
            for i, f in enumerate(features):
                t_mask[i, : len(f["teacher_input_ids"])] = 1
            batch["teacher_input_ids"] = t_ids
            batch["teacher_attention_mask"] = t_mask
            kl_t = [f["kl_teacher_pos"] for f in features]
            batch["kl_teacher_pos"] = torch.from_numpy(np.concatenate(kl_t)) if kl_t else torch.zeros(0, dtype=torch.long)
        return batch


# ==============================================================================
# TEACHER TOP-K EXTRACTION
# ==============================================================================
@torch.no_grad()
def teacher_topk(
    teacher,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kl_batch_idx: torch.Tensor,
    kl_teacher_pos: torch.Tensor,
    top_k: int,
    vocab_map: Optional[torch.Tensor],
    student_vocab_size: int,
    temperature: float = 1.0,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (ids [N, K] in *student* vocab space, raw logits [N, K] float32,
             coverage [N], full_logsumexp_at_temperature [N]).
    Entries with no student-vocab counterpart get logit -inf.
    """
    out = teacher(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits                                   # [B, L, Vt]; logits[p] predicts p+1
    B, L, Vt = logits.shape
    flat = logits.reshape(B * L, Vt)                      # contiguous -> free view
    flat_idx = kl_batch_idx * L + kl_teacher_pos
    selected = flat.index_select(0, flat_idx)             # [N, Vt], original model dtype
    del out, logits, flat

    if selected.size(0) == 0:
        k = min(top_k, min(Vt, student_vocab_size) if vocab_map is None else Vt)
        empty_ids = torch.empty((0, k), dtype=torch.long, device=selected.device)
        empty_vals = torch.empty((0, k), dtype=torch.float32, device=selected.device)
        empty_sites = torch.empty((0,), dtype=torch.float32, device=selected.device)
        return empty_ids, empty_vals, empty_sites, empty_sites

    all_ids, all_vals, all_cov, all_lse = [], [], [], []
    for start in range(0, selected.size(0), max(1, chunk_size)):
        rows = selected[start: start + max(1, chunk_size)].float()
        full_lse = torch.logsumexp(rows / temperature, dim=-1)

        if vocab_map is None:
            # Select the K strongest teacher targets the student can represent;
            # all other teacher mass is retained by the tail bucket.
            common_vocab = min(Vt, student_vocab_size)
            k = min(top_k, common_vocab)
            vals, ids = torch.topk(rows[:, :common_vocab], k=k, dim=-1)
            kept_lse = torch.logsumexp(vals / temperature, dim=-1)
        else:
            k = min(top_k, Vt)
            vals, ids = torch.topk(rows, k=k, dim=-1)
            ids = vocab_map[ids]
            invalid = ids < 0
            vals = vals.masked_fill(invalid, float("-inf"))
            ids = ids.masked_fill(invalid, 0)
            kept_lse = torch.logsumexp(vals / temperature, dim=-1)

        all_ids.append(ids)
        all_vals.append(vals)
        all_cov.append((kept_lse - full_lse).exp())
        all_lse.append(full_lse)

    del selected
    return (
        torch.cat(all_ids, dim=0),
        torch.cat(all_vals, dim=0),
        torch.cat(all_cov, dim=0),
        torch.cat(all_lse, dim=0),
    )


# ==============================================================================
# LOSSES
# ==============================================================================
def topk_kl_loss(
    student_logits: torch.Tensor,   # [B, L, V] -- unshifted; logits[p] predicts p+1
    kl_flat_idx: torch.Tensor,      # [N] flat index into B*L
    topk_ids: torch.Tensor,         # [N, K] student vocab ids
    topk_logits: torch.Tensor,      # [N, K] float32, -inf for invalid entries
    temperature: float,
    variant: str,
    chunk_size: int,
    teacher_logsumexp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean KL/surrogate over sites; ``tail_bucket`` is a proper compressed KL."""
    if kl_flat_idx.numel() == 0:
        return student_logits.sum() * 0.0

    V = student_logits.size(-1)
    flat = student_logits.reshape(-1, V)

    T = temperature
    tl = topk_logits.float() / T                           # [N, K]
    valid = torch.isfinite(tl)
    keep = valid.any(dim=-1)
    if not bool(keep.all()):
        kl_flat_idx = kl_flat_idx[keep]
        topk_ids = topk_ids[keep]
        tl = tl[keep]
        valid = valid[keep]
        if teacher_logsumexp is not None:
            teacher_logsumexp = teacher_logsumexp[keep]
    n_sites = kl_flat_idx.numel()
    if n_sites == 0:
        return student_logits.sum() * 0.0

    tl = tl.masked_fill(~valid, float("-inf"))
    if variant == "tail_bucket":
        if teacher_logsumexp is None:
            raise ValueError("tail_bucket KL requires teacher_logsumexp for every prediction site.")
        if teacher_logsumexp.numel() != n_sites:
            raise ValueError(
                f"teacher_logsumexp has {teacher_logsumexp.numel()} entries for {n_sites} KL sites."
            )
        log_p_t = tl - teacher_logsumexp.float().unsqueeze(-1)
    else:
        # Legacy truncated objectives condition the teacher on the retained K.
        log_p_t = tl - torch.logsumexp(tl, dim=-1, keepdim=True)
    p_t = log_p_t.exp()                                    # exact 0 on invalid entries
    log_p_t = torch.where(valid, log_p_t, torch.zeros_like(log_p_t))

    total = flat.new_zeros((), dtype=torch.float32)
    ids_safe = topk_ids.clamp_min(0)
    for start in range(0, n_sites, max(1, chunk_size)):
        end = min(start + chunk_size, n_sites)
        idx = kl_flat_idx[start:end]
        ids_c = ids_safe[start:end]
        pt_c = p_t[start:end]
        lpt_c = log_p_t[start:end]
        val_c = valid[start:end]

        rows = flat.index_select(0, idx)                    # [n, V]
        if variant in ("tail_bucket", "student_full"):
            log_p_s = F.log_softmax(rows.float() / T, dim=-1).gather(1, ids_c)
        else:
            sel = rows.gather(1, ids_c).float() / T
            sel = sel.masked_fill(~val_c, float("-inf"))
            log_p_s = sel - torch.logsumexp(sel, dim=-1, keepdim=True)

        log_ps_valid = torch.where(val_c, log_p_s, torch.zeros_like(log_p_s))
        term = pt_c * (lpt_c - log_ps_valid)
        subtotal = term.sum()
        if variant == "tail_bucket":
            # Both vocabularies are compressed to K named outcomes plus a
            # single "everything else" outcome. This preserves omitted mass.
            eps = torch.finfo(torch.float32).eps
            p_tail = (1.0 - pt_c.sum(dim=-1)).clamp(min=0.0, max=1.0)
            q_top = torch.where(val_c, log_p_s.exp(), torch.zeros_like(log_p_s)).sum(dim=-1)
            q_tail = (1.0 - q_top).clamp(min=0.0, max=1.0)
            tail_term = torch.where(
                p_tail > 0,
                p_tail * (p_tail.clamp_min(eps).log() - q_tail.clamp_min(eps).log()),
                torch.zeros_like(p_tail),
            )
            subtotal = subtotal + tail_term.sum()
        total = total + subtotal

    return total / n_sites


def hard_ce_loss(student_logits: torch.Tensor, shift_targets: torch.Tensor) -> torch.Tensor:
    """
    CE on the completion tokens only. Rows are selected before the float32 cast
    so the prompt's logits never get copied -- that matters a lot at V=152k.
    """
    V = student_logits.size(-1)
    flat = student_logits.reshape(-1, V)
    tgt = shift_targets.reshape(-1)
    sel = (tgt != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return student_logits.sum() * 0.0
    return F.cross_entropy(flat.index_select(0, sel).float(), tgt.index_select(0, sel))


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def resolve_attn(requested: str) -> str:
    if requested != "flash_attention_2":
        return requested
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            return "flash_attention_2"
    except Exception:
        pass
    log.warning("flash-attn unavailable -> falling back to attn_implementation='sdpa'.")
    return "sdpa"


def load_causal_lm(
    name: str,
    dtype: torch.dtype,
    attn: str,
    trainable: bool,
    revision: Optional[str] = None,
):
    from transformers import AutoModelForCausalLM

    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        use_cache=False,
        revision=revision,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation=attn, **kwargs)
    except (ValueError, ImportError, RuntimeError) as e:
        log.warning("attn_implementation='%s' rejected for %s (%s); retrying with 'eager'.", attn, name, e)
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", **kwargs)
    if not trainable:
        model.eval()
        model.requires_grad_(False)
    return model


def maybe_wrap_lora(model, args):
    if not args.lora:
        return model
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in targets if t in present] or ["q_proj", "v_proj"]
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def student_dtype_for(args) -> torch.dtype:
    if args.model_dtype == "bf16":
        return torch.bfloat16
    if args.model_dtype == "fp32":
        return torch.float32
    return torch.bfloat16 if args.lora else torch.float32


TEACHER_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ==============================================================================
# CACHE (precompute)
# ==============================================================================
def _json_hash(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def tokenizer_names(args) -> Tuple[str, str]:
    teacher_name = args.teacher_tokenizer or args.teacher
    if args.vocab_mode == "shared":
        student_name = args.student_tokenizer or teacher_name
    else:
        student_name = args.student_tokenizer or args.student
    return teacher_name, student_name


def model_contract(name: str, revision: Optional[str]) -> Dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
    cfg_dict = cfg.to_dict()
    return {
        "name": name,
        "requested_revision": revision,
        "resolved_commit": getattr(cfg, "_commit_hash", None),
        "model_type": getattr(cfg, "model_type", None),
        "vocab_size": int(getattr(cfg, "vocab_size", 0) or 0),
        "config_sha256": _json_hash(cfg_dict),
    }


def build_cache_contract(args, s_tok, t_tok, source_sha256: str, num_examples: int) -> Dict:
    t_name, s_name = tokenizer_names(args)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "teacher": model_contract(args.teacher, args.teacher_revision),
        "student": model_contract(args.student, args.student_revision),
        "teacher_tokenizer": {
            "name": t_name,
            "fingerprint": tokenizer_fingerprint(t_tok),
        },
        "student_tokenizer": {
            "name": s_name,
            "fingerprint": tokenizer_fingerprint(s_tok),
        },
        "vocab_mode": args.vocab_mode,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "input_format": args.input_format,
        "trace_colname": args.trace_colname,
        "problem_colname": args.problem_colname,
        "dataset_split": args.dataset_split,
        "max_train_samples": args.max_train_samples,
        "source_sha256": source_sha256,
        "num_examples": num_examples,
    }


def run_precompute(args, accelerator):
    import datasets as hf_datasets
    from accelerate.utils import broadcast_object_list

    if not args.cache_dir:
        raise ValueError("--cache_dir is required for --mode precompute.")
    if not args.train_traces:
        raise ValueError("--train_traces is required for --mode precompute.")

    cache_root = os.path.abspath(os.path.expanduser(args.cache_dir))
    has_existing = os.path.isdir(cache_root) and bool(os.listdir(cache_root))
    if has_existing and not args.overwrite_cache:
        raise FileExistsError(
            f"Cache already exists at {cache_root}. Use a new --cache_dir, or pass --overwrite_cache."
        )
    if accelerator.is_main_process and args.overwrite_cache and os.path.isdir(cache_root):
        shutil.rmtree(cache_root)
    accelerator.wait_for_everyone()
    os.makedirs(cache_root, exist_ok=True)
    args.cache_dir = cache_root

    source_hash_box = [source_fingerprint(args.train_traces, args.input_format)
                       if accelerator.is_main_process else None]
    broadcast_object_list(source_hash_box)
    source_sha256 = source_hash_box[0]

    s_tok, t_tok = build_tokenizers(args)
    pairs = load_pairs(args.train_traces, args, args.max_train_samples)
    examples = encode_all(pairs, args, s_tok, t_tok)

    student_vocab = student_vocab_size(args, s_tok)
    validate_student_token_ids(examples, student_vocab)
    cache_k_limit = min(teacher_vocab_size(args, t_tok), student_vocab) if args.vocab_mode == "shared" \
        else teacher_vocab_size(args, t_tok)
    if args.top_k > cache_k_limit:
        raise ValueError(
            f"--top_k={args.top_k} exceeds the cacheable vocabulary limit {cache_k_limit}. "
            "Choose a smaller K."
        )

    attn = resolve_attn(args.attn_implementation)
    teacher = load_causal_lm(
        args.teacher,
        TEACHER_DTYPES[args.teacher_dtype],
        attn,
        trainable=False,
        revision=args.teacher_revision,
    )
    teacher.to(accelerator.device)

    vocab_map = None
    if args.vocab_mode == "cross":
        vocab_map = build_vocab_map(t_tok, s_tok, teacher_vocab_size(args, t_tok),
                                    student_vocab).to(accelerator.device)

    # Shard across processes; each rank writes its own slice, then rank 0 merges.
    n = len(examples)
    rank, world = accelerator.process_index, accelerator.num_processes
    my_idx = list(range(rank, n, world))
    collator = Collator(s_tok.pad_token_id, t_tok.pad_token_id, cached=False)
    source = OnlineDataset(examples)

    rows = {"example_id": [], "student_input_ids": [], "student_labels": [],
            "kl_student_pos": [], "topk_ids": [], "topk_logits": [],
            "coverage": [], "teacher_logsumexp": []}

    bs = args.precompute_batch_size
    t0 = time.time()
    for b0 in range(0, len(my_idx), bs):
        chunk = my_idx[b0: b0 + bs]
        batch = collator([source[i] for i in chunk])
        batch = {k: (v.to(accelerator.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        ids, vals, cov, full_lse = teacher_topk(
            teacher, batch["teacher_input_ids"], batch["teacher_attention_mask"],
            batch["kl_batch_idx"], batch["kl_teacher_pos"], args.top_k, vocab_map, student_vocab,
            temperature=args.temperature,
            chunk_size=args.teacher_topk_chunk_size,
        )
        ids = ids.cpu().numpy().astype(np.int32)
        vals = vals.float().cpu().numpy()
        cov = cov.float().cpu().numpy()
        full_lse = full_lse.float().cpu().numpy()
        # -inf does not survive a round trip through Arrow; use a large negative
        # sentinel and restore it on load.
        vals = np.where(np.isfinite(vals), vals, -1e30).astype(np.float32)

        offset = 0
        bidx = batch["kl_batch_idx"].cpu().numpy()
        for local_i, ex_i in enumerate(chunk):
            m = int((bidx == local_i).sum())
            e = examples[ex_i]
            rows["example_id"].append(ex_i)
            rows["student_input_ids"].append(e.student_input_ids.astype(np.int32).tolist())
            rows["student_labels"].append(e.student_labels.astype(np.int32).tolist())
            rows["kl_student_pos"].append(e.kl_student_pos.astype(np.int32).tolist())
            rows["topk_ids"].append(ids[offset: offset + m].reshape(-1).tolist())
            rows["topk_logits"].append(vals[offset: offset + m].reshape(-1).tolist())
            rows["coverage"].append(cov[offset: offset + m].astype(np.float32).tolist())
            rows["teacher_logsumexp"].append(
                full_lse[offset: offset + m].astype(np.float32).tolist()
            )
            offset += m

        if accelerator.is_main_process and (b0 // max(bs, 1)) % 20 == 0:
            done = b0 + len(chunk)
            log.info("precompute %d/%d (%.1fs)", done, len(my_idx), time.time() - t0)

    shard_dir = os.path.join(args.cache_dir, f"_shard_{rank}")
    hf_datasets.Dataset.from_dict(rows).save_to_disk(shard_dir)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shards = [hf_datasets.load_from_disk(os.path.join(args.cache_dir, f"_shard_{r}")) for r in range(world)]
        merged = hf_datasets.concatenate_datasets(shards).sort("example_id")
        merged.save_to_disk(os.path.join(args.cache_dir, "data"))
        contract = build_cache_contract(args, s_tok, t_tok, source_sha256, len(merged))
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "signature": _json_hash(contract),
            "contract": contract,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_tmp = os.path.join(args.cache_dir, "cache_meta.json.tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        os.replace(meta_tmp, os.path.join(args.cache_dir, "cache_meta.json"))
        for r in range(world):
            shutil.rmtree(os.path.join(args.cache_dir, f"_shard_{r}"), ignore_errors=True)
        log.info("Wrote top-K cache for %d examples to %s", len(merged), args.cache_dir)
    accelerator.wait_for_everyone()


def load_cache(args, s_tok, t_tok):
    import datasets as hf_datasets

    meta_path = os.path.join(args.cache_dir, "cache_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No cache at {args.cache_dir}. Run --mode precompute first.")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("schema_version") != CACHE_SCHEMA_VERSION or "contract" not in meta:
        raise ValueError(
            f"Cache schema is missing or obsolete. Expected schema {CACHE_SCHEMA_VERSION}; rebuild it."
        )
    contract = meta["contract"]
    if meta.get("signature") != _json_hash(contract):
        raise ValueError("Cache metadata signature is invalid; the cache may be incomplete or modified.")

    t_name, s_name = tokenizer_names(args)
    expected = {
        "vocab_mode": args.vocab_mode,
        "top_k": args.top_k,
        "max_length": args.max_length,
    }
    mismatches = []
    for key, value in expected.items():
        if contract.get(key) != value:
            mismatches.append(f"{key}: cache={contract.get(key)!r}, run={value!r}")
    if contract.get("teacher", {}).get("name") != args.teacher:
        mismatches.append(f"teacher: cache={contract.get('teacher', {}).get('name')!r}, run={args.teacher!r}")
    if contract.get("student", {}).get("name") != args.student:
        mismatches.append(f"student: cache={contract.get('student', {}).get('name')!r}, run={args.student!r}")
    if contract.get("teacher_tokenizer", {}).get("name") != t_name:
        mismatches.append("teacher tokenizer name differs")
    if contract.get("student_tokenizer", {}).get("name") != s_name:
        mismatches.append("student tokenizer name differs")
    if contract.get("teacher_tokenizer", {}).get("fingerprint") != tokenizer_fingerprint(t_tok):
        mismatches.append("teacher tokenizer token-id fingerprint differs")
    if contract.get("student_tokenizer", {}).get("fingerprint") != tokenizer_fingerprint(s_tok):
        mismatches.append("student tokenizer token-id fingerprint differs")

    current_teacher = model_contract(args.teacher, args.teacher_revision)
    current_student = model_contract(args.student, args.student_revision)
    for label, current in (("teacher", current_teacher), ("student", current_student)):
        cached_model = contract.get(label, {})
        for key in ("resolved_commit", "config_sha256", "vocab_size"):
            if cached_model.get(key) != current.get(key):
                mismatches.append(
                    f"{label} {key}: cache={cached_model.get(key)!r}, run={current.get(key)!r}"
                )

    cache_temperature = float(contract.get("temperature", float("nan")))
    if args.kl_variant == "tail_bucket" and not math.isclose(
        cache_temperature, args.temperature, rel_tol=0.0, abs_tol=1e-12
    ):
        mismatches.append(
            f"temperature: cache={cache_temperature}, run={args.temperature}; tail_bucket requires an exact match"
        )
    elif not math.isclose(cache_temperature, args.temperature, rel_tol=0.0, abs_tol=1e-12):
        log.warning(
            "Cache coverage was measured at T=%g, while training uses T=%g. The legacy truncated loss "
            "can still run, but the logged coverage is not at the training temperature.",
            cache_temperature,
            args.temperature,
        )

    if args.train_traces:
        current_source = source_fingerprint(args.train_traces, args.input_format)
        if contract.get("source_sha256") != current_source:
            mismatches.append("source trace content fingerprint differs")
    if mismatches:
        raise ValueError("Cache contract mismatch; rebuild the cache:\n  - " + "\n  - ".join(mismatches))

    ds = hf_datasets.load_from_disk(os.path.join(args.cache_dir, "data"))
    required_columns = {
        "student_input_ids", "student_labels", "kl_student_pos", "topk_ids",
        "topk_logits", "coverage", "teacher_logsumexp",
    }
    missing = sorted(required_columns.difference(ds.column_names))
    if missing:
        raise ValueError(f"Cache data is incomplete; missing columns: {missing}. Rebuild it.")
    if len(ds) != int(contract.get("num_examples", -1)):
        raise ValueError(
            f"Cache row count is {len(ds)}, metadata says {contract.get('num_examples')}; rebuild it."
        )
    student_vocab = int(current_student.get("vocab_size", 0))
    if student_vocab > 0:
        for row_index, row in enumerate(ds):
            input_max = max(row["student_input_ids"], default=-1)
            target_max = max(row["topk_ids"], default=-1)
            if input_max >= student_vocab or target_max >= student_vocab:
                raise ValueError(
                    f"Cache row {row_index} contains student token id "
                    f"{max(input_max, target_max)} outside vocab size {student_vocab}; rebuild it."
                )
    log.info(
        "Loaded verified top-K cache: %d examples, K=%d, signature=%s.",
        len(ds), contract["top_k"], meta["signature"][:16],
    )
    return ds


def run_validate_cache(args) -> None:
    """Validate a cache without loading either language model's weights."""
    if not args.cache_dir:
        raise ValueError("--cache_dir is required for --mode validate_cache.")
    s_tok, t_tok = build_tokenizers(args)
    ds = load_cache(args, s_tok, t_tok)
    log.info("CACHE VALIDATION PASSED: %d examples at %s", len(ds), args.cache_dir)


# ==============================================================================
# SHARED HELPERS
# ==============================================================================
def build_tokenizers(args):
    t_name, s_name = tokenizer_names(args)
    t_revision = args.teacher_revision if t_name == args.teacher else None
    s_revision = args.student_revision if s_name == args.student else (
        args.teacher_revision if s_name == args.teacher else None
    )
    t_tok = load_tokenizer(t_name, revision=t_revision)
    s_tok = load_tokenizer(s_name, revision=s_revision) if s_name != t_name else t_tok
    log.info("Tokenizers: teacher='%s', student='%s'%s", t_name, s_name,
             "  (same object -- shared vocab)" if s_tok is t_tok else "")

    if args.vocab_mode == "shared":
        validate_shared_tokenizers(t_tok, s_tok, t_name, s_name)
    return s_tok, t_tok


def _config_vocab_size(name: str, tok, revision: Optional[str] = None) -> int:
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
        v = int(getattr(cfg, "vocab_size", 0))
        if v > 0:
            return v
    except Exception:
        pass
    return len(tok)


def student_vocab_size(args, s_tok) -> int:
    return _config_vocab_size(args.student, s_tok, args.student_revision)


def teacher_vocab_size(args, t_tok) -> int:
    return _config_vocab_size(args.teacher, t_tok, args.teacher_revision)


def make_scheduler(name, optimizer, num_warmup, num_training):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup, num_training)
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup, num_training)
    return get_constant_schedule_with_warmup(optimizer, num_warmup)


class EpochSeededSampler(Sampler):
    """Deterministic shuffle whose order can be reconstructed after a restart."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def _checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else -1


def latest_checkpoint(output_dir: str) -> Optional[str]:
    candidates = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if _checkpoint_step(path) >= 0 and os.path.isfile(os.path.join(path, CHECKPOINT_SUCCESS)):
            candidates.append(path)
    return max(candidates, key=_checkpoint_step) if candidates else None


def resolve_resume_checkpoint(args) -> Optional[str]:
    value = (args.resume_from_checkpoint or "none").strip()
    if value.lower() in ("none", "false", "off", "0"):
        return None
    if value.lower() == "auto":
        found = latest_checkpoint(args.output_dir)
        if found:
            log.info("Auto-resume selected %s.", found)
        return found
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, CHECKPOINT_SUCCESS)):
        raise FileNotFoundError(
            f"Resumable checkpoint is incomplete or missing: {path}. Expected {CHECKPOINT_SUCCESS}."
        )
    return path


def _torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch versions predating weights_only
        return torch.load(path, map_location=map_location)


def resume_metadata(checkpoint: str) -> Dict:
    path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint has no trainer_state.json: {checkpoint}")
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def checkpoint_contract(
    args,
    train_size: int,
    world: int,
    grad_accum: int,
    total_steps: int,
    data_signature: str,
    s_tok,
    t_tok,
) -> Dict:
    return {
        "teacher": args.teacher,
        "student": args.student,
        "teacher_revision": args.teacher_revision,
        "student_revision": args.student_revision,
        "teacher_tokenizer": args.teacher_tokenizer,
        "student_tokenizer": args.student_tokenizer,
        "vocab_mode": args.vocab_mode,
        "teacher_logits": args.teacher_logits,
        "data_signature": data_signature,
        "input_format": args.input_format,
        "dataset_split": args.dataset_split,
        "trace_colname": args.trace_colname,
        "problem_colname": args.problem_colname,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "kl_variant": args.kl_variant,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "per_device_batch_size": args.per_device_batch_size,
        "world_size": world,
        "grad_accum": grad_accum,
        "train_size": train_size,
        "total_steps": total_steps,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "model_dtype": args.model_dtype,
        "lora": args.lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "gradient_checkpointing": args.gradient_checkpointing,
        "teacher_model_contract": model_contract(args.teacher, args.teacher_revision),
        "student_model_contract": model_contract(args.student, args.student_revision),
        "teacher_tokenizer_fingerprint": tokenizer_fingerprint(t_tok),
        "student_tokenizer_fingerprint": tokenizer_fingerprint(s_tok),
    }


def validate_checkpoint_contract(saved: Dict, current: Dict) -> None:
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, run={value!r}"
        for key, value in current.items()
        if saved.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint contract mismatch; refusing an unsafe resume:\n  - " + "\n  - ".join(mismatches)
        )


def completed_run_contract(args, data_signature: str, s_tok, t_tok) -> Dict:
    """Identity fields that must match before an existing final model is accepted."""
    return {
        "teacher": model_contract(args.teacher, args.teacher_revision),
        "student": model_contract(args.student, args.student_revision),
        "teacher_tokenizer_fingerprint": tokenizer_fingerprint(t_tok),
        "student_tokenizer_fingerprint": tokenizer_fingerprint(s_tok),
        "data_signature": data_signature,
        "vocab_mode": args.vocab_mode,
        "teacher_logits": args.teacher_logits,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "kl_variant": args.kl_variant,
        "max_length": args.max_length,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "per_device_batch_size": args.per_device_batch_size,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "model_dtype": args.model_dtype,
        "lora": args.lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }


def validate_completed_run(final_dir: str, current: Dict) -> None:
    contract_path = os.path.join(final_dir, "training_contract.json")
    if not os.path.isfile(contract_path):
        raise FileNotFoundError(
            f"{final_dir} has {CHECKPOINT_SUCCESS} but no training_contract.json. "
            "Use a fresh --output_dir or inspect this incomplete/legacy result."
        )
    with open(contract_path, encoding="utf-8") as stream:
        saved = json.load(stream)
    mismatches = [
        f"{key}: completed={saved.get(key)!r}, run={value!r}"
        for key, value in current.items()
        if saved.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Completed output does not match this run; refusing to skip it:\n  - "
            + "\n  - ".join(mismatches)
        )


def _prune_checkpoints(output_dir: str, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = sorted(
        (p for p in glob.glob(os.path.join(output_dir, "checkpoint-*")) if _checkpoint_step(p) >= 0),
        key=_checkpoint_step,
    )
    for old in checkpoints[:-limit]:
        shutil.rmtree(old)
        log.info("Removed old checkpoint %s.", old)


def save_training_checkpoint(
    student,
    tokenizer,
    optimizer,
    scheduler,
    args,
    accelerator,
    global_step: int,
    next_epoch: int,
    next_batch: int,
    contract: Dict,
) -> str:
    """Save adapter/model, optimizer, scheduler, RNG, and exact dataloader position atomically."""
    final_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    temp_path = final_path + ".tmp"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)
    accelerator.wait_for_everyone()

    model = accelerator.unwrap_model(student)
    if accelerator.is_main_process:
        model_dir = os.path.join(temp_path, "adapter" if args.lora else "model")
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(os.path.join(temp_path, "tokenizer"))
        torch.save(optimizer.state_dict(), os.path.join(temp_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(temp_path, "scheduler.pt"))
        state = {
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "contract": contract,
        }
        with open(os.path.join(temp_path, "trainer_state.json"), "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)

    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(rng, os.path.join(temp_path, f"rng_rank{accelerator.process_index}.pt"))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        Path(os.path.join(temp_path, CHECKPOINT_SUCCESS)).write_text("ok\n", encoding="utf-8")
        if os.path.isdir(final_path):
            shutil.rmtree(final_path)
        os.replace(temp_path, final_path)
        _prune_checkpoints(args.output_dir, args.save_total_limit)
        log.info("Saved resumable checkpoint to %s.", final_path)
    accelerator.wait_for_everyone()
    return final_path


def restore_training_state(checkpoint, optimizer, scheduler, accelerator) -> Dict:
    state = resume_metadata(checkpoint)
    optimizer.load_state_dict(_torch_load(os.path.join(checkpoint, "optimizer.pt")))
    scheduler.load_state_dict(_torch_load(os.path.join(checkpoint, "scheduler.pt")))
    rng_path = os.path.join(checkpoint, f"rng_rank{accelerator.process_index}.pt")
    if not os.path.isfile(rng_path):
        raise FileNotFoundError(
            f"Checkpoint lacks RNG state for rank {accelerator.process_index}: {rng_path}"
        )
    rng = _torch_load(rng_path)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state


# ==============================================================================
# TRAINING
# ==============================================================================
def run_train(args, accelerator, grad_accum: int):
    from transformers import set_seed

    set_seed(args.seed)
    cached = args.teacher_logits == "cached"

    final_dir = os.path.join(args.output_dir, "final")
    final_success = os.path.join(final_dir, CHECKPOINT_SUCCESS)
    resume_path = resolve_resume_checkpoint(args)
    resume_info = resume_metadata(resume_path) if resume_path else None

    s_tok, t_tok = build_tokenizers(args)
    s_vocab_cfg = student_vocab_size(args, s_tok)

    # ---- data ----------------------------------------------------------
    if cached:
        cache_ds = load_cache(args, s_tok, t_tok)
        train_ds = CachedDataset(cache_ds, args.top_k)
        collator = Collator(s_tok.pad_token_id, t_tok.pad_token_id, cached=True)
        with open(os.path.join(args.cache_dir, "cache_meta.json"), encoding="utf-8") as stream:
            data_signature = json.load(stream)["signature"]
    else:
        if not args.train_traces:
            raise ValueError("--train_traces is required unless --teacher_logits cached.")
        pairs = load_pairs(args.train_traces, args, args.max_train_samples)
        train_examples = encode_all(pairs, args, s_tok, t_tok)
        validate_student_token_ids(train_examples, s_vocab_cfg)
        train_ds = OnlineDataset(train_examples)
        collator = Collator(s_tok.pad_token_id, t_tok.pad_token_id, cached=False)
        data_signature = source_fingerprint(args.train_traces, args.input_format)

    final_contract = completed_run_contract(args, data_signature, s_tok, t_tok)
    if (args.resume_from_checkpoint or "").lower() == "auto" and os.path.isfile(final_success):
        validate_completed_run(final_dir, final_contract)
        log.info("Verified completed run at %s; nothing to do.", final_dir)
        return

    eval_ds = None
    if args.eval_traces:
        eval_pairs = load_pairs(args.eval_traces, args, args.max_eval_samples)
        eval_examples = encode_all(eval_pairs, args, s_tok, t_tok)
        validate_student_token_ids(eval_examples, s_vocab_cfg)
        eval_ds = OnlineDataset(eval_examples)

    # ---- models --------------------------------------------------------
    attn = resolve_attn(args.attn_implementation)
    student_source = (
        os.path.join(resume_path, "model") if resume_path and not args.lora else args.student
    )
    student_revision = None if resume_path and not args.lora else args.student_revision
    student = load_causal_lm(
        student_source,
        student_dtype_for(args),
        attn,
        trainable=True,
        revision=student_revision,
    )
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
    if resume_path and args.lora:
        from peft import PeftModel

        student = PeftModel.from_pretrained(
            student, os.path.join(resume_path, "adapter"), is_trainable=True
        )
        log.info("Loaded trainable LoRA adapter from %s.", resume_path)
    else:
        student = maybe_wrap_lora(student, args)

    teacher = None
    vocab_map = None
    if not cached:
        teacher = load_causal_lm(
            args.teacher,
            TEACHER_DTYPES[args.teacher_dtype],
            attn,
            trainable=False,
            revision=args.teacher_revision,
        )
        teacher.to(accelerator.device)
        if args.vocab_mode == "cross":
            vocab_map = build_vocab_map(t_tok, s_tok, teacher_vocab_size(args, t_tok),
                                        s_vocab_cfg).to(accelerator.device)

    train_source = "cached" if cached else "online"
    # With a cached teacher the held-out set has no cached logits, so eval falls
    # back to hard-label CE (still a useful distillation-progress signal).
    eval_source = "none" if cached else "online"
    if cached and eval_ds is not None:
        log.warning("--teacher_logits cached: held-out evaluation reports CE only. "
                    "Precompute a cache for the held-out set if you want its KL too.")

    # ---- optimiser / schedule -----------------------------------------
    train_sampler = EpochSeededSampler(train_ds, args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.per_device_batch_size, sampler=train_sampler,
                          collate_fn=collator, num_workers=args.num_workers, drop_last=False)
    eval_dl = DataLoader(eval_ds, batch_size=args.per_device_batch_size, shuffle=False,
                         collate_fn=Collator(s_tok.pad_token_id, t_tok.pad_token_id, cached=False),
                         num_workers=args.num_workers) if eval_ds is not None else None

    decay, no_decay = [], []
    for n_, p_ in student.named_parameters():
        if not p_.requires_grad:
            continue
        (no_decay if (p_.ndim == 1 or n_.endswith(".bias")) else decay).append(p_)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )

    # The dataloader is sharded across processes by prepare(), so the schedule
    # length has to be derived from the *prepared* loader, not the raw one.
    student, optimizer, train_dl = accelerator.prepare(student, optimizer, train_dl)
    if eval_dl is not None:
        eval_dl = accelerator.prepare(eval_dl)

    steps_per_epoch = max(1, math.ceil(len(train_dl) / grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.num_epochs))
    # The scheduler is deliberately NOT passed to prepare(): a prepared scheduler
    # advances once per process per call, which makes the schedule length
    # world-size dependent. Stepping it only on sync_gradients gives exactly
    # `total_steps` LR updates regardless of how many GPUs are used.
    scheduler = make_scheduler(args.lr_scheduler_type, optimizer,
                               int(total_steps * args.warmup_ratio), total_steps)
    contract = checkpoint_contract(
        args,
        len(train_ds),
        accelerator.num_processes,
        grad_accum,
        total_steps,
        data_signature,
        s_tok,
        t_tok,
    )

    global_step = 0
    start_epoch = 0
    start_batch = 0
    if resume_path:
        validate_checkpoint_contract(resume_info.get("contract", {}), contract)
        restored = restore_training_state(resume_path, optimizer, scheduler, accelerator)
        global_step = int(restored["global_step"])
        start_epoch = int(restored["next_epoch"])
        start_batch = int(restored["next_batch"])
        if global_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {global_step} is already at/after total_steps={total_steps}, "
                "but the final completion marker is absent. Inspect the checkpoint before continuing."
            )
        log.info(
            "Resumed optimizer/scheduler/RNG at step %d; epoch=%d, next prepared batch=%d.",
            global_step,
            start_epoch,
            start_batch,
        )

    if accelerator.is_main_process:
        log.info("=" * 78)
        log.info("Soft top-K distillation | K=%d  T=%.2f  alpha=%.2f  variant=%s",
                 args.top_k, args.temperature, args.alpha, args.kl_variant)
        log.info("teacher=%s (%s)  student=%s", args.teacher,
                 "cached top-K" if cached else "online", args.student)
        log.info("vocab_mode=%s | examples=%d | grad_accum=%d | optimizer steps=%d",
                 args.vocab_mode, len(train_ds), grad_accum, total_steps)
        log.info("=" * 78)

    # ---- loop ----------------------------------------------------------
    running = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "cov": 0.0, "n": 0}
    student.train()
    done = False

    for epoch in range(start_epoch, math.ceil(args.num_epochs)):
        if done:
            break
        train_sampler.set_epoch(epoch)
        if hasattr(train_dl, "set_epoch"):
            train_dl.set_epoch(epoch)
        batch_offset = start_batch if epoch == start_epoch else 0
        epoch_dl = (
            accelerator.skip_first_batches(train_dl, batch_offset) if batch_offset else train_dl
        )
        for batch_index, batch in enumerate(epoch_dl, start=batch_offset):
            with accelerator.accumulate(student):
                loss, parts = compute_batch_loss(
                    student, teacher, batch, args, vocab_map, s_vocab_cfg, accelerator, train_source
                )
                accelerator.backward(loss)
                did_step = accelerator.sync_gradients
                if did_step and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                optimizer.step()
                if did_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            for k in ("loss", "kl", "ce", "cov"):
                running[k] += parts[k]
            running["n"] += 1

            if did_step:
                global_step += 1
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    n = max(1, running["n"])
                    log.info(
                        "step %d/%d | loss %.4f | kl %.4f | ce %.4f | topK cov %.3f | lr %.2e",
                        global_step, total_steps, running["loss"] / n, running["kl"] / n,
                        running["ce"] / n, running["cov"] / n, scheduler.get_last_lr()[0],
                    )
                    running = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "cov": 0.0, "n": 0}
                if args.eval_steps and eval_dl is not None and global_step % args.eval_steps == 0:
                    evaluate(student, teacher, eval_dl, args, vocab_map, s_vocab_cfg,
                             accelerator, eval_source)
                if args.save_steps and global_step % args.save_steps == 0:
                    next_epoch = epoch + 1 if batch_index + 1 >= len(train_dl) else epoch
                    next_batch = 0 if next_epoch != epoch else batch_index + 1
                    save_training_checkpoint(
                        student, s_tok, optimizer, scheduler, args, accelerator,
                        global_step, next_epoch, next_batch, contract,
                    )
                if args.stop_after_steps and global_step >= args.stop_after_steps:
                    next_epoch = epoch + 1 if batch_index + 1 >= len(train_dl) else epoch
                    next_batch = 0 if next_epoch != epoch else batch_index + 1
                    if not (args.save_steps and global_step % args.save_steps == 0):
                        save_training_checkpoint(
                            student, s_tok, optimizer, scheduler, args, accelerator,
                            global_step, next_epoch, next_batch, contract,
                        )
                    log.info("Stopped intentionally after %d optimizer steps.", global_step)
                    return
                if global_step >= total_steps:
                    done = True
                    break
        start_batch = 0

    if eval_dl is not None:
        evaluate(student, teacher, eval_dl, args, vocab_map, s_vocab_cfg, accelerator, eval_source)
    save_model(
        student,
        s_tok,
        args,
        accelerator,
        final_dir,
        final=True,
        training_contract=final_contract,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        log.info("Done. Student saved to %s", os.path.join(args.output_dir, "final"))


def compute_batch_loss(student, teacher, batch, args, vocab_map, s_vocab_cfg, accelerator,
                       teacher_source: str):
    """teacher_source: 'cached' | 'online' | 'none' (CE only)."""
    device = accelerator.device
    s_ids = batch["student_input_ids"].to(device)
    s_mask = batch["student_attention_mask"].to(device)
    s_lab = batch["student_labels"].to(device)
    kl_b = batch["kl_batch_idx"].to(device)
    kl_s = batch["kl_student_pos"].to(device)

    out = student(input_ids=s_ids, attention_mask=s_mask, use_cache=False)
    logits = out.logits                                # [B, L, V], logits[p] predicts p+1
    B, L, Vs = logits.shape

    # Target for position p is token p+1; the last position predicts nothing.
    shift_targets = torch.full_like(s_lab, IGNORE_INDEX)
    shift_targets[:, :-1] = s_lab[:, 1:]
    ce = hard_ce_loss(logits, shift_targets)

    coverage = 0.0
    teacher_lse = None
    if teacher_source == "none":
        kl = logits.sum() * 0.0
    else:
        if teacher_source == "cached":
            topk_ids = batch["topk_ids"].to(device)
            topk_logits = batch["topk_logits"].to(device).float()
            # -inf does not survive Arrow; the sentinel is restored here.
            topk_logits = torch.where(topk_logits <= -1e29,
                                      torch.full_like(topk_logits, float("-inf")), topk_logits)
            cov_t = batch["coverage"].to(device) if "coverage" in batch else None
            teacher_lse = batch["teacher_logsumexp"].to(device).float()
        else:
            topk_ids, topk_logits, cov_t, teacher_lse = teacher_topk(
                teacher,
                batch["teacher_input_ids"].to(device),
                batch["teacher_attention_mask"].to(device),
                kl_b,
                batch["kl_teacher_pos"].to(device),
                args.top_k,
                vocab_map,
                s_vocab_cfg,
                temperature=args.temperature,
                chunk_size=args.teacher_topk_chunk_size,
            )
        if cov_t is not None and cov_t.numel():
            coverage = float(cov_t.mean())

        # Guard: every id must be addressable in the student's output layer.
        if topk_ids.numel():
            bad = topk_ids >= Vs
            if bool(bad.any()):
                topk_ids = topk_ids.masked_fill(bad, 0)
                topk_logits = topk_logits.masked_fill(bad, float("-inf"))

        # Guard: a KL site must sit inside this batch's padded length.
        if kl_s.numel():
            in_range = kl_s < (L - 1)
            if not bool(in_range.all()):
                kl_b, kl_s = kl_b[in_range], kl_s[in_range]
                topk_ids, topk_logits = topk_ids[in_range], topk_logits[in_range]
                if teacher_lse is not None:
                    teacher_lse = teacher_lse[in_range]

        flat_idx = kl_b * L + kl_s
        kl = topk_kl_loss(logits, flat_idx, topk_ids, topk_logits,
                          args.temperature, args.kl_variant, args.kl_chunk_size,
                          teacher_logsumexp=teacher_lse)

    loss = args.alpha * (args.temperature ** 2) * kl + (1.0 - args.alpha) * ce
    parts = {"loss": float(loss.detach()), "kl": float(kl.detach()),
             "ce": float(ce.detach()), "cov": coverage}
    return loss, parts


@torch.no_grad()
def evaluate(student, teacher, eval_dl, args, vocab_map, s_vocab_cfg, accelerator, teacher_source):
    student.eval()
    # Summed on-device so the numbers can be reduced across processes: each rank
    # only sees its own shard of the held-out set.
    tot = torch.zeros(4, device=accelerator.device, dtype=torch.float32)
    for batch in eval_dl:
        _, parts = compute_batch_loss(student, teacher, batch, args, vocab_map,
                                      s_vocab_cfg, accelerator, teacher_source)
        tot += torch.tensor([parts["loss"], parts["kl"], parts["ce"], 1.0],
                            device=accelerator.device, dtype=torch.float32)
    tot = accelerator.reduce(tot, reduction="sum")
    n = max(1.0, float(tot[3]))
    if accelerator.is_main_process:
        log.info("[eval] loss %.4f | kl %.4f | ce %.4f | ppl %.3f%s",
                 float(tot[0]) / n, float(tot[1]) / n, float(tot[2]) / n,
                 math.exp(min(20.0, float(tot[2]) / n)),
                 "  (CE only: teacher not loaded)" if teacher_source == "none" else "")
    student.train()


def save_model(
    student,
    tokenizer,
    args,
    accelerator,
    path,
    final: bool,
    training_contract: Optional[Dict] = None,
):
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(student)
    if accelerator.is_main_process:
        save_path = path
        if final:
            save_path = path + ".tmp"
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        to_save = model
        # merge_and_unload() folds the adapter into the base weights in place, so
        # it is only safe once training is over. Mid-run checkpoints keep the
        # adapter separate.
        if final and args.lora and args.merge_lora_on_save and hasattr(model, "merge_and_unload"):
            to_save = model.merge_and_unload()
        if hasattr(to_save, "config"):
            to_save.config.use_cache = True
        to_save.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, "distill_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        if final:
            if training_contract is None:
                raise ValueError("A final model requires its training contract.")
            with open(os.path.join(save_path, "training_contract.json"), "w", encoding="utf-8") as f:
                json.dump(training_contract, f, indent=2, sort_keys=True)
            Path(os.path.join(save_path, CHECKPOINT_SUCCESS)).write_text("ok\n", encoding="utf-8")
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.replace(save_path, path)
        log.info("Saved to %s", path)
    accelerator.wait_for_everyone()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1.")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")
    if args.kl_variant == "tail_bucket" and args.vocab_mode != "shared":
        raise ValueError(
            "--kl_variant tail_bucket currently requires --vocab_mode shared because cross-tokenizer "
            "maps can contain duplicate or missing token identities. Use --kl_variant student_full "
            "for the explicitly approximate cross-tokenizer experiment."
        )
    if args.max_length < 2:
        raise ValueError("--max_length must be >= 2.")
    if args.teacher_topk_chunk_size < 1 or args.kl_chunk_size < 1:
        raise ValueError("--teacher_topk_chunk_size and --kl_chunk_size must be >= 1.")
    if args.batch_size < 1 or args.per_device_batch_size < 1 or args.precompute_batch_size < 1:
        raise ValueError("All batch sizes must be >= 1.")
    if args.num_epochs <= 0:
        raise ValueError("--num_epochs must be > 0.")
    if args.save_steps < 0 or args.save_total_limit < 0:
        raise ValueError("--save_steps and --save_total_limit cannot be negative.")
    if args.mode == "train" and args.teacher_logits == "cached" and not args.cache_dir:
        raise ValueError("--cache_dir is required with --teacher_logits cached.")
    if args.mode == "validate_cache" and not args.cache_dir:
        raise ValueError("--cache_dir is required with --mode validate_cache.")

    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.cache_dir:
        args.cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    if not args.teacher_revision or not args.student_revision:
        log.warning(
            "At least one model revision is unpinned. Cache metadata records resolved commits, but "
            "published runs should pass --teacher_revision and --student_revision explicitly."
        )

    if args.mode == "validate_cache":
        run_validate_cache(args)
        return

    from accelerate import Accelerator

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Read the launcher-provided world size without constructing PartialState.
    # Constructing PartialState first can lock Accelerate to CUDA and then make
    # a later CPU Accelerator fail in torchrun-based verification.
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError(f"Invalid WORLD_SIZE={os.environ.get('WORLD_SIZE')!r}.") from exc
    if world < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world}.")
    grad_accum = 1
    if args.mode == "train":
        per_step = args.per_device_batch_size * world
        if args.batch_size % per_step != 0:
            raise ValueError(
                f"--batch_size ({args.batch_size}) must be divisible by "
                f"per_device_batch_size * num_processes ({args.per_device_batch_size} * {world} "
                f"= {per_step})."
            )
        grad_accum = args.batch_size // per_step

    cpu_requested = os.environ.get("ACCELERATE_USE_CPU", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        cpu=cpu_requested,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "precompute":
        run_precompute(args, accelerator)
    else:
        run_train(args, accelerator, grad_accum)


if __name__ == "__main__":
    main()

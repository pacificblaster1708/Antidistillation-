# -*- coding: utf-8 -*-
"""Dataset loading, prompt construction and answer checking."""

from __future__ import annotations

import os
import re
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units "
    "within the brackets and fully evaluate arithmetic expressions.\n"
)

ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"

_MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #
def _split_70_30(ds: Dataset, split: str, seed: int = 42) -> Dataset:
    ds = ds.shuffle(seed=seed)
    cut = int(len(ds) * 0.7)
    if split == "train":
        return ds.select(range(cut))
    if split == "holdout":
        return ds.select(range(cut, len(ds)))
    raise ValueError(split)


def _load_gsm8k(split: str) -> Dataset:
    if split == "test":
        ds = load_dataset("madrylab/gsm8k-platinum", split="test")
        return ds.rename_columns({"question": "problem", "answer": "solution"})
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.rename_columns({"question": "problem", "answer": "solution"})
    return _split_70_30(ds, split)


def _load_hendrycks_math(split: str) -> Dataset:
    hf_split = "test" if split == "test" else "train"
    parts = [load_dataset("EleutherAI/hendrycks_math", s, split=hf_split) for s in _MATH_SUBSETS]
    ds = concatenate_datasets(parts)
    if split == "test":
        return ds
    return _split_70_30(ds, split)


def _mmlu_to_math_format(ds: Dataset) -> Dataset:
    def transform(ex):
        prompt = ex["question"] + "\n"
        prompt += "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ex["choices"]))
        return {"problem": prompt, "solution": "\\boxed{" + chr(65 + ex["answer"]) + "}"}

    return ds.map(transform, remove_columns=ds.column_names, desc="Formatting MMLU")


def _load_mmlu(split: str) -> Dataset:
    if split == "test":
        return _mmlu_to_math_format(load_dataset("cais/mmlu", "all", split="test"))
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    return _mmlu_to_math_format(_split_70_30(ds, split))


def _load_local(spec: str, split: str) -> Dataset:
    """local:<path>  -- a saved_to_disk dir, a *.jsonl file, or a dir of <split>.jsonl."""
    path = spec.split(":", 1)[1]
    if os.path.isfile(path):
        ds = load_dataset("json", data_files=path, split="train")
    elif os.path.isfile(os.path.join(path, f"{split}.jsonl")):
        ds = load_dataset("json", data_files=os.path.join(path, f"{split}.jsonl"), split="train")
    elif os.path.isdir(os.path.join(path, split)):
        ds = load_from_disk(os.path.join(path, split))
    else:
        ds = load_from_disk(path)
    missing = {"problem", "solution"} - set(ds.column_names)
    if missing:
        raise ValueError(f"local dataset {path!r} is missing column(s) {sorted(missing)}")
    return ds


def load_split(dataset: str, split: str, max_samples: Optional[int] = None) -> Dataset:
    """split is one of 'train' / 'holdout' / 'test'."""
    if split not in {"train", "holdout", "test"}:
        raise ValueError(f"split must be train/holdout/test, got {split!r}")
    if dataset.startswith("local:"):
        ds = _load_local(dataset, split)
    elif dataset == "gsm8k":
        ds = _load_gsm8k(split)
    elif dataset in {"hendrycks_math", "math"}:
        ds = _load_hendrycks_math(split)
    elif dataset == "mmlu":
        ds = _load_mmlu(split)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    ds = ds.select_columns(["problem", "solution"])
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def chat_messages(problem: str, response: Optional[str] = None) -> list:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip() + "\n"},
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": response.strip()})
    return msgs


# --------------------------------------------------------------------------- #
# answer checking
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _fallback_equal(prediction: str, solution: str) -> bool:
    """Last-resort comparison used when math_verify cannot parse either side."""
    def final(text: str) -> Optional[str]:
        hits = _BOXED.findall(text)
        if hits:
            return hits[-1].strip()
        hits = re.findall(r"####\s*(.+)", text)          # gsm8k gold format
        if hits:
            return hits[-1].strip()
        return None

    a, b = final(prediction), final(solution)
    if a is None or b is None:
        return False
    norm = lambda s: re.sub(r"[\s,$]|\\!|\\,|\\left|\\right", "", s).rstrip(".")
    if norm(a) == norm(b):
        return True
    try:
        return abs(float(norm(a)) - float(norm(b))) < 1e-6
    except ValueError:
        return False


def is_correct(prediction: str, solution: str) -> bool:
    """True when `prediction` ends in an answer equivalent to `solution`."""
    if not prediction:
        return False
    try:
        from math_verify import parse, verify

        gold = parse(solution)
        candidates = [prediction]
        if ANSWER_FORCE_STRING in prediction:
            # The forcing string splits the trace; the answer may be on either
            # side of it depending on how the model finished.
            parts = prediction.split(ANSWER_FORCE_STRING)
            candidates += [ANSWER_FORCE_STRING.join(parts[:-1]), parts[-1]]
        for cand in candidates:
            try:
                if verify(gold, parse(cand)):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return _fallback_equal(prediction, solution)

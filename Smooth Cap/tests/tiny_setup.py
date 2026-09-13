# -*- coding: utf-8 -*-
"""
Build tiny offline fixtures: two tokenizer families, three small models and a
local dataset. Everything is constructed from scratch -- no network, no
downloads -- so the test suite exercises the real code paths in seconds.
"""

from __future__ import annotations

import json
import os
import random


import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          Qwen2Config, Qwen2ForCausalLM)

# Two different chat formats, so the student really has to re-template the text.
TEACHER_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ m['content'] }}"
    "{% elif m['role'] == 'user' %}{{ '<|user|>' + m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ '<|assistant|>' + m['content'] + eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|><think>' }}{% endif %}"
)
STUDENT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ '[SYS]' + m['content'] }}"
    "{% elif m['role'] == 'user' %}{{ '[Q]' + m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ '[A]' + m['content'] + eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '[A]' }}{% endif %}"
)


def build_tokenizer(path: str, template: str, extra_specials, seed_marker: str):
    """A byte-level tokenizer with no merges: deterministic, lossless, tiny."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token=None,
        bos_token=None,
        chat_template=template,
    )
    tok.add_special_tokens({"additional_special_tokens": list(extra_specials)})
    os.makedirs(path, exist_ok=True)
    tok.save_pretrained(path)
    return tok


def build_models(root: str, teacher_vocab: int, proxy_vocab: int, student_vocab: int):
    torch.manual_seed(0)
    common = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=2048, tie_word_embeddings=False)

    # Teacher and proxy student are the same family (they share a tokenizer);
    # their configured vocab sizes deliberately differ from len(tokenizer) so
    # align_vocab() has real work to do.
    teacher = Qwen2ForCausalLM(Qwen2Config(vocab_size=teacher_vocab, **common))
    proxy = Qwen2ForCausalLM(Qwen2Config(vocab_size=proxy_vocab, **common))
    # The attacker's student is a different architecture with its own tokenizer.
    student = LlamaForCausalLM(LlamaConfig(vocab_size=student_vocab, **common))

    paths = {}
    for name, model in [("teacher", teacher), ("proxy_student", proxy), ("student", student)]:
        path = os.path.join(root, name)
        model.save_pretrained(path, safe_serialization=True)
        paths[name] = path
    return paths


PROBLEMS = [
    ("Tom has 3 apples and buys 4 more. How many apples?", "7"),
    ("A box holds 5 pens. How many pens in 3 boxes?", "15"),
    ("What is 12 minus 5?", "7"),
    ("Ann runs 2 km each day for 6 days. Total km?", "12"),
    ("Half of 18 is what number?", "9"),
    ("What is 4 times 4?", "16"),
    ("Sam had 10 sweets and ate 3. How many left?", "7"),
    ("Add 25 and 17.", "42"),
    ("What is 100 divided by 4?", "25"),
    ("Three friends split 21 marbles evenly. Each gets?", "7"),
    ("What is 9 plus 8?", "17"),
    ("A shelf has 6 rows of 3 books. How many books?", "18"),
    ("Subtract 14 from 30.", "16"),
    ("What is 7 times 5?", "35"),
    ("A jug holds 2 litres. How much do 8 jugs hold?", "16"),
    ("What is 45 minus 20?", "25"),
    ("Double 13.", "26"),
    ("What is 60 divided by 5?", "12"),
    ("Two packs of 9 stickers. Total stickers?", "18"),
    ("What is 33 plus 11?", "44"),
]


def build_dataset(path: str, sizes=(8, 6, 6)) -> str:
    os.makedirs(path, exist_ok=True)
    rng = random.Random(0)
    pool = list(PROBLEMS)
    rng.shuffle(pool)
    cursor = 0
    for split, size in zip(("train", "holdout", "test"), sizes):
        rows = []
        for _ in range(size):
            problem, answer = pool[cursor % len(pool)]
            cursor += 1
            rows.append({"problem": problem, "solution": "\\boxed{" + answer + "}"})
        with open(os.path.join(path, f"{split}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return "local:" + path


def build_all(root: str) -> dict:
    os.makedirs(root, exist_ok=True)
    teacher_tok_dir = os.path.join(root, "tok_teacher")
    student_tok_dir = os.path.join(root, "tok_student")

    teacher_tok = build_tokenizer(teacher_tok_dir, TEACHER_TEMPLATE,
                                  ["<|user|>", "<|assistant|>", "<think>", "</think>"], "T")
    student_tok = build_tokenizer(student_tok_dir, STUDENT_TEMPLATE,
                                  ["[SYS]", "[Q]", "[A]"], "S")

    paths = build_models(
        root,
        teacher_vocab=len(teacher_tok) + 7,     # intentionally too wide
        proxy_vocab=len(teacher_tok) + 3,       # intentionally too wide, differently
        student_vocab=len(student_tok),
    )
    # Models load their tokenizer from their own directory in some stages.
    for key, tok in [("teacher", teacher_tok), ("proxy_student", teacher_tok),
                     ("student", student_tok)]:
        tok.save_pretrained(paths[key])

    dataset = build_dataset(os.path.join(root, "data"))
    return {
        "root": root,
        "teacher": paths["teacher"],
        "proxy_student": paths["proxy_student"],
        "student": paths["student"],
        "teacher_tokenizer": teacher_tok_dir,
        "student_tokenizer": student_tok_dir,
        "dataset": dataset,
        "teacher_vocab": len(teacher_tok),
        "student_vocab": len(student_tok),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(build_all(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ads_fixtures"), indent=2))

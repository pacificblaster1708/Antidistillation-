# -*- coding: utf-8 -*-
"""
Unit tests for the parts that are easy to get subtly wrong.

Run:  python tests/test_units.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ads import (ADSLogitsProcessor, CompletionOnlyCollator, IncrementalLM, align_vocab,
                 apply_perturbation, count_label_tokens, load_causal_lm, load_tokenizer,
                 normalize_param_name)
from config import Config
from data import ANSWER_FORCE_STRING, is_correct
from tests.tiny_setup import build_all

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


# --------------------------------------------------------------------------- #
def test_config_modes():
    print("\n[config] mode selection")
    cfg = Config.load(["--ads=false", "--normal=true"])
    check("NORMAL selected", cfg.mode == "normal" and not cfg.ads)
    check("NORMAL forces lam=0", cfg.lam == 0.0 and cfg.eps == 0.0)

    cfg = Config.load(["--ads=true", "--normal=false", "--lam=0.2", "--eps=0.01"])
    check("ADS selected", cfg.mode == "ads" and cfg.ads)
    check("ADS keeps lam", cfg.lam == 0.2 and cfg.eps == 0.01)

    for bad in (["--ads=true", "--normal=true"], ["--ads=false", "--normal=false"]):
        try:
            Config.load(bad)
            check(f"rejects {bad}", False)
        except SystemExit:
            check(f"rejects {bad}", True)

    os.environ["ADS"], os.environ["NORMAL"] = "true", "false"
    try:
        check("reads env vars", Config.load([]).ads is True)
    finally:
        del os.environ["ADS"], os.environ["NORMAL"]


# --------------------------------------------------------------------------- #
def test_collator():
    print("\n[collator] completion-only masking")
    coll = CompletionOnlyCollator(pad_token_id=99, pad_to_multiple_of=None)
    batch = coll([{"input_ids": [1, 2, 3, 4, 5], "prompt_len": 2},
                  {"input_ids": [6, 7, 8], "prompt_len": 1}])
    check("padded to longest", list(batch["input_ids"].shape) == [2, 5])
    check("pad id used", batch["input_ids"][1].tolist() == [6, 7, 8, 99, 99])
    check("attention mask", batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0])
    check("prompt masked out", batch["labels"][0].tolist() == [-100, -100, 3, 4, 5])
    check("padding masked out", batch["labels"][1].tolist() == [-100, 7, 8, -100, -100])
    # after the causal shift, row0 supervises {3,4,5} and row1 supervises {7,8}
    check("label token count", count_label_tokens(batch["labels"]) == 5,
          f"got {count_label_tokens(batch['labels'])}")

    coll8 = CompletionOnlyCollator(pad_token_id=0, pad_to_multiple_of=8)
    b8 = coll8([{"input_ids": [1, 2, 3], "prompt_len": 1}])
    check("pads to multiple of 8", b8["input_ids"].shape[1] == 8)


# --------------------------------------------------------------------------- #
def test_answer_checking():
    print("\n[data] answer checking")
    check("boxed match", is_correct("so the answer is \\boxed{42}", "\\boxed{42}"))
    check("boxed mismatch", not is_correct("\\boxed{41}", "\\boxed{42}"))
    check("gsm8k gold format", is_correct("\\boxed{18}", "some working\n#### 18"))
    check("answer-forced trace",
          is_correct("thinking..." + ANSWER_FORCE_STRING + "7}\\]", "\\boxed{7}"))
    check("empty prediction", not is_correct("", "\\boxed{1}"))


# --------------------------------------------------------------------------- #
def test_vocab_alignment(fx):
    print("\n[ads] vocabulary alignment")
    tok = load_tokenizer(fx["teacher_tokenizer"])
    teacher = load_causal_lm(fx["teacher"], torch.float32, "eager")
    proxy = load_causal_lm(fx["proxy_student"], torch.float32, "eager")
    before = (teacher.get_input_embeddings().weight.shape[0],
              proxy.get_input_embeddings().weight.shape[0])
    check("fixtures start mismatched", before[0] != before[1], str(before))
    align_vocab(tok, teacher, proxy)
    after = (teacher.get_input_embeddings().weight.shape[0],
             proxy.get_input_embeddings().weight.shape[0])
    check("both resized to len(tokenizer)", after == (len(tok), len(tok)), str(after))
    return tok, teacher, proxy


# --------------------------------------------------------------------------- #
def test_incremental_cache(tok, proxy):
    print("\n[ads] incremental decoding matches a full forward")
    proxy.eval()
    ids = torch.tensor([[5, 9, 3, 7, 11, 2], [1, 4, 8, 6, 10, 12]])
    mask = torch.ones_like(ids)
    mask[1, 0] = 0                                        # a left-padded row

    with torch.inference_mode():
        reference = proxy(input_ids=ids, attention_mask=mask).logits[:, -1, :]

    inc = IncrementalLM(proxy)
    for step in range(3, ids.shape[1] + 1):               # prefix, then token by token
        got = inc.next_logits(ids[:, :step], mask[:, :step])
    check("cached == uncached", torch.allclose(got, reference, atol=1e-4),
          f"max delta {float((got - reference).abs().max()):.2e}")

    inc.reset()
    fresh = inc.next_logits(ids, mask)
    check("reset forces recompute", torch.allclose(fresh, reference, atol=1e-5))

    # A shorter batch after a longer one must not reuse the stale cache.
    inc.reset()
    inc.next_logits(ids, mask)
    small = inc.next_logits(ids[:1, :4], mask[:1, :4])
    with torch.inference_mode():
        small_ref = proxy(input_ids=ids[:1, :4], attention_mask=mask[:1, :4]).logits[:, -1, :]
    check("batch-size change invalidates cache", torch.allclose(small, small_ref, atol=1e-5))


# --------------------------------------------------------------------------- #
def test_ads_term(tok, teacher, proxy_path):
    print("\n[ads] the antidistillation term")
    lam, eps = 0.3, 1e-2
    plus_model = load_causal_lm(proxy_path, torch.float32, "eager")
    minus_model = load_causal_lm(proxy_path, torch.float32, "eager")
    align_vocab(tok, plus_model, minus_model)

    grads = {normalize_param_name(n): torch.randn_like(p) * 0.01
             for n, p in plus_model.named_parameters()}
    stats = apply_perturbation(plus_model, grads, +eps)
    apply_perturbation(minus_model, grads, -eps)
    check("perturbation touched every tensor", stats["num_tensors"] == len(grads))

    a = dict(plus_model.named_parameters())["model.layers.0.mlp.up_proj.weight"]
    b = dict(minus_model.named_parameters())["model.layers.0.mlp.up_proj.weight"]
    expected = 2 * eps * grads["model.layers.0.mlp.up_proj.weight"]
    check("plus/minus differ by 2*eps*g", torch.allclose(a - b, expected, atol=1e-6))

    plus_model.eval(); minus_model.eval()
    plus, minus = IncrementalLM(plus_model), IncrementalLM(minus_model)

    ids = torch.tensor([[5, 9, 3, 7, 11, 2]])
    mask = torch.ones_like(ids)
    proc = ADSLogitsProcessor(plus, minus, lam, eps, mask)
    scores = torch.zeros(1, len(tok))
    out = proc(ids, scores.clone())

    with torch.inference_mode():
        up = plus_model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
        down = minus_model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
    manual = scores + (lam / (2 * eps)) * (up - down)
    check("matches lam/(2*eps)*(f(+) - f(-))", torch.allclose(out, manual, atol=1e-4),
          f"max delta {float((out - manual).abs().max()):.2e}")
    check("term is non-trivial", float(out.abs().max()) > 1e-3,
          f"max |term| {float(out.abs().max()):.3e}")

    # A mismatched vocabulary must fail loudly rather than broadcast silently.
    proc.plus.reset(); proc.minus.reset()
    try:
        proc(ids, torch.zeros(1, len(tok) + 5))
        check("rejects vocab mismatch", False)
    except RuntimeError:
        check("rejects vocab mismatch", True)


# --------------------------------------------------------------------------- #
def test_gradient_sign(fx):
    """+eps*g must increase the holdout loss: that is what makes ADS 'anti'."""
    print("\n[grads] saved gradient points uphill in loss")
    tok = load_tokenizer(fx["teacher_tokenizer"], padding_side="right")
    model = load_causal_lm(fx["proxy_student"], torch.float32, "eager", use_cache=False)
    align_vocab(tok, model)

    ids = torch.randint(0, len(tok), (4, 24))
    labels = ids.clone()
    labels[:, :8] = -100
    mask = torch.ones_like(ids)

    model.train()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    base = float(out.loss)
    out.loss.backward()
    grads = {normalize_param_name(n): p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    def loss_at(scale):
        probe = load_causal_lm(fx["proxy_student"], torch.float32, "eager", use_cache=False)
        align_vocab(tok, probe)
        apply_perturbation(probe, grads, scale)
        probe.eval()
        with torch.no_grad():
            return float(probe(input_ids=ids, attention_mask=mask, labels=labels).loss)

    eps = 1e-3
    up, down = loss_at(+eps), loss_at(-eps)
    check("loss(+eps*g) > loss(theta)", up > base, f"{up:.6f} > {base:.6f}")
    check("loss(-eps*g) < loss(theta)", down < base, f"{down:.6f} < {base:.6f}")


# --------------------------------------------------------------------------- #
def main():
    fx = build_all("/tmp/ads_fixtures")
    test_config_modes()
    test_collator()
    test_answer_checking()
    tok, teacher, proxy = test_vocab_alignment(fx)
    test_incremental_cache(tok, proxy)
    test_ads_term(tok, teacher, fx["proxy_student"])
    test_gradient_sign(fx)

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print("  FAILED:", name)
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

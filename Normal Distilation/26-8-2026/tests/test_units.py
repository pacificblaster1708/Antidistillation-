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

from ads import (
    ADSLogitsProcessor,
    CompletionOnlyCollator,
    IncrementalLM,
    align_vocab,
    apply_perturbation,
    count_label_tokens,
    load_causal_lm,
    load_tokenizer,
    normalize_param_name,
)
from config import Config
from data import ANSWER_FORCE_STRING, is_correct
from emarc import EMARCLogitsProcessor, margin_floor, nucleus_mask
from horizon import finite_horizon_recurrence
from tests.tiny_setup import build_all

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}"
    )


# --------------------------------------------------------------------------- #
def test_config_modes():
    print("\n[config] mode selection")
    cfg = Config.load(["--ads=false", "--normal=true"])
    check("NORMAL selected", cfg.mode == "normal" and not cfg.ads and not cfg.emarc)
    check("NORMAL forces lam=0", cfg.lam == 0.0 and cfg.eps == 0.0)

    cfg = Config.load(["--ads=true", "--normal=false", "--lam=0.2", "--eps=0.01"])
    check("ADS selected", cfg.mode == "ads" and cfg.ads and not cfg.emarc)
    check("ADS keeps lam", cfg.lam == 0.2 and cfg.eps == 0.01)

    cfg = Config.load(["--emarc=true", "--ads=false", "--normal=false", "--eps=0.01"])
    check("EMARC selected", cfg.mode == "emarc" and cfg.emarc and not cfg.ads)
    check("EMARC disables static lam", cfg.lam == 0.0 and cfg.eps == 0.01)
    check(
        "EMARC direction is separately named",
        "emarc_direction_m1" in cfg.direction_path,
    )

    try:
        Config.load(["--ads=false", "--normal=true", "--top_p=0"])
        check("rejects invalid top_p", False)
    except SystemExit:
        check("rejects invalid top_p", True)

    for bad in (
        ["--ads=true", "--normal=true"],
        ["--ads=false", "--emarc=false", "--normal=false"],
        ["--ads=true", "--emarc=true", "--normal=false"],
    ):
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
class FixedLM:
    """Minimal IncrementalLM stand-in for deterministic controller tests."""

    def __init__(self, logits):
        self.logits = torch.as_tensor(logits, dtype=torch.float32).detach().clone()

    def next_logits(self, input_ids, attention_mask):
        del attention_mask
        return self.logits.to(input_ids.device).expand(input_ids.shape[0], -1)


def _emarc_processor(
    teacher_score, *, temperature=0.5, plus=None, minus=None, **overrides
):
    eps = 0.1
    direction = torch.tensor(teacher_score, dtype=torch.float32)
    plus_logits = (
        direction * eps if plus is None else torch.tensor(plus, dtype=torch.float32)
    )
    minus_logits = (
        -direction * eps if minus is None else torch.tensor(minus, dtype=torch.float32)
    )
    kwargs = dict(
        temperature=temperature,
        top_p=1.0,
        alpha_init=0.4,
        alpha_min=0.0,
        alpha_max=4.0,
        alpha_lr=0.0,
        target_flow=0.0,
        antiwindup=0.0,
        sigma=2.0,
        attacker_eta=0.5,
        attacker_iters=2,
        mass_min=0.1,
        mass_max=4.0,
        value_source="uniform",
        value_floor=0.1,
        beta_max=5.0,
        kl_cap=100.0,
        typicality_kappa=100.0,
        max_backtracks=12,
        margin=False,
        entropy_threshold=0.2,
        margin_gamma=0.05,
        margin_mass_min=0.0,
        safe_top_k=len(teacher_score),
        safe_logit_gap=10.0,
        protected_z=0.0,
        min_score_gap=0.0,
        margin_allow_kl_override=True,
        eos_token_id=None,
        protected_exclude_ids=(),
    )
    kwargs.update(overrides)
    return EMARCLogitsProcessor(
        FixedLM(plus_logits),
        FixedLM(minus_logits),
        eps,
        torch.ones(1, 2, dtype=torch.long),
        **kwargs,
    )


def test_emarc_math():
    print("\n[emarc] controller equations and safeguards")

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    mask = nucleus_mask(logits, temperature=1.0, top_p=0.8)
    check(
        "nucleus keeps boundary token",
        mask.tolist() == [[True, True, False, False]],
        str(mask.tolist()),
    )
    greedy = nucleus_mask(logits, temperature=0.0, top_p=0.8)
    check(
        "greedy nucleus is nominal argmax",
        greedy.tolist() == [[True, False, False, False]],
    )

    required = margin_floor(
        torch.tensor([2.0, 1.0, 0.0]),
        torch.tensor([0.0, 2.0, -1.0]),
        torch.tensor([True, True, True]),
        torch.tensor([False, True, False]),
        0.1,
    )
    check(
        "margin floor matches exact ratio",
        torch.allclose(required, torch.tensor(0.55)),
        f"got {float(required):.6f}",
    )
    impossible = margin_floor(
        torch.tensor([2.0, 1.0]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        0.1,
    )
    check("margin detects infeasible denominator", bool(torch.isinf(impossible)))

    proc = _emarc_processor([0.0, 1.0, -1.0], temperature=0.5)
    inputs = torch.tensor([[1, 2]])
    teacher = torch.tensor([[0.2, 0.1, -0.4]])
    out = proc(inputs, teacher.clone())
    stats = proc.totals()
    # First causal position has a_t=1 exactly, hence beta=tau*alpha*a/sigma.
    expected_beta = 0.5 * 0.4 / 2.0
    check(
        "continuous beta is tau*alpha*mass/sigma",
        abs(stats["maxima"]["beta"] - expected_beta) < 1e-6,
        f"{stats['maxima']['beta']:.8f} vs {expected_beta:.8f}",
    )
    check("positive score receives positive tilt", out[0, 1] > teacher[0, 1])
    check(
        "controller output stays finite on full support",
        bool(torch.isfinite(out).all()),
    )

    tight = _emarc_processor(
        [0.0, 20.0, -20.0], temperature=1.0, alpha_init=2.0, sigma=1.0, kl_cap=1e-4
    )
    tight(torch.tensor([[1, 2]]), torch.tensor([[0.0, 0.0, 0.0]]))
    tight_stats = tight.totals()
    check(
        "KL backtracking respects cap",
        tight_stats["maxima"]["kl"] <= 1e-4 + 1e-7,
        f"KL {tight_stats['maxima']['kl']:.3e}",
    )
    check("KL guard reports clipping", tight_stats["counts"]["guard_clipped"] == 1)

    hard_cap = _emarc_processor(
        [0.0, 100.0, -100.0],
        temperature=1.0,
        alpha_init=2.0,
        sigma=1.0,
        kl_cap=1e-8,
        max_backtracks=0,
    )
    hard_cap(torch.tensor([[1, 2]]), torch.tensor([[0.0, 0.0, 0.0]]))
    hard_stats = hard_cap.totals()
    check(
        "KL cap remains hard with zero backtracks",
        hard_stats["maxima"]["beta"] == 0.0 and hard_stats["maxima"]["kl"] <= 1e-8,
    )

    suppressed = _emarc_processor([0.0, 1.0], temperature=1.0)
    suppressed_out = suppressed(
        torch.tensor([[1, 2]]), torch.tensor([[0.0, -torch.inf]])
    )
    check(
        "suppressed -inf teacher token stays suppressed",
        bool(torch.isfinite(suppressed_out[0, 0]))
        and bool(torch.isneginf(suppressed_out[0, 1])),
    )

    greedy_proc = _emarc_processor(
        [0.0, 1.0, -1.0],
        temperature=0.0,
        alpha_init=0.4,
        sigma=1.0,
        margin=True,
        entropy_threshold=1.0,
        margin_mass_min=0.0,
        min_score_gap=0.1,
        protected_z=0.0,
        typicality_kappa=1.0,
        beta_max=2.0,
        safe_logit_gap=2.0,
    )
    greedy_logits = torch.tensor([[2.0, 1.8, 0.0]])
    greedy_out = greedy_proc(torch.tensor([[1, 2]]), greedy_logits)
    check(
        "greedy margin changes argmax into protected set",
        int(greedy_out.argmax(dim=-1)) == 1,
    )
    check(
        "greedy margin is certified",
        greedy_proc.totals()["counts"]["margin_certified"] == 1,
    )

    no_margin = _emarc_processor([0.0, 1.0], temperature=0.0, margin=False)
    unchanged = no_margin(torch.tensor([[1, 2]]), torch.tensor([[2.0, 1.9]]))
    check(
        "greedy continuous branch has beta zero",
        int(unchanged.argmax(dim=-1)) == 0
        and no_margin.totals()["maxima"]["beta"] == 0.0,
    )


def test_horizon_recurrence():
    print("\n[horizon] finite-step influence recurrence")
    gradient = {"w": torch.tensor([1.0, 2.0])}
    diagonal = torch.tensor([2.0, 3.0])

    def hvp(vector):
        return {"w": diagonal * vector["w"]}

    got = finite_horizon_recurrence(gradient, hvp, steps=3, lr=0.1, damping=0.2)
    manual = torch.zeros(2)
    for _ in range(3):
        manual = manual + 0.1 * (gradient["w"] - diagonal * manual - 0.2 * manual)
    check(
        "recurrence matches explicit diagonal calculation",
        torch.allclose(got["w"], manual, atol=1e-7),
        str(got["w"].tolist()),
    )
    one = finite_horizon_recurrence(gradient, hvp, steps=1, lr=0.1, damping=0.2)
    check("one-step recurrence is lr*g", torch.allclose(one["w"], 0.1 * gradient["w"]))


# --------------------------------------------------------------------------- #
def test_collator():
    print("\n[collator] completion-only masking")
    coll = CompletionOnlyCollator(pad_token_id=99, pad_to_multiple_of=None)
    batch = coll(
        [
            {"input_ids": [1, 2, 3, 4, 5], "prompt_len": 2},
            {"input_ids": [6, 7, 8], "prompt_len": 1},
        ]
    )
    check("padded to longest", list(batch["input_ids"].shape) == [2, 5])
    check("pad id used", batch["input_ids"][1].tolist() == [6, 7, 8, 99, 99])
    check("attention mask", batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0])
    check("prompt masked out", batch["labels"][0].tolist() == [-100, -100, 3, 4, 5])
    check("padding masked out", batch["labels"][1].tolist() == [-100, 7, 8, -100, -100])
    # after the causal shift, row0 supervises {3,4,5} and row1 supervises {7,8}
    check(
        "label token count",
        count_label_tokens(batch["labels"]) == 5,
        f"got {count_label_tokens(batch['labels'])}",
    )

    coll8 = CompletionOnlyCollator(pad_token_id=0, pad_to_multiple_of=8)
    b8 = coll8([{"input_ids": [1, 2, 3], "prompt_len": 1}])
    check("pads to multiple of 8", b8["input_ids"].shape[1] == 8)


# --------------------------------------------------------------------------- #
def test_answer_checking():
    print("\n[data] answer checking")
    check("boxed match", is_correct("so the answer is \\boxed{42}", "\\boxed{42}"))
    check("boxed mismatch", not is_correct("\\boxed{41}", "\\boxed{42}"))
    check("gsm8k gold format", is_correct("\\boxed{18}", "some working\n#### 18"))
    check(
        "answer-forced trace",
        is_correct("thinking..." + ANSWER_FORCE_STRING + "7}\\]", "\\boxed{7}"),
    )
    check("empty prediction", not is_correct("", "\\boxed{1}"))


# --------------------------------------------------------------------------- #
def test_vocab_alignment(fx):
    print("\n[ads] vocabulary alignment")
    tok = load_tokenizer(fx["teacher_tokenizer"])
    teacher = load_causal_lm(fx["teacher"], torch.float32, "eager")
    proxy = load_causal_lm(fx["proxy_student"], torch.float32, "eager")
    before = (
        teacher.get_input_embeddings().weight.shape[0],
        proxy.get_input_embeddings().weight.shape[0],
    )
    check("fixtures start mismatched", before[0] != before[1], str(before))
    align_vocab(tok, teacher, proxy)
    after = (
        teacher.get_input_embeddings().weight.shape[0],
        proxy.get_input_embeddings().weight.shape[0],
    )
    check("both resized to len(tokenizer)", after == (len(tok), len(tok)), str(after))
    return tok, teacher, proxy


# --------------------------------------------------------------------------- #
def test_incremental_cache(tok, proxy):
    print("\n[ads] incremental decoding matches a full forward")
    proxy.eval()
    ids = torch.tensor([[5, 9, 3, 7, 11, 2], [1, 4, 8, 6, 10, 12]])
    mask = torch.ones_like(ids)
    mask[1, 0] = 0  # a left-padded row

    with torch.inference_mode():
        reference = proxy(input_ids=ids, attention_mask=mask).logits[:, -1, :]

    inc = IncrementalLM(proxy)
    for step in range(3, ids.shape[1] + 1):  # prefix, then token by token
        got = inc.next_logits(ids[:, :step], mask[:, :step])
    check(
        "cached == uncached",
        torch.allclose(got, reference, atol=1e-4),
        f"max delta {float((got - reference).abs().max()):.2e}",
    )

    inc.reset()
    fresh = inc.next_logits(ids, mask)
    check("reset forces recompute", torch.allclose(fresh, reference, atol=1e-5))

    # A shorter batch after a longer one must not reuse the stale cache.
    inc.reset()
    inc.next_logits(ids, mask)
    small = inc.next_logits(ids[:1, :4], mask[:1, :4])
    with torch.inference_mode():
        small_ref = proxy(input_ids=ids[:1, :4], attention_mask=mask[:1, :4]).logits[
            :, -1, :
        ]
    check(
        "batch-size change invalidates cache",
        torch.allclose(small, small_ref, atol=1e-5),
    )


# --------------------------------------------------------------------------- #
def test_ads_term(tok, teacher, proxy_path):
    print("\n[ads] the antidistillation term")
    lam, eps = 0.3, 1e-2
    plus_model = load_causal_lm(proxy_path, torch.float32, "eager")
    minus_model = load_causal_lm(proxy_path, torch.float32, "eager")
    align_vocab(tok, plus_model, minus_model)

    grads = {
        normalize_param_name(n): torch.randn_like(p) * 0.01
        for n, p in plus_model.named_parameters()
    }
    stats = apply_perturbation(plus_model, grads, +eps)
    apply_perturbation(minus_model, grads, -eps)
    check("perturbation touched every tensor", stats["num_tensors"] == len(grads))

    a = dict(plus_model.named_parameters())["model.layers.0.mlp.up_proj.weight"]
    b = dict(minus_model.named_parameters())["model.layers.0.mlp.up_proj.weight"]
    expected = 2 * eps * grads["model.layers.0.mlp.up_proj.weight"]
    check("plus/minus differ by 2*eps*g", torch.allclose(a - b, expected, atol=1e-6))

    plus_model.eval()
    minus_model.eval()
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
    check(
        "matches lam/(2*eps)*(f(+) - f(-))",
        torch.allclose(out, manual, atol=1e-4),
        f"max delta {float((out - manual).abs().max()):.2e}",
    )
    check(
        "term is non-trivial",
        float(out.abs().max()) > 1e-3,
        f"max |term| {float(out.abs().max()):.3e}",
    )

    # A mismatched vocabulary must fail loudly rather than broadcast silently.
    proc.plus.reset()
    proc.minus.reset()
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
    base = float(out.loss.detach())
    out.loss.backward()
    grads = {
        normalize_param_name(n): p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)

    def loss_at(scale):
        probe = load_causal_lm(
            fx["proxy_student"], torch.float32, "eager", use_cache=False
        )
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
    test_emarc_math()
    test_horizon_recurrence()
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

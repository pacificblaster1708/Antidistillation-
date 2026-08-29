# -*- coding: utf-8 -*-
"""EMARC-ADS: causal entropy- and minimax-adaptive ADS control.

The processor keeps ADS's central finite-difference token score but replaces a
single static strength with a causal controller.  For each token position it:

1. estimates the adaptive reweighter's relative mass ``a_t`` with a causal
   KL-DRO best response;
2. applies the exact continuous optimum

       beta_t = tau * alpha_t * a_t / sigma,

   in *pre-temperature* logit space;
3. constrains the continuous action by per-token KL and typicality budgets;
4. in a low-entropy context, optionally raises beta to the smallest feasible
   margin floor over a likelihood-guarded protected token set; and
5. updates ``alpha_t`` with a deficit bank and anti-windup correction.

The likelihood guard is an implementable default, not a semantic safety
certificate.  A formal deployment claim requires replacing/augmenting it with
a task-specific verifier, as documented in README.md.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.nn import functional as F
from transformers import LogitsProcessor

from ads import IncrementalLM


def _one_hot_argmax(values: torch.Tensor) -> torch.Tensor:
    """Return a float one-hot tensor at each row's argmax."""
    out = torch.zeros_like(values, dtype=torch.float32)
    return out.scatter_(1, values.argmax(dim=-1, keepdim=True), 1.0)


def nucleus_mask(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Mask matching Hugging Face's top-p boundary convention.

    The boundary token that first reaches ``top_p`` is retained.  At greedy
    temperature only the nominal argmax belongs to the base support.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2, got shape {tuple(logits.shape)}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    finite = torch.isfinite(logits)
    if not bool(finite.any(dim=-1).all()):
        raise ValueError("every logits row needs at least one finite token")
    if temperature <= 0:
        return _one_hot_argmax(logits).bool()
    if top_p >= 1:
        return finite

    probs = torch.softmax(logits.float() / float(temperature), dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    keep_sorted = cumulative_before < float(top_p)
    keep_sorted[:, 0] = True
    keep = torch.zeros_like(keep_sorted)
    return keep.scatter_(1, sorted_indices, keep_sorted) & finite


def margin_floor(
    logits: torch.Tensor,
    scores: torch.Tensor,
    support: torch.Tensor,
    protected: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    r"""Smallest non-negative beta certifying a protected-set margin.

    For each protected candidate ``v`` this evaluates

        max_{u outside P} [l_u - l_v + gamma]_+ / (d_v - d_u)

    and returns the best feasible candidate.  A required non-positive
    denominator makes that candidate infeasible.  Only tokens in ``support``
    compete because all others are masked by the processor.
    """
    if logits.ndim != 1 or scores.ndim != 1:
        raise ValueError("margin_floor expects one-dimensional logits and scores")
    if (
        logits.shape != scores.shape
        or support.shape != logits.shape
        or protected.shape != logits.shape
    ):
        raise ValueError("margin_floor inputs must have identical shapes")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    candidates = torch.nonzero(protected & support, as_tuple=False).flatten()
    outside = support & ~protected
    if candidates.numel() == 0:
        return logits.new_tensor(float("inf"), dtype=torch.float32)
    if not bool(outside.any()):
        return logits.new_tensor(0.0, dtype=torch.float32)

    outside_logits = logits[outside].float()
    outside_scores = scores[outside].float()
    best = logits.new_tensor(float("inf"), dtype=torch.float32)
    tolerance = 1e-12
    for index in candidates:
        numerator = (outside_logits - logits[index].float() + float(gamma)).clamp_min(
            0.0
        )
        denominator = scores[index].float() - outside_scores
        needed = numerator > tolerance
        if bool((needed & (denominator <= 0)).any()):
            continue
        ratios = torch.where(
            needed,
            numerator / denominator.clamp_min(tolerance),
            torch.zeros_like(numerator),
        )
        best = torch.minimum(best, ratios.max())
    return best


class EMARCLogitsProcessor(LogitsProcessor):
    """Stateful, per-sequence EMARC-ADS controller for Hugging Face generation."""

    _SUM_KEYS = ("beta", "mass", "alpha", "kl", "typicality", "gain", "flow", "entropy")
    _COUNT_KEYS = (
        "margin_attempts",
        "margin_certified",
        "margin_infeasible",
        "guard_clipped",
        "beta_capped",
    )

    def __init__(
        self,
        plus: IncrementalLM,
        minus: IncrementalLM,
        eps: float,
        prompt_attention_mask: torch.LongTensor,
        *,
        temperature: float,
        top_p: float,
        alpha_init: float,
        alpha_min: float,
        alpha_max: float,
        alpha_lr: float,
        target_flow: float,
        antiwindup: float,
        sigma: float,
        attacker_eta: float,
        attacker_iters: int,
        mass_min: float,
        mass_max: float,
        value_source: str,
        value_floor: float,
        beta_max: float,
        kl_cap: float,
        typicality_kappa: float,
        max_backtracks: int,
        margin: bool,
        entropy_threshold: float,
        margin_gamma: float,
        margin_mass_min: float,
        safe_top_k: int,
        safe_logit_gap: float,
        protected_z: float,
        min_score_gap: float,
        margin_allow_kl_override: bool,
        eos_token_id: Optional[int] = None,
        protected_exclude_ids: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if sigma <= 0 or attacker_eta <= 0:
            raise ValueError("sigma and attacker_eta must be > 0")
        if attacker_iters < 1 or max_backtracks < 0:
            raise ValueError("iteration counts are invalid")

        self.plus, self.minus = plus, minus
        self.eps = float(eps)
        self.prompt_attention_mask = prompt_attention_mask
        self.prompt_width = int(prompt_attention_mask.shape[1])
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_lr = float(alpha_lr)
        self.target_flow = float(target_flow)
        self.antiwindup = float(antiwindup)
        self.sigma = float(sigma)
        self.attacker_eta = float(attacker_eta)
        self.attacker_iters = int(attacker_iters)
        self.mass_min = float(mass_min)
        self.mass_max = float(mass_max)
        self.value_source = value_source
        self.value_floor = float(value_floor)
        self.beta_max = float(beta_max)
        self.kl_cap = float(kl_cap)
        self.typicality_kappa = float(typicality_kappa)
        self.max_backtracks = int(max_backtracks)
        self.margin = bool(margin)
        self.entropy_threshold = float(entropy_threshold)
        self.margin_gamma = float(margin_gamma)
        self.margin_mass_min = float(margin_mass_min)
        self.safe_top_k = int(safe_top_k)
        self.safe_logit_gap = float(safe_logit_gap)
        self.protected_z = float(protected_z)
        self.min_score_gap = float(min_score_gap)
        self.margin_allow_kl_override = bool(margin_allow_kl_override)
        self.eos_token_id = eos_token_id
        self.protected_exclude_ids = tuple(
            int(x) for x in (protected_exclude_ids or ())
        )

        self.alpha: Optional[torch.Tensor] = None
        self.weight_sum: Optional[torch.Tensor] = None
        self.position_count: Optional[torch.Tensor] = None
        self.calls = 0
        self.active_tokens = 0
        self.sums: Dict[str, float] = {key: 0.0 for key in self._SUM_KEYS}
        self.maxima: Dict[str, float] = {
            "beta": 0.0,
            "kl": 0.0,
            "typicality": 0.0,
            "mass": 0.0,
        }
        self.counts: Dict[str, int] = {key: 0 for key in self._COUNT_KEYS}

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        if (
            self.alpha is not None
            and self.alpha.shape[0] == batch_size
            and self.alpha.device == device
        ):
            return
        self.alpha = torch.full(
            (batch_size,), self.alpha_init, device=device, dtype=torch.float32
        )
        self.weight_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
        self.position_count = torch.zeros(
            batch_size, device=device, dtype=torch.float32
        )

    def _active_rows(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if self.eos_token_id is None or input_ids.shape[1] <= self.prompt_width:
            return torch.ones(
                input_ids.shape[0], device=input_ids.device, dtype=torch.bool
            )
        generated = input_ids[:, self.prompt_width :]
        return ~(generated == int(self.eos_token_id)).any(dim=-1)

    def _nominal_value(
        self, full_probs: torch.Tensor, normalized_entropy: torch.Tensor
    ) -> torch.Tensor:
        if self.value_source == "uniform":
            raw = torch.ones_like(normalized_entropy)
        elif self.value_source == "low_entropy":
            raw = 1.0 - normalized_entropy
        elif self.value_source == "confidence":
            raw = full_probs.max(dim=-1).values
        else:  # configuration validation should make this unreachable
            raise RuntimeError(f"unknown EMARC value source {self.value_source!r}")
        return self.value_floor + (1.0 - self.value_floor) * raw.clamp(0.0, 1.0)

    def _supports(self, logits: torch.Tensor, normalized_entropy: torch.Tensor):
        nucleus = nucleus_mask(logits, self.temperature, self.top_p)
        k = min(self.safe_top_k, logits.shape[-1])
        top_indices = torch.topk(logits, k=k, dim=-1).indices
        safe = torch.zeros_like(nucleus)
        safe.scatter_(1, top_indices, True)
        safe &= logits >= (
            logits.max(dim=-1, keepdim=True).values - self.safe_logit_gap
        )
        valid_excluded = [
            x for x in self.protected_exclude_ids if 0 <= x < logits.shape[-1]
        ]
        if valid_excluded:
            safe[:, valid_excluded] = False
        low_entropy = normalized_entropy <= self.entropy_threshold
        expand = low_entropy[:, None] & self.margin
        support = nucleus | (safe & expand)
        return support, safe, low_entropy

    def _base_distribution(self, logits: torch.Tensor, support: torch.Tensor):
        masked = logits.masked_fill(~support, -torch.inf)
        if self.temperature > 0:
            logp = torch.log_softmax(masked / self.temperature, dim=-1)
            probs = torch.exp(logp)
        else:
            probs = _one_hot_argmax(masked)
            logp = torch.full_like(masked, -torch.inf)
            logp.scatter_(1, masked.argmax(dim=-1, keepdim=True), 0.0)
        return probs, logp

    def _metrics(
        self,
        beta: torch.Tensor,
        logits: torch.Tensor,
        scores: torch.Tensor,
        support: torch.Tensor,
        base_probs: torch.Tensor,
        base_log_probs: torch.Tensor,
        score_scale: torch.Tensor,
    ):
        adjusted = logits + beta[:, None] * scores
        masked = adjusted.masked_fill(~support, -torch.inf)
        if self.temperature > 0:
            logq = torch.log_softmax(masked / self.temperature, dim=-1)
            q = torch.exp(logq)
            kl_terms = torch.where(
                support, q * (logq - base_log_probs), torch.zeros_like(q)
            )
            kl = kl_terms.sum(dim=-1).clamp_min(0.0)
            surprisal = torch.where(
                support, -base_log_probs, torch.zeros_like(base_log_probs)
            )
            base_h = (base_probs * surprisal).sum(dim=-1)
            typicality = ((q * surprisal).sum(dim=-1) - base_h).abs()
        else:
            q = _one_hot_argmax(masked)
            kl = torch.zeros_like(beta)
            base_index = logits.masked_fill(~support, -torch.inf).argmax(
                dim=-1, keepdim=True
            )
            chosen_index = masked.argmax(dim=-1, keepdim=True)
            base_logit = logits.gather(1, base_index).squeeze(1)
            chosen_logit = logits.gather(1, chosen_index).squeeze(1)
            # Greedy decoding has no finite KL after an argmax switch.  The
            # nominal logit gap is the explicit intervention cost instead.
            typicality = (base_logit - chosen_logit).clamp_min(0.0)

        base_score = (base_probs * scores).sum(dim=-1)
        defended_score = (q * scores).sum(dim=-1)
        gain = (defended_score - base_score).clamp_min(0.0)
        gain_normalized = gain / score_scale
        return q, kl, typicality, gain, gain_normalized

    def _constrain(
        self,
        raw_beta: torch.Tensor,
        logits: torch.Tensor,
        scores: torch.Tensor,
        support: torch.Tensor,
        base_probs: torch.Tensor,
        base_log_probs: torch.Tensor,
        score_scale: torch.Tensor,
        active: torch.Tensor,
    ):
        beta = raw_beta.clamp(min=0.0, max=self.beta_max)
        if self.temperature > 0:
            for _ in range(self.max_backtracks):
                _, kl, typicality, _, _ = self._metrics(
                    beta,
                    logits,
                    scores,
                    support,
                    base_probs,
                    base_log_probs,
                    score_scale,
                )
                bad = active & (
                    (kl > self.kl_cap) | (typicality > self.typicality_kappa)
                )
                if not bool(bad.any()):
                    break
                beta = torch.where(bad, beta * 0.5, beta)
        metrics = self._metrics(
            beta, logits, scores, support, base_probs, base_log_probs, score_scale
        )
        if self.temperature > 0:
            final_bad = active & (
                (metrics[1] > self.kl_cap) | (metrics[2] > self.typicality_kappa)
            )
            if bool(final_bad.any()):
                # beta=0 is exactly the guarded base law. This makes both caps
                # hard even with zero backtracks or an extreme token score.
                beta = torch.where(final_bad, torch.zeros_like(beta), beta)
                metrics = self._metrics(
                    beta,
                    logits,
                    scores,
                    support,
                    base_probs,
                    base_log_probs,
                    score_scale,
                )
        return beta, metrics

    def _record(
        self,
        active: torch.Tensor,
        beta: torch.Tensor,
        mass: torch.Tensor,
        kl: torch.Tensor,
        typicality: torch.Tensor,
        gain: torch.Tensor,
        flow: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:
        n = int(active.sum().item())
        self.active_tokens += n
        if n == 0:
            return
        values = {
            "beta": beta,
            "mass": mass,
            "alpha": self.alpha,
            "kl": kl,
            "typicality": typicality,
            "gain": gain,
            "flow": flow,
            "entropy": entropy,
        }
        for key, tensor in values.items():
            self.sums[key] += float(tensor[active].sum().item())
        for key in self.maxima:
            self.maxima[key] = max(
                self.maxima[key], float(values[key][active].max().item())
            )

    @torch.inference_mode()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        pad = input_ids.shape[1] - self.prompt_attention_mask.shape[1]
        attention_mask = F.pad(self.prompt_attention_mask, (0, pad), value=1)
        up = self.plus.next_logits(input_ids, attention_mask).float()
        down = self.minus.next_logits(input_ids, attention_mask).float()
        if up.shape != down.shape or up.shape[-1] != scores.shape[-1]:
            raise RuntimeError(
                f"vocabulary mismatch: teacher emits {scores.shape[-1]} logits but EMARC's "
                f"proxy students emit {up.shape[-1]}. They must share a tokenizer."
            )
        if not bool(torch.isfinite(up).all() and torch.isfinite(down).all()):
            raise RuntimeError("non-finite proxy logits reached EMARC")
        if bool(torch.isnan(scores).any() or torch.isposinf(scores).any()):
            raise RuntimeError("NaN or +inf teacher logits reached EMARC")

        logits = scores.float()
        token_scores = (up - down) / (2.0 * self.eps)
        batch_size, vocab_size = logits.shape
        self._ensure_state(batch_size, logits.device)
        active = self._active_rows(input_ids)

        if self.temperature > 0:
            full_probs = torch.softmax(logits / self.temperature, dim=-1)
            full_log_probs = torch.log_softmax(logits / self.temperature, dim=-1)
            entropy_terms = torch.where(
                full_probs > 0,
                full_probs * full_log_probs,
                torch.zeros_like(full_probs),
            )
            full_entropy = -entropy_terms.sum(dim=-1)
            entropy_denom = math.log(max(2, vocab_size))
            normalized_entropy = (full_entropy / entropy_denom).clamp(0.0, 1.0)
        else:
            full_probs = _one_hot_argmax(logits)
            normalized_entropy = torch.zeros(batch_size, device=logits.device)

        support, safe, low_entropy = self._supports(logits, normalized_entropy)
        base_probs, base_log_probs = self._base_distribution(logits, support)
        base_score = (base_probs * token_scores).sum(dim=-1, keepdim=True)
        token_scores = token_scores - base_score  # shift-invariant, same sampling law
        support_count = support.sum(dim=-1).clamp_min(1).float()
        score_scale = torch.sqrt(
            (token_scores.square() * support).sum(dim=-1) / support_count
        ).clamp_min(1e-6)

        nominal_value = self._nominal_value(full_probs, normalized_entropy)
        next_count = self.position_count + 1.0
        mass = (
            next_count
            * nominal_value
            / (self.weight_sum + nominal_value).clamp_min(1e-12)
        )
        mass = mass.clamp(self.mass_min, self.mass_max)

        # Fixed-point best response: defended positions receive exponentially
        # less attacker weight, normalized causally over positions observed so far.
        defended_weight = nominal_value
        beta = torch.zeros_like(mass)
        raw_beta = torch.zeros_like(mass)
        metrics = None
        for _ in range(self.attacker_iters):
            raw_beta = self.temperature * self.alpha * mass / self.sigma
            raw_beta = torch.where(active, raw_beta, torch.zeros_like(raw_beta))
            beta, metrics = self._constrain(
                raw_beta,
                logits,
                token_scores,
                support,
                base_probs,
                base_log_probs,
                score_scale,
                active,
            )
            gain_normalized = metrics[4]
            exponent = (-gain_normalized / self.attacker_eta).clamp(-20.0, 0.0)
            defended_weight = nominal_value * torch.exp(exponent)
            proposed_mass = (
                next_count
                * defended_weight
                / (self.weight_sum + defended_weight).clamp_min(1e-12)
            )
            mass = torch.where(
                active, proposed_mass.clamp(self.mass_min, self.mass_max), mass
            )

        # Recompute once at the final fixed-point mass.
        raw_beta = self.temperature * self.alpha * mass / self.sigma
        raw_beta = torch.where(active, raw_beta, torch.zeros_like(raw_beta))
        beta, metrics = self._constrain(
            raw_beta,
            logits,
            token_scores,
            support,
            base_probs,
            base_log_probs,
            score_scale,
            active,
        )
        _, kl, typicality, gain, gain_normalized = metrics
        continuous_beta = beta.clone()
        final_exponent = (-gain_normalized / self.attacker_eta).clamp(-20.0, 0.0)
        defended_weight = nominal_value * torch.exp(final_exponent)

        self.counts["beta_capped"] += int(
            (active & (raw_beta > self.beta_max)).sum().item()
        )
        guard_limit = raw_beta.clamp(max=self.beta_max)
        self.counts["guard_clipped"] += int(
            (active & (continuous_beta + 1e-12 < guard_limit)).sum().item()
        )

        # Low-entropy activation margin.  Continuous beta vanishes with tau;
        # this branch is what remains meaningful at tau=0/greedy decoding.
        protected = safe & (token_scores >= self.protected_z * score_scale[:, None])
        nominal_top = logits.argmax(dim=-1, keepdim=True)
        top_score = token_scores.gather(1, nominal_top)
        protected &= token_scores >= (top_score + self.min_score_gap)
        eligible = active & low_entropy & (mass >= self.margin_mass_min) & self.margin
        for row in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            self.counts["margin_attempts"] += 1
            required = margin_floor(
                logits[row],
                token_scores[row],
                support[row],
                protected[row],
                self.margin_gamma,
            )
            if not bool(torch.isfinite(required)) or float(required) > self.beta_max:
                self.counts["margin_infeasible"] += 1
                continue
            candidate_beta = torch.maximum(beta[row], required)
            row_beta = beta.clone()
            row_beta[row] = candidate_beta
            candidate = self._metrics(
                row_beta,
                logits,
                token_scores,
                support,
                base_probs,
                base_log_probs,
                score_scale,
            )
            candidate_kl = candidate[1][row]
            candidate_typicality = candidate[2][row]
            typicality_ok = bool(candidate_typicality <= self.typicality_kappa)
            kl_ok = self.margin_allow_kl_override or bool(candidate_kl <= self.kl_cap)
            if typicality_ok and kl_ok:
                beta[row] = candidate_beta
                kl[row] = candidate_kl
                typicality[row] = candidate_typicality
                gain[row] = candidate[3][row]
                gain_normalized[row] = candidate[4][row]
                self.counts["margin_certified"] += 1
            else:
                self.counts["margin_infeasible"] += 1

        flow = mass * gain_normalized

        # Deficit-bank feedback.  Anti-windup feeds the continuous constraint
        # error back into alpha; a margin override is accounted separately and
        # therefore cannot spuriously wind the continuous controller upward.
        alpha_next = self.alpha + self.alpha_lr * (self.target_flow - flow)
        beta_per_alpha = self.temperature * mass / self.sigma
        windup = torch.where(
            beta_per_alpha > 1e-12,
            (continuous_beta - raw_beta) / beta_per_alpha.clamp_min(1e-12),
            torch.zeros_like(beta_per_alpha),
        )
        alpha_next = alpha_next + self.antiwindup * windup
        alpha_next = alpha_next.clamp(self.alpha_min, self.alpha_max)

        # State updates are only for unfinished sequences.
        self.weight_sum = torch.where(
            active, self.weight_sum + defended_weight, self.weight_sum
        )
        self.position_count = torch.where(active, next_count, self.position_count)
        self.alpha = torch.where(active, alpha_next, self.alpha)

        self.calls += 1
        self._record(active, beta, mass, kl, typicality, gain, flow, normalized_entropy)

        adjusted = logits + beta[:, None] * token_scores
        adjusted = adjusted.masked_fill(~support, -torch.inf)
        return torch.where(active[:, None], adjusted, logits)

    def totals(self) -> dict:
        """Raw additive diagnostics, suitable for merging across batches/ranks."""
        return {
            "calls": self.calls,
            "active_tokens": self.active_tokens,
            "sums": dict(self.sums),
            "maxima": dict(self.maxima),
            "counts": dict(self.counts),
        }


def merge_emarc_stats(items: Iterable[dict]) -> dict:
    """Merge processor diagnostics and return a JSON-friendly summary."""
    calls = active_tokens = 0
    sums = {key: 0.0 for key in EMARCLogitsProcessor._SUM_KEYS}
    maxima = {"beta": 0.0, "kl": 0.0, "typicality": 0.0, "mass": 0.0}
    counts = {key: 0 for key in EMARCLogitsProcessor._COUNT_KEYS}
    for item in items:
        calls += int(item.get("calls", 0))
        active_tokens += int(item.get("active_tokens", 0))
        for key in sums:
            sums[key] += float(item.get("sums", {}).get(key, 0.0))
        for key in maxima:
            maxima[key] = max(maxima[key], float(item.get("maxima", {}).get(key, 0.0)))
        for key in counts:
            counts[key] += int(item.get("counts", {}).get(key, 0))
    denom = max(1, active_tokens)
    return {
        "calls": calls,
        "active_tokens": active_tokens,
        "means": {key: value / denom for key, value in sums.items()},
        "maxima": maxima,
        **counts,
    }

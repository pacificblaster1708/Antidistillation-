# -*- coding: utf-8 -*-
"""
Core machinery shared by every stage.

The antidistillation term itself is ~15 lines (`ADSLogitsProcessor`); everything
else here is the plumbing that makes it run correctly:

  * tokenizer / model loading with a pad token and a working attention backend
  * vocabulary alignment, so the teacher's logits and the proxy student's logits
    are indexable by the same token ids
  * an explicit KV-cached incremental decoder for the two perturbed proxy
    students (they are called once per generated token, so a cache is essential)
  * a completion-only collator that masks the prompt using a recorded token
    count instead of string-matching a chat template
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, set_seed)

# --------------------------------------------------------------------------- #
# environment / devices
# --------------------------------------------------------------------------- #
def init_runtime(seed: int) -> None:
    set_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:      # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def resolve_dtype(name: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_attn(name: str) -> Optional[str]:
    """Pick an attention backend that is actually importable on this machine."""
    if name != "auto":
        return name
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(name_or_path: str, padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(
        name_or_path, use_fast=True, trust_remote_code=True, padding_side=padding_side
    )
    if tok.pad_token_id is None:
        # Reuse EOS rather than adding a brand-new token: adding one forces a
        # vocabulary resize that appends randomly initialised logits, and those
        # rows are reachable by sampling.
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        raise ValueError(f"tokenizer {name_or_path} has neither a pad nor an eos token")
    return tok


def strip_terminals(text: str, tokenizer) -> str:
    """Remove pad/eos/bos strings, keeping every other special token intact."""
    for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token):
        if tok:
            text = text.replace(tok, "")
    return text


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def load_causal_lm(path: str, dtype: torch.dtype, attn: Optional[str], device=None,
                   use_cache: bool = True):
    kwargs = dict(trust_remote_code=True, torch_dtype=dtype, use_cache=use_cache)
    for impl in [attn, "sdpa", "eager"]:
        if impl is None:
            continue
        try:
            model = AutoModelForCausalLM.from_pretrained(path, attn_implementation=impl, **kwargs)
            break
        except (ImportError, ValueError) as err:
            if impl == "eager":
                raise
            print(f"[ads] attn_implementation={impl!r} unavailable ({err}); falling back")
    else:                                                          # pragma: no cover
        raise RuntimeError("no usable attention implementation")
    if device is not None:
        model = model.to(device)
    return model


def vocab_rows(model) -> int:
    return model.get_input_embeddings().weight.shape[0]


def align_vocab(tokenizer, *models) -> int:
    """
    Resize every model to exactly len(tokenizer) rows.

    Two reasons this matters:
      * ADS adds the proxy student's logits to the teacher's logits elementwise,
        so both must have the same width;
      * trailing rows that no token maps to are still sampleable and decode to
        nothing, which silently truncates traces.
    """
    target = len(tokenizer)
    for model in models:
        if model is None:
            continue
        if vocab_rows(model) != target:
            model.resize_token_embeddings(target)
    return target


# --------------------------------------------------------------------------- #
# incremental decoding for the proxy students
# --------------------------------------------------------------------------- #
class IncrementalLM:
    """
    Thin KV-cache wrapper: feed the full prefix once, then one token per step.

    `reset()` must be called before every new `generate()` batch. The original
    reference implementation relies on "the new prompt is shorter than the last
    sequence" to invalidate the cache, which breaks whenever a batch finishes
    early or the final batch is a different size; here invalidation is explicit
    and also guards on batch size.
    """

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.cache = None
        self.cached_len = 0
        self.batch = 0

    @torch.inference_mode()
    def next_logits(self, input_ids: torch.LongTensor,
                    attention_mask: torch.LongTensor) -> torch.Tensor:
        bsz, length = input_ids.shape
        reusable = (self.cache is not None and self.batch == bsz
                    and length == self.cached_len + 1)
        if not reusable:
            self.reset()
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=True, return_dict=True)
            self.batch = bsz
        else:
            out = self.model(input_ids=input_ids[:, -1:], attention_mask=attention_mask,
                             past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        self.cached_len = length
        return out.logits[:, -1, :]


# --------------------------------------------------------------------------- #
# the antidistillation term
# --------------------------------------------------------------------------- #
class ADSLogitsProcessor(LogitsProcessor):
    r"""
    Adds the antidistillation term to the teacher's next-token logits:

        scores <- scores + lam_t * (logits_{theta + eps*g} - logits_{theta - eps*g}) / (2 * eps)

    `g` is the proxy student's gradient of the *holdout* language-modelling loss
    (see grads.py). The bracket is a central finite difference, so it estimates
    the directional derivative  g^T d/dtheta logits_theta(token)  scaled by 2*eps.
    Tokens whose likelihood under the proxy student *rises* along +g are boosted,
    and +g is the ascent direction of the student's loss -- i.e. the teacher is
    nudged toward tokens that would teach a distilling student the wrong lesson.

    `lam_t` is dynamic: after each token x_t is sampled we observe the scalar
    signal d_t = delta(x_t), the antidistillation push specifically on x_t.
    We maintain a Welford exponentially weighted moving average to estimate the
    mean and variance of the signal, then use a standard Z-score modulated by a
    sigmoid to bound lam_t.

    Note on ordering: HuggingFace merges custom processors *before* the
    temperature / top-p warpers, so the sampled distribution is
        softmax( top_p( (teacher_logits + ads_term) / tau ) )
    and the effective strength of the term is therefore lam_t/tau. This matches
    the reference implementation.
    """

    def __init__(self, plus: IncrementalLM, minus: IncrementalLM, lam_min: float = 0.01, eps: float = 1e-3,
                 prompt_attention_mask: Optional[torch.LongTensor] = None,
                 lam_max: float = 0.075, beta: float = 0.9, gamma: float = 1.0,
                 sigma2_prior: float = 1e-8, warmup_steps: int = 2, warmup_val: float = 0.04):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        self.plus, self.minus = plus, minus
        self.lam_min, self.lam_max, self.eps = float(lam_min), float(lam_max), float(eps)
        self.beta, self.gamma = float(beta), float(gamma)
        self.sigma2_prior = float(sigma2_prior)
        self.warmup_steps = int(warmup_steps)
        self.warmup_val = float(warmup_val)
        self.prompt_attention_mask = prompt_attention_mask
        self.calls = 0
        
        self._m: Optional[torch.Tensor] = None
        self._S: Optional[torch.Tensor] = None
        self._prev_delta: Optional[torch.Tensor] = None

    def _init_state(self, bsz: int, device: torch.device) -> None:
        self._m = torch.zeros(bsz, 1, device=device)
        self._S = torch.zeros(bsz, 1, device=device)
        self._prev_delta = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bsz = input_ids.shape[0]
        device = scores.device

        if self._m is None or self._m.shape[0] != bsz:
            self._init_state(bsz, device)

        pad = input_ids.shape[1] - self.prompt_attention_mask.shape[1]
        attention_mask = F.pad(self.prompt_attention_mask, (0, pad), value=1)

        up = self.plus.next_logits(input_ids, attention_mask).float()
        down = self.minus.next_logits(input_ids, attention_mask).float()
        if up.shape[-1] != scores.shape[-1]:
            raise RuntimeError(
                f"vocabulary mismatch: teacher emits {scores.shape[-1]} logits but the proxy "
                f"student emits {up.shape[-1]}. Teacher and proxy student must share a tokenizer."
            )

        delta = (up - down) / (2.0 * self.eps)

        if self._prev_delta is not None:
            last_tokens = input_ids[:, -1].unsqueeze(1)
            d = self._prev_delta.gather(1, last_tokens).float()

            delta_old = d - self._m
            self._m = self.beta * self._m + (1.0 - self.beta) * d
            delta_new = d - self._m
            self._S = self.beta * self._S + (1.0 - self.beta) * delta_old * delta_new

            t = float(self.calls + 1)
            m_hat = self._m / (1.0 - self.beta ** t)
            S_hat = self._S / (1.0 - self.beta ** t)

            if self.calls <= self.warmup_steps:
                lam_t = torch.full((bsz, 1), self.warmup_val, device=device)
            else:
                sigma = torch.sqrt(S_hat + self.sigma2_prior)
                z = m_hat / sigma
                alpha = torch.sigmoid(self.gamma * z)
                lam_t = self.lam_min + (self.lam_max - self.lam_min) * alpha
        else:
            lam_t = torch.full((bsz, 1), self.warmup_val, device=device)

        self._prev_delta = delta.detach()
        self.calls += 1
        return scores.float() + lam_t * delta


# --------------------------------------------------------------------------- #
# gradients: saving, loading, perturbing
# --------------------------------------------------------------------------- #
def normalize_param_name(name: str) -> str:
    """Strip wrappers that DDP / torch.compile / peft add to parameter names."""
    for prefix in ("module.", "_orig_mod.", "base_model.model."):
        while name.startswith(prefix):
            name = name[len(prefix):]
    return name


def save_grads(path: str, grads: Dict[str, torch.Tensor], meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "grads": {normalize_param_name(k): v.detach().to("cpu", torch.float32)
                  for k, v in grads.items()},
        "meta": meta,
    }
    torch.save(payload, path)


def load_grads(path: str) -> tuple:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "grads" in payload:
        return payload["grads"], payload.get("meta", {})
    # tolerate a bare state-dict-style file
    return {normalize_param_name(k): v for k, v in payload.items()}, {}


@torch.no_grad()
def apply_perturbation(model, grads: Dict[str, torch.Tensor], scale: float) -> dict:
    """theta <- theta + scale * g, in place, done in fp32 then cast back."""
    known = {normalize_param_name(name) for name, _ in model.named_parameters()}
    orphans = set(grads) - known
    if orphans:
        raise RuntimeError(
            f"{len(orphans)} saved gradients match no parameter of this model, "
            f"e.g. {sorted(orphans)[:3]}. Was the gradient file built with a different "
            f"proxy student?"
        )

    used, param_sq, grad_sq, n = 0, 0.0, 0.0, 0
    for name, param in model.named_parameters():
        key = normalize_param_name(name)
        if key not in grads:
            continue
        g = grads[key].to(device=param.device, dtype=torch.float32)
        if tuple(g.shape) != tuple(param.shape):
            raise RuntimeError(
                f"gradient for {key!r} has shape {tuple(g.shape)} but the proxy student "
                f"parameter has shape {tuple(param.shape)}. Regenerate the gradients with "
                f"the same proxy student and tokenizer."
            )
        p32 = param.data.to(torch.float32) + scale * g
        param.data = p32.to(param.data.dtype)
        param_sq += float(torch.sum(p32 * p32))
        grad_sq += float(torch.sum(g * g))
        n += param.numel()
        used += 1
    if used == 0:
        raise RuntimeError("no parameter was perturbed -- the gradient file does not match this model")
    return {
        "param_rms": math.sqrt(param_sq / n),
        "grad_rms": math.sqrt(grad_sq / n),
        "num_tensors": used,
        "num_params": n,
    }


# --------------------------------------------------------------------------- #
# completion-only collation
# --------------------------------------------------------------------------- #
@dataclass
class CompletionOnlyCollator:
    """
    Pads a batch of pre-tokenized examples and masks the prompt out of the loss.

    Each feature carries `input_ids` (prompt followed by completion) and
    `prompt_len`. Masking by that integer is exact; the reference implementation
    searches for a chat-template marker string in the token stream instead, which
    silently produces an all -100 batch whenever the marker tokenizes differently
    in context.
    """
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: Sequence[dict]) -> Dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            longest = int(math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            prompt_len = int(feature["prompt_len"])
            pad = longest - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append([self.label_pad_token_id] * prompt_len
                          + ids[prompt_len:]
                          + [self.label_pad_token_id] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def count_label_tokens(labels: torch.Tensor) -> int:
    """Number of positions that actually contribute to the causal-LM loss."""
    return int((labels[:, 1:] != -100).sum())


# --------------------------------------------------------------------------- #
# build-once-then-share
# --------------------------------------------------------------------------- #
def build_on_main(state, cache_dir: str, key: str, build):
    """
    Build a dataset once on rank 0 and let every other rank read it back.

    Ranks that each call `load_dataset` and `.map` themselves race on the shared
    HuggingFace cache and repeat the same tokenization N times. Materialising
    once under the experiment directory (rather than a fixed /tmp path, which
    collides between concurrent experiments) avoids both.
    """
    import hashlib
    import shutil

    from datasets import load_from_disk

    if getattr(state, "num_processes", 1) <= 1:
        return build()

    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"cache_{digest}")
    if state.is_main_process:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        build().save_to_disk(path)
    state.wait_for_everyone()
    return load_from_disk(path)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def describe(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    pct = lambda q: vals[min(n - 1, max(0, int(round(q * (n - 1)))))]
    return {"count": n, "mean": mean, "std": var ** 0.5, "min": vals[0],
            "p25": pct(0.25), "p50": pct(0.5), "p75": pct(0.75), "max": vals[-1]}


def banner(title: str, body: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")
    if body:
        print(body)

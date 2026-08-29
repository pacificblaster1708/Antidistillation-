# -*- coding: utf-8 -*-
"""
Configuration for the minimal antidistillation-sampling codebase.

Exactly one run mode, selected by three booleans:

    EMARC=false ADS=false NORMAL=true  -> plain distillation
    EMARC=false ADS=true NORMAL=false  -> static antidistillation sampling
    EMARC=true ADS=false NORMAL=false  -> adaptive EMARC-ADS

Any other combination is rejected.

Every field below can be set three ways (later wins):
    1. the default in this file
    2. an environment variable with the UPPER_CASE field name
    3. a command line flag  --field=value   (or  --field value)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

TRUE = {"1", "true", "t", "yes", "y", "on"}
FALSE = {"0", "false", "f", "no", "n", "off", ""}


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def _coerce(raw: Any, target_type: Any, name: str) -> Any:
    """Turn a string coming from argv/env into the type the dataclass declares."""
    if raw is None:
        return None
    origin = str(target_type)
    if target_type is bool or "bool" in origin:
        return to_bool(raw)
    if isinstance(raw, (int, float, bool)) and not isinstance(raw, bool):
        return raw
    s = str(raw).strip()
    if "Optional" in origin and s.lower() in {"none", "null", ""}:
        return None
    if target_type is int or "int" in origin:
        return int(float(s))  # tolerate "1e3"
    if target_type is float or "float" in origin:
        return float(s)
    return s


@dataclass
class Config:
    # ------------------------------------------------------------------ mode
    ads: bool = False  # antidistillation sampling
    emarc: bool = False  # adaptive EMARC-ADS controller
    normal: bool = True  # plain distillation

    # ---------------------------------------------------------------- models
    # Teacher: the model being protected. Its tokenizer is used for every
    # trace-generation and gradient step.
    teacher: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # Proxy student: the defender's stand-in for the attacker. MUST share the
    # teacher's tokenizer, because the ADS term is added to the teacher's logits
    # and therefore has to live on the same vocabulary.
    proxy_student: str = "Qwen/Qwen2.5-3B"
    # Student: the attacker's model. Free to be a different architecture and a
    # different tokenizer -- it only ever sees decoded text.
    student: str = "meta-llama/Llama-3.2-3B"
    student_tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

    # ------------------------------------------------------------------ data
    # "gsm8k" | "hendrycks_math" | "mmlu" | "local:/path/to/dir_or_jsonl"
    dataset: str = "gsm8k"
    max_train_samples: Optional[int] = None
    max_holdout_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    # ------------------------------------------------------------ generation
    tau: float = 1.0  # sampling temperature for the training traces (0 => greedy)
    holdout_tau: float = 0.0  # temperature for the holdout traces (greedy by default)
    eval_tau: float = 0.0  # temperature when scoring the student on the test split
    top_p: float = 0.95
    lam: float = 0.15  # ADS strength; only used when ads=true
    eps: float = 1e-2  # finite-difference step; ADS and EMARC
    gen_batch_size: int = 32
    max_new_tokens: int = 1024
    max_prompt_length: int = 512
    answer_force: bool = True  # append "**Final Answer** \boxed{" and greedily finish
    answer_force_tokens: int = 32

    # ---------------------------------------------------------------- grads
    grad_batch_size: int = 1

    # ------------------------------------------------------------- EMARC-ADS
    # EMARC uses beta_t = tau * alpha_t * a_t / sigma in pre-temperature
    # logit space, then optionally raises beta_t to a certified margin floor in
    # low-entropy contexts. alpha_t is a causal feedback controller; a_t is a
    # KL-DRO-inspired estimate of the adaptive reweighter's relative mass.
    emarc_alpha_init: float = 0.15
    emarc_alpha_min: float = 0.0
    emarc_alpha_max: float = 4.0
    emarc_alpha_lr: float = 0.05
    emarc_target_flow: float = 0.05
    emarc_antiwindup: float = 0.25
    emarc_sigma: float = 1.0  # constant utility/harm sensitivity
    emarc_attacker_eta: float = 0.5  # KL-DRO temperature for the reweighter
    emarc_attacker_iters: int = 3  # fixed-point refinements per token
    emarc_mass_min: float = 0.1
    emarc_mass_max: float = 4.0
    emarc_value_source: str = "confidence"  # confidence | low_entropy | uniform
    emarc_value_floor: float = 0.1
    emarc_beta_max: float = 2.0
    emarc_kl_cap: float = 0.08  # per-token KL above the top-p base law
    emarc_typicality_kappa: float = 2.0
    emarc_max_backtracks: int = 12

    # Low-entropy margin floor. The built-in safe set is likelihood guarded,
    # not a semantic verifier; see README before making a formal certificate.
    emarc_margin: bool = True
    emarc_entropy_threshold: float = 0.20  # normalized by log(vocab size)
    emarc_margin_gamma: float = 0.05
    emarc_margin_mass_min: float = 0.75
    emarc_safe_top_k: int = 32
    emarc_safe_logit_gap: float = 5.0
    emarc_protected_z: float = 0.5
    emarc_min_score_gap: float = 0.0
    emarc_margin_allow_kl_override: bool = True

    # Optional finite-horizon direction. m=1 is exactly the original ADS
    # gradient after norm matching. m>1 runs Hessian-vector recurrences offline.
    emarc_horizon_steps: int = 1
    emarc_horizon_lr: float = 1e-3
    emarc_horizon_damping: float = 1e-3
    emarc_horizon_max_batches: int = 0  # 0 => every holdout batch

    # --------------------------------------------------------- distillation
    lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lr: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    train_batch_size: int = 16  # global batch size
    per_device_batch_size: int = 2
    num_epochs: float = 3.0
    train_max_length: int = 4096
    do_eval: bool = True  # evaluate the student on clean holdout traces
    train_on_answer_forced: bool = False  # distil on the answer-forced trace instead

    # --------------------------------------------------------------- runtime
    seed: int = 42
    exp_dir: str = "experiments"
    attn_impl: str = "auto"  # auto | flash_attention_2 | sdpa | eager
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    num_proc: int = 0  # dataset .map workers; 0 => min(8, cpu_count)
    eval_teacher: bool = True  # also measure the teacher on the test split
    overwrite: bool = False  # ignore stage sentinels and redo everything
    num_gpus: int = 0  # 0 => every visible GPU (1 process on CPU)
    launcher: str = "auto"  # auto | accelerate | python

    # --------------------------------------------------- per-stage overrides
    # run.py fills these in when it launches a stage; you only set them by hand
    # if you want to call generate.py / distill.py directly.
    gen_model: str = ""  # "" => teacher
    gen_tokenizer: str = ""  # "" => tokenizer that ships with gen_model
    gen_split: str = "train"  # train | holdout | test
    gen_out: str = ""  # "" => derived from the split
    gen_use_ads: str = "auto"  # auto (follow --ads) | true | false
    gen_method: str = "auto"  # auto | plain | ads | emarc
    gen_max_samples: int = -1  # -1 => the max_*_samples field for the split
    gen_label: str = ""  # cosmetic, shown in the progress bar

    # ================================================================ helpers
    @property
    def mode(self) -> str:
        if self.emarc:
            return "emarc"
        return "ads" if self.ads else "normal"

    @property
    def run_name(self) -> str:
        if self.emarc:
            return (
                f"emarc_tau{self.tau:g}_alpha{self.emarc_alpha_init:g}_"
                f"m{self.emarc_horizon_steps}_eps{self.eps:g}"
            )
        if self.ads:
            return f"ads_tau{self.tau:g}_lam{self.lam:g}_eps{self.eps:g}"
        return f"normal_tau{self.tau:g}"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.exp_dir, self.run_name)

    @property
    def traces_dir(self) -> str:
        return os.path.join(self.exp_dir, "traces")

    @property
    def holdout_traces(self) -> str:
        # Shared between runs: always clean (no ADS), so a single copy is valid
        # for both modes and both the gradient step and student evaluation.
        return os.path.join(self.traces_dir, "holdout")

    @property
    def train_traces(self) -> str:
        return os.path.join(self.traces_dir, f"{self.run_name}_train")

    @property
    def grad_path(self) -> str:
        return os.path.join(self.exp_dir, "proxy_student_grads.pt")

    @property
    def direction_path(self) -> str:
        if not self.emarc:
            return self.grad_path
        name = (
            f"emarc_direction_m{self.emarc_horizon_steps}_"
            f"lr{self.emarc_horizon_lr:g}_damp{self.emarc_horizon_damping:g}_"
            f"b{self.emarc_horizon_max_batches}.pt"
        )
        return os.path.join(self.exp_dir, name)

    @property
    def model_path(self) -> str:
        return os.path.join(self.run_dir, "student")

    @property
    def student_final(self) -> str:
        return os.path.join(self.model_path, "final")

    @property
    def eval_student_traces(self) -> str:
        return os.path.join(self.traces_dir, f"{self.run_name}_eval_student")

    @property
    def eval_teacher_traces(self) -> str:
        return os.path.join(self.traces_dir, f"{self.run_name}_eval_teacher")

    @property
    def map_workers(self) -> int:
        return (
            self.num_proc if self.num_proc > 0 else max(1, min(8, os.cpu_count() or 1))
        )

    # =============================================================== loading
    @classmethod
    def load(cls, argv: Optional[list] = None) -> "Config":
        cfg = cls()
        type_by_name = {f.name: f.type for f in fields(cls)}

        # 2. environment
        for name, ftype in type_by_name.items():
            env = os.environ.get(name.upper())
            if env is not None:
                setattr(cfg, name, _coerce(env, ftype, name))

        # 3. command line
        argv = list(sys.argv[1:] if argv is None else argv)
        i = 0
        while i < len(argv):
            tok = argv[i]
            if not tok.startswith("--"):
                raise SystemExit(f"unexpected argument {tok!r} (use --key=value)")
            if "=" in tok:
                key, value = tok[2:].split("=", 1)
                i += 1
            else:
                key = tok[2:]
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    value, i = argv[i + 1], i + 2
                else:  # bare --flag means true
                    value, i = "true", i + 1
            key = key.replace("-", "_")
            if key not in type_by_name:
                raise SystemExit(
                    f"unknown option --{key}. known: {sorted(type_by_name)}"
                )
            setattr(cfg, key, _coerce(value, type_by_name[key], key))

        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.ads = to_bool(self.ads)
        self.emarc = to_bool(self.emarc)
        self.normal = to_bool(self.normal)
        selected = int(self.ads) + int(self.emarc) + int(self.normal)
        if selected != 1:
            raise SystemExit(
                "Set exactly one of EMARC / ADS / NORMAL.\n"
                "  EMARC=false ADS=false NORMAL=true  -> plain distillation\n"
                "  EMARC=false ADS=true  NORMAL=false -> static ADS\n"
                "  EMARC=true  ADS=false NORMAL=false -> adaptive EMARC-ADS\n"
                f"got EMARC={self.emarc} ADS={self.ads} NORMAL={self.normal}"
            )
        if self.tau < 0 or self.holdout_tau < 0 or self.eval_tau < 0:
            raise SystemExit("generation temperatures must be >= 0")
        if not 0 < self.top_p <= 1:
            raise SystemExit("top_p must be in (0, 1]")
        if self.gen_batch_size < 1 or self.grad_batch_size < 1:
            raise SystemExit("generation and gradient batch sizes must be >= 1")
        if self.ads:
            if self.lam <= 0:
                raise SystemExit(
                    "ADS mode needs lam > 0 (lam is the antidistillation strength)"
                )
            if self.eps <= 0:
                raise SystemExit("ADS mode needs eps > 0 (finite-difference step size)")
        elif self.emarc:
            self.lam = 0.0  # EMARC uses emarc_alpha_init, not static lam
            if self.eps <= 0:
                raise SystemExit(
                    "EMARC mode needs eps > 0 (finite-difference step size)"
                )
            if self.emarc_alpha_min < 0 or self.emarc_alpha_max <= 0:
                raise SystemExit("EMARC alpha bounds need 0 <= min and max > 0")
            if (
                not self.emarc_alpha_min
                <= self.emarc_alpha_init
                <= self.emarc_alpha_max
            ):
                raise SystemExit(
                    "emarc_alpha_init must lie inside [alpha_min, alpha_max]"
                )
            if self.emarc_alpha_lr < 0 or self.emarc_antiwindup < 0:
                raise SystemExit("EMARC feedback gains must be non-negative")
            if self.emarc_target_flow < 0:
                raise SystemExit("emarc_target_flow must be non-negative")
            if self.emarc_sigma <= 0 or self.emarc_attacker_eta <= 0:
                raise SystemExit("emarc_sigma and emarc_attacker_eta must be > 0")
            if self.emarc_attacker_iters < 1 or self.emarc_max_backtracks < 0:
                raise SystemExit("EMARC iteration counts are invalid")
            if not 0 < self.emarc_mass_min <= self.emarc_mass_max:
                raise SystemExit("EMARC mass bounds need 0 < min <= max")
            if self.emarc_value_source not in {"confidence", "low_entropy", "uniform"}:
                raise SystemExit(
                    "emarc_value_source must be confidence, low_entropy, or uniform"
                )
            if not 0 <= self.emarc_value_floor <= 1:
                raise SystemExit("emarc_value_floor must be in [0, 1]")
            if self.emarc_beta_max <= 0 or self.emarc_kl_cap < 0:
                raise SystemExit("emarc_beta_max must be > 0 and emarc_kl_cap >= 0")
            if self.emarc_typicality_kappa < 0:
                raise SystemExit("emarc_typicality_kappa must be >= 0")
            if not 0 <= self.emarc_entropy_threshold <= 1:
                raise SystemExit("emarc_entropy_threshold must be in [0, 1]")
            if self.emarc_margin_gamma < 0 or self.emarc_margin_mass_min < 0:
                raise SystemExit("EMARC margin thresholds must be non-negative")
            if self.emarc_protected_z < 0 or self.emarc_min_score_gap < 0:
                raise SystemExit(
                    "EMARC protected-set score thresholds must be non-negative"
                )
            if self.emarc_safe_top_k < 1 or self.emarc_safe_logit_gap < 0:
                raise SystemExit("EMARC safe-set top-k must be >= 1 and logit gap >= 0")
            if self.emarc_horizon_steps < 1 or self.emarc_horizon_lr <= 0:
                raise SystemExit("EMARC horizon needs steps >= 1 and lr > 0")
            if self.emarc_horizon_damping < 0 or self.emarc_horizon_max_batches < 0:
                raise SystemExit("EMARC horizon damping/max_batches cannot be negative")
        else:
            # In NORMAL mode the ADS term is switched off, full stop.
            self.lam = 0.0
            self.eps = 0.0
        if self.gen_method not in {"auto", "plain", "ads", "emarc"}:
            raise SystemExit("gen_method must be auto, plain, ads, or emarc")
        if self.train_batch_size % self.per_device_batch_size != 0:
            raise SystemExit(
                "train_batch_size must be a multiple of per_device_batch_size"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode
        d["run_name"] = self.run_name
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    def pretty(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

"""Dependency-free guardrails for the safety-critical repository wiring."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "soft_distill.py").read_text(encoding="utf-8")


def require(text: str) -> None:
    assert text in SOURCE, f"missing required implementation fragment: {text!r}"


ast.parse(SOURCE)
require("CACHE_SCHEMA_VERSION = 2")
require("def validate_shared_tokenizers")
require("def source_fingerprint")
require("def save_training_checkpoint")
require("def restore_training_state")
require("def validate_checkpoint_contract")
require("teacher_logsumexp")
require('choices=["tail_bucket", "student_full", "student_renorm"]')
require("--resume_from_checkpoint")
require('choices=["train", "precompute", "validate_cache"]')
require('"data_signature": data_signature')
require("def validate_completed_run")
require("--overwrite_cache")

for path in (ROOT / "slurm" / "top64_precompute.slurm", ROOT / "slurm" / "top64_train_array.slurm"):
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in text
    assert "/raid/mtanveer" in text

train = (ROOT / "slurm" / "top64_train_array.slurm").read_text(encoding="utf-8")
precompute = (ROOT / "slurm" / "top64_precompute.slurm").read_text(encoding="utf-8")
assert "--kl_variant tail_bucket" in train
assert "--resume_from_checkpoint auto" in train
assert "--train_traces" in train
assert "--mode validate_cache" in precompute

print("STATIC CONTRACT CHECKS PASSED")

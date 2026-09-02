#!/usr/bin/env python3
"""Fail-fast checks before submitting a long top-64 soft-distillation run."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--run_root", required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"PASS  {message}")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_dir).expanduser().resolve()
    traces = Path(args.traces).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()

    require(repo.is_dir(), f"repo exists: {repo}")
    require((repo / "soft_distill.py").is_file(), "soft_distill.py is present")
    require((repo / "slurm" / "top64_precompute.slurm").is_file(), "cache Slurm script is present")
    require((repo / "slurm" / "top64_train_array.slurm").is_file(), "training Slurm script is present")
    require(traces.exists(), f"trace source exists: {traces}")

    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", "soft_distill.py", "scripts/preflight.py"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    require(compile_result.returncode == 0, "Python sources compile")

    import torch
    import transformers
    import datasets
    import accelerate
    import peft

    print(
        "VERSIONS "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"datasets={datasets.__version__} accelerate={accelerate.__version__} "
        f"peft={getattr(peft, '__version__', 'unknown')}"
    )
    require(torch.cuda.is_available(), "CUDA is available to this environment")
    print(f"GPU   {torch.cuda.get_device_name(0)}")

    sys.path.insert(0, str(repo))
    from soft_distill import build_parser, load_pairs  # noqa: PLC0415

    check_args = build_parser().parse_args([
        "--top_k", "64", "--train_traces", str(traces), "--input_format", "repo",
    ])
    pairs = load_pairs(str(traces), check_args, 3)
    require(len(pairs) > 0, "at least one usable prompt/completion pair was read")
    require(all(p["prompt"].strip() and p["completion"].strip() for p in pairs),
            "sampled prompt/completion pairs are non-empty")

    run_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(run_root).free / (1024 ** 3)
    require(free >= 100, f"at least 100 GiB free under {run_root} (found {free:.1f} GiB)")
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

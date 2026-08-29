# -*- coding: utf-8 -*-
"""
The whole experiment, driven by three mutually exclusive mode booleans.

    EMARC=false ADS=false NORMAL=true  python run.py  # plain distillation
    EMARC=false ADS=true  NORMAL=false python run.py  # static ADS
    EMARC=true  ADS=false NORMAL=false python run.py  # adaptive EMARC-ADS

NORMAL runs:                          ADS runs:
    1. holdout traces (teacher)           1. holdout traces (teacher, greedy, no ADS)
    2. training traces (teacher)          2. proxy-student gradients on those traces
    3. distil the student                 3. training traces (teacher + ADS term)
    4. score the student on test          4. distil the student
    5. score the teacher on test          5. score the student on test
                                          6. score the teacher on test (with ADS)

Each stage is a separate process so it can be launched under `accelerate` on
multiple GPUs, and each writes a sentinel so a re-run resumes instead of
repeating work (pass --overwrite=true to force).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import List, Optional

from config import Config


# --------------------------------------------------------------------------- #
def _flags(cfg: Config, **overrides) -> List[str]:
    payload = cfg.to_dict()
    payload.pop("mode", None)
    payload.pop("run_name", None)
    payload.update(overrides)
    return [f"--{k}={'none' if v is None else v}" for k, v in sorted(payload.items())]


def _launcher(cfg: Config) -> List[str]:
    choice = cfg.launcher
    if choice == "auto":
        try:
            import torch

            gpus = cfg.num_gpus or torch.cuda.device_count()
        except Exception:
            gpus = 0
        choice = "accelerate" if gpus > 1 else "python"
    if choice == "python":
        return [sys.executable]
    try:
        import torch

        gpus = cfg.num_gpus or torch.cuda.device_count()
    except Exception:
        gpus = 1
    return [
        "accelerate",
        "launch",
        f"--num_processes={max(1, gpus)}",
        "--mixed_precision=no",
    ]


def _done(path: str) -> bool:
    if path.endswith(".pt") or path.endswith(".json"):
        return os.path.exists(path)
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "dataset_info.json")
    )


def _run_stage(
    cfg: Config,
    name: str,
    script: str,
    sentinel: str,
    force_python: bool = False,
    **overrides,
) -> None:
    if not cfg.overwrite and (
        _done(sentinel) or os.path.exists(os.path.join(sentinel, "config.json"))
    ):
        print(f"\n⏭  skip {name}: {sentinel} already exists")
        return
    launcher = [sys.executable] if force_python else _launcher(cfg)
    cmd = (
        launcher
        + [os.path.join(os.path.dirname(os.path.abspath(__file__)), script)]
        + _flags(cfg, **overrides)
    )
    print(f"\n▶  {name}\n   {' '.join(cmd[:4])} ... ({len(cmd)} argv)")
    started = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        raise SystemExit(f"stage {name!r} failed with exit code {result.returncode}")
    print(f"✅ {name} finished in {time.time() - started:.1f}s")


def _read_summary(path: str) -> Optional[dict]:
    meta = path + ".json"
    if not os.path.exists(meta):
        return None
    with open(meta) as fh:
        return json.load(fh).get("summary")


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = Config.load()
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(cfg.traces_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.run_dir, "config.json"))

    print("=" * 78)
    print(
        f"MODE: {cfg.mode.upper()}   "
        f"(EMARC={cfg.emarc}, ADS={cfg.ads}, NORMAL={cfg.normal})"
    )
    print(f"run  : {cfg.run_name}")
    print(f"dir  : {os.path.abspath(cfg.run_dir)}")
    print("=" * 78)
    print(cfg.pretty())

    started = time.time()

    # 1. holdout traces -- clean teacher traces. ADS needs them for the gradient;
    #    both modes use them as the student's evaluation set.
    if cfg.ads or cfg.emarc or cfg.do_eval:
        _run_stage(
            cfg,
            "holdout traces",
            "generate.py",
            cfg.holdout_traces,
            gen_split="holdout",
            gen_model=cfg.teacher,
            gen_tokenizer=cfg.teacher,
            gen_out=cfg.holdout_traces,
            gen_method="plain",
            tau=cfg.holdout_tau,
            gen_label="holdout",
        )

    # 2. proxy-student gradients (both protected modes)
    if cfg.ads or cfg.emarc:
        _run_stage(cfg, "proxy-student gradients", "grads.py", cfg.grad_path)

    # 2b. optional multi-step direction (EMARC only). Exact HVPs use
    # autograd.grad and eager attention, so this deliberately runs one process.
    if cfg.emarc:
        _run_stage(
            cfg,
            "EMARC finite-horizon direction",
            "horizon.py",
            cfg.direction_path,
            force_python=True,
        )

    # 3. training traces -- the stage that distinguishes all three modes
    _run_stage(
        cfg,
        f"training traces ({cfg.mode})",
        "generate.py",
        cfg.train_traces,
        gen_split="train",
        gen_model=cfg.teacher,
        gen_tokenizer=cfg.teacher,
        gen_out=cfg.train_traces,
        gen_method=cfg.mode if cfg.mode != "normal" else "plain",
        gen_label=f"train/{cfg.mode}",
    )

    # 4. distillation
    _run_stage(cfg, "distillation", "distill.py", cfg.student_final)

    # 5. student on the test split (never uses ADS -- the attacker has no reason to)
    _run_stage(
        cfg,
        "evaluate student",
        "generate.py",
        cfg.eval_student_traces,
        gen_split="test",
        gen_model=cfg.student_final,
        gen_tokenizer=cfg.student_final,
        gen_out=cfg.eval_student_traces,
        gen_method="plain",
        tau=cfg.eval_tau,
        gen_label="test/student",
    )

    # 6. teacher on the test split, sampling exactly as it did for the training
    #    traces -- this is the utility the defence costs.
    if cfg.eval_teacher:
        _run_stage(
            cfg,
            "evaluate teacher",
            "generate.py",
            cfg.eval_teacher_traces,
            gen_split="test",
            gen_model=cfg.teacher,
            gen_tokenizer=cfg.teacher,
            gen_out=cfg.eval_teacher_traces,
            gen_method=cfg.mode if cfg.mode != "normal" else "plain",
            gen_label=f"test/teacher/{cfg.mode}",
        )

    # ------------------------------------------------------------- results
    results = {
        "mode": cfg.mode,
        "run_name": cfg.run_name,
        "tau": cfg.tau,
        "lam": cfg.lam,
        "eps": cfg.eps,
        "emarc_alpha_init": cfg.emarc_alpha_init if cfg.emarc else None,
        "emarc_horizon_steps": cfg.emarc_horizon_steps if cfg.emarc else None,
        "train_traces": _read_summary(cfg.train_traces),
        "holdout_traces": _read_summary(cfg.holdout_traces),
        "student_test": _read_summary(cfg.eval_student_traces),
        "teacher_test": _read_summary(cfg.eval_teacher_traces),
        "distillation": (_read_summary(cfg.model_path) or {}),
        "wall_seconds": round(time.time() - started, 1),
    }
    out = os.path.join(cfg.run_dir, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    def line(label: str, summary: Optional[dict]) -> str:
        if not summary:
            return f"  {label:<28} --"
        acc = summary.get("accuracy_af", summary.get("accuracy"))
        return f"  {label:<28} {acc * 100:6.2f}%   (n={summary['n']})"

    print("\n" + "=" * 78)
    print(f"RESULTS [{cfg.mode}]  tau={cfg.tau:g} lam={cfg.lam:g} eps={cfg.eps:g}")
    print("=" * 78)
    print(line("teacher, training traces", results["train_traces"]))
    print(line("teacher on test", results["teacher_test"]))
    print(line("distilled student on test", results["student_test"]))
    print(f"\nwritten to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

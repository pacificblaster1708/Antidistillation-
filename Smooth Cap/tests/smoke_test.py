# -*- coding: utf-8 -*-
"""
End-to-end smoke test: runs both pipelines, top to bottom, on tiny offline
fixtures. Every stage really executes -- generation, the ADS logits processor,
the proxy-student gradient, LoRA SFT, merging, and evaluation.

Run:  python tests/smoke_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tests.tiny_setup import build_all  # noqa: E402

FIXTURES = "/tmp/ads_fixtures"
EXP = "/tmp/ads_smoke"

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


def common_flags(fx):
    return [
        f"--exp_dir={EXP}",
        f"--dataset={fx['dataset']}",
        f"--teacher={fx['teacher']}",
        f"--proxy_student={fx['proxy_student']}",
        f"--student={fx['student']}",
        f"--student_tokenizer={fx['student_tokenizer']}",
        "--dtype=float32", "--attn_impl=eager", "--num_proc=1", "--launcher=python",
        "--tau=1.0", "--top_p=1.0", "--holdout_tau=0.0", "--eval_tau=0.0",
        "--gen_batch_size=4", "--max_new_tokens=12", "--max_prompt_length=512",
        "--answer_force=true", "--answer_force_tokens=4",
        "--grad_batch_size=2",
        "--train_batch_size=2", "--per_device_batch_size=2", "--num_epochs=1",
        "--lora=true", "--lora_r=4", "--lora_alpha=8", "--lr=1e-3",
        "--train_max_length=768", "--do_eval=true", "--eval_teacher=true",
        "--seed=123",
    ]


def run_pipeline(fx, mode_flags, title):
    cmd = [sys.executable, os.path.join(ROOT, "run.py")] + mode_flags + common_flags(fx)
    print(f"\n$ {title}")
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    took = time.time() - started
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:])
        raise SystemExit(f"{title} failed with exit code {proc.returncode}")
    print(f"  ({took:.1f}s)")
    return proc.stdout


def load_results(run_name):
    with open(os.path.join(EXP, run_name, "results.json")) as fh:
        return json.load(fh)


def main():
    shutil.rmtree(EXP, ignore_errors=True)
    fx = build_all(FIXTURES)

    # ------------------------------------------------------------- NORMAL
    out_normal = run_pipeline(fx, ["--ads=false", "--normal=true"], "ADS=false NORMAL=true")
    print("\n[normal] pipeline")
    check("mode banner says NORMAL", "MODE: NORMAL" in out_normal)
    normal = load_results("normal_tau1")
    check("results.json written", normal["mode"] == "normal")
    check("lam forced to 0", normal["lam_min"] == 0.0)
    check("training traces produced", normal["train_traces"]["n"] == 8)
    check("training traces are ADS-free", normal["train_traces"]["use_ads"] is False)
    check("student scored on test", normal["student_test"]["n"] == 6)
    check("teacher scored on test", normal["teacher_test"]["n"] == 6)
    check("student weights saved",
          os.path.exists(os.path.join(EXP, "normal_tau1", "student", "final", "config.json")))
    check("no proxy gradients in NORMAL mode",
          not os.path.exists(os.path.join(EXP, "proxy_student_grads.pt")))
    check("SFT loss recorded", "train_loss" in normal["distillation"]["metrics"])
    check("SFT evaluated on holdout", "eval_loss" in normal["distillation"]["metrics"])

    # ---------------------------------------------------------------- ADS
    out_ads = run_pipeline(fx, ["--ads=true", "--normal=false",
                                "--lam_min=0.01", "--lam_max=0.5", "--eps=0.01"],
                           "ADS=true NORMAL=false")
    print("\n[ads] pipeline")
    check("mode banner says ADS", "MODE: ADS" in out_ads)
    ads = load_results("ads_tau1_lmin0.01_lmax0.5_eps0.01")
    check("results.json written", ads["mode"] == "ads")
    check("proxy gradients saved", os.path.exists(os.path.join(EXP, "proxy_student_grads.pt")))
    check("training traces used ADS", ads["train_traces"]["use_ads"] is True)
    check("teacher eval used ADS", ads["teacher_test"]["use_ads"] is True)
    check("student eval did NOT use ADS", ads["student_test"]["use_ads"] is False)
    check("student weights saved",
          os.path.exists(os.path.join(EXP, "ads_tau1_lmin0.01_lmax0.5_eps0.01", "student",
                                      "final", "config.json")))

    import torch
    payload = torch.load(os.path.join(EXP, "proxy_student_grads.pt"),
                         map_location="cpu", weights_only=False)
    check("gradient file carries metadata", payload["meta"]["proxy_student"] == fx["proxy_student"])
    check("gradient is non-zero", payload["meta"]["grad_norm"] > 0,
          f"norm {payload['meta']['grad_norm']:.3e}")
    check("gradient covers every parameter tensor", len(payload["grads"]) > 10,
          f"{len(payload['grads'])} tensors")

    # ------------------------------------------- the two modes really differ
    print("\n[compare] the ADS term changes what the teacher writes")
    from datasets import load_from_disk
    plain = load_from_disk(os.path.join(EXP, "traces", "normal_tau1_train"))["completion"]
    poisoned = load_from_disk(os.path.join(EXP, "traces",
                                           "ads_tau1_lmin0.01_lmax0.5_eps0.01_train"))["completion"]
    differing = sum(1 for a, b in zip(plain, poisoned) if a != b)
    check("same number of traces", len(plain) == len(poisoned))
    check("ADS traces differ from plain traces", differing > 0,
          f"{differing}/{len(plain)} completions changed")
    check("holdout traces shared between runs",
          normal["holdout_traces"]["path"] == ads["holdout_traces"]["path"])

    # ------------------------------------------------------- resumability
    print("\n[resume] a second run skips completed stages")
    again = run_pipeline(fx, ["--ads=true", "--normal=false",
                               "--lam_min=0.01", "--lam_max=0.5", "--eps=0.01"],
                         "ADS=true NORMAL=false (resume)")
    check("skipped previously finished stages", again.count("skip") >= 5,
          f"{again.count('skip')} stages skipped")

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print("  FAILED:", name)
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

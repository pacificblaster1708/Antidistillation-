#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_agreement.py -- how close did the student actually get to the teacher?

Training loss tells you the objective is going down. This tells you what that
bought you, in units you can put in a table:

  top-1 agreement   fraction of positions where the student's argmax equals the
                    teacher's argmax. The headline number.
  top-K overlap     mean |student top-K  n  teacher top-K| / K. Rewards getting the
                    shortlist right even when the ordering differs.
  forward KL        mean KL(teacher top-K || student), the training objective
                    itself, evaluated on held-out data.
  teacher CE        cross-entropy of the student against the teacher's argmax.
  coverage          share of the teacher's full-vocab probability mass that its
                    own top-K holds -- tells you whether K is large enough.

Run it on the base student and on the distilled student and compare.

  python scripts/evaluate_agreement.py \
      --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --student ./experiments/models/soft_k50/final \
      --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --traces ./experiments/traces/holdout \
      --top_k 50 --max_samples 200

Add --baseline_student Qwen/Qwen2.5-3B to print a before/after table in one go.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soft_distill import (  # noqa: E402
    Collator, OnlineDataset, build_vocab_map, encode_all, load_pairs,
    load_tokenizer, resolve_attn, teacher_topk,
)


def parse_args():
    p = argparse.ArgumentParser(description="Measure student/teacher agreement.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True, help="Distilled student to score.")
    p.add_argument("--baseline_student", default=None,
                   help="Optional second student (e.g. the un-distilled base) for a before/after table.")
    p.add_argument("--tokenizer", default=None, help="Teacher tokenizer. Defaults to --teacher.")
    p.add_argument("--student_tokenizer", default=None, help="Only needed for --vocab_mode cross.")
    p.add_argument("--vocab_mode", choices=["shared", "cross"], default="shared")
    p.add_argument("--traces", required=True, help="load_from_disk path or .jsonl")
    p.add_argument("--input_format", choices=["repo", "jsonl"], default="repo")
    p.add_argument("--trace_colname", default="auto")
    p.add_argument("--problem_colname", default="auto")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    return p.parse_args()


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@torch.no_grad()
def score(student_path, args, examples, s_tok, t_tok, teacher, vocab_map, dtype, device):
    from transformers import AutoModelForCausalLM

    attn = resolve_attn(args.attn_implementation)
    student = AutoModelForCausalLM.from_pretrained(
        student_path, trust_remote_code=True, torch_dtype=dtype, attn_implementation=attn
    ).to(device).eval()
    s_vocab = student.config.vocab_size

    coll = Collator(s_tok.pad_token_id, t_tok.pad_token_id, cached=False)
    ds = OnlineDataset(examples)

    n = hits = 0
    overlap = kl_sum = ce_sum = cov_sum = 0.0
    for i in range(0, len(examples), args.batch_size):
        feats = [ds[j] for j in range(i, min(i + args.batch_size, len(examples)))]
        b = coll(feats)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        t_ids, t_vals, cov, full_lse = teacher_topk(
            teacher, b["teacher_input_ids"], b["teacher_attention_mask"],
            b["kl_batch_idx"], b["kl_teacher_pos"], args.top_k, vocab_map, s_vocab,
            temperature=args.temperature,
        )
        out = student(input_ids=b["student_input_ids"], attention_mask=b["student_attention_mask"])
        L = out.logits.size(1)
        flat = out.logits.reshape(-1, out.logits.size(-1))
        rows = flat.index_select(0, b["kl_batch_idx"] * L + b["kl_student_pos"]).float()

        valid = torch.isfinite(t_vals)
        keep = valid.any(-1)
        if not bool(keep.any()):
            continue
        rows, t_ids, t_vals, valid, cov, full_lse = (
            rows[keep], t_ids[keep], t_vals[keep], valid[keep], cov[keep], full_lse[keep]
        )

        # The teacher's own argmax is the first valid entry of its sorted top-K.
        first_valid = valid.float().argmax(dim=-1)
        t_top1 = t_ids.gather(1, first_valid.unsqueeze(1)).squeeze(1)
        s_top1 = rows.argmax(-1)
        hits += int((s_top1 == t_top1).sum())

        k_eff = int(valid.sum(-1).clamp(min=1).float().mean().round())
        s_topk = rows.topk(min(args.top_k, rows.size(-1)), dim=-1).indices
        for r in range(rows.size(0)):
            tset = set(t_ids[r][valid[r]].tolist())
            sset = set(s_topk[r].tolist())
            overlap += len(tset & sset) / max(1, len(tset))

        log_p_t = t_vals / args.temperature - full_lse.unsqueeze(-1)
        p_t = torch.where(valid, log_p_t.exp(), torch.zeros_like(log_p_t))
        log_p_s = torch.log_softmax(rows / args.temperature, dim=-1).gather(1, t_ids.clamp_min(0))
        term = p_t * (log_p_t - log_p_s)
        eps = torch.finfo(torch.float32).eps
        p_tail = (1.0 - p_t.sum(-1)).clamp(min=0.0, max=1.0)
        q_tail = (1.0 - torch.where(valid, log_p_s.exp(), torch.zeros_like(log_p_s)).sum(-1)).clamp(
            min=0.0, max=1.0
        )
        tail_term = torch.where(
            p_tail > 0,
            p_tail * (p_tail.clamp_min(eps).log() - q_tail.clamp_min(eps).log()),
            torch.zeros_like(p_tail),
        )
        kl_sum += float((torch.where(valid, term, torch.zeros_like(term)).sum(-1) + tail_term).sum())
        ce_sum += float(torch.nn.functional.cross_entropy(rows, t_top1, reduction="sum"))
        cov_sum += float(cov.sum())
        n += rows.size(0)

    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = max(1, n)
    return {"positions": n, "top1_agreement": hits / n, "topk_overlap": overlap / n,
            "forward_kl": kl_sum / n, "teacher_ce": ce_sum / n, "coverage": cov_sum / n}


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM

    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    t_name = args.tokenizer or args.teacher
    t_tok = load_tokenizer(t_name)
    s_tok = t_tok if args.vocab_mode == "shared" else load_tokenizer(args.student_tokenizer or args.student)

    shim = argparse.Namespace(vocab_mode=args.vocab_mode, max_length=args.max_length,
                              input_format=args.input_format, trace_colname=args.trace_colname,
                              problem_colname=args.problem_colname, dataset_split=None)
    pairs = load_pairs(args.traces, shim, args.max_samples)
    examples = encode_all(pairs, shim, s_tok, t_tok)

    attn = resolve_attn(args.attn_implementation)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, trust_remote_code=True, torch_dtype=dtype, attn_implementation=attn
    ).to(device).eval()

    vocab_map = None
    if args.vocab_mode == "cross":
        vocab_map = build_vocab_map(t_tok, s_tok, teacher.config.vocab_size,
                                    2 ** 31 - 1).to(device)

    results = {}
    if args.baseline_student:
        results["baseline"] = score(args.baseline_student, args, examples, s_tok, t_tok,
                                    teacher, vocab_map, dtype, device)
    results["distilled"] = score(args.student, args, examples, s_tok, t_tok,
                                 teacher, vocab_map, dtype, device)

    cols = ["positions", "top1_agreement", "topk_overlap", "forward_kl", "teacher_ce", "coverage"]
    width = max(len(c) for c in cols) + 2
    print("\n" + "=" * 78)
    print(f"Agreement with teacher {args.teacher}   (K={args.top_k})")
    print("=" * 78)
    print("metric".ljust(width) + "".join(name.rjust(14) for name in results))
    for c in cols:
        row = c.ljust(width)
        for name in results:
            v = results[name][c]
            row += (f"{v:14d}" if isinstance(v, int) else f"{v:14.4f}")
        print(row)
    print("=" * 78)
    print("top1_agreement and topk_overlap: higher is better.")
    print("forward_kl and teacher_ce:       lower is better.")
    print("coverage: teacher mass inside its own top-K. Below ~0.9, consider a larger K.")


if __name__ == "__main__":
    main()

"""Functional test: does soft top-K distillation actually move the student
toward the teacher's distribution? Measures top-1 agreement and KL before/after."""
import atexit, os, shutil, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIXTURES, REPO_ROOT, TMP  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer
from soft_distill import (build_parser, load_pairs, encode_all, OnlineDataset, Collator,
                          teacher_topk)

FIX = FIXTURES
OUT = tempfile.mkdtemp(prefix="learning_run_", dir=TMP)
atexit.register(shutil.rmtree, OUT, ignore_errors=True)


def agreement(student_path, args, examples, s_tok, t_tok):
    st = AutoModelForCausalLM.from_pretrained(student_path, attn_implementation="sdpa",
                                              torch_dtype=torch.float32).eval()
    te = AutoModelForCausalLM.from_pretrained(f"{FIX}/teacher_qwen", attn_implementation="sdpa",
                                              torch_dtype=torch.float32).eval()
    coll = Collator(s_tok.pad_token_id, t_tok.pad_token_id, cached=False)
    ds = OnlineDataset(examples)
    hits = tot = 0
    kls = []
    with torch.no_grad():
        for i in range(0, len(examples), 4):
            b = coll([ds[j] for j in range(i, min(i + 4, len(examples)))])
            ids, vals, _, _ = teacher_topk(
                te, b["teacher_input_ids"], b["teacher_attention_mask"],
                b["kl_batch_idx"], b["kl_teacher_pos"], args.top_k,
                None, st.config.vocab_size, temperature=args.temperature,
            )
            out = st(input_ids=b["student_input_ids"], attention_mask=b["student_attention_mask"])
            L = out.logits.size(1)
            flat = out.logits.reshape(-1, out.logits.size(-1))
            rows = flat[b["kl_batch_idx"] * L + b["kl_student_pos"]].float()
            hits += int((rows.argmax(-1) == ids[:, 0]).sum())
            tot += rows.size(0)
            p_t = torch.softmax(vals, -1)
            lp_s = torch.log_softmax(rows, -1).gather(1, ids)
            kls.append((p_t * (torch.log(p_t.clamp_min(1e-30)) - lp_s)).sum(-1).mean())
    return hits / tot, float(torch.stack(kls).mean())


def main():
    args = build_parser().parse_args([
        "--top_k", "8", "--temperature", "1.0", "--max_length", "256", "--vocab_mode", "shared",
        "--teacher", f"{FIX}/teacher_qwen", "--student", f"{FIX}/student_qwen_big",
        "--train_traces", f"{FIX}/traces",
    ])
    s_tok = AutoTokenizer.from_pretrained(f"{FIX}/tok_qwen")
    s_tok.padding_side = "right"
    pairs = load_pairs(f"{FIX}/traces", args, None)
    examples = encode_all(pairs, args, s_tok, s_tok)

    a0, k0 = agreement(f"{FIX}/student_qwen_big", args, examples, s_tok, s_tok)
    print(f"BEFORE  top-1 agreement with teacher: {a0:.3f}   mean top-K KL: {k0:.4f}")

    cmd = [sys.executable, os.path.join(REPO_ROOT, "soft_distill.py"), "--mode", "train", "--teacher_logits", "online",
           "--vocab_mode", "shared", "--teacher", f"{FIX}/teacher_qwen",
           "--teacher_tokenizer", f"{FIX}/tok_qwen", "--student", f"{FIX}/student_qwen_big",
           "--train_traces", f"{FIX}/traces", "--top_k", "8", "--temperature", "1.0",
           "--alpha", "1.0", "--max_length", "256", "--num_epochs", "160",
           "--batch_size", "8", "--per_device_batch_size", "4", "--no-lora",
           "--lr", "3e-3", "--model_dtype", "fp32", "--teacher_dtype", "fp32",
           "--attn_implementation", "sdpa", "--num_workers", "0", "--logging_steps", "80",
           "--output_dir", OUT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
    for line in r.stdout.splitlines():
        if "step " in line or "====" in line:
            print("   ", line.split("] ")[-1])

    a1, k1 = agreement(os.path.join(OUT, "final"), args, examples, s_tok, s_tok)
    print(f"AFTER   top-1 agreement with teacher: {a1:.3f}   mean top-K KL: {k1:.4f}")

    ok = a1 > a0 and k1 < k0 and a1 > 0.85 and k1 < 0.05
    print("\nLEARNING TEST PASSED" if ok else "\nLEARNING TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

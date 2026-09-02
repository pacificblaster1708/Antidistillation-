"""tests/test_kl.py -- numeric checks for the top-K KL against slow, obvious reference code."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: F401,E402

import torch, torch.nn.functional as F  # noqa: E402
from soft_distill import topk_kl_loss, hard_ce_loss, IGNORE_INDEX

torch.manual_seed(0)
B, L, V, K = 3, 7, 40, 5


def reference_kl(student_logits, flat_idx, ids, tlog, T, variant):
    """Loop-based ground truth, one site at a time."""
    flat = student_logits.reshape(-1, V)
    vals = []
    for n in range(flat_idx.numel()):
        tl = tlog[n] / T
        valid = torch.isfinite(tl)
        if not valid.any():
            continue
        p_t = torch.softmax(tl.masked_fill(~valid, float("-inf")), dim=-1)
        row = flat[flat_idx[n]].float() / T
        if variant == "student_full":
            lps = torch.log_softmax(row, dim=-1)[ids[n]]
        else:
            sel = row[ids[n]].masked_fill(~valid, float("-inf"))
            lps = sel - torch.logsumexp(sel, dim=-1)
        lpt = torch.log(p_t.clamp_min(1e-45))
        kl = ((p_t * (lpt - lps))[valid]).sum()
        vals.append(kl)
    return torch.stack(vals).sum() / len(vals)


ok = True
for variant in ("student_full", "student_renorm"):
    for T in (1.0, 2.0, 4.0):
        s_logits = torch.randn(B, L, V, dtype=torch.float64).float().requires_grad_(True)
        n_sites = 9
        flat_idx = torch.randint(0, B * L, (n_sites,))
        ids = torch.stack([torch.randperm(V)[:K] for _ in range(n_sites)])
        tlog = torch.randn(n_sites, K) * 3
        # make some entries invalid (unmapped in cross-tokenizer mode)
        tlog[0, 2:] = float("-inf")
        tlog[3, 1] = float("-inf")
        tlog[5, :] = float("-inf")          # whole site dropped

        for chunk in (1, 4, 1000):
            got = topk_kl_loss(s_logits, flat_idx, ids, tlog, T, variant, chunk)
            want = reference_kl(s_logits, flat_idx, ids, tlog, T, variant)
            d = abs(float(got) - float(want))
            if d > 2e-5:
                ok = False
                print(f"MISMATCH variant={variant} T={T} chunk={chunk}: {float(got)} vs {float(want)}")
        print(f"  {variant:15s} T={T}  kl={float(got):.6f}  ref={float(want):.6f}  chunk-invariant OK")

# --- proper K+tail KL matches an explicit (K+1)-category reference ----------
for T in (1.0, 2.0, 4.0):
    n_sites = 9
    s_logits = torch.randn(B, L, V).requires_grad_(True)
    flat_idx = torch.randint(0, B * L, (n_sites,))
    teacher_full = torch.randn(n_sites, V) * 3
    tvals, tids = torch.topk(teacher_full, K, dim=-1)
    full_lse = torch.logsumexp(teacher_full / T, dim=-1)

    got = topk_kl_loss(
        s_logits, flat_idx, tids, tvals, T, "tail_bucket", 3,
        teacher_logsumexp=full_lse,
    )
    srows = s_logits.reshape(-1, V)[flat_idx] / T
    log_ps = torch.log_softmax(srows, -1).gather(1, tids)
    log_pt = tvals / T - full_lse[:, None]
    pt = log_pt.exp()
    eps = torch.finfo(torch.float32).eps
    ptail = (1 - pt.sum(-1)).clamp(min=0.0)
    qtail = (1 - log_ps.exp().sum(-1)).clamp(min=0.0)
    tail = torch.where(
        ptail > 0,
        ptail * (ptail.clamp_min(eps).log() - qtail.clamp_min(eps).log()),
        torch.zeros_like(ptail),
    )
    want = ((pt * (log_pt - log_ps)).sum(-1) + tail).mean()
    delta = abs(float(got) - float(want))
    print(f"  {'tail_bucket':15s} T={T}  kl={float(got):.6f}  ref={float(want):.6f}")
    ok &= delta < 2e-5

# Self-distillation is exactly zero after aggregating the common tail.
self_logits = torch.randn(1, 1, V)
self_vals, self_ids = torch.topk(self_logits.reshape(1, V), K, dim=-1)
self_lse = torch.logsumexp(self_logits.reshape(1, V), dim=-1)
self_tail = float(topk_kl_loss(
    self_logits, torch.tensor([0]), self_ids, self_vals, 1.0,
    "tail_bucket", 8, teacher_logsumexp=self_lse,
))
print(f"  self-distillation KL (tail bucket, should be 0): {self_tail:.3e}")
ok &= abs(self_tail) < 1e-6

# --- KL is zero when the student already matches the teacher exactly ---------
s_logits = torch.randn(B, L, V)
flat_idx = torch.arange(0, B * L, 3)
n = flat_idx.numel()
tvals, tids = torch.topk(s_logits.reshape(-1, V)[flat_idx], k=K, dim=-1)
z = float(topk_kl_loss(s_logits, flat_idx, tids, tvals, 1.0, "student_renorm", 8))
print(f"  self-distillation KL (renorm, should be 0): {z:.3e}")
ok &= abs(z) < 1e-6
z_full = float(topk_kl_loss(s_logits, flat_idx, tids, tvals, 1.0, "student_full", 8))
print(f"  self-distillation KL (student_full, = mass outside top-K, >0): {z_full:.4f}")
ok &= z_full > 0

# --- KL must be non-negative and shrink as the student is fitted -------------
target = torch.randn(1, 1, V)
tv, ti = torch.topk(target.reshape(1, V), k=K, dim=-1)
student = torch.zeros(1, 1, V, requires_grad=True)
opt = torch.optim.Adam([student], lr=0.3)
first = None
for step in range(400):
    opt.zero_grad()
    loss = topk_kl_loss(student, torch.tensor([0]), ti, tv, 1.0, "student_full", 8)
    loss.backward(); opt.step()
    if step == 0:
        first = float(loss)
print(f"  gradient descent on KL: {first:.4f} -> {float(loss):.6f}")
ok &= float(loss) < first and float(loss) < 1e-2

# --- the fitted student's top-K must equal the teacher's top-K ---------------
got_ids = set(torch.topk(student.detach().reshape(-1), K).indices.tolist())
print(f"  recovered teacher top-K ids: {got_ids == set(ti.reshape(-1).tolist())}")
ok &= got_ids == set(ti.reshape(-1).tolist())

# --- CE helper matches F.cross_entropy on the masked subset -----------------
logits = torch.randn(2, 5, V)
tgt = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 9, IGNORE_INDEX],
                    [IGNORE_INDEX, 7, 2, 1, IGNORE_INDEX]])
a = float(hard_ce_loss(logits, tgt))
b = float(F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=IGNORE_INDEX))
print(f"  CE helper vs F.cross_entropy: {a:.6f} vs {b:.6f}")
ok &= abs(a - b) < 1e-5

print("\nALL KL/CE CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)

# Antidistillation Sampling + EMARC-ADS

A stripped-down, working reimplementation of [`locuslab/antidistillation-sampling`](https://antidistillation.com).
It now has three mutually exclusive modes:

```bash
EMARC=false ADS=false NORMAL=true  python run.py  # plain distillation
EMARC=false ADS=true  NORMAL=false python run.py  # static ADS baseline
EMARC=true  ADS=false NORMAL=false python run.py  # adaptive EMARC-ADS
```

Any other combination is rejected. `NORMAL` forces `lam=eps=0`; static `ADS`
uses the original fixed `lam`; `EMARC` uses a causal per-token controller and
an optional finite-horizon influence direction. Models, data, SFT recipe and
evaluation remain identical so the modes are directly comparable.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, GPU only:
pip install flash-attn --no-build-isolation
```

Verified end to end on torch 2.6.0 / transformers 4.51.3 / datasets 5.0.1 /
accelerate 1.14.0 / peft 0.20.0 / math-verify 0.9.0, Python 3.11.
No `trl`, no `hydra`, no `wandb` — the stages that used them are plain
`transformers.Trainer` and plain argparse now.

## Run

```bash
# plain distillation, 8 GPUs
ADS=false NORMAL=true python run.py --exp_dir=experiments --dataset=gsm8k --tau=1.0

# antidistillation, same everything else
EMARC=false ADS=true NORMAL=false python run.py --exp_dir=experiments --dataset=gsm8k \
    --tau=1.0 --lam=0.15 --eps=1e-2

# EMARC-ADS, runnable reference setting (m=1 exactly reuses the ADS direction)
EMARC=true ADS=false NORMAL=false python run.py \
    --exp_dir=experiments --dataset=gsm8k --tau=1.0 --eps=1e-2 \
    --emarc_alpha_init=0.15 --emarc_horizon_steps=1
```

Both write `experiments/<run_name>/results.json`. Compare them:

```bash
jq '{mode, teacher: .teacher_test.accuracy_af, student: .student_test.accuracy_af}' \
   experiments/*/results.json
```

The result you are looking for is a better worst-case student-damage versus
teacher-utility frontier, not merely a larger internal controller value. Compare
all modes at matched teacher accuracy, trace quality and wall-clock budget.

Every flag is settable as an env var (`TAU=1.0`), a CLI flag (`--tau=1.0`), or by
editing the default in `config.py`. `python run.py --help` is not implemented;
`config.py` is the reference and every field is commented.

### Resuming

Each stage writes a sentinel and is skipped on a re-run. Force a redo with
`--overwrite=true`, or delete the specific artefact.

### One stage at a time

The stages are ordinary scripts; `run.py` just launches them in order.

```bash
python generate.py --ads=true --normal=false --gen_split=train --gen_use_ads=true \
       --gen_out=experiments/traces/my_traces --lam=0.15 --eps=1e-2
python grads.py    --ads=true --normal=false
python horizon.py  --emarc=true --ads=false --normal=false --emarc_horizon_steps=4
python distill.py  --ads=false --normal=true
```

---

## What runs, in order

| # | stage | NORMAL | static ADS | EMARC-ADS | script |
|---|-------|--------|------------|-----------|--------|
| 1 | clean holdout traces | eval set | eval + gradient | eval + gradient | `generate.py` |
| 2 | proxy gradient | — | ✓ | ✓ | `grads.py` |
| 2b | finite-horizon direction | — | — | ✓ (`m=1` is an exact copy) | `horizon.py` |
| 3 | teacher training traces | plain | fixed ADS term | adaptive EMARC term | `generate.py` |
| 4 | student LoRA SFT | ✓ | ✓ | ✓ | `distill.py` |
| 5 | student test | plain | plain | plain | `generate.py` |
| 6 | teacher test | plain | fixed ADS | adaptive EMARC | `generate.py` |

Stage 1 is shared between runs (same path, same clean traces), so a second run in
the same `exp_dir` reuses it.

### The mechanism, in three lines

`grads.py` computes `g`, the gradient of a proxy student's loss on the teacher's
own holdout traces — the direction in the student's parameter space that makes it
*worse*. `generate.py` loads two copies of that proxy student, perturbed to
`θ + εg` and `θ − εg`. At every generated token:

```
teacher_logits  +=  (lam / (2ε)) · ( logits_{θ+εg}(token) − logits_{θ−εg}(token) )
```

The bracket is a central finite difference, so the added term is `lam` times the
directional derivative of the student's token log-likelihood along `g`. Tokens
that would push a distilling student uphill in loss get boosted; `lam` sets how
much utility the teacher is willing to trade for that.

HuggingFace applies custom logits processors *before* the temperature and top-p
warpers, so the sampled distribution is `softmax(top_p((teacher_logits + term)/τ))`
and the effective strength is `lam/τ`. This matches the reference implementation.

### EMARC-ADS controller

`emarc.py` uses the same finite-difference score `d_t`, but chooses a separate
pre-temperature strength for every token position:

```text
beta_cont,t = tau * alpha_t * a_hat_t / sigma
q_t         = softmax((teacher_logits + beta_t * d_t) / tau)
```

This is the stationary point of the local objective
`a_hat_t E_q[d_t] - mu sigma KL(q || p)` with `alpha=1/mu`. The implementation
uses a causal KL-DRO best response: positions with high nominal learning value
and low delivered normalized defense gain receive more estimated attacker mass.
The built-in nominal-value proxies are `confidence`, `low_entropy`, and
`uniform`; `confidence` is the default.

The continuous action is backtracked until both its per-token KL and typicality
budgets pass. A deficit bank updates `alpha_t` toward `emarc_target_flow`; an
anti-windup term prevents a clipped action from making the bank diverge.

When normalized entropy is below `emarc_entropy_threshold`, EMARC can impose
the exact activation floor

```text
beta_margin = min_{v in P} max_{u outside P}
              [logit_u - logit_v + gamma]_+ / (d_v - d_u).
```

Infeasible/non-positive denominators are rejected. The default protected set is
restricted to non-special tokens in the teacher's top-k and within a nominal
logit gap. This lets the implementation remain effective at `tau=0`: the
continuous branch becomes zero and only a feasible guarded margin can change
the greedy argmax.

Every EMARC trace summary records mean/max `beta`, attacker mass, `alpha`, KL,
typicality, defense gain, entropy, clipping counts, and margin
attempt/certificate/failure counts under `summary.controller`.

Important scope: the built-in safe set is a likelihood/typicality guard, **not a
semantic verifier**. It supports a reproducible experiment, but a formal safety
certificate requires a task-specific verifier or retention critic. Likewise,
`confidence` is an observable reweighting proxy, not privileged access to an
attacker's true weights.

### Finite-horizon direction

`horizon.py` optionally replaces the one-step gradient with

```text
v_0     = 0
v_{j+1} = v_j + eta * (g - (H + damping I) v_j).
```

The result is norm-matched to `g`, so `--emarc_horizon_steps=1` is tensor-exact
with static ADS and is the safest first run. For `m>1`, exact Hessian-vector
products run offline in one process with eager attention. Start with a small
subset before paying for the full holdout set:

```bash
--emarc_horizon_steps=4 --emarc_horizon_max_batches=8
```

Key EMARC controls:

| flag | default | role |
|---|---:|---|
| `emarc_alpha_init` | `0.15` | initial continuous defense budget |
| `emarc_target_flow` / `emarc_alpha_lr` | `0.05` / `0.05` | deficit-bank target and feedback rate |
| `emarc_kl_cap` | `0.08` | continuous per-token KL ceiling |
| `emarc_beta_max` | `2.0` | hard pre-temperature strength ceiling |
| `emarc_attacker_eta` | `0.5` | adaptive reweighter's KL-DRO temperature |
| `emarc_value_source` | `confidence` | nominal retention/learning-value proxy |
| `emarc_margin` | `true` | enable the low-entropy margin branch |
| `emarc_safe_top_k` / `emarc_safe_logit_gap` | `32` / `5.0` | likelihood safe-set restriction |
| `emarc_typicality_kappa` | `2.0` | typicality/intervention-cost ceiling |
| `emarc_horizon_steps` | `1` | influence horizon (`1` is exact ADS direction) |

Do not tune these against student test labels. Select them on a separate
validation split, then report the worst-case student result over the declared
prefill/reweighting attack set at matched teacher utility.

---

## Files

```
config.py     every knob, plus NORMAL/ADS/EMARC validation
data.py       gsm8k / hendrycks_math / mmlu / local loaders, prompts, answer checking
ads.py        the ADS logits processor, KV cache, vocab alignment, collator, perturbation
emarc.py      adaptive controller, margin solver, guards, diagnostics
horizon.py    optional exact finite-horizon/Hessian-vector direction stage
generate.py   stage: sample traces (plain, static ADS, or EMARC)
grads.py      stage: proxy-student gradient (ADS and EMARC)
distill.py    stage: LoRA SFT of the attacker's student
run.py        orchestrator: the table above
tests/        offline unit + end-to-end tests
```

### Models

- **teacher** — the model being protected. Its tokenizer is used for every
  generation and gradient step. Default `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.
- **proxy_student** — the defender's stand-in for the attacker. **Must share the
  teacher's tokenizer**, since its logits are added to the teacher's. Default
  `Qwen/Qwen2.5-3B`. The code checks this and fails loudly on a mismatch.
- **student** — the attacker's model. Free to be a different architecture with a
  different tokenizer; it only ever sees the decoded text. Default
  `meta-llama/Llama-3.2-3B`.

### Custom data

Point `--dataset` at a directory holding `train.jsonl` / `holdout.jsonl` /
`test.jsonl` with `problem` and `solution` fields:

```bash
--dataset=local:/path/to/my_data
```

---

## Hardware

Sized for one 8×H100 node, as in the reference. The heavy stages:

- **generation with ADS or EMARC** runs three models at once (teacher + two perturbed
  proxy students) and calls both proxy students once per generated token. Budget
  roughly 3× the memory and 2–3× the wall time of plain generation; drop
  `--gen_batch_size` first if you OOM (the reference uses 192 with ADS vs 512
  without).
- **`grads.py`** holds the proxy student in fp32 plus an fp32 gradient
  accumulator — about 3× the model's parameter bytes. fp32 is deliberate: `eps`
  is small and the perturbation has to survive the round trip.
- **`horizon.py` with `m>1`** additionally performs `m-1` exact HVP sweeps. It
  runs on one device in fp32/eager mode. Use `m=1` first; then raise the horizon
  and/or `emarc_horizon_max_batches` only after measuring memory and runtime.

To shrink the whole thing for a first run:

```bash
ADS=true NORMAL=false python run.py \
  --teacher=Qwen/Qwen2.5-1.5B-Instruct --proxy_student=Qwen/Qwen2.5-0.5B \
  --student=Qwen/Qwen2.5-0.5B --student_tokenizer=Qwen/Qwen2.5-0.5B-Instruct \
  --max_train_samples=200 --max_holdout_samples=100 --max_test_samples=100 \
  --max_new_tokens=512 --gen_batch_size=8 --lam=0.15 --eps=1e-2
```

For the analogous EMARC run, replace the mode flags and static `lam` with:

```bash
EMARC=true ADS=false NORMAL=false python run.py \
  --teacher=Qwen/Qwen2.5-1.5B-Instruct --proxy_student=Qwen/Qwen2.5-0.5B \
  --student=Qwen/Qwen2.5-0.5B --student_tokenizer=Qwen/Qwen2.5-0.5B-Instruct \
  --max_train_samples=200 --max_holdout_samples=100 --max_test_samples=100 \
  --max_new_tokens=512 --gen_batch_size=8 --eps=1e-2 \
  --emarc_alpha_init=0.15 --emarc_horizon_steps=1
```

CPU works (slowly) for smoke-testing: everything falls back to `sdpa`/fp32
automatically.

Multi-GPU is automatic — `run.py` uses `accelerate launch` when more than one GPU
is visible. Force it with `--launcher=accelerate --num_gpus=N`, or force
single-process with `--launcher=python`.

---

## Tests

```bash
python tests/test_units.py    # 52 checks
python tests/smoke_test.py    # 37 checks, all three pipelines end to end
```

Both are fully offline: they build two tokenizer families, three tiny random
models (Qwen2 teacher + Qwen2 proxy student sharing a tokenizer, Llama student
with its own) and a small local dataset, then run every stage for real. No
downloads, ~90 seconds on 2 CPU cores.

What they actually pin down:

- the ADS term equals `lam/(2ε)·(f(θ+εg) − f(θ−εg))` against an independently
  computed value, and is non-zero
- `+εg` **increases** the proxy student's holdout loss and `−εg` decreases it —
  i.e. the saved gradient really points the anti-distillation way
- KV-cached incremental decoding matches a full forward pass to 4e-8
- gradients accumulated across 2 processes are bit-identical to 1 process
- the completion-only collator masks exactly the prompt and the padding
- ADS traces differ from plain traces given the same seed, model and temperature
- invalid zero-mode and multi-mode flag combinations are rejected
- the EMARC continuous solution is exactly `tau*alpha*mass/sigma`
- top-p boundary semantics, KL backtracking and greedy margin activation
- finite-horizon recurrence and tensor-exact `m=1` direction compatibility

---

## Differences from the reference repo

Behaviour is intended to match; these are fixes and simplifications made while
reproducing it.

**Bugs that stop the reference from running**

1. `gentraces.py` reads `cfg.repetition_penalty`, which is defined in neither
   `gen_config.yaml` nor any pipeline script — the generation loop raises
   `ConfigAttributeError` on the first batch.
2. `bos_token` is only bound in the non-Llama branch of the tokenizer setup, but
   the answer-forcing block uses it unconditionally → `NameError` on Llama.
3. `af_accuracy` reads the `is_af_correct` column unconditionally, so
   `answer_force=false` raises at the stats step.

**Correctness**

4. The proxy students' KV cache is never explicitly reset between batches; it is
   invalidated only by "this prompt is shorter than the last sequence", which
   fails when a batch finishes early or the last batch has a different size —
   yielding a stale cache with the wrong batch dimension. Reset is now explicit
   and guards on batch size.
5. Loss masking used `DataCollatorForCompletionOnlyLM` searching for a hardcoded
   `<｜Assistant｜>` marker in the token stream; if the marker tokenizes
   differently in context the collator silently emits an all-`-100` batch and
   the gradient is zero. Prompt length is now recorded at generation time and
   masking is exact — and tokenizer-agnostic.
6. Gradient accumulation weighted each batch by its example count while the
   model's loss is a mean over *tokens*, so the saved gradient depended on batch
   composition. It is now a plain token-weighted mean over the dataset,
   verified identical across 1 and 2 processes.
7. Saved gradient keys assumed a DDP `module.` prefix; single-GPU runs produced
   keys that matched nothing and the assertion fired. Names are normalised.
8. Adding a `[PAD]` token forced a vocabulary resize that appends a randomly
   initialised, sampleable logit row. The pad token now reuses EOS, and models
   are resized only to `len(tokenizer)`.
9. `/tmp/cached_proc_dataset` and `/tmp/cached_ds` are fixed global paths shared
   by every concurrent experiment on the machine. Caches now live under
   `exp_dir/.cache`, keyed by content.
10. Shard temp directories were deleted by non-main ranks while rank 0 was still
    reading them.

**Simplification**

11. `trl`'s `SFTTrainer` → `transformers.Trainer` with a local collator. The TRL
    API for this changed repeatedly across 0.16–0.20 (`tokenizer` →
    `processing_class`, `max_seq_length` → `max_length`,
    `DataCollatorForCompletionOnlyLM` removed) and was the single most fragile
    dependency.
12. `hydra` + `omegaconf` → one dataclass. Missing keys are now impossible.
13. `wandb` removed; every stage writes `<artifact>.json` and the run writes
    `results.json`.
14. `grid.py`'s hostname-sharded hyperparameter sweep is gone — this runs one
    `(tau, lam, eps)` point, per the brief. Loop over it in your own shell for a
    sweep; the sentinels make that safe.

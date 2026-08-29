# Antidistillation Sampling — minimal, two-mode


```bash
ADS=false NORMAL=true  python run.py     # plain distillation
ADS=true  NORMAL=false python run.py     # antidistillation sampling
```

Any other combination of the two flags is rejected. In `NORMAL` mode `lam` and
`eps` are forced to `0` and the proxy student is never loaded; in `ADS` mode the
gradient stage runs and the teacher samples with the antidistillation term.
Everything else — models, data, SFT recipe, evaluation — is identical between
the two, so the difference in the distilled student is attributable to the
sampling and nothing else.

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
ADS=true NORMAL=false python run.py --exp_dir=experiments --dataset=gsm8k \
    --tau=1.0 --lam=0.15 --eps=1e-2
```

Both write `experiments/<run_name>/results.json`. Compare them:

```bash
jq '{mode, teacher: .teacher_test.accuracy_af, student: .student_test.accuracy_af}' \
   experiments/*/results.json
```

The result you are looking for: **teacher accuracy roughly unchanged between the
two runs, student accuracy lower in the ADS run.** That gap is the defence.

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
python distill.py  --ads=false --normal=true
```

---

## What runs, in order

| # | stage | NORMAL | ADS | script |
|---|-------|--------|-----|--------|
| 1 | holdout traces — clean teacher traces, greedy | ✓ (eval set) | ✓ (eval set **and** gradient input) | `generate.py` |
| 2 | proxy-student gradient on those traces | — | ✓ | `grads.py` |
| 3 | training traces from the teacher | plain sampling | **+ ADS term** | `generate.py` |
| 4 | distil the student (LoRA SFT) | ✓ | ✓ | `distill.py` |
| 5 | student on the test split | ✓ | ✓ | `generate.py` |
| 6 | teacher on the test split | plain sampling | **+ ADS term** | `generate.py` |

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

---

## Files

```
config.py     every knob, plus the ADS/NORMAL validation
data.py       gsm8k / hendrycks_math / mmlu / local loaders, prompts, answer checking
ads.py        the ADS logits processor, KV cache, vocab alignment, collator, perturbation
generate.py   stage: sample traces (with or without ADS)
grads.py      stage: proxy-student gradient  (ADS only)
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

- **generation with ADS** runs three models at once (teacher + two perturbed
  proxy students) and calls both proxy students once per generated token. Budget
  roughly 3× the memory and 2–3× the wall time of plain generation; drop
  `--gen_batch_size` first if you OOM (the reference uses 192 with ADS vs 512
  without).
- **`grads.py`** holds the proxy student in fp32 plus an fp32 gradient
  accumulator — about 3× the model's parameter bytes. fp32 is deliberate: `eps`
  is small and the perturbation has to survive the round trip.

To shrink the whole thing for a first run:

```bash
ADS=true NORMAL=false python run.py \
  --teacher=Qwen/Qwen2.5-1.5B-Instruct --proxy_student=Qwen/Qwen2.5-0.5B \
  --student=Qwen/Qwen2.5-0.5B --student_tokenizer=Qwen/Qwen2.5-0.5B-Instruct \
  --max_train_samples=200 --max_holdout_samples=100 --max_test_samples=100 \
  --max_new_tokens=512 --gen_batch_size=8 --lam=0.15 --eps=1e-2
```

CPU works (slowly) for smoke-testing: everything falls back to `sdpa`/fp32
automatically.

Multi-GPU is automatic — `run.py` uses `accelerate launch` when more than one GPU
is visible. Force it with `--launcher=accelerate --num_gpus=N`, or force
single-process with `--launcher=python`.

---

## Tests

```bash
python tests/test_units.py    # 31 checks
python tests/smoke_test.py    # 25 checks, both pipelines end to end
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
- `ADS=true NORMAL=true` and `ADS=false NORMAL=false` are both rejected

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

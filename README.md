# Top-K Soft Distillation

This repository trains a `Qwen/Qwen2.5-3B` student from traces produced by
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. It is the soft-distillation
baseline to report alongside your hard-distillation, ADS, and EMARC-ADS runs.

The recommended experiment is **Qwen student, top-64 teacher logits, cached
teacher targets, three random seeds**. It does not replace ADS or EMARC-ADS;
it measures what an attacker can achieve with stronger standard student
training.

## What this version fixes

- The soft objective includes the omitted teacher probability mass. Each site
  uses K explicit tokens and one tail category:

  \[
  \mathcal L_{\rm soft}=T^2D_{\rm KL}((p_1,\ldots,p_K,p_{\rm tail})
  \Vert(q_1,\ldots,q_K,q_{\rm tail})).
  \]

- Cache contracts bind the cache to the model configuration, tokenizer token-ID
  contract, traces, context length, K, and temperature. The Slurm cache job
  validates an existing cache before reusing it.
- Checkpoints include the model/LoRA adapter, optimizer, scheduler, rank-local
  RNG state, and exact dataloader position, so interrupted jobs resume safely.
- Final model directories are written atomically and receive `_SUCCESS` only
  after the model, tokenizer, arguments, and training contract are complete.
- Trace columns are auto-detected from `trace`, `completion_af`, `completion`,
  `response`, and `text`.
- The DGX Slurm workflow writes models, caches, logs, and temporary data under
  `/raid`, not the quota-limited home filesystem.

## Recommended configuration

| Item | Value |
|---|---|
| Teacher | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Student | `Qwen/Qwen2.5-3B` |
| Vocabulary mode | `shared` |
| K | `64` |
| Temperature | `2.0` |
| Soft-loss weight | `alpha=0.9` |
| Context length | `2048` initially |
| Training | LoRA rank 128, 3 epochs, effective batch 16 |
| Seeds | `42, 123, 456` |

Use normal, unpoisoned teacher traces for the normal soft-distillation
baseline. Use ADS traces only when that is the experimental condition you
intend to study.

## Install

```bash
python -m pip install -r requirements.txt
```

Flash Attention is optional; PyTorch SDPA is used automatically if it is not
installed.

```bash
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Inspect traces first

```bash
python scripts/convert_traces.py inspect /path/to/traces
python scripts/convert_traces.py validate /path/to/traces
```

For nonstandard data, pass `--trace_colname` and `--problem_colname`
explicitly. Do not guess the completion column.

## Cached one-GPU workflow

Build the teacher cache once:

```bash
TRAIN_TRACES=/path/to/normal_teacher_traces \
K=64 TEMPERATURE=2.0 MAX_LENGTH=2048 PRECOMPUTE_BS=1 \
ACCEL_CONFIG=configs/accelerate_1gpu.yaml \
bash scripts/precompute_cache.sh
```

Train one seed from that cache:

```bash
CACHE_DIR=./experiments/topk_cache_k64 \
K=64 TEMPERATURE=2.0 MAX_LENGTH=2048 \
ACCEL_CONFIG=configs/accelerate_1gpu.yaml \
bash scripts/train_cached.sh
```

The `tail_bucket` objective requires the same temperature used while
precomputing. This is required to reconstruct the teacher tail probability
correctly.

## DGX: all three seeds on two GPUs

The launcher first builds one cache, then starts a three-seed array with at
most two concurrent GPUs. Each seed writes independently and resumes from its
newest complete checkpoint.

```bash
cd /path/to/topk-soft-distillation
source /raid/mtanveer/antidistill_env/bin/activate

export TRAIN_TRACES=/raid/mtanveer/path/to/normal_teacher_traces
export RUN_ROOT=/raid/mtanveer/top64_soft_distillation
export VENV_DIR=/raid/mtanveer/antidistill_env
export HF_HOME=/raid/mtanveer/model_cache/huggingface

bash scripts/submit_slurm.sh
```

Monitor it with:

```bash
squeue -u "$USER"
find /raid/mtanveer/top64_soft_distillation/models -name _SUCCESS -print
```

The launcher checks Python dependencies, CUDA visibility, source-trace
readability, source compilation, and at least 100 GiB free storage before it
submits anything.

## Resumption

Every checkpoint contains:

```text
checkpoint-100/
  adapter/                 # LoRA weights
  optimizer.pt
  scheduler.pt
  rng_rank0.pt
  trainer_state.json
  _SUCCESS                 # written only after the checkpoint is complete
```

`--resume_from_checkpoint auto` resumes only a complete checkpoint. The code
refuses a resume if model, tokenizer, source data, K, batch/world size, or
other training-contract fields changed. A completed output is also accepted
only when its saved contract matches the requested run.

## Evaluation

Compare base Qwen to the soft-distilled student on held-out traces:

```bash
python scripts/evaluate_agreement.py \
  --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --student /raid/mtanveer/top64_soft_distillation/models/top64_seed42/final \
  --baseline_student Qwen/Qwen2.5-3B \
  --traces /path/to/heldout_teacher_traces \
  --top_k 64 --temperature 2.0 --max_length 2048
```

For GSM8K answer accuracy, use the exact same answer-forced evaluation protocol
as the ADS repository. The paper table should include hard distillation, normal
top-64 soft distillation, ADS, and EMARC-ADS, always with teacher utility and
student answer-forced accuracy.

## Scope

`shared` mode is the main Qwen-to-Qwen experiment. It verifies every common
token ID before GPU work. The teacher's slightly larger padded vocabulary is
handled through the tail category.

`cross` mode remains only for exploratory Qwen-to-Llama experiments. It aligns
character offsets and maps tokens by surface form, so it is approximate and
uses `student_full`, not the tail-bucket KL. Do not use it as the main baseline.

## Verification

Run these on the same environment used for Slurm:

```bash
python -m py_compile soft_distill.py scripts/*.py
python tests/test_kl.py
bash tests/test_end_to_end.sh
bash tests/test_distributed.sh
bash verify.sh --full
```

The tests use tiny local models; they do not download the 7B teacher. They
exercise online and cached paths, cache-contract failures, cross/shared modes,
and interruption/resume correctness.

## Reproducibility

Pin Hub commits for a final paper run:

```bash
--teacher_revision <commit> --student_revision <commit>
```

The cache records resolved commit/configuration fingerprints even when you do
not pass revisions, but explicit pinning is stronger.

## License

MIT. Model checkpoints and trace data retain their original licenses.

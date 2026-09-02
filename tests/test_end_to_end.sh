#!/usr/bin/env bash
# ==============================================================================
# tests/test_end_to_end.sh -- exercise every code path on tiny CPU models.
#
# Covers: shared/cross vocab x online/cached teacher logits, LoRA on/off, both
# KL variants, jsonl input, mid-run checkpointing, and every guarded error path.
# Also asserts that the cached path reproduces the online path exactly.
#
#   bash tests/test_end_to_end.sh
#
# Runs on CPU in a couple of minutes. No GPU, no network, no model downloads.
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
FIX="$ROOT/tests/fixtures"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/topk-soft-e2e.XXXXXX")
trap 'rm -rf "${TMP}"' EXIT

if [[ ! -d "$FIX/teacher_qwen" ]]; then
    echo "Fixtures missing -- building them first."
    python3 tests/make_fixtures.py > "$TMP/fixtures.log" 2>&1 || { cat "$TMP/fixtures.log"; exit 1; }
fi

COMMON="--max_length 256 --num_epochs 1 --model_dtype fp32 --teacher_dtype fp32 \
 --attn_implementation sdpa --num_workers 0 --logging_steps 3 --top_k 8"
T="--teacher $FIX/teacher_qwen --teacher_tokenizer $FIX/tok_qwen"

pass=0; fail=0
run() {
    local name="$1"; shift
    printf '%-34s' "$name"
    if "$@" > "$TMP/log_$name.txt" 2>&1; then
        echo "PASS"; pass=$((pass+1))
    else
        echo "FAIL"; fail=$((fail+1)); tail -25 "$TMP/log_$name.txt"
    fi
}
expect_error() {
    local name="$1"; local needle="$2"; shift 2
    printf '%-34s' "$name"
    if "$@" > "$TMP/log_$name.txt" 2>&1; then
        echo "FAIL (expected an error)"; fail=$((fail+1))
    elif grep -q "$needle" "$TMP/log_$name.txt"; then
        echo "PASS"; pass=$((pass+1))
    else
        echo "FAIL (wrong error)"; fail=$((fail+1)); tail -5 "$TMP/log_$name.txt"
    fi
}

echo "=============================================================="
echo " TRAINING PATHS"
echo "=============================================================="

run shared_online_lora python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --eval_traces "$FIX/holdout" --batch_size 4 --per_device_batch_size 2 \
  --lora --lora_r 8 --lora_alpha 16 --output_dir "$TMP/t1" $COMMON

run shared_online_full_renorm python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --batch_size 2 --per_device_batch_size 1 --no-lora --kl_variant student_renorm \
  --temperature 1.0 --alpha 1.0 --lr 1e-4 --output_dir "$TMP/t2" $COMMON

run precompute_shared python3 soft_distill.py --mode precompute --vocab_mode shared \
  $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --cache_dir "$TMP/cache_shared" --precompute_batch_size 3 --output_dir "$TMP/t3" $COMMON

run cached_shared python3 soft_distill.py --mode train --teacher_logits cached \
  --vocab_mode shared $T --student "$FIX/student_qwen" --cache_dir "$TMP/cache_shared" \
  --eval_traces "$FIX/holdout" --batch_size 4 --per_device_batch_size 2 \
  --lora --lora_r 8 --lora_alpha 16 --output_dir "$TMP/t3" $COMMON

run cross_online python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode cross $T --student "$FIX/student_llama" --student_tokenizer "$FIX/tok_llama" \
  --kl_variant student_full \
  --train_traces "$FIX/traces" --eval_traces "$FIX/holdout" --batch_size 4 \
  --per_device_batch_size 2 --lora --lora_r 8 --lora_alpha 16 --output_dir "$TMP/t4" $COMMON

run precompute_cross python3 soft_distill.py --mode precompute --vocab_mode cross \
  $T --student "$FIX/student_llama" --student_tokenizer "$FIX/tok_llama" \
  --kl_variant student_full \
  --train_traces "$FIX/traces" --cache_dir "$TMP/cache_cross" --precompute_batch_size 3 \
  --output_dir "$TMP/t5" $COMMON

run cached_cross python3 soft_distill.py --mode train --teacher_logits cached \
  --vocab_mode cross $T --student "$FIX/student_llama" --student_tokenizer "$FIX/tok_llama" \
  --kl_variant student_full \
  --cache_dir "$TMP/cache_cross" --batch_size 4 --per_device_batch_size 2 \
  --lora --lora_r 8 --lora_alpha 16 --output_dir "$TMP/t5" $COMMON

run jsonl_input python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --input_format jsonl \
  --train_traces "$FIX/pairs.jsonl" --batch_size 2 --per_device_batch_size 2 \
  --lora --lora_r 8 --lora_alpha 16 --output_dir "$TMP/t6" $COMMON

run mid_run_checkpoints python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --eval_traces "$FIX/holdout" --batch_size 4 --per_device_batch_size 2 \
  --lora --lora_r 8 --lora_alpha 16 --save_steps 2 --eval_steps 2 \
  --output_dir "$TMP/t7" $COMMON

run resume_stage1 python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --batch_size 4 --per_device_batch_size 2 --lora --lora_r 8 --lora_alpha 16 \
  --save_steps 1 --stop_after_steps 2 --output_dir "$TMP/resume" $COMMON

run resume_stage2 python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --batch_size 4 --per_device_batch_size 2 --lora --lora_r 8 --lora_alpha 16 \
  --save_steps 1 --resume_from_checkpoint auto --output_dir "$TMP/resume" $COMMON

run edge_top_k_1 python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --batch_size 2 --per_device_batch_size 2 --lora --lora_r 8 --lora_alpha 16 \
  --output_dir "$TMP/k1" --max_length 256 --num_epochs 1 --model_dtype fp32 \
  --teacher_dtype fp32 --attn_implementation sdpa --num_workers 0 --top_k 1

run edge_top_k_gt_vocab python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --batch_size 2 --per_device_batch_size 2 --lora --lora_r 8 --lora_alpha 16 \
  --output_dir "$TMP/kbig" --max_length 256 --num_epochs 1 --model_dtype fp32 \
  --teacher_dtype fp32 --attn_implementation sdpa --num_workers 0 --top_k 5000

run edge_alpha_zero python3 soft_distill.py --mode train --teacher_logits online \
  --vocab_mode shared $T --student "$FIX/student_qwen" --train_traces "$FIX/traces" \
  --batch_size 2 --per_device_batch_size 2 --lora --lora_r 8 --lora_alpha 16 \
  --alpha 0.0 --output_dir "$TMP/a0" $COMMON

echo
echo "=============================================================="
echo " CACHED == ONLINE EQUIVALENCE"
echo "=============================================================="
# Compare the training-step lines only. The [eval] line is deliberately
# different: with a cached teacher the held-out set has no cached logits, so
# evaluation reports hard-label CE alone.
printf '%-34s' "losses_match"
a=$(grep -E "^\[.*\] step [0-9]+/" "$TMP/log_shared_online_lora.txt" | grep -oE "loss [0-9.]+ \| kl [0-9.]+ \| ce [0-9.]+" | tr '\n' ' ')
b=$(grep -E "^\[.*\] step [0-9]+/" "$TMP/log_cached_shared.txt" | grep -oE "loss [0-9.]+ \| kl [0-9.]+ \| ce [0-9.]+" | tr '\n' ' ')
if [[ -n "$a" && "$a" == "$b" ]]; then echo "PASS"; pass=$((pass+1))
else echo "FAIL"; echo "  online: $a"; echo "  cached: $b"; fail=$((fail+1)); fi

echo
echo "=============================================================="
echo " ERROR PATHS (each must fail with a clear message)"
echo "=============================================================="

expect_error err_tokenizer_mismatch "identical token-id meanings" \
  python3 soft_distill.py --mode train --teacher_logits online --vocab_mode shared $T \
  --student "$FIX/student_qwen" --student_tokenizer "$FIX/tok_llama" \
  --train_traces "$FIX/traces" --batch_size 2 --per_device_batch_size 2 \
  --output_dir "$TMP/e1" $COMMON

expect_error err_batch_not_divisible "must be divisible" \
  python3 soft_distill.py --mode train --teacher_logits online --vocab_mode shared $T \
  --student "$FIX/student_qwen" --train_traces "$FIX/traces" --batch_size 5 \
  --per_device_batch_size 2 --output_dir "$TMP/e2" $COMMON

expect_error err_missing_cache "Run --mode precompute first" \
  python3 soft_distill.py --mode train --teacher_logits cached --vocab_mode shared $T \
  --student "$FIX/student_qwen" --cache_dir "$TMP/does_not_exist" --batch_size 2 \
  --per_device_batch_size 2 --output_dir "$TMP/e3" $COMMON

expect_error err_cache_k_mismatch "rebuild" \
  python3 soft_distill.py --mode train --teacher_logits cached --vocab_mode shared $T \
  --student "$FIX/student_qwen" --cache_dir "$TMP/cache_shared" --batch_size 2 \
  --per_device_batch_size 2 --output_dir "$TMP/e4" --max_length 256 --num_epochs 1 \
  --model_dtype fp32 --teacher_dtype fp32 --attn_implementation sdpa --num_workers 0 --top_k 4

expect_error err_bad_column "Pass --trace_colname" \
  python3 soft_distill.py --mode train --teacher_logits online --vocab_mode shared $T \
  --student "$FIX/student_qwen" --train_traces "$FIX/traces" --trace_colname nope \
  --batch_size 2 --per_device_batch_size 2 --output_dir "$TMP/e5" $COMMON

expect_error err_top_k_zero "top_k must be >= 1" \
  python3 soft_distill.py --mode train --teacher_logits online --vocab_mode shared $T \
  --student "$FIX/student_qwen" --train_traces "$FIX/traces" --batch_size 2 \
  --per_device_batch_size 2 --output_dir "$TMP/e6" --max_length 256 --num_epochs 1 \
  --model_dtype fp32 --teacher_dtype fp32 --attn_implementation sdpa --num_workers 0 --top_k 0

expect_error err_alpha_range "alpha must be in" \
  python3 soft_distill.py --mode train --teacher_logits online --vocab_mode shared $T \
  --student "$FIX/student_qwen" --train_traces "$FIX/traces" --alpha 1.5 --batch_size 2 \
  --per_device_batch_size 2 --output_dir "$TMP/e7" $COMMON

echo
echo "=============================================================="
echo " SAVED ARTIFACTS"
echo "=============================================================="
printf '%-34s' "final_is_merged_model"
if [[ -f "$TMP/t7/final/config.json" && -f "$TMP/t7/final/model.safetensors" \
      && -f "$TMP/t7/final/training_contract.json" ]]; then
    echo "PASS"; pass=$((pass+1)); else echo "FAIL"; fail=$((fail+1)); fi
printf '%-34s' "checkpoint_is_lora_adapter"
if [[ -f "$TMP/t7/checkpoint-2/adapter/adapter_config.json" \
      && -f "$TMP/t7/checkpoint-2/optimizer.pt" \
      && -f "$TMP/t7/checkpoint-2/trainer_state.json" \
      && -f "$TMP/t7/checkpoint-2/_SUCCESS" ]]; then
    echo "PASS"; pass=$((pass+1)); else echo "FAIL"; fail=$((fail+1)); fi
printf '%-34s' "resume_reaches_final"
if [[ -f "$TMP/resume/final/_SUCCESS" ]] \
   && grep -q "Resumed optimizer/scheduler/RNG at step 2" "$TMP/log_resume_stage2.txt"; then
    echo "PASS"; pass=$((pass+1)); else echo "FAIL"; fail=$((fail+1)); fi

echo
echo "=============================================================="
echo "  $pass passed, $fail failed"
echo "=============================================================="
exit $(( fail > 0 ))

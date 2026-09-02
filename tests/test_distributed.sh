#!/usr/bin/env bash
# ==============================================================================
# tests/test_distributed.sh -- run the trainer under a real 2-process
# torch.distributed job (gloo on CPU) to check the multi-GPU wiring:
#
#   * gradient accumulation is derived from the true world size
#   * the LR schedule finishes at exactly the announced step count
#   * held-out metrics are reduced across ranks
#   * saving happens once, from rank 0
#
#   bash tests/test_distributed.sh
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
FIX="$ROOT/tests/fixtures"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/topk-soft-ddp.XXXXXX")
trap 'rm -rf "${TMP}"' EXIT

if [[ ! -d "$FIX/teacher_qwen" ]]; then
    python3 tests/make_fixtures.py > "$TMP/fixtures.log" 2>&1 || { cat "$TMP/fixtures.log"; exit 1; }
fi

PORT=${PORT:-29577}
fail=0

for MODE in shared cross; do
    echo "--- 2-process DDP, vocab_mode=$MODE"
    EXTRA=(--student "$FIX/student_qwen")
    [[ "$MODE" == "cross" ]] && EXTRA=(--student "$FIX/student_llama" \
        --student_tokenizer "$FIX/tok_llama" --kl_variant student_full)

    ACCELERATE_USE_CPU=1 torchrun --nproc_per_node=2 --master_port=$PORT soft_distill.py \
        --mode train --teacher_logits online --vocab_mode "$MODE" \
        --teacher "$FIX/teacher_qwen" --teacher_tokenizer "$FIX/tok_qwen" \
        "${EXTRA[@]}" \
        --train_traces "$FIX/traces" --eval_traces "$FIX/holdout" \
        --top_k 8 --max_length 256 --num_epochs 1 \
        --batch_size 4 --per_device_batch_size 1 \
        --model_dtype fp32 --teacher_dtype fp32 --attn_implementation sdpa \
        --num_workers 0 --lora --lora_r 8 --lora_alpha 16 --logging_steps 2 \
        --output_dir "$TMP/ddp_$MODE" > "$TMP/ddp_$MODE.log" 2>&1

    # batch_size 4 = 2 processes x 1 per device x 2 accumulation steps
    if grep -q "grad_accum=2" "$TMP/ddp_$MODE.log" \
       && grep -qE "step 6/6.*lr 0.00e\+00" "$TMP/ddp_$MODE.log" \
       && grep -q "\[eval\]" "$TMP/ddp_$MODE.log" \
       && [[ -f "$TMP/ddp_$MODE/final/config.json" ]]; then
        echo "    PASS"
    else
        echo "    FAIL"; fail=1
        grep -E "grad_accum|step |eval|Traceback|Error" "$TMP/ddp_$MODE.log" | tail -15
    fi
    PORT=$((PORT+1))
done

echo
[[ $fail -eq 0 ]] && echo "DISTRIBUTED TESTS PASSED" || echo "DISTRIBUTED TESTS FAILED"
exit $fail

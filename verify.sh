#!/usr/bin/env bash
# ==============================================================================
# verify.sh -- prove the code works before you spend GPU hours on it.
#
# Builds tiny local models and tokenizers, then runs four layers of checks:
#   1. numerics   -- the KL matches a slow reference implementation
#   2. paths      -- every mode/flag combination runs, and errors are clear
#   3. distributed-- a real 2-process job wires up correctly
#   4. learning   -- the objective actually drives a student to match a teacher
#
# Everything runs on CPU with no network access and no model downloads.
#   bash verify.sh              # layers 1-3 (about 3 minutes)
#   bash verify.sh --full       # all four layers (about 8 minutes)
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")"

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

hdr() { echo; echo "######################################################################"; echo "# $1"; echo "######################################################################"; }
fail=0

hdr "0/5  Static contracts: parser, cache, checkpoint, and Slurm wiring"
python3 tests/test_static_contract.py || { echo "static contract checks FAILED"; exit 1; }

hdr "1/5  Building test fixtures (tiny models + two disagreeing tokenizers)"
python3 tests/make_fixtures.py || { echo "fixture build FAILED"; exit 1; }

hdr "2/5  Numerics: top-K KL vs a reference implementation"
python3 tests/test_kl.py || fail=1

hdr "3/5  End-to-end: every mode, flag and error path"
bash tests/test_end_to_end.sh || fail=1

hdr "4/5  Distributed: real 2-process torchrun job"
bash tests/test_distributed.sh || fail=1

if [[ $FULL -eq 1 ]]; then
    hdr "5/5  Learning: does the objective actually distill?"
    python3 tests/test_learning.py || fail=1
else
    hdr "5/5  Learning: SKIPPED (re-run with: bash verify.sh --full)"
fi

echo
if [[ $fail -eq 0 ]]; then
    echo "######################################################################"
    echo "#  ALL CHECKS PASSED"
    echo "######################################################################"
else
    echo "######################################################################"
    echo "#  SOME CHECKS FAILED -- see the output above"
    echo "######################################################################"
fi
exit $fail

# EMARC-ADS verification record

This record covers the repository state packaged with it. The checks ran in a
fresh Python 3.12 environment with the versions resolved from
`requirements.txt` on 2026-08-29 (PyTorch 2.13.0, Transformers 4.57.6,
Datasets 5.0.1, Accelerate 1.14.0 and PEFT 0.20.0).

## Pass 1 — equation and interface review

The new code was reviewed from configuration through orchestration and back:

- exactly one of `normal`, `ads`, and `emarc` is accepted;
- static ADS still calls the unchanged `ADSLogitsProcessor`;
- EMARC loads `direction_path`, not the static-gradient path by accident;
- the finite difference is `(f(theta+eps*v)-f(theta-eps*v))/(2*eps)`;
- the continuous pre-temperature action is exactly
  `beta=tau*alpha*mass/sigma`;
- the top-p boundary token is retained and EMARC does not apply top-p twice;
- KL/typicality backtracking can only lower the continuous action;
- non-positive margin denominators are rejected as infeasible;
- at `tau=0` the continuous action is zero and only a feasible guarded margin
  can change the argmax;
- finished sequences do not update controller state;
- alpha, attacker mass and generation KV caches reset at each new batch;
- `m=1` saves a tensor-exact copy of the ADS gradient;
- `m>1` uses the stated Hessian-vector recurrence and norm matches the result;
- summaries and sentinels use mode-specific paths and remain resumable.

This review exposed and fixed a portability issue in the original stages:
`datasets.map(num_proc=1)` can start a process manager in current Datasets.
Every stage now passes `num_proc=None` for a true single-process map.

## Pass 2 — static and unit verification

Commands:

```bash
python -m compileall -q .
ruff check --select E4,E7,E9,F,I \
  config.py emarc.py horizon.py generate.py run.py \
  tests/test_units.py tests/smoke_test.py
python tests/test_units.py
```

Results:

- compilation: passed;
- selected Ruff correctness/import checks: passed;
- unit suite: **52 passed, 0 failed**.

The unit suite independently checks the controller equation, top-p support,
exact margin ratio, infeasibility handling, KL cap, greedy activation, finite
horizon recurrence, original ADS term/sign, cache correctness, masking, mode
validation, and vocabulary alignment.

## Pass 3 — real three-mode pipeline

The offline smoke suite built local random Qwen2 teacher/proxy models, a local
Llama student, two local tokenizers, and a local dataset. It then ran every
stage for `normal`, static `ads`, and `emarc`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python tests/smoke_test.py
```

Result: **37 passed, 0 failed**.

This included real trace generation, gradient saving, the `m=1` direction
stage, adaptive EMARC decoding, LoRA SFT and merge, student/teacher evaluation,
JSON diagnostics, mode comparison, and resume sentinels. Both ADS and EMARC
changed all 8 tiny training completions relative to normal generation. The
`m=1` saved direction was tensor-exact with the ADS gradient.

## Additional finite-horizon stress check

The exact HVP stage was separately run with `m=2` and one real holdout batch,
then that direction was loaded for an EMARC generation probe. Both completed
successfully; the probe recorded finite controller diagnostics and respected
its KL budget.

## Research-scope caveat

These checks establish implementation consistency and end-to-end execution;
they do not establish that a particular hyperparameter point is the best
defense. The built-in protected set is likelihood/typicality guarded, not a
semantic certificate, and the default confidence feature estimates rather than
observes an attacker's reweighting mass. Comparative defense claims still need
the adaptive-prefill/reweighting evaluation described in the README.

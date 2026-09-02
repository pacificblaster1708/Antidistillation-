#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_traces.py -- inspect and convert training data between the two formats
soft_distill.py accepts.

  repo   a HuggingFace dataset directory written by datasets.save_to_disk, with a
         `problem` column and a `trace` column holding the teacher's full
         prompt+response string (this is what gentraces.py in the
         antidistillation-sampling repo produces).
  jsonl  one JSON object per line: {"prompt": ..., "completion": ...}

Subcommands
-----------
  inspect   print columns, row count, length statistics and one full example
  to-jsonl  repo dataset  ->  .jsonl
  to-repo   .jsonl        ->  repo dataset directory
  validate  check that every row yields a usable prompt/completion pair

Examples
--------
  python scripts/convert_traces.py inspect  ./experiments/traces/holdout
  python scripts/convert_traces.py to-jsonl ./experiments/traces/holdout out.jsonl
  python scripts/convert_traces.py to-repo  out.jsonl ./my_traces
  python scripts/convert_traces.py validate ./experiments/traces/holdout
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soft_distill import (  # noqa: E402
    DEEPSEEK_ASSISTANT_MARKER, DEEPSEEK_EOS_MARKER, TRACE_COLUMN_CANDIDATES,
    _choose_column, load_pairs,
)


def _shim(fmt, trace_col, problem_col):
    return argparse.Namespace(input_format=fmt, trace_colname=trace_col,
                              problem_colname=problem_col, dataset_split=None)


def cmd_inspect(args):
    if args.path.endswith(".jsonl"):
        pairs = load_pairs(args.path, _shim("jsonl", args.trace_colname, args.problem_colname), None)
        print(f"format: jsonl\nrows:   {len(pairs)}")
    else:
        import datasets

        ds = datasets.load_from_disk(args.path)
        print(f"format:  repo (datasets.load_from_disk)")
        print(f"rows:    {len(ds)}")
        print(f"columns: {ds.column_names}")
        pairs = load_pairs(args.path, _shim("repo", args.trace_colname, args.problem_colname), None)

    plens = [len(p["prompt"]) for p in pairs]
    clens = [len(p["completion"]) for p in pairs]
    def stats(name, xs):
        xs = sorted(xs)
        print(f"  {name:11s} min {xs[0]:7d}  median {int(statistics.median(xs)):7d}  "
              f"p95 {xs[int(0.95 * (len(xs) - 1))]:7d}  max {xs[-1]:7d}")
    print("\ncharacter lengths:")
    stats("prompt", plens)
    stats("completion", clens)
    print("\n--- example row -----------------------------------------------------")
    print("PROMPT:\n" + pairs[0]["prompt"][:1200])
    print("\nCOMPLETION:\n" + pairs[0]["completion"][:2000])
    print("---------------------------------------------------------------------")
    print("\nRough token estimate (chars/3.5): "
          f"prompt ~{int(statistics.median(plens) / 3.5)}, "
          f"completion ~{int(statistics.median(clens) / 3.5)} median tokens.")


def cmd_to_jsonl(args):
    fmt = "jsonl" if args.src.endswith(".jsonl") else "repo"
    pairs = load_pairs(args.src, _shim(fmt, args.trace_colname, args.problem_colname), None)
    with open(args.dst, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} rows to {args.dst}")


def cmd_to_repo(args):
    import datasets

    pairs = load_pairs(args.src, _shim("jsonl", args.trace_colname, args.problem_colname), None)
    rows = [{"problem": p["prompt"],
             "trace": f"{DEEPSEEK_ASSISTANT_MARKER}{p['completion']}{DEEPSEEK_EOS_MARKER}"}
            for p in pairs]
    ds = datasets.Dataset.from_list(rows)
    ds.save_to_disk(args.dst)
    print(f"wrote {len(rows)} rows to {args.dst} (columns: {ds.column_names})")


def cmd_validate(args):
    fmt = "jsonl" if args.path.endswith(".jsonl") else "repo"
    pairs = load_pairs(args.path, _shim(fmt, args.trace_colname, args.problem_colname), None)
    bad = [i for i, p in enumerate(pairs)
           if not p["prompt"].strip() or not p["completion"].strip()]
    empty_marker = 0
    if fmt == "repo":
        import datasets

        ds = datasets.load_from_disk(args.path)
        trace_col = _choose_column(ds.column_names, args.trace_colname,
                                   TRACE_COLUMN_CANDIDATES, "trace")
        empty_marker = sum(1 for r in ds
                           if r.get(trace_col) and
                           DEEPSEEK_ASSISTANT_MARKER not in r[trace_col])
    print(f"usable pairs:            {len(pairs)}")
    print(f"blank prompt/completion: {len(bad)}")
    if fmt == "repo":
        print(f"rows missing '{DEEPSEEK_ASSISTANT_MARKER}': {empty_marker} "
              "(these fall back to using the whole trace as the completion)")
    print("OK" if not bad else f"PROBLEM: rows {bad[:10]} are unusable")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace_colname", default="auto")
    p.add_argument("--problem_colname", default="auto")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect"); s.add_argument("path"); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("to-jsonl"); s.add_argument("src"); s.add_argument("dst"); s.set_defaults(fn=cmd_to_jsonl)
    s = sub.add_parser("to-repo"); s.add_argument("src"); s.add_argument("dst"); s.set_defaults(fn=cmd_to_repo)
    s = sub.add_parser("validate"); s.add_argument("path"); s.set_defaults(fn=cmd_validate)

    args = p.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()

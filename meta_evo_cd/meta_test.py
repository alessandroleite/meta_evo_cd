from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from typing import List, Dict
import numpy as np
from numpy.random import default_rng

from .pipelines import PipelineConfig
from .run import evaluate_cfg_on_tasks
from .meta_protocol import sample_test_tasks_shifted, make_shift_suite
from .logging_utils import append_jsonl, append_csv, ensure_dir

def load_selected_cfgs(final_selected_jsonl: str, top_k: int = 5) -> List[PipelineConfig]:
    # final_selected.jsonl contains one JSON object with {"seed":..., "final":[...]}
    last = None
    with open(final_selected_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            last = json.loads(line)
    if last is None or "final" not in last:
        raise ValueError("Could not find final configs in file.")
    cfg_dicts = last["final"][:top_k]
    return [PipelineConfig(**d) for d in cfg_dicts]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g. runs/run0")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--test_tasks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ensure_dir(args.run_dir)
    cfgs = load_selected_cfgs(f"{args.run_dir}/final_selected.jsonl", top_k=args.top_k)

    out_jsonl = f"{args.run_dir}/meta_test.jsonl"
    out_csv = f"{args.run_dir}/meta_test.csv"
    header = ["shift","cfg_rank","mean_shd","mean_ace_err","mean_stability","mean_time",
              "skeleton","alpha","max_cond_set","corr_thresh","max_edges","orient","meek_passes","prune_marginal_alpha"]

    rng = default_rng(args.seed)
    shifts = make_shift_suite()

    for shift in shifts:
        tasks = sample_test_tasks_shifted(rng, args.test_tasks, shift=shift)
        for r, cfg in enumerate(cfgs):
            ind = evaluate_cfg_on_tasks(cfg, tasks)
            row = {"shift": shift, "cfg_rank": r, **ind.meta, **asdict(cfg)}
            append_jsonl(out_jsonl, row)
            append_csv(out_csv, row, header=header)
            print(f"[{shift}] cfg#{r} meta={ind.meta}")

    print(f"\nSaved meta-test results to {out_csv} and {out_jsonl}")

if __name__ == "__main__":
    main()
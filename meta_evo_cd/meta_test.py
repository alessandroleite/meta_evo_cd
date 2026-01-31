from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from typing import List
from collections import Counter

from numpy.random import default_rng

from .pipelines import PipelineConfig
from .run import evaluate_cfg_on_tasks
from .meta_protocol import sample_test_tasks_shifted, make_shift_suite
from .logging_utils import append_jsonl, append_csv, ensure_dir


FAM_COLS = [
    "task_linear",
    "task_tanh_quad_clip",
    "task_tanh_quad_bounded",
    "task_poly3_clip",
    "task_post_nonlinear",
    "task_relu_smooth_clip",
    "task_frac_nonlinear",
]


def summarize_task_batch(tasks) -> dict:
    """
    Summarize the sampled task batch (shift batch) so meta-test logs are interpretable.
    Tasks without nonlinear_spec are counted as 'linear'.
    """
    fams = []
    for t in tasks:
        nl = getattr(t, "nonlinear_spec", None)
        fams.append(nl.family if nl is not None else "linear")
    c = Counter(fams)
    return {str(k): int(v) for k, v in c.items()}


def family_cols(task_family_counts: dict, total_tasks: int) -> dict:
    nonlinear_total = total_tasks - int(task_family_counts.get("linear", 0))
    return {
        "task_linear": int(task_family_counts.get("linear", 0)),
        "task_tanh_quad_clip": int(task_family_counts.get("tanh_quad_clip", 0)),
        "task_tanh_quad_bounded": int(task_family_counts.get("tanh_quad_bounded", 0)),
        "task_poly3_clip": int(task_family_counts.get("poly3_clip", 0)),
        "task_post_nonlinear": int(task_family_counts.get("post_nonlinear", 0)),
        "task_relu_smooth_clip": int(task_family_counts.get("relu_smooth_clip", 0)),
        "task_frac_nonlinear": float(nonlinear_total) / max(1, total_tasks),
    }


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

    base_header = [
        "shift", "cfg_rank",
        "mean_shd", "mean_ace_err", "mean_stability", "mean_time",
        "skeleton", "alpha", "max_cond_set", "corr_thresh", "max_edges",
        "orient", "meek_passes", "prune_marginal_alpha",
    ]
    header = base_header + FAM_COLS

    rng = default_rng(args.seed)
    shifts = make_shift_suite()

    for shift in shifts:
        tasks = sample_test_tasks_shifted(rng, args.test_tasks, shift=shift)

        # Summarize the batch once per shift (same tasks used for all cfgs)
        task_family_counts = summarize_task_batch(tasks)
        extras = family_cols(task_family_counts, total_tasks=len(tasks))

        for r, cfg in enumerate(cfgs):
            ind = evaluate_cfg_on_tasks(cfg, tasks)

            row_csv = {"shift": shift, "cfg_rank": r, **ind.meta, **asdict(cfg), **extras}
            row_jsonl = {**row_csv, "task_family_counts": task_family_counts}

            append_jsonl(out_jsonl, row_jsonl)
            append_csv(out_csv, row_csv, header=header)

            print(f"[{shift}] cfg#{r} meta={ind.meta} task_mix={task_family_counts}")

    print(f"\nSaved meta-test results to {out_csv} and {out_jsonl}")


if __name__ == "__main__":
    main()
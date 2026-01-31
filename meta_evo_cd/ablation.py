from __future__ import annotations
import argparse
import os
from dataclasses import asdict
from typing import List
from collections import Counter

import numpy as np
from numpy.random import default_rng

from .pipelines import PipelineConfig, random_config, mutate, crossover
from .run import evaluate_cfg_on_tasks
from .evo import select_nsga2, Individual
from .meta_protocol import sample_train_tasks, sample_test_tasks_shifted, make_shift_suite
from .logging_utils import ensure_dir, append_jsonl, append_csv
from .meta_test import load_selected_cfgs


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
    Summarize the sampled task batch so logs are interpretable.
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


# ---- helpers to constrain the genome ----

def constrain_cfg(cfg: PipelineConfig, mode: str, rng: np.random.Generator) -> PipelineConfig:
    """
    mode examples:
      - "full": no constraints
      - "no_pc": force corr skeleton
      - "no_orient": force orient none
      - "pc_only": force pc_lite
      - "orient_only": force v_meek
      - "no_prune": disable marginal pruning
      - "small_k": cap max_cond_set=1
    """
    d = cfg.__dict__.copy()
    if mode == "full":
        pass
    elif mode == "no_pc":
        d["skeleton"] = "corr"
    elif mode == "pc_only":
        d["skeleton"] = "pc_lite"
    elif mode == "no_orient":
        d["orient"] = "none"
    elif mode == "orient_only":
        d["orient"] = "v_meek"
    elif mode == "no_prune":
        d["prune_marginal_alpha"] = 1.5
    elif mode == "small_k":
        d["max_cond_set"] = 1
    else:
        raise ValueError(mode)
    d["seed"] = int(rng.integers(0, 10**9))
    return PipelineConfig(**d)


def evolve_one(mode: str, outdir: str, seed: int, generations: int, pop: int, task_batch: int):
    ensure_dir(outdir)
    rng = default_rng(seed)

    pop_cfgs = [constrain_cfg(random_config(rng), mode, rng) for _ in range(pop)]

    for gen in range(generations):
        tasks = sample_train_tasks(rng, task_batch)
        task_family_counts = summarize_task_batch(tasks)
        extras = family_cols(task_family_counts, total_tasks=len(tasks))

        evaled: List[Individual] = [evaluate_cfg_on_tasks(cfg, tasks) for cfg in pop_cfgs]
        sel = select_nsga2(evaled, n_keep=max(2, pop // 3))

        sel_sorted = sorted(sel, key=lambda ind: (ind.obj[0], ind.obj[1], ind.obj[3]))
        best = sel_sorted[0]
        print(f"[{mode} gen {gen}] best meta={best.meta} cfg={asdict(best.cfg)} task_mix={task_family_counts}")

        # log best only (keeps ablation lightweight)
        row_jsonl = {"mode": mode, "gen": gen, **best.meta, **asdict(best.cfg), **extras, "task_family_counts": task_family_counts}
        append_jsonl(f"{outdir}/train_best.jsonl", row_jsonl)

        # offspring
        new_pop = [ind.cfg for ind in sel]
        while len(new_pop) < pop:
            p1, p2 = rng.choice(sel).cfg, rng.choice(sel).cfg
            c1, c2 = crossover(p1, p2, rng)
            c1 = constrain_cfg(mutate(c1, rng), mode, rng)
            c2 = constrain_cfg(mutate(c2, rng), mode, rng)
            new_pop.append(c1)
            if len(new_pop) < pop:
                new_pop.append(c2)
        pop_cfgs = new_pop

    # save final selected configs (top 5 from last sel)
    final_cfgs = [asdict(ind.cfg) for ind in sel_sorted[:5]]
    append_jsonl(f"{outdir}/final_selected.jsonl", {"seed": seed, "mode": mode, "final": final_cfgs})


def meta_test_run(run_dir: str, seed: int, test_tasks: int = 24, top_k: int = 5):
    rng = default_rng(seed)

    # load final configs
    cfgs = load_selected_cfgs(f"{run_dir}/final_selected.jsonl", top_k=top_k)
    shifts = make_shift_suite()

    out_jsonl = f"{run_dir}/meta_test.jsonl"
    out_csv = f"{run_dir}/meta_test.csv"

    base_header = [
        "shift", "cfg_rank",
        "mean_shd", "mean_ace_err", "mean_stability", "mean_time",
        "skeleton", "alpha", "max_cond_set", "corr_thresh", "max_edges",
        "orient", "meek_passes", "prune_marginal_alpha",
    ]
    header = base_header + FAM_COLS

    for shift in shifts:
        tasks = sample_test_tasks_shifted(rng, test_tasks, shift=shift)

        # summarize once per shift (same task batch for all cfgs)
        task_family_counts = summarize_task_batch(tasks)
        extras = family_cols(task_family_counts, total_tasks=len(tasks))

        for r, cfg in enumerate(cfgs):
            ind = evaluate_cfg_on_tasks(cfg, tasks)

            row_csv = {"shift": shift, "cfg_rank": r, **ind.meta, **asdict(cfg), **extras}
            row_jsonl = {**row_csv, "task_family_counts": task_family_counts}

            append_jsonl(out_jsonl, row_jsonl)
            append_csv(out_csv, row_csv, header=header)

            print(f"[meta-test {os.path.basename(run_dir)}] {shift} cfg#{r} meta={ind.meta} task_mix={task_family_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", type=str, default="runs/ablations")
    ap.add_argument("--modes", type=str, default="full,no_pc,no_orient,no_prune,small_k")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--task_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test_tasks", type=int, default=24)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ensure_dir(args.outroot)

    for i, mode in enumerate(modes):
        run_dir = f"{args.outroot}/{mode}"
        evolve_one(
            mode, run_dir,
            seed=args.seed + 100 * i,
            generations=args.generations,
            pop=args.pop,
            task_batch=args.task_batch,
        )
        meta_test_run(
            run_dir,
            seed=args.seed + 999 + 100 * i,
            test_tasks=args.test_tasks,
            top_k=5,
        )

    print(f"\nAblations done. See {args.outroot}/<mode>/train_best.jsonl, meta_test.csv, and meta_test.jsonl")


if __name__ == "__main__":
    main()
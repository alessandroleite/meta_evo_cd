from __future__ import annotations
import argparse
import os
from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional
import numpy as np
from numpy.random import default_rng

from .pipelines import PipelineConfig, random_config, mutate, crossover
from .run import evaluate_cfg_on_tasks
from .evo import select_nsga2, Individual
from .meta_protocol import sample_train_tasks
from .logging_utils import ensure_dir, append_jsonl
from .meta_test import load_selected_cfgs
from .meta_protocol import sample_test_tasks_shifted, make_shift_suite

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
        evaled: List[Individual] = [evaluate_cfg_on_tasks(cfg, tasks) for cfg in pop_cfgs]
        sel = select_nsga2(evaled, n_keep=max(2, pop // 3))

        sel_sorted = sorted(sel, key=lambda ind: (ind.obj[0], ind.obj[1], ind.obj[3]))
        best = sel_sorted[0]
        print(f"[{mode} gen {gen}] best meta={best.meta} cfg={asdict(best.cfg)}")

        # log best only (keeps ablation lightweight)
        append_jsonl(f"{outdir}/train_best.jsonl", {"mode": mode, "gen": gen, **best.meta, **asdict(best.cfg)})

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
    out = f"{run_dir}/meta_test.jsonl"
    for shift in shifts:
        tasks = sample_test_tasks_shifted(rng, test_tasks, shift=shift)
        for r, cfg in enumerate(cfgs):
            ind = evaluate_cfg_on_tasks(cfg, tasks)
            append_jsonl(out, {"shift": shift, "cfg_rank": r, **ind.meta, **asdict(cfg)})
            print(f"[meta-test {os.path.basename(run_dir)}] {shift} cfg#{r} meta={ind.meta}")

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
        evolve_one(mode, run_dir, seed=args.seed + 100*i, generations=args.generations,
                  pop=args.pop, task_batch=args.task_batch)
        meta_test_run(run_dir, seed=args.seed + 999 + 100*i, test_tasks=args.test_tasks, top_k=5)

    print(f"\nAblations done. See {args.outroot}/<mode>/train_best.jsonl and meta_test.jsonl")

if __name__ == "__main__":
    main()
from __future__ import annotations
import argparse
from dataclasses import asdict
from numpy.random import default_rng

from .pipelines import random_config, mutate, crossover, discover_graph
from .evo import select_nsga2
from .budget import evaluate_cfg_budgeted
from .meta_protocol import sample_task
from .logging_utils import ensure_dir, append_jsonl
from .plotting import plot_pareto

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs/run_budgeted0")
    ap.add_argument("--budget_s", type=float, default=20.0, help="Time budget per pipeline evaluation")
    ap.add_argument("--min_tasks", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    rng = default_rng(args.seed)
    pop = [random_config(rng) for _ in range(args.pop)]

    log_path = f"{args.outdir}/train_budgeted.jsonl"

    for gen in range(args.generations):
        evaled = [
            evaluate_cfg_budgeted(
                cfg, discover_fn=discover_graph, task_sampler=sample_task,
                seed=args.seed + 10_000 * gen + i,
                time_budget_s=args.budget_s,
                min_tasks=args.min_tasks
            )
            for i, cfg in enumerate(pop)
        ]

        sel = select_nsga2(evaled, n_keep=max(2, args.pop // 3))
        sel_sorted = sorted(sel, key=lambda ind: (ind.obj[0], ind.obj[1], ind.obj[3]))
        best = sel_sorted[0]

        print(f"\n[gen {gen}] best obj={best.obj} meta={best.meta}")
        print(f" cfg={asdict(best.cfg)}")

        # log top 10
        for rank, ind in enumerate(sel_sorted[: min(10, len(sel_sorted))]):
            append_jsonl(log_path, {"gen": gen, "rank": rank, **ind.meta, **asdict(ind.cfg)})

        # pareto snapshot
        pts = [(ind.meta["mean_shd"], ind.meta["mean_ace_err"]) for ind in evaled]
        plot_pareto(pts, f"{args.outdir}/pareto_gen{gen}.png", title=f"Budgeted Gen {gen}")

        # offspring
        new_pop = [ind.cfg for ind in sel]
        while len(new_pop) < args.pop:
            p1, p2 = rng.choice(sel).cfg, rng.choice(sel).cfg
            c1, c2 = crossover(p1, p2, rng)
            c1 = mutate(c1, rng)
            c2 = mutate(c2, rng)
            new_pop.append(c1)
            if len(new_pop) < args.pop:
                new_pop.append(c2)
        pop = new_pop

    print(f"\nSaved budgeted logs/plots to {args.outdir}")

if __name__ == "__main__":
    main()
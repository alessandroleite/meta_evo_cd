# from __future__ import annotations
# import argparse
# import time
# from dataclasses import asdict
# from typing import List
# import numpy as np
# from numpy.random import default_rng

# from .envs import sample_task, random_dag, simulate_sem
# from .pipelines import random_config, mutate, crossover, discover_graph, PipelineConfig
# from .metrics import evaluate_on_task
# from .evo import Individual, select_nsga2

# def evaluate_cfg_on_tasks(cfg: PipelineConfig, tasks) -> Individual:
#     shds, aces, stabs, times = [], [], [], []
#     for t in tasks:
#         rng = default_rng(t.seed)
#         A_true = random_dag(t.d, t.expected_degree, rng)
#         X = simulate_sem(A_true, t.n, t.sem_type, t.noise_type, t.noise_scale, rng)

#         t0 = time.perf_counter()
#         dag_hat = discover_graph(X, cfg)
#         dt = time.perf_counter() - t0

#         m = evaluate_on_task(t, A_true, X, dag_hat, dt, discover_graph, cfg)
#         shds.append(m["shd"]); aces.append(m["ace_err"]); stabs.append(m["stability"]); times.append(m["time"])

#     meta = {
#         "mean_shd": float(np.mean(shds)),
#         "mean_ace_err": float(np.mean(aces)),
#         "mean_stability": float(np.mean(stabs)),
#         "mean_time": float(np.mean(times)),
#     }
#     # objectives all minimized:
#     obj = (meta["mean_shd"], meta["mean_ace_err"], -meta["mean_stability"], meta["mean_time"])
#     return Individual(cfg=cfg, obj=obj, meta=meta)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--generations", type=int, default=8)
#     ap.add_argument("--pop", type=int, default=24)
#     ap.add_argument("--task_batch", type=int, default=8)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     rng = default_rng(args.seed)
#     pop = [random_config(rng) for _ in range(args.pop)]

#     for gen in range(args.generations):
#         tasks = [sample_task(rng) for _ in range(args.task_batch)]

#         evaled = [evaluate_cfg_on_tasks(cfg, tasks) for cfg in pop]
#         sel = select_nsga2(evaled, n_keep=max(2, args.pop // 3))

#         # print a few pareto-front members
#         sel_sorted = sorted(sel, key=lambda ind: (ind.obj[0], ind.obj[1], ind.obj[3]))
#         best = sel_sorted[0]
#         print(f"\n[gen {gen}] sample best (lex): obj={best.obj} meta={best.meta}")
#         print(f" cfg={asdict(best.cfg)}")

#         # offspring
#         new_pop = [ind.cfg for ind in sel]
#         while len(new_pop) < args.pop:
#             p1, p2 = rng.choice(sel).cfg, rng.choice(sel).cfg
#             c1, c2 = crossover(p1, p2, rng)
#             c1 = mutate(c1, rng)
#             c2 = mutate(c2, rng)
#             new_pop.append(c1)
#             if len(new_pop) < args.pop:
#                 new_pop.append(c2)

#         pop = new_pop

# if __name__ == "__main__":
#     main()

from __future__ import annotations
import argparse
import time
from dataclasses import asdict
from typing import List
import numpy as np
from numpy.random import default_rng

from .pipelines import random_config, mutate, crossover, discover_graph, PipelineConfig
from .metrics import evaluate_on_task
from .evo import Individual, select_nsga2
from .envs import random_dag, simulate_sem
from .meta_protocol import sample_train_tasks
from .logging_utils import append_jsonl, append_csv, ensure_dir
from .plotting import plot_pareto

def evaluate_cfg_on_tasks(cfg: PipelineConfig, tasks) -> Individual:
    shds, aces, stabs, times = [], [], [], []
    for t in tasks:
        rng = default_rng(t.seed)
        A_true = random_dag(t.d, t.expected_degree, rng)
        X = simulate_sem(A_true, t.n, t.sem_type, t.noise_type, t.noise_scale, rng)

        t0 = time.perf_counter()
        dag_hat = discover_graph(X, cfg)
        dt = time.perf_counter() - t0

        m = evaluate_on_task(t, A_true, X, dag_hat, dt, discover_graph, cfg)
        shds.append(m["shd"]); aces.append(m["ace_err"]); stabs.append(m["stability"]); times.append(m["time"])

    meta = {
        "mean_shd": float(np.mean(shds)),
        "mean_ace_err": float(np.mean(aces)),
        "mean_stability": float(np.mean(stabs)),
        "mean_time": float(np.mean(times)),
    }
    obj = (meta["mean_shd"], meta["mean_ace_err"], -meta["mean_stability"], meta["mean_time"])
    return Individual(cfg=cfg, obj=obj, meta=meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--task_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs/run0")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    rng = default_rng(args.seed)
    pop = [random_config(rng) for _ in range(args.pop)]

    jsonl_path = f"{args.outdir}/train_log.jsonl"
    csv_path = f"{args.outdir}/train_log.csv"

    # fixed CSV header for easy plotting later
    header = [
        "gen","rank","mean_shd","mean_ace_err","mean_stability","mean_time",
        "skeleton","alpha","max_cond_set","corr_thresh","max_edges","orient","meek_passes","prune_marginal_alpha"
    ]

    for gen in range(args.generations):
        tasks = sample_train_tasks(rng, args.task_batch)

        evaled = [evaluate_cfg_on_tasks(cfg, tasks) for cfg in pop]
        sel = select_nsga2(evaled, n_keep=max(2, args.pop // 3))

        # sort selected for reporting (lex)
        sel_sorted = sorted(sel, key=lambda ind: (ind.obj[0], ind.obj[1], ind.obj[3]))
        best = sel_sorted[0]

        print(f"\n[gen {gen}] best (lex) obj={best.obj} meta={best.meta}")
        print(f" cfg={asdict(best.cfg)}")

        # log all selected individuals + some of evaled (optional)
        for rank, ind in enumerate(sel_sorted[: min(len(sel_sorted), 10)]):
            row = {"gen": gen, "rank": rank, **ind.meta, **asdict(ind.cfg)}
            append_jsonl(jsonl_path, row)
            append_csv(csv_path, row, header=header)

        # plot a pareto snapshot for SHD vs ACE error using entire population
        pts = [(ind.meta["mean_shd"], ind.meta["mean_ace_err"]) for ind in evaled]
        plot_pareto(pts, f"{args.outdir}/pareto_gen{gen}.png", title=f"Gen {gen}: population")

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

    # store final selected configs
    final_cfgs = [asdict(ind.cfg) for ind in sel_sorted]
    append_jsonl(f"{args.outdir}/final_selected.jsonl", {"seed": args.seed, "final": final_cfgs})
    print(f"\nSaved logs/plots to {args.outdir}")

if __name__ == "__main__":
    main()
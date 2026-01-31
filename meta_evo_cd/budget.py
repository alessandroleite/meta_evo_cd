from __future__ import annotations
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
from numpy.random import default_rng

from .envs import sample_task, random_dag, simulate_sem
from .metrics import evaluate_on_task
from .evo import Individual

def evaluate_cfg_budgeted(cfg,
                          discover_fn,
                          task_sampler: Callable[[np.random.Generator], object],
                          seed: int,
                          time_budget_s: float,
                          min_tasks: int = 4,
                          max_tasks: int = 10_000) -> Individual:
    """
    Evaluate cfg by sampling tasks until wall-clock budget is used.
    Ensures at least min_tasks tasks (unless max_tasks smaller).
    Returns an Individual where meta metrics are averaged over evaluated tasks.

    Objectives minimized: (mean_shd, mean_ace_err, -mean_stability, mean_time_per_task)
    """
    rng = default_rng(seed)
    shds, aces, stabs, times = [], [], [], []
    start = time.perf_counter()
    n_done = 0

    while n_done < max_tasks:
        t = task_sampler(rng)
        # simulate data
        rng_t = default_rng(t.seed)
        A_true = random_dag(t.d, t.expected_degree, rng_t)
        X = simulate_sem(A_true, t.n, t.sem_type, t.noise_type, t.noise_scale, rng_t)

        t0 = time.perf_counter()
        dag_hat = discover_fn(X, cfg)
        dt = time.perf_counter() - t0

        m = evaluate_on_task(t, A_true, X, dag_hat, dt, discover_fn, cfg)
        shds.append(m["shd"]); aces.append(m["ace_err"]); stabs.append(m["stability"]); times.append(m["time"])
        n_done += 1

        elapsed = time.perf_counter() - start
        if n_done >= min_tasks and elapsed >= time_budget_s:
            break

    meta = {
        "mean_shd": float(np.mean(shds)) if shds else float("nan"),
        "mean_ace_err": float(np.mean(aces)) if aces else float("nan"),
        "mean_stability": float(np.mean(stabs)) if stabs else float("nan"),
        "mean_time": float(np.mean(times)) if times else float("nan"),  # time per task
        "n_tasks": float(n_done),
        "budget_s": float(time_budget_s),
    }
    obj = (meta["mean_shd"], meta["mean_ace_err"], -meta["mean_stability"], meta["mean_time"])
    return Individual(cfg=cfg, obj=obj, meta=meta)
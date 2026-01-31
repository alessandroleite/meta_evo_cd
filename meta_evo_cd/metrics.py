from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx

from .envs import TaskSpec, true_ace_mc

def shd_directed(A_true: np.ndarray, A_hat: np.ndarray) -> int:
    # simple directed mismatch count (reversal counts 2)
    return int(np.sum(np.abs(A_true - A_hat)))

def bootstrap_stability(X: np.ndarray, discover_fn, cfg, B: int = 10, frac: float = 0.8) -> float:
    rng = np.random.default_rng(cfg.seed + 12345)
    d = X.shape[1]
    counts = np.zeros((d, d), dtype=float)
    n = X.shape[0]
    m = max(2, int(frac * n))
    for b in range(B):
        idx = rng.choice(n, size=m, replace=True)
        cfg_b = cfg.__class__(**{**cfg.__dict__, "seed": cfg.seed + 1000 + b})
        A = discover_fn(X[idx], cfg_b)
        counts += A
    p = counts / B
    return float(1.0 - np.mean(4.0 * p * (1.0 - p)))

def _ols_coef(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    X1 = np.column_stack([np.ones(n), X]) if X.size else np.ones((n, 1))
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta

def ace_via_adjustment(X: np.ndarray, dag_hat: np.ndarray, treat: int, outcome: int,
                       a: float = 2.0, b: float = -2.0) -> float:
    """
    Estimate ACE using a simple backdoor adjustment heuristic:
    use Z = parents of treat in learned DAG if treat->... and no obvious issue.
    We just fit linear regression Y ~ X + Z and compute beta_X*(a-b).
    """
    pa = np.where(dag_hat[:, treat] == 1)[0]
    Z = X[:, pa] if len(pa) else np.empty((X.shape[0], 0))
    Xreg = np.column_stack([X[:, treat], Z]) if Z.size else X[:, [treat]]
    beta = _ols_coef(X[:, outcome], Xreg)
    beta_x = float(beta[1]) if Z.size else float(beta[1])  # after intercept
    return beta_x * (a - b)

def evaluate_on_task(task: TaskSpec, A_true: np.ndarray, X: np.ndarray,
                     dag_hat: np.ndarray, discover_time: float,
                     discover_fn, cfg) -> Dict[str, float]:
    d = task.d
    # sample ACE queries
    rngq = np.random.default_rng(task.seed + 4242)
    ace_errs = []
    for _ in range(task.n_ace_queries):
        treat = int(rngq.integers(0, d))
        outcome = int(rngq.integers(0, d))
        while outcome == treat:
            outcome = int(rngq.integers(0, d))

        true = true_ace_mc(task, A_true, treat, outcome)
        pred = ace_via_adjustment(X, dag_hat, treat, outcome,
                                  a=task.intervention_strength, b=-task.intervention_strength)
        denom = max(1e-6, abs(true))
        ace_errs.append(abs(true - pred) / denom)

    stab = bootstrap_stability(X, discover_fn, cfg, B=8, frac=0.8)
    return {
        "shd": float(shd_directed(A_true, dag_hat)),
        "ace_err": float(np.mean(ace_errs)),
        "stability": float(stab),
        "time": float(discover_time),
    }
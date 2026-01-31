## `meta_evo_cd/envs.py`
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from numpy.random import default_rng
import networkx as nx

@dataclass(frozen=True)
class TaskSpec:
    d: int
    expected_degree: float
    n: int
    sem_type: str       # "linear" or "nonlinear"
    noise_type: str     # "gauss"|"laplace"|"student"
    noise_scale: float
    seed: int
    n_ace_queries: int = 3
    ace_mc: int = 2000
    intervention_strength: float = 2.0

def random_dag(d: int, expected_degree: float, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(d)
    inv = np.empty(d, dtype=int)
    inv[order] = np.arange(d)
    p = min(1.0, expected_degree / max(1, d - 1))
    A = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(d):
            if i != j and inv[i] < inv[j] and rng.random() < p:
                A[i, j] = 1
    return A

def _noise(n: int, noise_type: str, scale: float, rng: np.random.Generator, *, nonlinear=False) -> np.ndarray:
    if noise_type == "gauss":
        return rng.normal(0.0, scale, size=n)
    if noise_type == "laplace":
        return rng.laplace(0.0, scale, size=n)
    if noise_type == "student":
        # return rng.standard_t(df=3, size=n) * scale
        # Heavy-tailed but numerically safe Student-t
        eps = rng.standard_t(df=3, size=n) * scale
        if nonlinear:
            eps = np.clip(eps, -10.0 * scale, 10.0 * scale)
        # Clip extreme tails to avoid NaNs downstream
        # eps = np.clip(eps, -10.0 * scale, 10.0 * scale)
        return eps

    raise ValueError(noise_type)

def simulate_sem(A: np.ndarray, n: int, sem_type: str, noise_type: str, noise_scale: float,
                 rng: np.random.Generator) -> np.ndarray:
    d = A.shape[0]
    topo = list(nx.topological_sort(nx.DiGraph(A)))
    X = np.zeros((n, d), dtype=float)

    if sem_type == "nonlinear":
        W = rng.uniform(0.2, 0.8, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
    else: 
        W = rng.uniform(0.5, 2.0, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))

    for j in topo:
        pa = np.where(A[:, j] == 1)[0]
        eps = _noise(n, noise_type, noise_scale, rng)
        if len(pa) == 0:
            X[:, j] = eps
        else:
            # lin = sum(W[p, j] * X[:, p] for p in pa)
            lin = np.sum(X[:, pa] * W[pa, j], axis=1)
            if sem_type == "linear":
                X[:, j] = lin + eps
            elif sem_type == "nonlinear":
                # X[:, j] = np.tanh(lin) + 0.3 * (lin ** 2) + eps              

                # --- numeric stabilization ---
                # Prevent overflow in lin**2, especially with heavy-tailed noise.
                lin = np.clip(lin, -10.0, 10.0)              # adjust bounds if needed
                quad = 0.3 * (lin * lin)                     # safe after clip
                X[:, j] = np.tanh(lin) + quad + eps
            else:
                raise ValueError(sem_type)
            
            xj = X[:, j]
            # Replace any non-finite values (rare but can happen with extreme noise)
            if not np.all(np.isfinite(xj)):
                xj = np.where(np.isfinite(xj), xj, np.nan)
                # try a cheap fallback: re-draw eps once and recompute xj
                eps2 = _noise(n, noise_type, noise_scale, rng)
                if sem_type == "linear":
                    xj = lin + eps2
                else:
                    xj = np.tanh(lin) + quad + eps2                    
                # final guard: force finite
                xj = np.where(np.isfinite(xj), xj, 0.0)

            m = np.nanmean(xj)
            s = np.nanstd(xj)
            if np.isfinite(m) and np.isfinite(s) and s > 1e-8:
                X[:, j] = (xj - m) / s
            else:
                # fallback: if degenerate or non-finite, just center safely
                # X[:, j] = xj - np.nanmean(xj)
                mm = np.nanmean(xj)
                if not np.isfinite(mm):
                    X[:, j] = 0.0
                else: 
                    X[:, j] = xj - mm

    return X

def do_intervention_samples(A: np.ndarray, sem_type: str, noise_type: str, noise_scale: float,
                            treat: int, treat_value: float, n: int, rng: np.random.Generator) -> np.ndarray:
    d = A.shape[0]
    topo = list(nx.topological_sort(nx.DiGraph(A)))
    X = np.zeros((n, d), dtype=float)
    # W = rng.uniform(0.5, 2.0, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
    # Match simulate_sem() weight scales to avoid blow-ups under nonlinear SEMs
    if sem_type == "nonlinear":
        W = rng.uniform(0.2, 0.8, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
    else:
        W = rng.uniform(0.5, 2.0, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))        
    for j in topo:
        if j == treat:
            X[:, j] = treat_value
            continue
        pa = np.where(A[:, j] == 1)[0]
        eps = _noise(n, noise_type, noise_scale, rng, nonlinear=(sem_type == "nonlinear"))
        if len(pa) == 0:
            X[:, j] = eps
        else:
            # lin = sum(W[p, j] * X[:, p] for p in pa)
            # faster + numerically cleaner than Python sum()
            lin = np.sum(X[:, pa] * W[pa, j], axis=1)
            if sem_type == "linear":
                X[:, j] = lin + eps
            elif sem_type == "nonlinear":
                # X[:, j] = np.tanh(lin) + 0.3 * (lin ** 2) + eps
                # --- numeric stabilization (same as simulate_sem) ---
                lin = np.clip(lin, -10.0, 10.0)
                quad = 0.3 * (lin * lin)
                X[:, j] = np.tanh(lin) + quad + eps
            else:
                raise ValueError(sem_type)
            
            # per-node standardization + non-finite guard
            xj = X[:, j]
            if not np.all(np.isfinite(xj)):
                xj = np.where(np.isfinite(xj), xj, np.nan)
                eps2 = _noise(n, noise_type, noise_scale, rng, nonlinear=(sem_type == "nonlinear"))
                if sem_type == "linear":
                    xj = lin + eps2
                else:
                    quad = 0.3 * (lin * lin)
                    xj = np.tanh(lin) + quad + eps2
                xj = np.where(np.isfinite(xj), xj, 0.0)

            m = np.nanmean(xj)
            s = np.nanstd(xj)
            if np.isfinite(m) and np.isfinite(s) and s > 1e-8:
                X[:, j] = (xj - m) / s
            else:
                mm = np.nanmean(xj)
                X[:, j] = 0.0 if not np.isfinite(mm) else (xj - mm)            

    return X

def true_ace_mc(task: TaskSpec, A_true: np.ndarray, treat: int, outcome: int) -> float:
    rng = default_rng(task.seed + 10_000 + 31 * treat + 17 * outcome)
    a = task.intervention_strength
    b = -task.intervention_strength
    Xa = do_intervention_samples(A_true, task.sem_type, task.noise_type, task.noise_scale,
                                 treat, a, task.ace_mc, rng)
    Xb = do_intervention_samples(A_true, task.sem_type, task.noise_type, task.noise_scale,
                                 treat, b, task.ace_mc, rng)
    return float(Xa[:, outcome].mean() - Xb[:, outcome].mean())

def sample_task(rng: np.random.Generator) -> TaskSpec:
    d = int(rng.choice([10, 20, 30]))
    n = int(rng.choice([200, 500, 1000, 5000]))
    expected_degree = float(rng.choice([1.0, 2.0, 3.0]))
    sem_type = str(rng.choice(["linear", "nonlinear"]))
    noise_type = str(rng.choice(["gauss", "laplace", "student"]))
    noise_scale = float(rng.choice([0.5, 1.0, 2.0]))
    seed = int(rng.integers(0, 10**9))
    return TaskSpec(d=d, expected_degree=expected_degree, n=n, sem_type=sem_type,
                    noise_type=noise_type, noise_scale=noise_scale, seed=seed)
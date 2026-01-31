## `meta_evo_cd/envs.py`
from __future__ import annotations
import numpy as np
import networkx as nx
from dataclasses import dataclass
from numpy.random import default_rng
from typing import Optional, List, Literal, Tuple


# ---------------------------
# Nonlinear family definition
# ---------------------------

NonlinearFamily = Literal[
    "tanh_quad_clip",      # tanh(lin_b) + a*(lin_b^2) + eps
    "tanh_quad_bounded",   # tanh(lin_raw) + a*(lin_raw^2/(1+lin_raw^2)) + eps
    "poly3_clip",          # b1*lin_b + b2*lin_b^2 + b3*lin_b^3 + eps
    "post_nonlinear",      # g( f(lin_raw) + eps ), where g is tanh or sigmoid-ish
    "relu_smooth_clip",    # softplus(lin_b) + a*(lin_b^2)/(1+lin_b^2) + eps
]

@dataclass(frozen=True)
class NonlinearSpec:
    family: NonlinearFamily = "tanh_quad_clip"
    clip_lin: float = 10.0           # used in *_clip families
    quad_a: float = 0.3              # quadratic coefficient or bounded quad scale
    poly_b1: float = 1.0
    poly_b2: float = 0.1
    poly_b3: float = 0.01
    post_g: Literal["tanh", "sigmoid"] = "tanh"
    post_scale: float = 1.0          # scale inside post-nonlinear g()
    softplus_beta: float = 1.0       # for smooth relu (softplus)

@dataclass(frozen=True)
class TaskSpec:
    d: int
    expected_degree: float
    n: int
    sem_type: str       # "linear" or "nonlinear"
    noise_type: str     # "gauss"|"laplace"|"student"
    noise_scale: float
    seed: int
    # sampled nonlinearity spec for nonlinear tasks
    nonlinear_spec: Optional[NonlinearSpec] = None

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

def _softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    # stable softplus: log(1+exp(beta x))/beta
    z = beta * x
    # prevent overflow in exp
    z = np.clip(z, -50.0, 50.0)
    return np.log1p(np.exp(z)) / beta

def _post_g(u: np.ndarray, kind: str, scale: float) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(u / max(1e-8, scale))
    if kind == "sigmoid":
        z = np.clip(u / max(1e-8, scale), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(kind)

def _nonlinear_eval(lin_raw: np.ndarray, eps: np.ndarray, spec: NonlinearSpec) -> np.ndarray:
    """
    Nonlinear structural equation xj = h(lin_raw, eps; spec)
    Must be numerically stable; clip-based families use explicit bounded predictors.
    """
    fam = spec.family

    if fam == "tanh_quad_clip":
        # explicit bounded predictor invariant
        lin_b = np.clip(lin_raw, -spec.clip_lin, spec.clip_lin)
        quad_b = spec.quad_a * (lin_b * lin_b)
        return np.tanh(lin_b) + quad_b + eps

    if fam == "tanh_quad_bounded":
        # bounded quad without clipping (clean SEM semantics)
        # quad term is in [0, quad_a)
        denom = 1.0 + lin_raw * lin_raw
        quad = spec.quad_a * (lin_raw * lin_raw) / denom
        return np.tanh(lin_raw) + quad + eps

    if fam == "poly3_clip":
        lin_b = np.clip(lin_raw, -spec.clip_lin, spec.clip_lin)
        return (spec.poly_b1 * lin_b +
                spec.poly_b2 * (lin_b * lin_b) +
                spec.poly_b3 * (lin_b * lin_b * lin_b) +
                eps)

    if fam == "post_nonlinear":
        # f is tanh(lin_raw), then add eps, then apply g
        u = np.tanh(lin_raw) + eps
        return _post_g(u, spec.post_g, spec.post_scale)

    if fam == "relu_smooth_clip":
        lin_b = np.clip(lin_raw, -spec.clip_lin, spec.clip_lin)
        quad = spec.quad_a * (lin_b * lin_b) / (1.0 + lin_b * lin_b)
        return _softplus(lin_b, beta=spec.softplus_beta) + quad + eps

    raise ValueError(f"Unknown nonlinear family: {fam}")

def _sample_weights(d: int, sem_type: str, rng: np.random.Generator) -> np.ndarray:
    """Sample SEM weights; use smaller magnitudes for nonlinear to avoid blow-ups."""
    if sem_type == "nonlinear":
        lo, hi = 0.2, 0.8
    else:
        lo, hi = 0.5, 2.0
    return rng.uniform(lo, hi, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))

# def _sample_weights(d: int, sem_type: str, rng: np.random.Generator) -> np.ndarray:
#     """Sample SEM weights; use smaller magnitudes for nonlinear to avoid blow-ups."""
#     if sem_type == "nonlinear":
#         lo, hi = 0.2, 0.8
#     else:
#         lo, hi = 0.5, 2.0
#     return rng.uniform(lo, hi, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))

def _standardize_safe(x: np.ndarray) -> np.ndarray:
    """Per-node standardization with robust non-finite handling."""
    x = np.where(np.isfinite(x), x, np.nan)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if np.isfinite(m) and np.isfinite(s) and s > 1e-8:
        z = (x - m) / s
        return np.where(np.isfinite(z), z, 0.0)
    if not np.isfinite(m):
        return np.zeros_like(x)
    z = x - m
    return np.where(np.isfinite(z), z, 0.0)


def _node_update(
    X: np.ndarray,
    W: np.ndarray,
    A: np.ndarray,
    j: int,
    sem_type: str,
    noise_type: str,
    noise_scale: float,
    rng: np.random.Generator,
    *,
    nonlinear_spec: Optional[NonlinearSpec] = None,
    standardize: bool = True,
) -> None:
    """
    Compute node j given current X (parents already filled due to topological order),
    using explicit bounded predictor variables in nonlinear families and a non-finite guard.
    Updates X[:, j] in-place.
    """
    n = X.shape[0]
    pa = np.where(A[:, j] == 1)[0]

    eps = _noise(n, noise_type, noise_scale, rng, nonlinear=(sem_type == "nonlinear"))

    if len(pa) == 0:
        xj = eps
        X[:, j] = _standardize_safe(xj) if standardize else np.where(np.isfinite(xj), xj, 0.0)
        return

    lin_raw = np.sum(X[:, pa] * W[pa, j], axis=1)

    if sem_type == "linear":
        xj = lin_raw + eps
        if not np.all(np.isfinite(xj)):
            eps2 = _noise(n, noise_type, noise_scale, rng, nonlinear=False)
            xj = lin_raw + eps2
            xj = np.where(np.isfinite(xj), xj, 0.0)

    elif sem_type == "nonlinear":
        spec = nonlinear_spec or NonlinearSpec()  # default family/params
        xj = _nonlinear_eval(lin_raw, eps, spec)

        # non-finite guard: resample eps once, reuse same lin_raw
        if not np.all(np.isfinite(xj)):
            eps2 = _noise(n, noise_type, noise_scale, rng, nonlinear=True)
            xj = _nonlinear_eval(lin_raw, eps2, spec)
            xj = np.where(np.isfinite(xj), xj, 0.0)

    else:
        raise ValueError(sem_type)

    X[:, j] = _standardize_safe(xj) if standardize else np.where(np.isfinite(xj), xj, 0.0)

# def simulate_sem(A: np.ndarray, n: int, sem_type: str, noise_type: str, noise_scale: float,
#                  rng: np.random.Generator) -> np.ndarray:
#     d = A.shape[0]
#     topo = list(nx.topological_sort(nx.DiGraph(A)))
#     X = np.zeros((n, d), dtype=float)

#     if sem_type == "nonlinear":
#         W = rng.uniform(0.2, 0.8, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
#     else: 
#         W = rng.uniform(0.5, 2.0, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))

#     for j in topo:
#         pa = np.where(A[:, j] == 1)[0]
#         eps = _noise(n, noise_type, noise_scale, rng)
#         if len(pa) == 0:
#             X[:, j] = eps
#         else:
#             # lin = sum(W[p, j] * X[:, p] for p in pa)
#             lin = np.sum(X[:, pa] * W[pa, j], axis=1)
#             if sem_type == "linear":
#                 X[:, j] = lin + eps
#             elif sem_type == "nonlinear":
#                 # X[:, j] = np.tanh(lin) + 0.3 * (lin ** 2) + eps              

#                 # --- numeric stabilization ---
#                 # Prevent overflow in lin**2, especially with heavy-tailed noise.
#                 lin = np.clip(lin, -10.0, 10.0)              # adjust bounds if needed
#                 quad = 0.3 * (lin * lin)                     # safe after clip
#                 X[:, j] = np.tanh(lin) + quad + eps
#             else:
#                 raise ValueError(sem_type)
            
#             xj = X[:, j]
#             # Replace any non-finite values (rare but can happen with extreme noise)
#             if not np.all(np.isfinite(xj)):
#                 xj = np.where(np.isfinite(xj), xj, np.nan)
#                 # try a cheap fallback: re-draw eps once and recompute xj
#                 eps2 = _noise(n, noise_type, noise_scale, rng)
#                 if sem_type == "linear":
#                     xj = lin + eps2
#                 else:
#                     xj = np.tanh(lin) + quad + eps2                    
#                 # final guard: force finite
#                 xj = np.where(np.isfinite(xj), xj, 0.0)

#             m = np.nanmean(xj)
#             s = np.nanstd(xj)
#             if np.isfinite(m) and np.isfinite(s) and s > 1e-8:
#                 X[:, j] = (xj - m) / s
#             else:
#                 # fallback: if degenerate or non-finite, just center safely
#                 # X[:, j] = xj - np.nanmean(xj)
#                 mm = np.nanmean(xj)
#                 if not np.isfinite(mm):
#                     X[:, j] = 0.0
#                 else: 
#                     X[:, j] = xj - mm

#     return X

def simulate_sem(
    A: np.ndarray,
    n: int,
    sem_type: str,
    noise_type: str,
    noise_scale: float,
    rng: np.random.Generator,
    *,
    nonlinear_spec: Optional[NonlinearSpec] = None,
    standardize: bool = True,
) -> np.ndarray:
    d = A.shape[0]
    topo = list(nx.topological_sort(nx.DiGraph(A)))
    X = np.zeros((n, d), dtype=float)

    W = _sample_weights(d, sem_type, rng)

    for j in topo:
        _node_update(
            X, W, A, j,
            sem_type=sem_type,
            noise_type=noise_type,
            noise_scale=noise_scale,
            rng=rng,
            nonlinear_spec=nonlinear_spec,
            standardize=standardize,
        )
    return X


def do_intervention_samples(
    A: np.ndarray,
    sem_type: str,
    noise_type: str,
    noise_scale: float,
    treat: int,
    treat_value: float,
    n: int,
    rng: np.random.Generator,
    *,
    nonlinear_spec: Optional[NonlinearSpec] = None,
    standardize: bool = True,
) -> np.ndarray:
    d = A.shape[0]
    topo = list(nx.topological_sort(nx.DiGraph(A)))
    X = np.zeros((n, d), dtype=float)

    W = _sample_weights(d, sem_type, rng)

    for j in topo:
        if j == treat:
            xj = np.full(n, treat_value, dtype=float)
            X[:, j] = _standardize_safe(xj) if standardize else xj
            continue

        _node_update(
            X, W, A, j,
            sem_type=sem_type,
            noise_type=noise_type,
            noise_scale=noise_scale,
            rng=rng,
            nonlinear_spec=nonlinear_spec,
            standardize=standardize,
        )
    return X



# def do_intervention_samples(A: np.ndarray, sem_type: str, noise_type: str, noise_scale: float,
#                             treat: int, treat_value: float, n: int, rng: np.random.Generator) -> np.ndarray:
#     d = A.shape[0]
#     topo = list(nx.topological_sort(nx.DiGraph(A)))
#     X = np.zeros((n, d), dtype=float)
#     # W = rng.uniform(0.5, 2.0, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
#     # Match simulate_sem() weight scales to avoid blow-ups under nonlinear SEMs
#     if sem_type == "nonlinear":
#         W = rng.uniform(0.2, 0.8, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
#     else:
#         W = rng.uniform(0.5, 2.0, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))        
#     for j in topo:
#         if j == treat:
#             X[:, j] = treat_value
#             continue
#         pa = np.where(A[:, j] == 1)[0]
#         eps = _noise(n, noise_type, noise_scale, rng, nonlinear=(sem_type == "nonlinear"))
#         if len(pa) == 0:
#             X[:, j] = eps
#         else:
#             # lin = sum(W[p, j] * X[:, p] for p in pa)
#             # faster + numerically cleaner than Python sum()
#             lin = np.sum(X[:, pa] * W[pa, j], axis=1)
#             if sem_type == "linear":
#                 X[:, j] = lin + eps
#             elif sem_type == "nonlinear":
#                 # X[:, j] = np.tanh(lin) + 0.3 * (lin ** 2) + eps
#                 # --- numeric stabilization (same as simulate_sem) ---
#                 lin = np.clip(lin, -10.0, 10.0)
#                 quad = 0.3 * (lin * lin)
#                 X[:, j] = np.tanh(lin) + quad + eps
#             else:
#                 raise ValueError(sem_type)
            
#             # per-node standardization + non-finite guard
#             xj = X[:, j]
#             if not np.all(np.isfinite(xj)):
#                 xj = np.where(np.isfinite(xj), xj, np.nan)
#                 eps2 = _noise(n, noise_type, noise_scale, rng, nonlinear=(sem_type == "nonlinear"))
#                 if sem_type == "linear":
#                     xj = lin + eps2
#                 else:
#                     quad = 0.3 * (lin * lin)
#                     xj = np.tanh(lin) + quad + eps2
#                 xj = np.where(np.isfinite(xj), xj, 0.0)

#             m = np.nanmean(xj)
#             s = np.nanstd(xj)
#             if np.isfinite(m) and np.isfinite(s) and s > 1e-8:
#                 X[:, j] = (xj - m) / s
#             else:
#                 mm = np.nanmean(xj)
#                 X[:, j] = 0.0 if not np.isfinite(mm) else (xj - mm)            

#     return X

def true_ace_mc(task: TaskSpec, A_true: np.ndarray, treat: int, outcome: int) -> float:
    rng = default_rng(task.seed + 10_000 + 31 * treat + 17 * outcome)
    a = task.intervention_strength
    b = -task.intervention_strength
    Xa = do_intervention_samples(
        A_true, task.sem_type, task.noise_type, task.noise_scale,
        treat, a, task.ace_mc, rng,
        nonlinear_spec=task.nonlinear_spec)
    Xb = do_intervention_samples(
        A_true, task.sem_type, task.noise_type, task.noise_scale,
        treat, b, task.ace_mc, rng, 
        nonlinear_spec=task.nonlinear_spec)
    return float(Xa[:, outcome].mean() - Xb[:, outcome].mean())


def sample_nonlinear_spec(rng: np.random.Generator) -> NonlinearSpec:
    """
    Sample a heterogeneous nonlinear family + parameters.
    Keep it simple and stable; you can widen later.
    """
    family = str(rng.choice([
        "tanh_quad_clip",
        "tanh_quad_bounded",
        "poly3_clip",
        "post_nonlinear",
        "relu_smooth_clip",
    ]))

    # base spec with a sensible clip
    spec = NonlinearSpec(
        family=family,
        clip_lin=float(rng.choice([6.0, 8.0, 10.0])),
        quad_a=float(rng.choice([0.1, 0.2, 0.3])),
        softplus_beta=float(rng.choice([0.5, 1.0, 2.0])),
        post_scale=float(rng.choice([0.7, 1.0, 1.5])),
        post_g=str(rng.choice(["tanh", "sigmoid"])),
        poly_b1=float(rng.choice([0.8, 1.0, 1.2])),
        poly_b2=float(rng.choice([0.05, 0.1, 0.2])),
        poly_b3=float(rng.choice([0.005, 0.01, 0.02])),
    )
    return spec

def sample_task(rng: np.random.Generator) -> TaskSpec:
    d = int(rng.choice([10, 20, 30]))
    n = int(rng.choice([200, 500, 1000, 5000]))
    expected_degree = float(rng.choice([1.0, 2.0, 3.0]))
    sem_type = str(rng.choice(["linear", "nonlinear"]))
    noise_type = str(rng.choice(["gauss", "laplace", "student"]))
    noise_scale = float(rng.choice([0.5, 1.0, 2.0]))
    seed = int(rng.integers(0, 10**9))

    nl_spec = sample_nonlinear_spec(rng) if sem_type == "nonlinear" else None


    return TaskSpec(
        d=d, 
        expected_degree=expected_degree,
        n=n,
        sem_type=sem_type,
        noise_type=noise_type,
        noise_scale=noise_scale,
        seed=seed,
        nonlinear_spec=nl_spec,
    )
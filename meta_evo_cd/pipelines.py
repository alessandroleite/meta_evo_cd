from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from numpy.random import default_rng

from .ci import PCSpec, pc_lite_skeleton
from .orient import orient_vstructures, meek_rules, cpdag_to_dag

@dataclass
class PipelineConfig:
    # skeleton
    skeleton: str          # "pc_lite" or "corr"
    alpha: float
    max_cond_set: int
    corr_thresh: float
    max_edges: int

    # orientation
    orient: str            # "v_meek" or "none"
    meek_passes: int

    # pruning (cheap post filter)
    prune_marginal_alpha: float  # >1 disables

    seed: int

def corr_skeleton(X: np.ndarray, corr_thresh: float, max_edges: int) -> np.ndarray:
    d = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    und = np.zeros((d, d), dtype=int)
    edges = []
    for i in range(d):
        for j in range(i+1, d):
            v = abs(C[i, j])
            if np.isfinite(v) and v >= corr_thresh:
                edges.append((v, i, j))
    edges.sort(reverse=True)
    for _, i, j in edges[:max_edges]:
        und[i, j] = und[j, i] = 1
    return und

def prune_marginal(X: np.ndarray, dag: np.ndarray, alpha: float) -> np.ndarray:
    if alpha > 1.0:
        return dag
    from scipy import stats
    d = dag.shape[0]
    new = dag.copy()
    for i in range(d):
        for j in range(d):
            if new[i, j] == 1:
                _r, p = stats.pearsonr(X[:, i], X[:, j])
                if not np.isfinite(p) or p > alpha:
                    new[i, j] = 0
    return new

def discover_graph(X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    rng = default_rng(cfg.seed)
    if cfg.skeleton == "pc_lite":
        und = pc_lite_skeleton(X, PCSpec(alpha=cfg.alpha, max_cond_set=cfg.max_cond_set, max_edges=cfg.max_edges))
    elif cfg.skeleton == "corr":
        und = corr_skeleton(X, cfg.corr_thresh, cfg.max_edges)
    else:
        raise ValueError(cfg.skeleton)

    if cfg.orient == "none":
        # orient by topological heuristic: i<j (just to get a DAG)
        d = und.shape[0]
        dag = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(i+1, d):
                if und[i, j] == 1:
                    dag[i, j] = 1
    elif cfg.orient == "v_meek":
        A = orient_vstructures(und)
        A = meek_rules(A, max_passes=cfg.meek_passes)
        dag = cpdag_to_dag(A, seed=cfg.seed + 777)
    else:
        raise ValueError(cfg.orient)

    dag = prune_marginal(X, dag, cfg.prune_marginal_alpha)
    return dag.astype(int)

def random_config(rng: np.random.Generator) -> PipelineConfig:
    return PipelineConfig(
        skeleton=str(rng.choice(["pc_lite", "corr"])),
        alpha=float(rng.choice([0.01, 0.05, 0.1])),
        max_cond_set=int(rng.choice([1, 2, 3])),
        corr_thresh=float(rng.choice([0.2, 0.3, 0.4, 0.5])),
        max_edges=int(rng.choice([10, 20, 40, 80])),
        orient=str(rng.choice(["v_meek", "none"])),
        meek_passes=int(rng.choice([2, 5, 10])),
        prune_marginal_alpha=float(rng.choice([0.01, 0.05, 0.1, 1.5])),
        seed=int(rng.integers(0, 10**9)),
    )

def mutate(cfg: PipelineConfig, rng: np.random.Generator, p: float = 0.3) -> PipelineConfig:
    d = asdict(cfg)
    if rng.random() < p: d["skeleton"] = str(rng.choice(["pc_lite", "corr"]))
    if rng.random() < p: d["alpha"] = float(rng.choice([0.01, 0.05, 0.1]))
    if rng.random() < p: d["max_cond_set"] = int(rng.choice([1, 2, 3]))
    if rng.random() < p: d["corr_thresh"] = float(rng.choice([0.2, 0.3, 0.4, 0.5]))
    if rng.random() < p: d["max_edges"] = int(rng.choice([10, 20, 40, 80]))
    if rng.random() < p: d["orient"] = str(rng.choice(["v_meek", "none"]))
    if rng.random() < p: d["meek_passes"] = int(rng.choice([2, 5, 10]))
    if rng.random() < p: d["prune_marginal_alpha"] = float(rng.choice([0.01, 0.05, 0.1, 1.5]))
    d["seed"] = int(rng.integers(0, 10**9))
    return PipelineConfig(**d)

def crossover(a: PipelineConfig, b: PipelineConfig, rng: np.random.Generator) -> tuple[PipelineConfig, PipelineConfig]:
    da, db = asdict(a), asdict(b)
    keys = ["skeleton","alpha","max_cond_set","corr_thresh","max_edges","orient","meek_passes","prune_marginal_alpha"]
    for k in keys:
        if rng.random() < 0.5:
            da[k], db[k] = db[k], da[k]
    da["seed"] = int(rng.integers(0, 10**9))
    db["seed"] = int(rng.integers(0, 10**9))
    return PipelineConfig(**da), PipelineConfig(**db)
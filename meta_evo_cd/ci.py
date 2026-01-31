from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, Set
import itertools
import numpy as np
from scipy import stats

def _regress_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of y after linear regression on X (with intercept)."""
    n = y.shape[0]
    if X.size == 0:
        # return y - y.mean()
        m = np.nanmean(y)
        if not np.isfinite(m):
             # If y is all-nan or inf, return zeros so the test becomes "independent" downstream
            return np.zeros_like(y)
        out = y - m
        # Replace any remaining non-finite values
        out = np.where(np.isfinite(out), out, 0.0)
        return out
    
    X1 = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    # return y - X1 @ beta
    res = y - X1 @ beta
    res = np.where(np.isfinite(res), res, 0.0)
    return res

def partial_corr_test(x: np.ndarray, y: np.ndarray, Z: np.ndarray, alpha: float) -> Tuple[bool, float]:
    """
    H0: x ⟂ y | Z under linear Gaussian; use residual correlation + t-test.
    Returns (independent?, p_value).
    """
    rx = _regress_residual(x, Z)
    ry = _regress_residual(y, Z)
    # If numerically broken, fail safe as "independent" (so PC removes edges less aggressively)
    if not (np.all(np.isfinite(rx)) and np.all(np.isfinite(ry))):
        return True, 1.0
    r, p = stats.pearsonr(rx, ry)
    if not np.isfinite(p):
        p = 1.0
    return (p > alpha), float(p)

@dataclass
class PCSpec:
    alpha: float = 0.05
    max_cond_set: int = 2
    max_edges: Optional[int] = None  # optional hard cap after skeleton

def pc_lite_skeleton(X: np.ndarray, spec: PCSpec) -> np.ndarray:
    """
    PC-lite skeleton discovery:
    - start fully connected undirected graph
    - remove edges if conditional independence found with |S| <= max_cond_set
    """
    n, d = X.shape
    # undirected adjacency
    adj = np.ones((d, d), dtype=int) - np.eye(d, dtype=int)
    sep_sets: Dict[Tuple[int,int], Set[int]] = {}

    # neighbors
    def nbrs(i: int) -> list[int]:
        return [j for j in range(d) if adj[i, j] == 1]

    for l in range(0, spec.max_cond_set + 1):
        changed = True
        while changed:
            changed = False
            for i in range(d):
                Ni = nbrs(i)
                for j in list(Ni):
                    if i == j or adj[i, j] == 0:
                        continue
                    Nij = [k for k in Ni if k != j]
                    if len(Nij) < l:
                        continue
                    # try conditioning sets of size l
                    found_sep = False
                    for S in itertools.combinations(Nij, l):
                        Z = X[:, S] if len(S) else np.empty((n, 0))
                        indep, _p = partial_corr_test(X[:, i], X[:, j], Z, spec.alpha)
                        if indep:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            sep_sets[(i, j)] = set(S)
                            sep_sets[(j, i)] = set(S)
                            changed = True
                            found_sep = True
                            break
                    if found_sep:
                        continue

    # optional edge cap (keep strongest marginal correlations)
    if spec.max_edges is not None:
        C = np.corrcoef(X, rowvar=False)
        edges = []
        for i in range(d):
            for j in range(i+1, d):
                if adj[i, j] == 1:
                    edges.append((abs(C[i, j]), i, j))
        edges.sort(reverse=True)
        keep = set((i, j) for _, i, j in edges[:spec.max_edges])
        for i in range(d):
            for j in range(i+1, d):
                if adj[i, j] == 1 and (i, j) not in keep:
                    adj[i, j] = adj[j, i] = 0

    return adj
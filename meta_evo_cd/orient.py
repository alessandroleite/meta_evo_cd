from __future__ import annotations
import numpy as np
import networkx as nx
from typing import List, Tuple, Set

def _is_adj(und: np.ndarray, i: int, j: int) -> bool:
    return und[i, j] == 1

def orient_vstructures(und: np.ndarray) -> np.ndarray:
    """
    From undirected skeleton, orient v-structures i -> k <- j where i and j not adjacent.
    Returns CPDAG-like directed adjacency with 0/1 edges (may contain both directions for undirected).
    Convention:
      - If undirected i--j, represent as both i->j and j->i set to 1.
      - If directed i->j, set i->j=1 and j->i=0.
    """
    d = und.shape[0]
    A = np.zeros((d, d), dtype=int)

    # start with all undirected edges represented as bidirected
    for i in range(d):
        for j in range(d):
            if und[i, j] == 1:
                A[i, j] = 1

    # v-structures
    for k in range(d):
        nbrs = [i for i in range(d) if und[i, k] == 1]
        for idx in range(len(nbrs)):
            for jdx in range(idx+1, len(nbrs)):
                i = nbrs[idx]
                j = nbrs[jdx]
                if und[i, j] == 0:
                    # orient i -> k and j -> k
                    A[i, k] = 1; A[k, i] = 0
                    A[j, k] = 1; A[k, j] = 0
    return A

def meek_rules(A: np.ndarray, max_passes: int = 10) -> np.ndarray:
    """
    Apply a subset of Meek rules to orient CPDAG edges without creating cycles.
    Uses bidirected encoding for undirected edges.
    """
    d = A.shape[0]

    def is_dir(i, j) -> bool:
        return A[i, j] == 1 and A[j, i] == 0

    def is_und(i, j) -> bool:
        return A[i, j] == 1 and A[j, i] == 1

    def orient(i, j):
        A[i, j] = 1
        A[j, i] = 0

    for _ in range(max_passes):
        changed = False

        # R1: i -> j - k and i not adjacent k => orient j -> k
        for i in range(d):
            for j in range(d):
                if not is_dir(i, j):
                    continue
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if is_und(j, k) and A[i, k] == 0 and A[k, i] == 0:
                        # avoid cycle
                        if not nx.has_path(nx.DiGraph((A == 1) & (A.T == 0)), k, j):
                            orient(j, k)
                            changed = True

        # R2: i - j, i -> k -> j => orient i -> j
        for i in range(d):
            for j in range(d):
                if not is_und(i, j):
                    continue
                for k in range(d):
                    if is_dir(i, k) and is_dir(k, j):
                        if not nx.has_path(nx.DiGraph((A == 1) & (A.T == 0)), j, i):
                            orient(i, j)
                            changed = True
                            break

        # R3 (restricted): i - j, i - k, i - l, k -> j, l -> j, k not adj l => orient i -> j
        for i in range(d):
            for j in range(d):
                if not is_und(i, j):
                    continue
                preds = [k for k in range(d) if is_dir(k, j) and is_und(i, k)]
                for a in range(len(preds)):
                    for b in range(a+1, len(preds)):
                        k, l = preds[a], preds[b]
                        if A[k, l] == 0 and A[l, k] == 0:
                            if not nx.has_path(nx.DiGraph((A == 1) & (A.T == 0)), j, i):
                                orient(i, j)
                                changed = True

        if not changed:
            break

    return A

def cpdag_to_dag(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Turn partially directed graph into a DAG by breaking remaining undirected edges
    randomly but consistently with acyclicity.
    """
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    # directed-only adjacency
    D = ((A == 1) & (A.T == 0)).astype(int)
    # list undirected edges (i<j)
    und_edges = [(i, j) for i in range(d) for j in range(i+1, d) if (A[i, j] == 1 and A[j, i] == 1)]
    rng.shuffle(und_edges)

    G = nx.DiGraph(D)
    for i, j in und_edges:
        # try i->j else j->i
        G1 = G.copy()
        G1.add_edge(i, j)
        if nx.is_directed_acyclic_graph(G1):
            G = G1
            continue
        G2 = G.copy()
        G2.add_edge(j, i)
        if nx.is_directed_acyclic_graph(G2):
            G = G2
            continue
        # if neither works, skip (rare)
    return nx.to_numpy_array(G, dtype=int).astype(int)
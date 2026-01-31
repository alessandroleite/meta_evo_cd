from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class Individual:
    cfg: object
    obj: Tuple[float, float, float, float]  # (shd, ace_err, -stability, time) all minimized
    meta: Dict[str, float]

def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_nondominated_sort(pop: List[Individual]) -> List[List[int]]:
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p].obj, pop[q].obj):
                S[p].add(q)
            elif dominates(pop[q].obj, pop[p].obj):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front: List[Individual], idxs: List[int]) -> Dict[int, float]:
    if len(idxs) == 0:
        return {}
    m = len(front[idxs[0]].obj)
    dist = {i: 0.0 for i in idxs}
    for k in range(m):
        idxs_sorted = sorted(idxs, key=lambda i: front[i].obj[k])
        dist[idxs_sorted[0]] = float("inf")
        dist[idxs_sorted[-1]] = float("inf")
        fmin = front[idxs_sorted[0]].obj[k]
        fmax = front[idxs_sorted[-1]].obj[k]
        if fmax == fmin:
            continue
        for t in range(1, len(idxs_sorted) - 1):
            prevv = front[idxs_sorted[t - 1]].obj[k]
            nextv = front[idxs_sorted[t + 1]].obj[k]
            dist[idxs_sorted[t]] += (nextv - prevv) / (fmax - fmin)
    return dist

def select_nsga2(pop: List[Individual], n_keep: int) -> List[Individual]:
    fronts = fast_nondominated_sort(pop)
    selected: List[Individual] = []
    for f in fronts:
        if len(selected) + len(f) <= n_keep:
            selected.extend([pop[i] for i in f])
        else:
            # fill remainder by crowding distance
            d = crowding_distance(pop, f)
            f_sorted = sorted(f, key=lambda i: d[i], reverse=True)
            rem = n_keep - len(selected)
            selected.extend([pop[i] for i in f_sorted[:rem]])
            break
    return selected
from __future__ import annotations
from dataclasses import replace
from typing import List, Callable, Optional
import numpy as np
from numpy.random import default_rng

from .envs import TaskSpec, sample_task

def sample_train_tasks(rng: np.random.Generator, k: int) -> List[TaskSpec]:
    # mixed distribution (baseline)
    return [sample_task(rng) for _ in range(k)]

def sample_test_tasks_shifted(rng: np.random.Generator, k: int, shift: str) -> List[TaskSpec]:
    """
    Held-out distributions to test meta-generalization.
    shift:
      - "bigger_d": train d<=20, test d=30
      - "heavy_tail": test noise=student more often
      - "nonlinear_only": test sem=nonlinear only
      - "low_n": test small sample only
    """
    tasks = []
    for _ in range(k):
        t = sample_task(rng)
        if shift == "bigger_d":
            t = replace(t, d=30)
        elif shift == "heavy_tail":
            t = replace(t, noise_type="student", noise_scale=float(rng.choice([1.0, 2.0])))
        elif shift == "nonlinear_only":
            t = replace(t, sem_type="nonlinear")
        elif shift == "low_n":
            t = replace(t, n=int(rng.choice([200, 500])))
        else:
            raise ValueError(f"Unknown shift={shift}")
        tasks.append(t)
    return tasks

def make_shift_suite() -> list[str]:
    return ["bigger_d", "heavy_tail", "nonlinear_only", "low_n"]
from __future__ import annotations
import os
from typing import List, Tuple, Optional
import numpy as np

def plot_pareto(points: List[Tuple[float, float]], out_png: str, title: str = "") -> None:
    """
    points: (x,y) pairs. Saves a png.
    Uses matplotlib if installed; otherwise saves a minimal CSV next to it.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.figure()
        plt.scatter(xs, ys)
        plt.xlabel("mean_shd (lower better)")
        plt.ylabel("mean_ace_err (lower better)")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        # fallback: save CSV
        csv_path = out_png.replace(".png", ".csv")
        arr = np.array(points, dtype=float)
        np.savetxt(csv_path, arr, delimiter=",", header="mean_shd,mean_ace_err", comments="")
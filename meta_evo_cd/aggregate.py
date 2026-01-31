from __future__ import annotations
import argparse
import csv
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Performance metrics we always summarize (if present)
METRICS = ["mean_shd", "mean_ace_err", "mean_stability", "mean_time"]

# Task mix columns (new; summarize if present)
TASK_MIX_COLS = [
    "task_linear",
    "task_tanh_quad_clip",
    "task_tanh_quad_bounded",
    "task_poly3_clip",
    "task_post_nonlinear",
    "task_relu_smooth_clip",
    "task_frac_nonlinear",
]

@dataclass
class Row:
    shift: str
    mode: str
    cfg_rank: int

    # metrics (may be nan if missing)
    mean_shd: float
    mean_ace_err: float
    mean_stability: float
    mean_time: float

    # task mix (optional; may be nan if missing)
    task_linear: float = float("nan")
    task_tanh_quad_clip: float = float("nan")
    task_tanh_quad_bounded: float = float("nan")
    task_poly3_clip: float = float("nan")
    task_post_nonlinear: float = float("nan")
    task_relu_smooth_clip: float = float("nan")
    task_frac_nonlinear: float = float("nan")

def _safe_float(x: Optional[str]) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _infer_mode_from_path(p: str) -> str:
    # e.g. runs/ablations/no_pc/meta_test.csv -> mode=no_pc
    parent = os.path.basename(os.path.dirname(p))
    return parent if parent else "run"

def read_meta_test_csvs(root: str) -> List[Row]:
    """
    Looks for:
      - {root}/meta_test.csv
      - {root}/**/meta_test.csv (ablations)
    Mode inferred from directory name unless column 'mode' exists.
    """
    paths: List[str] = []
    if os.path.isfile(root) and root.endswith(".csv"):
        paths = [root]
    else:
        paths = glob.glob(os.path.join(root, "meta_test.csv"))
        paths += glob.glob(os.path.join(root, "**", "meta_test.csv"), recursive=True)
        paths = sorted(set(paths))

    rows: List[Row] = []
    for p in paths:
        inferred_mode = _infer_mode_from_path(p)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "shift" not in r:
                    continue
                mode = str(r.get("mode", inferred_mode))
                rows.append(Row(
                    shift=str(r["shift"]),
                    mode=mode,
                    cfg_rank=int(float(r.get("cfg_rank", 0))),
                    mean_shd=_safe_float(r.get("mean_shd")),
                    mean_ace_err=_safe_float(r.get("mean_ace_err")),
                    mean_stability=_safe_float(r.get("mean_stability")),
                    mean_time=_safe_float(r.get("mean_time")),

                    # task mix (optional)
                    task_linear=_safe_float(r.get("task_linear")),
                    task_tanh_quad_clip=_safe_float(r.get("task_tanh_quad_clip")),
                    task_tanh_quad_bounded=_safe_float(r.get("task_tanh_quad_bounded")),
                    task_poly3_clip=_safe_float(r.get("task_poly3_clip")),
                    task_post_nonlinear=_safe_float(r.get("task_post_nonlinear")),
                    task_relu_smooth_clip=_safe_float(r.get("task_relu_smooth_clip")),
                    task_frac_nonlinear=_safe_float(r.get("task_frac_nonlinear")),
                ))
    return rows

def _stat(vals: List[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

def summarize(rows: List[Row], group_by_cfg_rank: bool) -> List[Dict[str, str]]:
    """
    Returns list of dict rows for CSV printing.
    Groups:
      - (mode, shift) if group_by_cfg_rank=False
      - (mode, shift, cfg_rank) if True
    Also summarizes task mix columns if present.
    """
    groups: Dict[Tuple, List[Row]] = defaultdict(list)
    for r in rows:
        key = (r.mode, r.shift, r.cfg_rank) if group_by_cfg_rank else (r.mode, r.shift)
        groups[key].append(r)

    out: List[Dict[str, str]] = []
    for key, rs in sorted(groups.items(), key=lambda kv: kv[0]):
        if group_by_cfg_rank:
            mode, shift, cfg_rank = key
        else:
            mode, shift = key
            cfg_rank = -1

        shd_m, shd_s = _stat([x.mean_shd for x in rs])
        ace_m, ace_s = _stat([x.mean_ace_err for x in rs])
        stb_m, stb_s = _stat([x.mean_stability for x in rs])
        tim_m, tim_s = _stat([x.mean_time for x in rs])

        row: Dict[str, str] = {
            "mode": str(mode),
            "shift": str(shift),
            "n": str(len(rs)),
            "mean_shd": f"{shd_m:.3f} ± {shd_s:.3f}",
            "mean_ace_err": f"{ace_m:.3f} ± {ace_s:.3f}",
            "mean_stability": f"{stb_m:.3f} ± {stb_s:.3f}",
            "mean_time": f"{tim_m:.4f} ± {tim_s:.4f}",
        }
        if group_by_cfg_rank:
            row["cfg_rank"] = str(cfg_rank)

        # Include task mix summaries only if at least one finite value exists for that column
        for col in TASK_MIX_COLS:
            vals = [getattr(x, col) for x in rs]
            m, s = _stat(vals)
            if np.isfinite(m):
                # task_* counts are integers but stored as float; format compactly
                if col.startswith("task_") and col != "task_frac_nonlinear":
                    row[col] = f"{m:.2f} ± {s:.2f}"
                else:
                    row[col] = f"{m:.3f} ± {s:.3f}"

        out.append(row)
    return out

def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    # union header across rows (since task mix cols may be missing for older runs)
    header = []
    for r in rows:
        for k in r.keys():
            if k not in header:
                header.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Run dir (runs/run0) or ablation root (runs/ablations) or a meta_test.csv path")
    ap.add_argument("--out", type=str, default="runs/summary_meta_test.csv")
    ap.add_argument("--by_cfg_rank", action="store_true", help="Also separate results by cfg_rank")
    args = ap.parse_args()

    rows = read_meta_test_csvs(args.root)
    if not rows:
        raise SystemExit(f"No meta_test.csv files found under {args.root}")

    summ = summarize(rows, group_by_cfg_rank=args.by_cfg_rank)
    write_csv(args.out, summ)

    print(f"Wrote: {args.out}")
    for r in summ[: min(12, len(summ))]:
        print(r)

if __name__ == "__main__":
    main()
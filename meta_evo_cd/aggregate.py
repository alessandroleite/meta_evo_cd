from __future__ import annotations
import argparse
import csv
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

METRICS = ["mean_shd", "mean_ace_err", "mean_stability", "mean_time"]

@dataclass
class Row:
    shift: str
    mode: str
    cfg_rank: int
    mean_shd: float
    mean_ace_err: float
    mean_stability: float
    mean_time: float

def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def read_meta_test_csvs(root: str) -> List[Row]:
    """
    Looks for:
      - {root}/meta_test.csv
      - {root}/**/meta_test.csv (ablations)
    Mode is inferred as:
      - directory name containing meta_test.csv (e.g., runs/ablations/no_pc/meta_test.csv -> mode=no_pc)
      - if root points directly at a run_dir, mode="run"
    """
    paths = []
    if os.path.isfile(root) and root.endswith(".csv"):
        paths = [root]
    else:
        # match meta_test.csv directly and under subdirs
        paths = glob.glob(os.path.join(root, "meta_test.csv"))
        paths += glob.glob(os.path.join(root, "**", "meta_test.csv"), recursive=True)
        # de-dup
        paths = sorted(set(paths))

    rows: List[Row] = []
    for p in paths:
        parent = os.path.basename(os.path.dirname(p))
        mode = parent if parent else "run"
        # If root itself is a run_dir and contains meta_test.csv, parent is run0 etc; keep that.
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "shift" not in r:
                    continue
                rows.append(Row(
                    shift=str(r["shift"]),
                    mode=str(r.get("mode", mode)),
                    cfg_rank=int(float(r.get("cfg_rank", 0))),
                    mean_shd=_safe_float(r.get("mean_shd", "nan")),
                    mean_ace_err=_safe_float(r.get("mean_ace_err", "nan")),
                    mean_stability=_safe_float(r.get("mean_stability", "nan")),
                    mean_time=_safe_float(r.get("mean_time", "nan")),
                ))
    return rows

def summarize(rows: List[Row], group_by_cfg_rank: bool) -> List[Dict[str, str]]:
    """
    Returns list of dict rows for CSV printing.
    Groups:
      - (mode, shift) if group_by_cfg_rank=False
      - (mode, shift, cfg_rank) if True
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

        def stat(vals: List[float]) -> Tuple[float, float]:
            a = np.asarray(vals, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

        shd_m, shd_s = stat([x.mean_shd for x in rs])
        ace_m, ace_s = stat([x.mean_ace_err for x in rs])
        stb_m, stb_s = stat([x.mean_stability for x in rs])
        tim_m, tim_s = stat([x.mean_time for x in rs])

        row = {
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
        out.append(row)
    return out

def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    # stable header
    header = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

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
    # Print a small preview
    for r in summ[: min(12, len(summ))]:
        print(r)

if __name__ == "__main__":
    main()
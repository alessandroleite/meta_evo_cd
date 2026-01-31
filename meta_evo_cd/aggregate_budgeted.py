from __future__ import annotations
import argparse
import json
import os
import csv
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})

def _plot_series(xs: List[int], ys: List[float], out_png: str, xlabel: str, ylabel: str, title: str = "") -> None:
    _ensure_dir(os.path.dirname(out_png) or ".")
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        # fallback CSV
        csv_path = out_png.replace(".png", ".csv")
        arr = np.column_stack([np.asarray(xs, int), np.asarray(ys, float)])
        np.savetxt(csv_path, arr, delimiter=",", header=f"{xlabel},{ylabel}", comments="")

def _score_lex(row: Dict[str, Any]) -> Tuple[float, float, float]:
    # smaller is better: (shd, ace_err, time)
    return (float(row.get("mean_shd", 1e9)),
            float(row.get("mean_ace_err", 1e9)),
            float(row.get("mean_time", 1e9)))

def _score_scalar(row: Dict[str, Any], w_shd=1.0, w_ace=1.0, w_time=0.2, w_stab=0.5) -> float:
    # larger is better (a simple proxy, not true hypervolume)
    shd = float(row.get("mean_shd", 1e9))
    ace = float(row.get("mean_ace_err", 1e9))
    tim = float(row.get("mean_time", 1e9))
    stb = float(row.get("mean_stability", 0.0))
    return -w_shd*shd - w_ace*ace - w_time*np.log1p(tim) + w_stab*stb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., runs/run_budgeted0 or runs/run0")
    ap.add_argument("--jsonl", type=str, default="", help="Override jsonl path. If empty, autodetect.")
    ap.add_argument("--out", type=str, default="", help="Output dir; default <run_dir>/curves")
    ap.add_argument("--mode", type=str, default="lex", choices=["lex", "scalar"],
                    help="How to pick best per gen: lex or scalar proxy.")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out or os.path.join(run_dir, "curves")
    _ensure_dir(out_dir)

    # autodetect
    candidates = []
    if args.jsonl:
        candidates = [args.jsonl]
    else:
        for name in ["train_budgeted.jsonl", "train_log.jsonl", "train_best.jsonl", "train_log.jsonl"]:
            p = os.path.join(run_dir, name)
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            raise SystemExit(f"No training jsonl found in {run_dir}. Expected train_budgeted.jsonl or train_log.jsonl.")

    path = candidates[0]
    rows = _read_jsonl(path)

    # group by gen
    by_gen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "gen" not in r:
            continue
        by_gen[int(r["gen"])].append(r)

    gens = sorted(by_gen.keys())
    if not gens:
        raise SystemExit(f"No 'gen' field found in {path}")

    best_rows = []
    for g in gens:
        cand = by_gen[g]
        if args.mode == "lex":
            best = min(cand, key=_score_lex)
        else:
            best = max(cand, key=_score_scalar)
        best_rows.append({"gen": g, **best})

    # write best-per-gen CSV
    header = ["gen", "mean_shd", "mean_ace_err", "mean_stability", "mean_time", "n_tasks", "budget_s", "rank"]
    # include any config keys if present
    extra_keys = []
    for k in best_rows[0].keys():
        if k not in header and k != "cfg_rank" and k != "shift":
            extra_keys.append(k)
    header = header + [k for k in extra_keys if k not in header]

    out_csv = os.path.join(out_dir, "best_per_gen.csv")
    _write_csv(out_csv, best_rows, header=header)

    # plots
    xs = [int(r["gen"]) for r in best_rows]
    def series(key: str) -> List[float]:
        out = []
        for r in best_rows:
            v = r.get(key, float("nan"))
            try:
                out.append(float(v))
            except Exception:
                out.append(float("nan"))
        return out

    _plot_series(xs, series("mean_shd"), os.path.join(out_dir, "best_shd.png"),
                 xlabel="generation", ylabel="mean_shd", title="Best per gen")
    _plot_series(xs, series("mean_ace_err"), os.path.join(out_dir, "best_ace_err.png"),
                 xlabel="generation", ylabel="mean_ace_err", title="Best per gen")
    _plot_series(xs, series("mean_time"), os.path.join(out_dir, "best_time.png"),
                 xlabel="generation", ylabel="mean_time", title="Best per gen")
    _plot_series(xs, series("mean_stability"), os.path.join(out_dir, "best_stability.png"),
                 xlabel="generation", ylabel="mean_stability", title="Best per gen")

    print(f"Read: {path}")
    print(f"Wrote: {out_csv}")
    print(f"Plots in: {out_dir}")

if __name__ == "__main__":
    main()
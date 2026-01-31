from __future__ import annotations
import argparse
import os
import csv
from typing import Dict, Any, List

from .aggregate import read_meta_test_csvs, summarize, write_csv
from .aggregate_budgeted import main as aggregate_curves_main

def _print_table_preview(csv_path: str, n: int = 12) -> None:
    if not os.path.exists(csv_path):
        print(f"(missing) {csv_path}")
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    print(f"\nPreview: {csv_path}")
    for row in rows[: min(n, len(rows))]:
        print(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations_root", type=str, default="runs/ablations", help="Root containing mode subdirs")
    ap.add_argument("--run_dir", type=str, default="", help="Single run dir to aggregate curves (optional)")
    ap.add_argument("--budgeted_run_dir", type=str, default="", help="Budgeted run dir to aggregate curves (optional)")
    ap.add_argument("--outdir", type=str, default="runs/reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) ablation meta-test summary
    if os.path.exists(args.ablations_root):
        rows = read_meta_test_csvs(args.ablations_root)
        if rows:
            summ = summarize(rows, group_by_cfg_rank=False)
            out = os.path.join(args.outdir, "ablations_meta_test_summary.csv")
            write_csv(out, summ)
            _print_table_preview(out, n=20)
        else:
            print(f"No meta_test.csv found under {args.ablations_root}")
    else:
        print(f"Ablations root not found: {args.ablations_root}")

    # 2) aggregate curves for run_dir
    if args.run_dir:
        import sys
        sys.argv = ["aggregate_budgeted", "--run_dir", args.run_dir, "--mode", "lex"]
        aggregate_curves_main()
        _print_table_preview(os.path.join(args.run_dir, "curves", "best_per_gen.csv"), n=12)

    # 3) aggregate curves for budgeted_run_dir
    if args.budgeted_run_dir:
        import sys
        sys.argv = ["aggregate_budgeted", "--run_dir", args.budgeted_run_dir, "--mode", "lex"]
        aggregate_curves_main()
        _print_table_preview(os.path.join(args.budgeted_run_dir, "curves", "best_per_gen.csv"), n=12)

    print(f"\nReport outputs are under: {args.outdir}")

if __name__ == "__main__":
    main()
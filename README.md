# Meta-Evolution of Causal Discovery Algorithms

This repository implements a **meta-evolutionary framework for causal discovery**, where *entire causal discovery pipelines* (rather than graphs) are evolved and evaluated across heterogeneous causal environments.

The code supports:
- standard meta-evolution experiments,
- systematic ablation studies,
- time-budgeted evolution,
- and fully reproducible aggregation and analysis.

All experiments are orchestrated via a `Makefile`.

## Quickstart (Makefile-based workflow)

This repo supports three main experimental workflows:

- **Main results:** `run.py → meta_test.py → aggregate.py`
- **Ablations:** `ablation.py → aggregate.py`
- **Budgeted evolution:** `run_budgeted.py → aggregate_budgeted.py`

## 1) Environment setup

We recommend using the provided Makefile to manage a local virtual environment.

### Create a virtual environment

```bash
make venv
```

### Install dependencies

The project supports both `pyproject.toml` and `requirements.txt`.

You can control the installation behavior via `INSTALL_MODE`:

| Mode | Behavior |
|---|---|
| `auto` (default) | Use `pyproject.toml` if present, else `requirements.txt` |
| `pyproject` | `pip install -e .` |
| `reqs` | `pip install -r requirements.txt` |
| `both` | Install editable package and requirements |

Examples:

```bash
make install                     # auto
make install INSTALL_MODE=pyproject
make install INSTALL_MODE=reqs
make install INSTALL_MODE=both
```

### Sanity check

```bash
make doctor
```

This prints Python/pip versions and checks that `meta_evo_cd` imports correctly.

---

## 2) Smoke tests (fast sanity checks)

Run minimal configurations to verify everything works:

```bash
make smoke_main
make smoke_ablations
make smoke_budgeted
```

These complete in a few minutes and generate small output directories under `runs/`.

---

## 3) Main experiment workflow (paper results)

This is the primary experimental pipeline.

### Step 1 — Meta-evolution

```bash
make run RUN_DIR=runs/run0 SEED=0 GENERATIONS=10 POP=32 TASK_BATCH=16
```

Outputs:

```text
runs/run0/
 ├── train_log.jsonl
 ├── final_selected.jsonl
```

### Step 2 — Meta-test under distribution shifts

```bash
make meta_test RUN_DIR=runs/run0
```

Outputs:

```text
runs/run0/
 ├── meta_test.csv
 ├── meta_test.jsonl
```

### Step 3 — Aggregate results

```bash
make aggregate_main RUN_DIR=runs/run0 MAIN_SUMMARY=runs/run0_summary.csv
```

Final summary:

```text
runs/run0_summary.csv
```

---

## 4) Ablation studies

Ablations are self-contained: each mode runs training and meta-testing internally.

### Run ablations

```bash
make ablations
```

This produces:

```text
runs/ablations/<mode>/
 ├── train_best.jsonl
 ├── final_selected.jsonl
 ├── meta_test.csv
 └── meta_test.jsonl
```

### Aggregate ablation results

```bash
make aggregate_ablations
```

Final summary:

```text
runs/ablations_summary.csv
```

---

## 5) Budgeted evolution (time-constrained)

This evaluates pipelines under a fixed wall-clock budget per evaluation.

### Run budgeted evolution

```bash
make budgeted BUD_DIR=runs/run_budgeted0 BUDGET_S=20 MIN_TASKS=4 REF_TASKS=12
```

Outputs:

```text
runs/run_budgeted0/
 ├── train_budgeted.jsonl
 ├── pareto_gen*.png
 └── curves/
     ├── best_per_gen.csv
     ├── best_shd.png
     ├── best_ace_err.png
     ├── task_frac_nonlinear.png
```

---

## 6) Cleaning utilities

```bash
make clean_runs        # remove runs/run*
make clean_ablations   # remove runs/ablations
make clean_budgeted    # remove budgeted runs
make clean_venv        # remove .venv
make clean_all         # remove all generated files
```

---

## Notes on reproducibility

- All experiments are fully seeded.
- Task distributions include heterogeneous nonlinear SEM families, logged explicitly.
- CSV outputs are designed for direct inclusion in plots and tables.
- JSONL logs preserve full metadata for post-hoc analysis.

---

## Expected directory structure

```text
meta_evo_cd/
Makefile
README.md
pyproject.toml / requirements.txt
runs/
```

---

## Citation

If you use this codebase, please cite the corresponding paper:

> *Meta-Evolution of Causal Discovery Algorithms*,  
> [authors], [venue], [year].

---

## Questions / extensions

If you are interested in:
- adding new SEM families,
- evolving constraint rules or scoring functions,
- logging per-task evaluation traces,
- or extending the framework to interventional design,

feel free to reach out or open an issue.

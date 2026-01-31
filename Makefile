# Makefile for meta_evo_cd experiments + environment setup
# --------------------------------------------------------
# Usage:
#   make venv
#   make install
#   make main
#   make ablations && make aggregate_ablations
#   make budgeted
#
# By default, this Makefile uses a local virtual environment: .venv
# You can override PY to use your own python: make main PY=python3.11

SHELL := /bin/bash

# ---------
# Python / venv
# ---------
VENV ?= .venv
PY_SYS ?= python3
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

# If you prefer to run without venv, you can do:
#   make main PY=python
# but then install deps yourself.

# Install mode: auto | pyproject | reqs | both
INSTALL_MODE ?= auto

# Detect project files
PYPROJECT := $(wildcard pyproject.toml)
REQS := $(wildcard requirements.txt)

# -------------------------
# Main run (non-budgeted)
# -------------------------
RUN_DIR ?= runs/run0
SEED ?= 0
GENERATIONS ?= 10
POP ?= 32
TASK_BATCH ?= 16

TOP_K ?= 5
TEST_TASKS ?= 24
META_TEST_SEED ?= 123

# -------------------------
# Ablations
# -------------------------
ABL_ROOT ?= runs/ablations
ABL_SEED ?= 7
ABL_GENERATIONS ?= 6
ABL_POP ?= 20
ABL_TASK_BATCH ?= 8
ABL_TEST_TASKS ?= 24
ABL_MODES ?= full,no_pc,no_orient,no_prune,small_k

# -------------------------
# Budgeted run
# -------------------------
BUD_DIR ?= runs/run_budgeted0
BUD_SEED ?= 42
BUD_GENERATIONS ?= 8
BUD_POP ?= 24
BUDGET_S ?= 20
MIN_TASKS ?= 4
REF_TASKS ?= 12

# -------------------------
# Aggregation outputs
# -------------------------
MAIN_SUMMARY ?= runs/run0_summary.csv
ABL_SUMMARY ?= runs/ablations_summary.csv

.PHONY: help dirs \
        venv install reinstall upgrade freeze doctor \
        run meta_test aggregate_main main \
        ablations aggregate_ablations \
        budgeted aggregate_budgeted \
        smoke_main smoke_ablations smoke_budgeted \
        clean_runs clean_ablations clean_budgeted clean_venv clean_all

help:
	@echo ""
	@echo "Environment:"
	@echo "  make venv         # create .venv"	
	@echo "  make install INSTALL_MODE=auto|pyproject|reqs|both"
	@echo "  make install_pyproject | install_reqs | install_both"
	@echo "  make reinstall    # force re-install"
	@echo "  make upgrade      # upgrade pip/setuptools/wheel"
	@echo "  make freeze       # write runs/requirements_freeze.txt"
	@echo "  make doctor       # print python/pip versions + module import check"
	@echo ""
	@echo "Main workflow:"
	@echo "  make main                 # run + meta-test + aggregate for RUN_DIR"
	@echo "  make run                  # only train (run.py)"
	@echo "  make meta_test            # only meta-test selected configs"
	@echo "  make aggregate_main        # aggregate meta_test.csv under RUN_DIR"
	@echo ""
	@echo "Ablations:"
	@echo "  make ablations            # run ablations (includes per-mode meta-test)"
	@echo "  make aggregate_ablations   # aggregate meta_test.csv under ABL_ROOT"
	@echo ""
	@echo "Budgeted:"
	@echo "  make budgeted             # budgeted evolution + curve aggregation"
	@echo "  make aggregate_budgeted    # only curve aggregation for BUD_DIR"
	@echo ""
	@echo "Smoke tests:"
	@echo "  make smoke_main"
	@echo "  make smoke_ablations"
	@echo "  make smoke_budgeted"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean_runs | clean_ablations | clean_budgeted | clean_venv | clean_all"
	@echo ""
	@echo "Common overrides:"
	@echo "  make main RUN_DIR=runs/run1 SEED=1 GENERATIONS=5 POP=20 TASK_BATCH=8"
	@echo "  make ablations ABL_MODES=full,no_pc ABL_GENERATIONS=3 ABL_POP=10"
	@echo "  make budgeted BUD_DIR=runs/bud1 BUDGET_S=10 MIN_TASKS=3"
	@echo ""

dirs:
	@mkdir -p runs
	@mkdir -p $(RUN_DIR)
	@mkdir -p $(ABL_ROOT)
	@mkdir -p $(BUD_DIR)

# -------------------------
# Environment targets
# -------------------------
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating venv at $(VENV) using $(PY_SYS)"; \
		$(PY_SYS) -m venv $(VENV); \
	else \
		echo "Venv already exists at $(VENV)"; \
	fi

upgrade: venv
	@echo "Upgrading pip/setuptools/wheel..."
	@$(PIP) install --upgrade pip setuptools wheel


install_pyproject: upgrade
	@if [ -z "$(PYPROJECT)" ]; then \
		echo "ERROR: pyproject.toml not found."; \
		exit 1; \
	fi
	@echo "Installing via pyproject.toml (editable): pip install -e ."
	@$(PIP) install -e .

install_reqs: upgrade
	@if [ -z "$(REQS)" ]; then \
		echo "ERROR: requirements.txt not found."; \
		exit 1; \
	fi
	@echo "Installing via requirements.txt: pip install -r requirements.txt"
	@$(PIP) install -r requirements.txt

install_both: upgrade
	@echo "Installing both pyproject.toml (editable) and requirements.txt (if present)..."
	@if [ -n "$(PYPROJECT)" ]; then \
		echo " - pyproject.toml found -> pip install -e ."; \
		$(PIP) install -e .; \
	else \
		echo " - pyproject.toml not found -> skipping"; \
	fi
	@if [ -n "$(REQS)" ]; then \
		echo " - requirements.txt found -> pip install -r requirements.txt"; \
		$(PIP) install -r requirements.txt; \
	else \
		echo " - requirements.txt not found -> skipping"; \
	fi

install: upgrade
	@echo "INSTALL_MODE=$(INSTALL_MODE)"
	@if [ "$(INSTALL_MODE)" = "pyproject" ]; then \
		$(MAKE) install_pyproject; \
	elif [ "$(INSTALL_MODE)" = "reqs" ]; then \
		$(MAKE) install_reqs; \
	elif [ "$(INSTALL_MODE)" = "both" ]; then \
		$(MAKE) install_both; \
	elif [ "$(INSTALL_MODE)" = "auto" ]; then \
		if [ -n "$(PYPROJECT)" ]; then \
			$(MAKE) install_pyproject; \
		elif [ -n "$(REQS)" ]; then \
			$(MAKE) install_reqs; \
		else \
			echo "ERROR: Neither pyproject.toml nor requirements.txt found."; \
			exit 1; \
		fi; \
	else \
		echo "ERROR: Unknown INSTALL_MODE=$(INSTALL_MODE). Use auto|pyproject|reqs|both."; \
		exit 1; \
	fi

reinstall: upgrade
	@echo "Reinstalling (force) with INSTALL_MODE=$(INSTALL_MODE)..."
	@if [ "$(INSTALL_MODE)" = "pyproject" ]; then \
		$(PIP) install -e . --force-reinstall; \
	elif [ "$(INSTALL_MODE)" = "reqs" ]; then \
		$(PIP) install -r requirements.txt --force-reinstall; \
	elif [ "$(INSTALL_MODE)" = "both" ]; then \
		if [ -n "$(PYPROJECT)" ]; then $(PIP) install -e . --force-reinstall; fi; \
		if [ -n "$(REQS)" ]; then $(PIP) install -r requirements.txt --force-reinstall; fi; \
	else \
		# auto
		if [ -n "$(PYPROJECT)" ]; then $(PIP) install -e . --force-reinstall; \
		elif [ -n "$(REQS)" ]; then $(PIP) install -r requirements.txt --force-reinstall; \
		else echo "ERROR: Neither pyproject.toml nor requirements.txt found."; exit 1; fi; \
	fi

freeze: venv
	@mkdir -p runs
	@echo "Freezing environment to runs/requirements_freeze.txt"
	@$(PIP) freeze > runs/requirements_freeze.txt
	@echo "Wrote runs/requirements_freeze.txt"

doctor: venv
	@echo "Python: $$($(PY) --version)"
	@echo "Pip:    $$($(PIP) --version)"
	@echo "Import check:"
	@$(PY) -c "import meta_evo_cd; print('meta_evo_cd import: OK')"

# -------------------------
# Main workflow
# -------------------------
run: dirs
	$(PY) -m meta_evo_cd.run \
	  --outdir $(RUN_DIR) \
	  --generations $(GENERATIONS) \
	  --pop $(POP) \
	  --task_batch $(TASK_BATCH) \
	  --seed $(SEED)

meta_test: dirs
	$(PY) -m meta_evo_cd.meta_test \
	  --run_dir $(RUN_DIR) \
	  --top_k $(TOP_K) \
	  --test_tasks $(TEST_TASKS) \
	  --seed $(META_TEST_SEED)

aggregate_main: dirs
	$(PY) -m meta_evo_cd.aggregate \
	  --root $(RUN_DIR) \
	  --out $(MAIN_SUMMARY)

main: run meta_test aggregate_main
	@echo ""
	@echo "Done: main workflow"
	@echo "  Run dir:     $(RUN_DIR)"
	@echo "  Meta-test:   $(RUN_DIR)/meta_test.csv"
	@echo "  Summary:     $(MAIN_SUMMARY)"
	@echo ""

# -------------------------
# Ablation workflow
# -------------------------
ablations: dirs
	$(PY) -m meta_evo_cd.ablation \
	  --outroot $(ABL_ROOT) \
	  --modes $(ABL_MODES) \
	  --generations $(ABL_GENERATIONS) \
	  --pop $(ABL_POP) \
	  --task_batch $(ABL_TASK_BATCH) \
	  --seed $(ABL_SEED) \
	  --test_tasks $(ABL_TEST_TASKS)

aggregate_ablations: dirs
	$(PY) -m meta_evo_cd.aggregate \
	  --root $(ABL_ROOT) \
	  --out $(ABL_SUMMARY)

# -------------------------
# Budgeted workflow
# -------------------------
budgeted: dirs
	$(PY) -m meta_evo_cd.run_budgeted \
	  --outdir $(BUD_DIR) \
	  --generations $(BUD_GENERATIONS) \
	  --pop $(BUD_POP) \
	  --seed $(BUD_SEED) \
	  --budget_s $(BUDGET_S) \
	  --min_tasks $(MIN_TASKS) \
	  --ref_tasks $(REF_TASKS)
	$(MAKE) aggregate_budgeted

aggregate_budgeted: dirs
	$(PY) -m meta_evo_cd.aggregate_budgeted \
	  --run_dir $(BUD_DIR) \
	  --plot_task_mix

# -------------------------
# Smoke tests (fast sanity)
# -------------------------
smoke_main:
	$(MAKE) main RUN_DIR=runs/smoke_run SEED=0 GENERATIONS=2 POP=8 TASK_BATCH=4 TOP_K=3 TEST_TASKS=8 META_TEST_SEED=999 MAIN_SUMMARY=runs/smoke_run_summary.csv

smoke_ablations:
	$(MAKE) ablations ABL_ROOT=runs/smoke_ablations ABL_MODES=full,no_pc ABL_GENERATIONS=2 ABL_POP=8 ABL_TASK_BATCH=4 ABL_TEST_TASKS=8 ABL_SEED=999
	$(MAKE) aggregate_ablations ABL_ROOT=runs/smoke_ablations ABL_SUMMARY=runs/smoke_ablations_summary.csv

smoke_budgeted:
	$(MAKE) budgeted BUD_DIR=runs/smoke_budgeted BUD_SEED=0 BUD_GENERATIONS=2 BUD_POP=10 BUDGET_S=5 MIN_TASKS=2 REF_TASKS=8

# -------------------------
# Cleaning
# -------------------------
clean_runs:
	@rm -rf runs/run* runs/smoke_run*

clean_ablations:
	@rm -rf $(ABL_ROOT) runs/smoke_ablations

clean_budgeted:
	@rm -rf runs/run_budgeted* runs/smoke_budgeted

clean_venv:
	@rm -rf $(VENV)

clean_all:
	@rm -rf runs
	@rm -rf $(VENV)
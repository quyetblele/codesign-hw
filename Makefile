PYTHON ?= .venv/bin/python
POLICY ?= round-robin
JOBS ?= 1000
WINDOW ?= 5
UPDATE ?= 1
RUNS ?= 100
REPEATS ?= 3
DEVICE ?= both
RUN ?=
ARGS ?=
export PYTHONPATH := $(CURDIR)/src
export RUN

.PHONY: run profile report test help

run:
	@$(PYTHON) -m edge_scheduler.main --live --scheduler $(POLICY) --jobs $(JOBS) --window $(WINDOW) --update $(UPDATE) $(ARGS)

profile:
	@$(PYTHON) -m edge_scheduler.inference.profiling --device $(DEVICE) --runs $(RUNS) --repeats $(REPEATS) $(ARGS)

report:
	@$(PYTHON) -m edge_scheduler.reporting $(ARGS)

test:
	@$(PYTHON) -m unittest discover -s tests -v

help:
	@echo 'make run [POLICY=round-robin JOBS=1000 WINDOW=5 UPDATE=1]'
	@echo 'make profile [DEVICE=both RUNS=100 REPEATS=3]'
	@echo 'make report [RUN=logs/pipeline/<run_id>]'
	@echo 'Optional CLI arguments: ARGS="--warmup 1"'

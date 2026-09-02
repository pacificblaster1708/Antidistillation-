# Convenience targets. Everything here also works as a plain command -- see the
# README if you would rather not use make.

PYTHON ?= python3

.PHONY: help install install-flash fixtures test test-static test-kl test-e2e test-ddp test-learning verify lint clean

help:
	@echo "make install        install Python dependencies"
	@echo "make install-flash  additionally install flash-attention (needs a GPU + CUDA toolkit)"
	@echo "make verify         build fixtures and run the fast test layers"
	@echo "make test           same as verify, plus the slower learning test"
	@echo "make test-static    dependency-free source and Slurm contract check"
	@echo "make test-kl        numeric checks on the KL only"
	@echo "make test-e2e       every mode / flag / error path"
	@echo "make test-ddp       2-process distributed check"
	@echo "make lint           pyflakes over the source"
	@echo "make clean          remove test fixtures, temp files and __pycache__"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-flash:
	$(PYTHON) -m pip install flash-attn==2.7.4.post1 --no-build-isolation

fixtures:
	$(PYTHON) tests/make_fixtures.py

test-static:
	$(PYTHON) tests/test_static_contract.py

test-kl:
	$(PYTHON) tests/test_kl.py

test-e2e: fixtures
	bash tests/test_end_to_end.sh

test-ddp: fixtures
	bash tests/test_distributed.sh

test-learning: fixtures
	$(PYTHON) tests/test_learning.py

verify:
	bash verify.sh

test:
	bash verify.sh --full

lint:
	$(PYTHON) -m pyflakes soft_distill.py scripts/*.py tests/*.py || true

clean:
	rm -rf tests/fixtures tests/_tmp
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

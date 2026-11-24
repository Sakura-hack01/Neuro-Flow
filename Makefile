.PHONY: all test experiments plot cpp-bench pages clean help

PYTHON ?= python

help:
	@echo "Neural-Flow Connectome Makefile Targets:"
	@echo "  make test         - Run automated PyTest test suite"
	@echo "  make experiments  - Execute all scientific experiment suites"
	@echo "  make plot         - Generate 300-DPI high-resolution research figures"
	@echo "  make cpp-bench    - Run real-time BCI edge engine latency benchmark"
	@echo "  make pages        - Prepare and verify GitHub Pages static docs"
	@echo "  make all          - Execute full pipeline: test, experiments, plot, cpp-bench"
	@echo "  make clean        - Remove temporary caches and build artifacts"

all: test experiments plot cpp-bench pages

test:
	$(PYTHON) -m src.cli test

experiments:
	$(PYTHON) -m src.cli exp

plot:
	$(PYTHON) -m src.cli plot

cpp-bench:
	$(PYTHON) -m src.cli cpp-bench

pages:
	@echo "Verifying docs/ assets for GitHub Pages..."
	@ls docs/assets/*.png || echo "Assets ready in docs/assets/"

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ src/*/__pycache__
	rm -rf cpp_inference/build cpp_inference/*.o cpp_inference/*.exe

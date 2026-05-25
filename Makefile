# PRAXIS - AI Clinical Documentation for NHS General Practice
# Satyawan Singh, MSc AI for Business Intelligence, University of Leicester

# Python interpreter for test, serve, and benchmark targets. Defaults to the
# project venv so a stale system python3 does not silently miss pytest/uvicorn.
# Override on the command line if needed: `make test PYTHON=python3`.
PYTHON ?= ./venv/bin/python

.PHONY: install install-minimal install-clinical test demo serve website \
        benchmark-asr benchmark-der benchmark-soap benchmark-all eval-tables \
        sample clean-test-venv

# ── Setup ──

install:
	python3 -m pip install -r requirements.txt

install-minimal:
	python3 -m pip install -r requirements-minimal.txt

install-clinical:
	python3 -m pip install -r optional-requirements.txt
	@echo "Clinical NLP stack installed. scispaCy model (separate):"
	@echo "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"

# ── Testing ──

test:
	$(PYTHON) -m pytest tests/ -q

# ── Demo ──

sample:
	python3 scripts/generate_sample.py

demo:
	@if [ ! -f data/samples/sample_consultation.wav ]; then \
		echo "No sample audio found. Generating..."; \
		python3 scripts/generate_sample.py; \
	fi
	python3 scripts/run_full_pipeline.py data/samples/sample_consultation.wav --offline

demo-full:
	@if [ ! -f data/samples/sample_consultation.wav ]; then \
		echo "No sample audio found. Generating..."; \
		python3 scripts/generate_sample.py; \
	fi
	python3 scripts/run_full_pipeline.py data/samples/sample_consultation.wav

# ── Benchmarks ──

benchmark-asr:
	$(PYTHON) -m src.evaluation.wer_benchmark --model praxis-conformer

benchmark-der:
	$(PYTHON) -m src.evaluation.der_benchmark --model praxis-ecapa

benchmark-soap:
	@echo "SOAP benchmark requires pipeline run outputs in data/pipeline_runs/"
	$(PYTHON) -m src.evaluation.soap_quality --notes-dir data/pipeline_runs/

benchmark-all: benchmark-asr benchmark-der benchmark-soap eval-tables
	@echo "All benchmarks complete. Results in data/evaluation_results/"

eval-tables:
	$(PYTHON) -m src.evaluation.generate_tables

# ── API Server ──

serve:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

website:
	@echo "Starting API server and opening website..."
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
	@sleep 2
	open http://localhost:8000/site/

# ── Cleanup ──

clean-test-venv:
	rm -rf test_venv

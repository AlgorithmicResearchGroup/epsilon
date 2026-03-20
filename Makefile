PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: venv install test-core docker-build publish-ghcr bench-report bench-scout clean

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install pytest-mock

test-core:
	PYTHONPATH=. $(BIN)/pytest -q \
		tests/test_bash_tool.py \
		tests/test_sql_tool.py \
		tests/test_call_llm_tool.py \
		tests/test_litellm_client.py \
		tests/test_byoa_sdk.py \
		tests/test_byoa_function_runner.py \
		tests/test_byoa_runner.py \
		tests/test_broker_reliability.py \
		tests/test_scale_topologies.py \
		tests/test_hf_entity_graph_demo.py \
		tests/test_benchmark_scout_demo.py \
		tests/test_benchmark_report_demo.py

docker-build:
	docker build -t epsilon:local .

publish-ghcr:
	scripts/publish_ghcr.sh

bench-report:
	PYTHONPATH=. $(BIN)/python examples/benchmark_report/run_demo.py $(ARGS)

bench-scout:
	PYTHONPATH=. $(BIN)/python examples/benchmark_scout/run_demo.py $(ARGS)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# ---------- cross-platform venv helpers ----------
ifeq ($(OS),Windows_NT)
	VENV_PY = .venv/Scripts/python.exe
else
	VENV_PY = .venv/bin/python
endif

.PHONY: env lint test lint\ test compose clean

env:
	python -m venv .venv
	$(VENV_PY) -m pip install -U pip wheel
	$(VENV_PY) -m pip install -r requirements.txt

lint:
	$(VENV_PY) -m ruff .
	$(VENV_PY) -m black --check --line-length 100 .

test:
	$(VENV_PY) -m pytest -q

lint\ test: lint test

compose:
	docker compose up -d airflow duckdb neo4j

clean:
	docker compose down
	@if exist .venv rmdir /s /q .venv 
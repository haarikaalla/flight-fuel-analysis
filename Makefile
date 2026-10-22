PYTHON ?= python3
VENV ?= .venv
RUN = $(VENV)/bin/python

.PHONY: setup test advanced stream serve clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(RUN) -m pip install --upgrade pip
	$(RUN) -m pip install -e .

test:
	$(RUN) -m unittest discover -v -s tests

advanced:
	$(RUN) -m flight_fuel.advanced

stream:
	$(RUN) -m flight_fuel.streaming --limit 20 --withhold-labels

serve: advanced
	$(RUN) -m http.server 8000

clean:
	find src tests -name '__pycache__' -type d -prune -exec rm -rf {} +

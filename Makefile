VENV = venv
PYTHON = $(VENV)/bin/python
ACTIVATE = . $(VENV)/bin/activate

.PHONY: docs examples

docs:
	mkdocs build

test:
	pytest tests/unit/
	pytest tests/issues/
	pytest --nbval tests/notebooks/
	ydata_profiling -h

test_spark:
	pytest tests/backends/spark_backend/
	ydata_profiling -h

test_cov:
	pytest --cov=. tests/unit/
	pytest --cov=. --cov-append tests/issues/
	pytest --cov=. --cov-append --nbval tests/notebooks/
	ydata_profiling -h

examples:
	find ./examples -maxdepth 2 -type f -name "*.py" -execdir python {} \;

package:
	rm -rf build dist
	echo "$(version)" > VERSION
	python setup.py sdist bdist_wheel
	twine check dist/*

install_dev_pandas:
	rm -rf $(VENV)
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

install_dev_spark:
	rm -rf $(VENV)
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-spark.txt
	$(PYTHON) -m pip install -e .

install_pandas:
	pip install -e ".[pandas, test, notebook]"

install_spark:
	pip install -e ".[spark, test, notebook]"

install-docs: install ### Installs regular and docs dependencies
	pip install -r requirements-docs.txt

# compile requirements files
compile:
	$(PYTHON) -m piptools compile -o requirements.txt pyproject.toml
	$(PYTHON) -m piptools compile --extra dev --extra test --extra pandas -o requirements-dev.txt pyproject.toml
	$(PYTHON) -m piptools compile --extra dev --extra test --extra spark -o requirements-spark.txt pyproject.toml

publish-docs: examples ### Publishes the documentation
	mkdir docs/examples
	rsync -R examples/*/*.html docs
	mike deploy --push --update-aliases $(version) latest

lint:
	pre-commit run --all-files

clean:
	git rm --cached `git ls-files -i --exclude-from=.gitignore`

all:
	make lint
	make install
	make examples
	make docs
	make test

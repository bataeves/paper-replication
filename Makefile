install-local:
	pip install -U "poetry<=1.8.0"
    poetry install
    pre-commit install

lock:
	poetry lock

format:
	pre-commit run -a

test:
	python -m pytest tests

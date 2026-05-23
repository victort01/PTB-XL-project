.PHONY: setup test lint clean-artifacts

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest tests

lint:
	python -m ruff check src tests

clean-artifacts:
	python -c "from pathlib import Path; roots=[Path('reports/figures'),Path('reports/tables'),Path('reports/model_cards'),Path('models')]; [p.unlink() for r in roots for p in r.glob('*') if p.name != '.gitkeep' and p.is_file()]"

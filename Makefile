.PHONY: setup setup-tcc2 test lint smoke-tcc2 audit-external fetch-external prepare-features-tcc2 train-tcc2 freeze-tcc2 evaluate-test-tcc2 clean-artifacts

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

setup-tcc2:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e ".[dev,tcc2]"

test:
	python -m pytest tests

lint:
	python -m ruff check src tests

smoke-tcc2:
	python scripts/train_tcc2.py smoke --config configs/tcc2_multilabel.yaml

audit-external:
	python scripts/audit_external_dataset.py --config configs/tcc2_multilabel.yaml

fetch-external:
	python scripts/fetch_external_repositories.py --config configs/external_repositories.yaml

prepare-features-tcc2:
	python scripts/prepare_tcc2_features.py --config configs/tcc2_multilabel.yaml

MODEL ?= tcn
train-tcc2:
	python scripts/train_tcc2.py train --config configs/tcc2_multilabel.yaml --model $(MODEL)

CANDIDATE ?=
freeze-tcc2:
	python scripts/train_tcc2.py freeze --config configs/tcc2_multilabel.yaml --candidate-manifest $(CANDIDATE)

FROZEN ?=
evaluate-test-tcc2:
	python scripts/train_tcc2.py evaluate-test --config configs/tcc2_multilabel.yaml --frozen-manifest $(FROZEN)

clean-artifacts:
	python -c "from pathlib import Path; roots=[Path('reports/figures'),Path('reports/tables'),Path('reports/model_cards'),Path('models')]; [p.unlink() for r in roots for p in r.glob('*') if p.name != '.gitkeep' and p.is_file()]"

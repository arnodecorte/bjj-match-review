.PHONY: install dev train download-data lint

install:
	pip install -r requirements.txt

dev:
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

download-data:
	bash scripts/download_vicos.sh

train:
	python -m classifier.train --data-dir data/vicos --epochs 50 --output models/classifier.pt

lint:
	python -m py_compile classifier/labels.py classifier/model.py classifier/preprocess.py \
	    classifier/inference.py classifier/train.py data/vicos.py api/server.py
	@echo "Syntax OK"

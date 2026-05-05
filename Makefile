.PHONY: install dev train train-quick evaluate download-data lint

install:
	pip install -r requirements.txt

dev:
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

download-data:
	bash scripts/download_vicos.sh

train:
	python -m classifier.train --data-dir data/vicos --epochs 50 --output models/classifier.pt

train-quick:
	python -m classifier.train --data-dir data/vicos --epochs 10 --max-samples 20000 --batch-size 512 --device cuda --output models/classifier.quick.pt

evaluate:
	python scripts/evaluate_phase0.py --video sample_match.mp4 --model models/classifier.pt --fps 2.0 --output reports/phase0_eval.json

lint:
	python -m py_compile classifier/labels.py classifier/model.py classifier/preprocess.py \
	    classifier/inference.py classifier/train.py data/vicos.py api/server.py
	@echo "Syntax OK"

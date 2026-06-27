.PHONY: install test lint data eval-smoke

install:
	pip install -e .

test:
	pytest -q

lint:
	ruff check .

data:
	python scripts/00_generate_countdown_dataset.py --out_dir data/countdown --train 5000 --valid 500 --test 500 --ood 500 --seed 42

eval-smoke:
	python scripts/03_eval.py --model_name Qwen/Qwen2.5-0.5B-Instruct --data_path data/countdown/test_in_dist.jsonl --limit 8 --output_dir outputs/eval_smoke

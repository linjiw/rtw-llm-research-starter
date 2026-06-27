RUN_DIR ?= outputs/grpo_rtw_cuda_smoke_50

.PHONY: install test lint data eval-smoke m4-check eval-m4-smoke check-run-health

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

m4-check:
	python -c 'import torch; mps = getattr(torch.backends, "mps", None); print("cuda_available={}".format(torch.cuda.is_available())); print("mps_available={}".format(bool(mps and mps.is_available()))); print("mps_built={}".format(bool(mps and mps.is_built()))); print("torch_version={}".format(torch.__version__))'

eval-m4-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/03_eval.py --model_name Qwen/Qwen2.5-0.5B-Instruct --engine hf --device mps --data_path data/countdown/test_in_dist.jsonl --output_dir outputs/eval_m4_base_smoke --limit 16 --batch_size 1 --max_new_tokens 64

check-run-health:
	python scripts/05_check_run_health.py --run_dir $(RUN_DIR)

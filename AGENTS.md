# Agent instructions for this repository

This repo is designed to be agent-legible. Preserve the following invariants:

1. The verifier in `src/rtw_llm/countdown.py` is the source of truth for task correctness.
2. Never count an output as correct unless it passes the verifier.
3. Keep reward components separately logged. Do not only log total reward.
4. When adding a new task, include a deterministic generator, verifier, dataset card, and tests.
5. When changing prompts, keep `prompt_low`, `prompt_mid`, and `prompt_high` fields so harness-shift experiments remain possible.
6. The primary reward must remain final task success. Auxiliary rewards are training wheels, not the objective.

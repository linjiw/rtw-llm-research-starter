# Auto-Research Program: Curriculum Learning for LLM Agentic-Task RL Tuning

Created: 2026-07-08. This is the operating program for autonomous research
iterations in this repo, adapted from the karpathy/autoresearch pattern
(fixed experiment budget → one objective metric → keep-or-discard → repeat).
An agent (Claude Code) executes iterations of the loop below; a human reviews
the ledger and this program between sessions and improves the program itself.

## 1. North-star goal

> Show that adaptive curricula — RTW-style reward curricula and GACL-style task
> curricula — make RL post-training of LLMs on verifier-based (agentic) tasks
> more sample-efficient, less reward-hacking-prone, and more robust to harness
> and distribution shift than fixed/manual/random schedules.

Paper 1 (current): the Countdown verifier harness, Stable-RTW + verifier-guided
best-of-N (v0.9B is the strongest archived result). Paper 2 (later): the same
teacher pattern on coding tasks. See `docs/PROJECT_DESIGN.md` for the full
roadmap and `docs/CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md` for the
current paper-safe claim.

## 2. The objective metric

Every experiment is scored on a fixed, comparable protocol. The analogue of
autoresearch's `val_bpb` here is:

**Primary:** `reranked_exact@8` on the frozen 50-task validation set
(`outputs/v09_task_ids_validation_limit50.txt`), practical selector, paired
per-task against the current best method.

**Guardrails (a win on primary does not count if these regress):**

- `reranked_exact@8` on frozen `test_in_dist` task IDs (no meaningful drop);
- `selected_valid` and `selected_number_f1` (legality must not collapse);
- `reward_hacking_candidate` rate (must not rise);
- generated tokens + wall-clock (cost per exact must be reported);
- `oracle_exact@8` (did the method create exact candidates, or only rank them?).

**Decision rule:** a variant is KEPT if the paired validation advantage at N=8
is positive with discordant-pair evidence (McNemar-style counts; with one seed,
Stable-only vs baseline-only counts must favor it clearly), and no guardrail
regresses. Otherwise DISCARDED — record why in the ledger and move on. Multi-seed
(0/1/2) expansion is only spent on variants that survive seed 0.

## 3. Fixed experiment budget (do not vary between candidates)

```text
model:            Qwen/Qwen2.5-0.5B-Instruct
training:         GRPO + LoRA, 300 steps, batch 2, grad_accum 8, 4 generations,
                  max_prompt 768, max_completion 256, seed 0 first
eval harness:     scripts/07_best_of_n_rerank.py, N=1,4,8 (max_n 8),
                  temperature 0.7, top_p 0.95, sampling seed 0,
                  frozen task IDs per split, limit 50
verifier:         src/rtw_llm/countdown.py (source of truth, never bypass)
selector:         practical_score (never uses exact correctness as input)
```

## 4. Frozen vs mutable (the `prepare.py` / `train.py` split)

**Frozen — do not modify during autonomous iterations:**

- the verifier (`src/rtw_llm/countdown.py` verification semantics);
- frozen task-ID files and committed datasets under `data/countdown/`;
- the practical selector in `scripts/07_best_of_n_rerank.py`;
- eval/sampling configuration above;
- reward-component logging (components must stay separately logged).

**Mutable — this is where research happens:**

- `src/rtw_llm/teacher.py`: new teacher strategies (curriculum controllers);
- task *sampling* during training (GACL-style difficulty distribution, grounding
  probability ε) — the training-side curriculum, not the eval sets;
- training hyperparameters, within the fixed budget above;
- SFT warmup usage;
- analysis/diagnostic scripts (additive only).

## 5. The loop (one iteration)

```text
1. PICK    take the top item from the queue (section 6)
2. COMMIT  commit the code state before any CUDA run (archival invariant)
3. TRAIN   run the 300-step GRPO variant under the fixed budget
4. HEALTH  scripts/05_check_run_health.py — abort iteration if unhealthy
5. EVAL    best-of-N on frozen validation + test_in_dist task IDs, N=1,4,8
6. SCORE   paired comparison vs current best; check guardrails
7. RECORD  append a row to docs/EXPERIMENT_LEDGER.md (keep or discard + why),
           update docs status if the best method changed, commit
8. UPDATE  reorder/extend the queue based on what was learned; if evidence
           contradicts a queue hypothesis, rewrite it rather than running it
```

Rules of engagement for the agent:

- one variable per iteration; no drive-by changes to frozen components;
- never count an output correct unless the strict verifier passes;
- a null/negative result is a valid, ledger-worthy outcome — record and proceed;
- if two consecutive iterations of a theme fail (e.g., teacher tweaks), stop
  that theme and move to the next queue block (this rule killed v0.7/v0.8-era
  teacher tweaking in favor of best-of-N, correctly);
- keep runs sequential — one A10G; long jobs run under nohup with logs in the
  run directory.

## 6. Prioritized queue (as of 2026-07-08)

### Gate 0 — local baseline ladder (REQUIRED before any new method)

Reproduce locally what the archived v0.9B numbers rest on, per
`docs/BASELINE_INVENTORY_AND_NEXT_EXPERIMENTS_20260708.md`:

```text
base model      best-of-N @ N=1,4,8   (base N=1 already collected)
static  seed0   300-step train → best-of-N
stable  seed0   300-step train → best-of-N
```

Runner: `scripts/run_gate0_baseline_ladder.sh`. Success = local stable-vs-static
ordering is directionally consistent with archived v0.9B on validation. This
gives every future variant a same-machine ladder: base → static → Stable-RTW →
candidate.

### v0.10 — GACL-style task curriculum (the main new-method bet)

This is the direct "curriculum learning in agentic-task RL tuning" experiment.
Add training-time difficulty control to the teacher action:

```text
a_teacher = (w_t, P_t(difficulty), epsilon_grounding)
```

- Difficulty knobs already exist in the generator: operand count, operator set,
  target range, solution depth.
- Arms: uniform sampling (current), manual easy→hard, adaptive difficulty
  (success-rate-targeted, e.g. keep batch exact/legality rate near a band),
  adaptive + grounding ε against the reference distribution.
- Each arm runs on top of Stable-RTW reward weights (best known), fixed budget,
  scored per section 2. Hypothesis: task curriculum attacks the exact-search
  bottleneck that reward shaping alone did not (v0.7/v0.8 evidence).

### v0.11 — joint RTW × GACL

Only if v0.10 shows any arm ≥ Stable-RTW-uniform: combine reward and task
curricula; test complementarity (the RQ3 hypothesis in PROJECT_DESIGN.md).

### Supporting audits (interleave when GPU is busy or results are ambiguous)

- Mechanism audit: failure taxonomy over local candidate banks (valid-but-wrong,
  clipping, selector near-misses, oracle-practical gap).
- Cost audit: controlled token/wall-clock comparison, identical cache state.
- Throughput: batch generation or vLLM path for best-of-N (~6 s/example at 256
  tokens is the current bottleneck; a >2× win here doubles experiment rate).

### Later gates (do not start early)

- Seeds 1/2 + OOD splits for any KEPT variant (preserve frozen protocol).
- Harness-shift matrix (prompt_high/mid/low) for the kept method.
- v1.0 coding pilot (HumanEval/MBPP-scale, per PROJECT_DESIGN.md §9) only after
  the Countdown curriculum result is frozen for the paper.

## 7. Self-improvement / continual-learning plan

What must be updated at the end of every iteration, so knowledge compounds
instead of living in one chat session:

1. **`docs/EXPERIMENT_LEDGER.md`** — one row per experiment: id, hypothesis,
   config delta, primary metric, guardrails, KEEP/DISCARD, one-line lesson.
2. **Current-best pointer** — if the best method changed, update
   `docs/CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md` and README status.
3. **This program** — queue reordered, dead hypotheses struck through with a
   reason, new hypotheses appended. The program document is itself a research
   artifact and should improve every cycle (the autoresearch `program.md` role).
4. **Agent memory** — durable cross-session facts (current best, active gate,
   open runs) go to Claude's memory directory so a fresh session resumes in
   seconds.
5. **Commits** — every iteration ends with a commit; runs are attributable to
   code states.

Escalate to the human (rather than deciding autonomously) only for: changing
anything in the frozen list, spending multi-day compute, or changing the paper's
main claim.

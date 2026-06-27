# Literature positioning notes

This project is positioned as a direct transfer of two robotics curriculum-learning ideas into LLM post-training.

## RTW transfer

Original RTW idea:

- Student receives primary reward plus weighted auxiliary rewards.
- Teacher observes performance history and reward history.
- Teacher outputs auxiliary reward weights.
- Training wheels should help early learning and gradually matter less as the student becomes capable.

LLM version:

- Primary reward: verifier says final answer is correct.
- Auxiliary rewards: output format, parser validity, uses required numbers/tools, follows operator/tool constraints, brevity.
- Teacher observes recent verifier outcomes and updates auxiliary weights.
- Desired behavior: format/validity rewards matter early; final correctness dominates later.

## GACL transfer

Original GACL idea:

- Teacher tracks task history and student performance.
- Curriculum must remain grounded in target-domain reference tasks.
- Generated tasks should challenge the student without drifting away from deployment relevance.

LLM version:

- Task history: difficulty bins, operator set, harness level, prompt schema.
- Student performance: exact success and component failures by difficulty/harness.
- Grounding: mix synthetic generated tasks with held-out reference/benchmark tasks.
- Future action space: difficulty distribution + harness informativeness + reward weights.

## Why the first experiment is Countdown

Countdown is small but useful because correctness is non-subjective and verifier-based. This avoids human preference noise while testing the core research object: adaptive auxiliary rewards during LLM post-training.

After this works, the natural second domain is code generation with unit-test harnesses.

# Mechanism audit: where exact candidates are lost (local seed-0 banks)

Date: 2026-07-09. Inputs: validation candidate banks (50 tasks × 8) for
base/static/stable (Gate 0) and v10c2, all loop-mode, frozen protocol.
Artifacts: `outputs/failure_taxonomy_local_banks_validation.json`.
Purpose (NEXT_STEPS Step 2): locate the exact-search bottleneck after the
v0.10-C2 mediator finding, and settle the C2-revision question.

## Failure taxonomy (per-candidate primary class, validation)

| class | base | static | stable | v10c2 |
|---|---:|---:|---:|---:|
| no_answer_span | 0.75 | 0.11 | 0.11 | **0.32** |
| illegal extra/repeated number | 0.13 | 0.33 | 0.30 | 0.25 |
| missing required number | 0.02 | 0.31 | 0.36 | 0.17 |
| legal_but_wrong_value | 0.01 | 0.13 | 0.11 | 0.13 |
| parse_failure | 0.09 | 0.07 | 0.07 | 0.09 |
| exact_correct | 0.00 | 0.03 | 0.02 | 0.02 |

## The four load-bearing findings

1. **Ranking is not the bottleneck; candidate formation is.** Selector
   near-misses = 0 on every bank: whenever an exact candidate exists in a
   task's bank, the practical selector picks it (oracle == practical on all
   local runs — legality features are enough to find the needle). Only 5–7
   tasks of 50 ever produce an exact candidate at N=8.

2. **Number-multiset legality is the dominant trained failure (~60%).**
   Missing-required plus extra/repeated numbers account for 0.53–0.64 of
   all trained candidates. Value search comes after: legal-but-wrong is
   only ~0.12, and of those only ~20% land within ±5 of the target.

3. **Exactness is almost entirely easy-tier.** Exact candidates by tier:
   static 9 easy / 1 medium / 0 hard; stable 7 / 0 / 0; v10c2 6 / 0 / 1.
   Hard-tier valid rate is 3–4/144 for static/stable. v10c2 lifted hard-tier legality to 16/144
   (the curriculum did teach hard-tier syntax) with zero exact gain —
   consistent with the mediator finding: mixing difficulty differently
   cannot fix a legality/search failure.

4. **`no_answer_span` ≡ clipping.** 41–42 of 42–43 no-span candidates in
   static/stable are 256-token-capped rambles; v10c2 has 127 (3×), which is
   the same phenomenon as its 1.8× cost regression — longer generations
   that never close `</answer>`. C2's extra tokens bought truncation, not
   search.

## Consequences

- **C2 revision: NO.** A competence-signal retune changes the difficulty
  mix; findings 2–4 say the losses are number-set legality, truncation, and
  easy-tier-confined search — none of which a mix change addresses. The
  curriculum theme stops at strike one + informative mechanism (recorded).
- What the audit says would matter instead (queue candidates, in order of
  expected leverage):
  a. **Number-legality-targeted training signal** — e.g. increase the
     number_multiset_f1 weight floor, or a hard legality gate on aux reward
     (teacher-side, mutable). This attacks the 60% class directly.
  b. **Truncation control** — brevity/format shaping or an explicit
     close-tag reward so capped rambles become scored attempts; v10c2 shows
     the cost of ignoring it.
  c. Value-search improvements (only after legality): the legal-but-wrong
     pool is small and mostly far from target, so search help is premature.

## Overclaims to avoid

- "The selector is perfect" — it is perfect *on these banks* because exact
  candidates so far always carry clean legality features; a method that
  produces messier exact candidates could reopen the gap.
- "Curricula are useless here" — one arm (C2 adaptive) at one scale; the
  audit says difficulty mix is the wrong lever *for this failure profile*.
- Any cross-method claim from these one-seed banks beyond failure-mode
  structure (the classes are 400-candidate proportions and stable across
  methods, unlike the 50-task exact counts).

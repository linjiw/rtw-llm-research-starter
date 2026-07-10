# Paper 1 assets (auto-generated from committed evidence)

Regenerate: `.venv/bin/python scripts/17_paper1_assets.py`. Reads only
committed score JSONs + candidate banks. Claims per
`RESEARCH_GOAL_AND_PLANS_20260709.md` §2.1.

## C1 — Inference-time selection saturates

reranked@8 == oracle@8 in 53/53 banks (selection saturated)

| bank | n_tasks | reranked@8 | oracle@8 | gap |
|---|---|---|---|---|
| base_local_seed0_test_in_dist_limit50_n8 | 50 | 3 | 3 | 0 |
| base_local_seed0_validation_limit50_n8 | 50 | 0 | 0 | 0 |
| base_qwen05b_seed0_test_in_dist_limit50_n1 | 50 | 1 | 1 | 0 |
| base_qwen05b_seed0_validation_limit50_n1 | 50 | 0 | 0 | 0 |
| harness_stable_seed0_test_in_dist_prompt_high_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_stable_seed0_test_in_dist_prompt_mid_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_stable_seed0_validation_prompt_high_limit50_n8 | 50 | 6 | 6 | 0 |
| harness_stable_seed0_validation_prompt_mid_limit50_n8 | 50 | 1 | 1 | 0 |
| harness_stable_seed1_validation_prompt_high_limit50_n8 | 50 | 5 | 5 | 0 |
| harness_stable_seed1_validation_prompt_mid_limit50_n8 | 50 | 1 | 1 | 0 |
| harness_stable_seed2_validation_prompt_high_limit50_n8 | 50 | 4 | 4 | 0 |
| harness_stable_seed2_validation_prompt_mid_limit50_n8 | 50 | 4 | 4 | 0 |
| harness_static_seed0_test_in_dist_prompt_high_limit50_n8 | 50 | 5 | 5 | 0 |
| harness_static_seed0_test_in_dist_prompt_mid_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_static_seed0_validation_prompt_high_limit50_n8 | 50 | 7 | 7 | 0 |
| harness_static_seed0_validation_prompt_mid_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_static_seed1_test_in_dist_prompt_high_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_static_seed1_validation_prompt_high_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_static_seed1_validation_prompt_mid_limit50_n8 | 50 | 4 | 4 | 0 |
| harness_static_seed2_validation_prompt_high_limit50_n8 | 50 | 2 | 2 | 0 |
| harness_static_seed2_validation_prompt_mid_limit50_n8 | 50 | 2 | 2 | 0 |
| ood_base_test_ood_division_limit50_n8 | 50 | 0 | 0 | 0 |
| ood_base_test_ood_long_limit50_n8 | 50 | 0 | 0 | 0 |
| ood_stable_seed0_test_ood_division_limit50_n8 | 50 | 0 | 0 | 0 |
| ood_stable_seed0_test_ood_long_limit50_n8 | 50 | 0 | 0 | 0 |
| ood_static_seed0_test_ood_division_limit50_n8 | 50 | 1 | 1 | 0 |
| ood_static_seed0_test_ood_long_limit50_n8 | 50 | 0 | 0 | 0 |
| ood_v13sft_seed0_test_ood_division_limit50_n8 | 50 | 1 | 1 | 0 |
| ood_v13sft_seed0_test_ood_long_limit50_n8 | 50 | 0 | 0 | 0 |
| stable_local_seed0_test_in_dist_limit50_n8 | 50 | 2 | 2 | 0 |
| stable_local_seed0_validation_limit50_n8 | 50 | 6 | 6 | 0 |
| stable_local_seed1_test_in_dist_limit50_n8 | 50 | 6 | 6 | 0 |
| stable_local_seed1_validation_limit50_n8 | 50 | 5 | 5 | 0 |
| stable_local_seed2_test_in_dist_limit50_n8 | 50 | 8 | 8 | 0 |
| stable_local_seed2_validation_limit50_n8 | 50 | 4 | 4 | 0 |
| static_local_seed0_test_in_dist_limit50_n8 | 50 | 5 | 5 | 0 |
| static_local_seed0_validation_limit50_n8 | 50 | 7 | 7 | 0 |
| static_local_seed1_test_in_dist_limit50_n8 | 50 | 2 | 2 | 0 |
| static_local_seed1_validation_limit50_n8 | 50 | 2 | 2 | 0 |
| static_local_seed2_test_in_dist_limit50_n8 | 50 | 6 | 6 | 0 |
| static_local_seed2_validation_limit50_n8 | 50 | 2 | 2 | 0 |
| v10c2_local_seed0_test_in_dist_limit50_n8 | 50 | 7 | 7 | 0 |
| v10c2_local_seed0_validation_limit50_n8 | 50 | 5 | 5 | 0 |
| v12legality_seed0_test_in_dist_limit50_n8 | 50 | 8 | 8 | 0 |
| v12legality_seed0_validation_limit50_n8 | 50 | 6 | 6 | 0 |
| v13sft_seed0_test_in_dist_limit50_n8 | 50 | 25 | 25 | 0 |
| v13sft_seed0_validation_limit50_n8 | 50 | 22 | 22 | 0 |
| v13sft_seed1_test_in_dist_limit50_n8 | 50 | 21 | 21 | 0 |
| v13sft_seed1_validation_limit50_n8 | 50 | 18 | 18 | 0 |
| v13sft_seed2_test_in_dist_limit50_n8 | 50 | 26 | 26 | 0 |
| v13sft_seed2_validation_limit50_n8 | 50 | 21 | 21 | 0 |
| v13sftonly_seed0_test_in_dist_limit50_n8 | 50 | 16 | 16 | 0 |
| v13sftonly_seed0_validation_limit50_n8 | 50 | 11 | 11 | 0 |

## C2 — Shaping moves intermediates, not success (two strikes)

```
{
  "v12legality": {
    "oracle@8": 6,
    "stable_oracle@8": 6,
    "n_tasks": 50
  },
  "note": "Two pre-registered reward-shaping strikes; details in V10/V12 plan docs + ledger."
}
```

## C3 — SFT capability lever moves both walls (~5×)

seeds present in banks: [0, 1, 2]

```
{
  "validation": {
    "easy_legality_arm": 1.0,
    "easy_legality_baseline_pooled": 0.22303921568627452,
    "all_tier_p_exact_given_legal": 0.2347560975609756,
    "v13_oracle@8_by_seed": {
      "0": {
        "oracle@8": 22,
        "reranked@8": 22,
        "n_tasks": 50
      },
      "1": {
        "oracle@8": 18,
        "reranked@8": 18,
        "n_tasks": 50
      },
      "2": {
        "oracle@8": 21,
        "reranked@8": 21,
        "n_tasks": 50
      }
    },
    "stable_oracle@8_by_seed": [
      6,
      5,
      4
    ],
    "stable_oracle@8_mean": 5
  },
  "test_in_dist": {
    "easy_legality_arm": 1.0,
    "easy_legality_baseline_pooled": 0.21944444444444444,
    "all_tier_p_exact_given_legal": 0.256797583081571,
    "v13_oracle@8_by_seed": {
      "0": {
        "oracle@8": 25,
        "reranked@8": 25,
        "n_tasks": 50
      },
      "1": {
        "oracle@8": 21,
        "reranked@8": 21,
        "n_tasks": 50
      },
      "2": {
        "oracle@8": 26,
        "reranked@8": 26,
        "n_tasks": 50
      }
    },
    "stable_oracle@8_by_seed": [
      2,
      6,
      8
    ],
    "stable_oracle@8_mean": 5.33
  },
  "seeds012_validation_present": true,
  "seeds012_test_in_dist_present": true
}
```

## C5 — Cost: stable ~0.58× tokens at equal exact

| split | static_tok_mean | stable_tok_mean | ratio_stable_static | gap_over_noise |
|---|---|---|---|---|
| validation | 116.9 | 67.6 | 0.578 | 3.3 |
| test_in_dist | 107.8 | 64.9 | 0.601 | 2.7 |

## C6 — Robustness (pre-registered; may be pending)

```
{
  "harness_shift": "present",
  "ood": "see json"
}
```


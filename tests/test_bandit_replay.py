"""Tests for the S5 stronger-teacher replay (scripts/19_bandit_replay.py).

Pre-registered guards from docs/S5_BANDIT_TEACHER_REPLAY_PLAN.md.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from rtw_llm.teacher import AUX_KEYS

# load the numbered script as a module
_SPEC = importlib.util.spec_from_file_location(
    "bandit_replay", Path(__file__).resolve().parent.parent / "scripts" / "19_bandit_replay.py"
)
br = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(br)


def _comp(correct=0.0, **aux):
    c = {k: 0.0 for k in AUX_KEYS}
    c.update(aux)
    c["correct"] = correct
    return c


def test_total_reward_matches_formula():
    comps = _comp(correct=1.0, format=1.0, brevity=0.5)
    w = {k: 0.2 for k in AUX_KEYS}
    # 1.0 (primary) + 0.2*1.0 (format) + 0.2*0.5 (brevity) = 1.3
    assert abs(br.total_reward(comps, w) - 1.3) < 1e-9


def test_realistic_correct_completion_cannot_be_demoted():
    # The mechanism core (empirical, not a hard primary-only guarantee): a
    # genuinely correct Countdown completion ALSO satisfies its aux components
    # (valid expression, right numbers/ops, numeric_distance=1.0 since it hits
    # the target). So it scores primary + near-full aux (~2.2, cf. the
    # bottleneck diagnosis) while any incorrect completion gets aux-only
    # (<=~1.2). Reweighting cannot demote it.
    correct = _comp(correct=1.0, **{k: 1.0 for k in AUX_KEYS})
    wrong = _comp(correct=0.0, format=1.0, brevity=1.0, valid_expression=0.5)
    group = {"components": [correct, wrong, wrong, wrong]}
    cand = br.candidate_weight_vectors(n_random=50, seed=0)
    m = br.oracle_group_metrics(group, cand)
    assert m["uniform_correct_on_top"] is True
    assert m["correct_demotable"] is False


def test_adversarial_zero_aux_correct_CAN_be_outranked():
    # Documents the subtlety the diagnosis relies on NOT happening in practice:
    # because uniform aux budget (0.2*6=1.2) exceeds primary_weight (1.0), an
    # unrealistic correct-but-zero-aux completion CAN be outranked by an
    # aux-perfect wrong one. Real correct completions never have zero aux.
    correct_no_aux = _comp(correct=1.0)  # 1.0 total under uniform
    perfect_aux_wrong = _comp(correct=0.0, **{k: 1.0 for k in AUX_KEYS})  # 1.2
    group = {"components": [correct_no_aux, perfect_aux_wrong, correct_no_aux, correct_no_aux]}
    m = br.oracle_group_metrics(group, br.candidate_weight_vectors(n_random=20, seed=0))
    assert m["uniform_correct_on_top"] is False  # the 1.2 wrong one wins under uniform


def test_oracle_detects_a_top1_flip_among_non_correct():
    # Two legal-but-wrong completions differ on which aux key they satisfy;
    # reweighting should flip which is top-1.
    a = _comp(correct=0.0, format=1.0)
    b = _comp(correct=0.0, brevity=1.0)
    group = {"components": [a, b, a, b]}
    cand = br.candidate_weight_vectors(n_random=50, seed=0)
    m = br.oracle_group_metrics(group, cand)
    assert m["top1_flip"] is True


def test_group_reconstruction_asserts_shared_id(tmp_path):
    p = tmp_path / "rc.jsonl"
    rows = []
    # one clean group of 4 with shared id, one batch
    for _ in range(4):
        rows.append({"reward_batch_index": 0, "id": "t1", "difficulty": "easy", "components": _comp()})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    groups = br.load_groups(str(p))
    assert len(groups) == 1
    assert groups[0]["id"] == "t1"


def test_group_reconstruction_rejects_mixed_ids(tmp_path):
    p = tmp_path / "rc.jsonl"
    rows = [
        {"reward_batch_index": 0, "id": "t1", "difficulty": "easy", "components": _comp()},
        {"reward_batch_index": 0, "id": "t1", "difficulty": "easy", "components": _comp()},
        {"reward_batch_index": 0, "id": "t2", "difficulty": "easy", "components": _comp()},
        {"reward_batch_index": 0, "id": "t2", "difficulty": "easy", "components": _comp()},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    # 4 rows, group size 4, but two distinct ids -> must raise
    with pytest.raises(AssertionError):
        br.load_groups(str(p))


def test_group_reconstruction_rejects_bad_batch_size(tmp_path):
    p = tmp_path / "rc.jsonl"
    rows = [{"reward_batch_index": 0, "id": "t1", "components": _comp()} for _ in range(6)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(AssertionError):
        br.load_groups(str(p))


def test_exp3_bandit_is_deterministic_under_seed():
    groups = [{"components": [_comp(format=1.0), _comp(brevity=1.0), _comp(), _comp()]}
              for _ in range(40)]
    for g in groups:
        g["batch_index"] = 0
    # give them distinct batch indices so rounds form
    for i, g in enumerate(groups):
        g["batch_index"] = i
    r1 = br.run_bandit(groups, round_bandit=10, seed=7)
    r2 = br.run_bandit(groups, round_bandit=10, seed=7)
    assert r1["final_weights"] == r2["final_weights"]
    r3 = br.run_bandit(groups, round_bandit=10, seed=8)
    # different seed generally diverges (not asserted equal); just runs cleanly
    assert set(r3["final_weights"]) == set(AUX_KEYS)


def test_bandit_movement_is_bounded_not_a_ratchet():
    # Advisor guard: the renormalization fix must remove the increment-only
    # ratchet (which drove L1=0.80 to the cap regardless of signal). After the
    # fix, movement is bounded well below that. NOTE (honest finding): at these
    # arm-reward magnitudes the EXP3 updates are tiny vs the exploration term,
    # so the realized trajectory is exploration-dominated and NOT materially
    # signal-driven — which is exactly why the analysis DEMOTES the bandit to a
    # non-load-bearing check and leads with the assumption-free oracle ceiling.
    a = _comp(format=1.0)
    b = _comp(brevity=1.0)
    sig_groups = [{"batch_index": i, "components": [a, b, a, b]} for i in range(40)]
    r_sig = br.run_bandit(sig_groups, round_bandit=10, seed=3)
    assert r_sig["final_vs_init_l1"] < 0.4  # not the 0.80 ratchet-to-cap
    assert set(r_sig["final_weights"]) == set(AUX_KEYS)


def test_candidate_set_includes_vertices():
    cand = br.candidate_weight_vectors(n_random=5, seed=0)
    # uniform + primary-only + 6 pure + 5 random = 13
    assert len(cand) == 13
    assert {k: 0.2 for k in AUX_KEYS} in cand
    assert {k: 0.0 for k in AUX_KEYS} in cand

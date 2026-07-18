"""Tests for pnyx.baselines — the four static-pool baselines computed over
Pass-1 independent beliefs: mean pool, median pool, log opinion pool, and
Platt-style calibrated stacking.

All fixtures are hand-built ``BeliefEvent``s (no event log on disk, no LLM
calls) so exact values can be hand-verified.
"""

import math
import warnings

import numpy as np
import pytest

from pnyx import baselines
from pnyx.schemas import Belief, BeliefEvent, TurnKey

CONDITION = "BASELINE_TEST"
SEED = 0


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _key(qid, agent_id, *, phase="independent", round_=0, condition=CONDITION):
    return TurnKey(
        condition=condition, seed=SEED, question_id=qid, phase=phase,
        round=round_, agent_id=agent_id,
    )


def _belief(qid, agent_id, prob, *, phase="independent", round_=0, condition=CONDITION):
    return BeliefEvent(
        key=_key(qid, agent_id, phase=phase, round_=round_, condition=condition),
        belief=Belief(prob=prob, rationale="r"),
        parse_failed=False,
        prompt_version="p3-v1",
        ts=0.0,
    )


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# 1. mean_pool / median_pool / log_opinion_pool: exact hand-computed values
# ---------------------------------------------------------------------------


def test_mean_pool_exact():
    events = [_belief("q0", "a0", 0.2), _belief("q0", "a1", 0.5), _belief("q0", "a2", 0.8)]
    assert baselines.mean_pool(events) == pytest.approx(0.5)


def test_mean_pool_uneven_exact():
    events = [_belief("q0", "a0", 0.1), _belief("q0", "a1", 0.9), _belief("q0", "a2", 0.4)]
    assert baselines.mean_pool(events) == pytest.approx((0.1 + 0.9 + 0.4) / 3.0)


def test_median_pool_odd_count():
    events = [_belief("q0", "a0", 0.9), _belief("q0", "a1", 0.1), _belief("q0", "a2", 0.5)]
    assert baselines.median_pool(events) == pytest.approx(0.5)


def test_median_pool_even_count_averages_middle_two():
    events = [
        _belief("q0", "a0", 0.1),
        _belief("q0", "a1", 0.3),
        _belief("q0", "a2", 0.7),
        _belief("q0", "a3", 0.9),
    ]
    assert baselines.median_pool(events) == pytest.approx(0.5)


def test_log_opinion_pool_symmetric_around_half():
    # logit(0.5) = 0, so a symmetric pair averages to logit 0 -> sigmoid 0.5.
    events = [_belief("q0", "a0", 0.2), _belief("q0", "a1", 0.8)]
    assert baselines.log_opinion_pool(events) == pytest.approx(0.5, abs=1e-9)


def test_log_opinion_pool_exact_value():
    probs = [0.1, 0.6, 0.95]
    events = [_belief("q0", f"a{i}", p) for i, p in enumerate(probs)]
    expected = _sigmoid(sum(_logit(p) for p in probs) / len(probs))
    assert baselines.log_opinion_pool(events) == pytest.approx(expected, abs=1e-9)


def test_log_opinion_pool_clamps_extreme_probs():
    # 0.0 and 1.0 would blow up logit() without clamping to [1e-6, 1-1e-6].
    events = [_belief("q0", "a0", 0.0), _belief("q0", "a1", 1.0)]
    result = baselines.log_opinion_pool(events)
    assert math.isfinite(result)
    assert result == pytest.approx(0.5, abs=1e-5)


def test_pools_raise_on_empty():
    with pytest.raises(ValueError):
        baselines.mean_pool([])
    with pytest.raises(ValueError):
        baselines.median_pool([])
    with pytest.raises(ValueError):
        baselines.log_opinion_pool([])


# ---------------------------------------------------------------------------
# 2. Phase guard: raises on a market-phase event, for ALL four functions
# ---------------------------------------------------------------------------


def test_assert_pass1_raises_on_market_phase():
    contaminated = _belief("q0", "a0", 0.5, phase="market", round_=1)
    with pytest.raises(ValueError):
        baselines.assert_pass1(contaminated)


def test_assert_pass1_accepts_independent():
    clean = _belief("q0", "a0", 0.5)
    baselines.assert_pass1(clean)  # must not raise


def test_mean_pool_phase_guard():
    contaminated = [_belief("q0", "a0", 0.5, phase="market", round_=1)]
    with pytest.raises(ValueError):
        baselines.mean_pool(contaminated)


def test_median_pool_phase_guard():
    contaminated = [_belief("q0", "a0", 0.5, phase="market", round_=1)]
    with pytest.raises(ValueError):
        baselines.median_pool(contaminated)


def test_log_opinion_pool_phase_guard():
    contaminated = [_belief("q0", "a0", 0.5, phase="market", round_=1)]
    with pytest.raises(ValueError):
        baselines.log_opinion_pool(contaminated)


def test_calibrated_stack_phase_guard_on_train():
    contaminated_train = [
        ([_belief("q0", "a0", 0.5, phase="market", round_=1)], 1),
    ]
    test = [_belief("q1", "a0", 0.5)]
    with pytest.raises(ValueError):
        baselines.calibrated_stack(contaminated_train, test)


def test_calibrated_stack_phase_guard_on_test():
    train = [([_belief("q0", "a0", 0.5)], 1)]
    contaminated_test = [_belief("q1", "a0", 0.5, phase="market", round_=1)]
    with pytest.raises(ValueError):
        baselines.calibrated_stack(train, contaminated_test)


def test_calibrated_stack_phase_guard_fires_even_when_degenerate():
    # Degenerate (all-1) training outcomes take the fallback branch, but the
    # guard must still fire on contaminated events before the fallback is
    # reached (scientific-integrity invariant is unconditional).
    contaminated_train = [
        ([_belief("q0", "a0", 0.5, phase="market", round_=1)], 1),
        ([_belief("q1", "a0", 0.6)], 1),
    ]
    test = [_belief("q2", "a0", 0.5)]
    with pytest.raises(ValueError):
        baselines.calibrated_stack(contaminated_train, test)


# ---------------------------------------------------------------------------
# 3. calibrated_stack: degenerate-outcome fallback
# ---------------------------------------------------------------------------


def test_calibrated_stack_degenerate_all_ones_falls_back_to_uncalibrated():
    train = [
        ([_belief("q0", "a0", 0.6)], 1),
        ([_belief("q1", "a0", 0.9)], 1),
        ([_belief("q2", "a0", 0.7)], 1),
    ]
    test = [_belief("q3", "a0", 0.3), _belief("q3", "a1", 0.4)]
    result = baselines.calibrated_stack(train, test)
    assert result == pytest.approx(baselines.log_opinion_pool(test), abs=1e-9)


def test_calibrated_stack_degenerate_all_zeros_falls_back_to_uncalibrated():
    train = [
        ([_belief("q0", "a0", 0.6)], 0),
        ([_belief("q1", "a0", 0.9)], 0),
    ]
    test = [_belief("q2", "a0", 0.55)]
    result = baselines.calibrated_stack(train, test)
    assert result == pytest.approx(baselines.log_opinion_pool(test), abs=1e-9)


def test_calibrated_stack_empty_train_falls_back_to_uncalibrated():
    test = [_belief("q0", "a0", 0.42)]
    result = baselines.calibrated_stack([], test)
    assert result == pytest.approx(baselines.log_opinion_pool(test), abs=1e-9)


# ---------------------------------------------------------------------------
# 4. calibrated_stack: recovers a~=1, b~=0 when the data is an exact root of
#    the score equations at those parameters (see report for derivation:
#    each group's rational success-fraction is chosen so k_j == n_j*p_j
#    exactly, so (a=1, b=0) exactly zeroes the log-likelihood gradient and
#    is therefore the unique MLE for this strictly concave problem).
# ---------------------------------------------------------------------------

_EXACT_ROOT_GROUPS = [
    # (p, n_replicas, n_successes) with n_replicas * p == n_successes exactly.
    (0.5, 2, 1),
    (0.75, 4, 3),
    (0.25, 4, 1),
    (0.9, 10, 9),
    (0.1, 10, 1),
    (0.8, 5, 4),
    (0.2, 5, 1),
]


def _exact_root_train():
    train = []
    qi = 0
    for p, n, k in _EXACT_ROOT_GROUPS:
        for j in range(n):
            outcome = 1 if j < k else 0
            train.append(([_belief(f"q{qi}", "a0", p)], outcome))
            qi += 1
    return train


def test_calibrated_stack_recovers_identity_transform():
    train = _exact_root_train()
    x_test = 0.37
    p_test = _sigmoid(x_test)
    test = [_belief("qtest", "a0", p_test)]

    result = baselines.calibrated_stack(train, test)

    assert result == pytest.approx(p_test, abs=1e-3)


def test_fit_platt_recovers_a1_b0_directly():
    train = _exact_root_train()
    xs = np.array([_logit(p) for p, n, _ in _EXACT_ROOT_GROUPS for _ in range(n)])
    ys = np.array(
        [1.0 if j < k else 0.0 for p, n, k in _EXACT_ROOT_GROUPS for j in range(n)]
    )
    a, b = baselines._fit_platt(xs, ys)
    assert a == pytest.approx(1.0, abs=1e-3)
    assert b == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. loo_calibrated_stack: one prediction per question, fit on the other n-1
# ---------------------------------------------------------------------------


def test_loo_calibrated_stack_returns_one_prediction_per_question():
    questions = _exact_root_train()
    predictions = baselines.loo_calibrated_stack(questions)
    assert len(predictions) == len(questions)
    for p in predictions:
        assert 0.0 <= p <= 1.0


def test_loo_calibrated_stack_matches_manual_leave_one_out():
    questions = _exact_root_train()[:10]
    predictions = baselines.loo_calibrated_stack(questions)
    for i, (events, _) in enumerate(questions):
        train = questions[:i] + questions[i + 1:]
        expected = baselines.calibrated_stack(train, events)
        assert predictions[i] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. Separation robustness: non-degenerate but perfectly separable training
#    data. Outcomes vary (unlike section 3's all-0/all-1 case) but are
#    perfectly predicted by x, so the true MLE is still at infinity.
#    Newton-Raphson previously either overflowed exp() (a crash under
#    -W error) or "converged" to an arbitrary huge fit once every point's
#    raw (unclipped) sigmoid saturated. Reviewer's construction: 60 points,
#    y = 1{x > 0}.
# ---------------------------------------------------------------------------


def _separable_train_and_test():
    xs = np.linspace(-3.0, 3.0, 60)
    train = [
        ([_belief(f"sep{i}", "a0", _sigmoid(x))], 1 if x > 0 else 0)
        for i, x in enumerate(xs)
    ]
    test = [_belief("septest", "a0", _sigmoid(0.5))]
    return train, test


def test_fit_platt_separable_nondegenerate_returns_none():
    xs = np.linspace(-3.0, 3.0, 60)
    ys = np.array([1.0 if x > 0 else 0.0 for x in xs])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = baselines._fit_platt(xs, ys)
    assert fit is None


def test_calibrated_stack_separable_nondegenerate_falls_back_no_overflow():
    """Must return a finite in-range probability with NO warnings (this
    used to raise ``RuntimeWarning: overflow encountered in exp`` under
    ``-W error``, or silently return a diverged garbage fit — reviewer
    measured a=698.4, b=9.7 on this exact construction), and must equal
    the uncalibrated fallback exactly (same contract as the degenerate-
    outcome fallback in section 3)."""
    train, test = _separable_train_and_test()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = baselines.calibrated_stack(train, test)

    assert math.isfinite(result)
    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(baselines.log_opinion_pool(test), abs=1e-9)


# ---------------------------------------------------------------------------
# 7. Hessian regression coverage with genuinely noisy data. Sections 4's
#    exact-root construction starts Newton-Raphson at a point where the
#    gradient is already ~0 (a=1, b=0 is already the MLE) — that exercises
#    convergence but gives zero signal on whether the HESSIAN itself is
#    correct, since any invertible-but-wrong Hessian still "converges"
#    immediately from a zero gradient. This fits real Bernoulli-sampled
#    noise so a mis-weighted or sign-flipped Hessian would visibly fail.
# ---------------------------------------------------------------------------


def test_fit_platt_recovers_noisy_logistic_parameters():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    true_a, true_b = 2.3, -0.7
    p_true = 1.0 / (1.0 + np.exp(-(true_a * x + true_b)))
    y = rng.binomial(1, p_true).astype(float)

    fit = baselines._fit_platt(x, y)

    assert fit is not None
    a, b = fit
    assert a == pytest.approx(true_a, abs=0.2)
    assert b == pytest.approx(true_b, abs=0.2)

"""Tests for pnyx.env — generative environment + exact Bayes oracle.

The exact-value tests below carry their algebraic derivations in comments;
the brute-force enumeration cross-check (``test_bruteforce_matches_closed_form``)
is the proof that the closed-form oracle equals the true joint posterior.
"""

import itertools
import math

import numpy as np
import pytest
from pydantic import ValidationError

from pnyx.env import (
    EnvConfig,
    generate_question,
    posterior,
    subset_key,
)

# ---------------------------------------------------------------------------
# Exact-value oracle tests (hand-computed)
# ---------------------------------------------------------------------------


def test_independent_two_signals_exact():
    # Two conditionally-independent signals (lam=0 => z is irrelevant, the
    # sum over z collapses because P_priv does not depend on z and
    # sum_z P(z|s) = 1). Accuracies a=(0.8, 0.7), both observed = 1, prior 0.5.
    #
    #   L(s=1) = a0 * a1           = 0.8 * 0.7 = 0.56
    #   L(s=0) = (1-a0)*(1-a1)     = 0.2 * 0.3 = 0.06
    #   P(s=1 | x) = 0.56 / (0.56 + 0.06) = 0.56 / 0.62
    expected = 0.56 / 0.62
    got = posterior({0: 1, 1: 1}, accuracies=[0.8, 0.7], lams=[0.0, 0.0], a_z=0.8)
    assert got == pytest.approx(expected, abs=1e-15)


def test_single_signal_posteriors_exact():
    # Single independent signal x=1, lam=0: sum over z collapses, posterior = a.
    #   P(s=1 | x0=1) = 0.8 / (0.8 + 0.2) = 0.8
    #   P(s=1 | x1=1) = 0.7 / (0.7 + 0.3) = 0.7
    p0 = posterior({0: 1}, accuracies=[0.8, 0.7], lams=[0.0, 0.0], a_z=0.8)
    p1 = posterior({1: 1}, accuracies=[0.8, 0.7], lams=[0.0, 0.0], a_z=0.8)
    assert p0 == pytest.approx(0.8, abs=1e-15)
    assert p1 == pytest.approx(0.7, abs=1e-15)


def test_conflicting_equal_accuracy_is_half():
    # Two independent signals of equal accuracy disagree (x0=1, x1=0).
    #   L(s=1) = a*(1-a),  L(s=0) = (1-a)*a  => equal => posterior 0.5.
    got = posterior({0: 1, 1: 0}, accuracies=[0.75, 0.75], lams=[0.0, 0.0], a_z=0.8)
    assert got == pytest.approx(0.5, abs=1e-15)


def test_lambda_one_second_signal_adds_zero_information():
    # Both signals deterministically copy z (lam=1). Given the realized values
    # agree (both=1, as they must when both copy the same z), the second
    # observation is redundant: it is fully determined by the first.
    #
    #   With lam=1: P(x_i | s,z) = 1{x_i = z}. Both x=1 => only z=1 survives.
    #   L(s=1) = P(z=1|s=1) = a_z = 0.8 ; L(s=0) = P(z=1|s=0) = 1-a_z = 0.2
    #   posterior(both) = 0.8 / (0.8+0.2) = 0.8 == posterior(first alone).
    acc = [0.9, 0.6]  # accuracies are irrelevant when lam=1
    lams = [1.0, 1.0]
    p_first = posterior({0: 1}, accuracies=acc, lams=lams, a_z=0.8)
    p_both = posterior({0: 1, 1: 1}, accuracies=acc, lams=lams, a_z=0.8)
    assert p_first == pytest.approx(0.8, abs=1e-15)
    assert p_both == pytest.approx(p_first, abs=1e-15)


def test_empty_subset_is_prior():
    assert posterior({}, accuracies=[0.8], lams=[0.0], a_z=0.8) == 0.5


# ---------------------------------------------------------------------------
# Brute-force enumeration cross-check (the correctness proof)
# ---------------------------------------------------------------------------


def _brute_force_posterior(observed, accuracies, lams, a_z):
    """P(s=1 | observed) by explicit enumeration of the full latent joint
    (s, z, per-signal copy/private mechanism, and the private draw y_i).

    Structurally independent of the closed form: the private branch here
    sums over the latent private value y with an indicator 1{x_i = y}
    rather than folding it into a_i vs 1-a_i.
    """
    num_s1 = 0.0
    total = 0.0
    for s in (0, 1):
        ps = 0.5
        for z in (0, 1):
            pz = a_z if z == s else 1.0 - a_z
            prob_x = 1.0
            for i, x in observed.items():
                a = accuracies[i]
                lam = lams[i]
                # copy branch: signal = z
                pi = lam * (1.0 if x == z else 0.0)
                # private branch: signal = y, y = s w.p. a
                for y in (0, 1):
                    py = a if y == s else 1.0 - a
                    pi += (1.0 - lam) * py * (1.0 if x == y else 0.0)
                prob_x *= pi
            joint = ps * pz * prob_x
            total += joint
            if s == 1:
                num_s1 += joint
    return num_s1 / total


def test_bruteforce_matches_closed_form():
    rng = np.random.default_rng(20260712)
    n = 3
    for _ in range(200):
        accuracies = list(rng.uniform(0.55, 0.95, size=n))
        lams = list(rng.uniform(0.0, 0.9, size=n))
        a_z = float(rng.uniform(0.55, 0.95))
        for bits in itertools.product((0, 1), repeat=n):
            observed = {i: bits[i] for i in range(n)}
            closed = posterior(observed, accuracies, lams, a_z)
            brute = _brute_force_posterior(observed, accuracies, lams, a_z)
            assert abs(closed - brute) < 1e-12


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------


def test_envconfig_defaults_and_validation():
    cfg = EnvConfig(
        n_signals=3,
        accuracies=[0.7, 0.7, 0.7],
        lams=[0.1, 0.1, 0.1],
        a_z=0.8,
        difficulty="easy",
    )
    assert cfg.min_single_shard_margin == 0.1
    assert cfg.max_rejection_tries == 1000

    # length mismatch is rejected
    with pytest.raises(ValidationError):
        EnvConfig(
            n_signals=3,
            accuracies=[0.7, 0.7],
            lams=[0.1, 0.1, 0.1],
            a_z=0.8,
            difficulty="easy",
        )
    # out-of-range accuracy rejected
    with pytest.raises(ValidationError):
        EnvConfig(
            n_signals=1,
            accuracies=[1.5],
            lams=[0.0],
            a_z=0.8,
            difficulty="easy",
        )


# ---------------------------------------------------------------------------
# generate_question
# ---------------------------------------------------------------------------


def _favored(p):
    return max(p, 1.0 - p)


def test_generate_easy_respects_band_and_margin():
    cfg = EnvConfig(
        n_signals=6,
        accuracies=[0.75] * 6,
        lams=[0.1] * 6,
        a_z=0.8,
        difficulty="easy",
    )
    rng = np.random.default_rng(1)
    q = generate_question(cfg, rng, "q-easy")

    p_all = q.posterior_table[subset_key(range(6))]
    assert _favored(p_all) >= 0.85

    # single-shard margin holds for every signal
    for i in range(6):
        p_single = q.posterior_table[subset_key([i])]
        assert abs(p_all - p_single) >= cfg.min_single_shard_margin

    # empty subset is the prior
    assert q.posterior_table[""] == 0.5
    # full table present: all 2^n subsets
    assert len(q.posterior_table) == 2**6


def test_generate_hard_respects_band():
    # Hard band (favored posterior in [0.60, 0.75]) with the single-shard
    # margin is only reachable when individual signals are weak (single-signal
    # posteriors sit well away from the aggregate); accuracy 0.55 achieves this.
    cfg = EnvConfig(
        n_signals=6,
        accuracies=[0.55] * 6,
        lams=[0.1] * 6,
        a_z=0.7,
        difficulty="hard",
    )
    rng = np.random.default_rng(7)
    q = generate_question(cfg, rng, "q-hard")
    p_all = q.posterior_table[subset_key(range(6))]
    assert 0.60 <= _favored(p_all) <= 0.75
    for i in range(6):
        p_single = q.posterior_table[subset_key([i])]
        assert abs(p_all - p_single) >= cfg.min_single_shard_margin


def test_generate_deterministic_given_seed():
    cfg = EnvConfig(
        n_signals=6,
        accuracies=[0.75] * 6,
        lams=[0.1] * 6,
        a_z=0.8,
        difficulty="easy",
    )
    q1 = generate_question(cfg, np.random.default_rng(42), "q")
    q2 = generate_question(cfg, np.random.default_rng(42), "q")
    assert q1.latent_state == q2.latent_state
    assert [s.value for s in q1.signals] == [s.value for s in q2.signals]
    assert q1.posterior_table == q2.posterior_table


def test_generate_table_matches_direct_oracle():
    cfg = EnvConfig(
        n_signals=6,
        accuracies=[0.75] * 6,
        lams=[0.1] * 6,
        a_z=0.8,
        difficulty="easy",
    )
    q = generate_question(cfg, np.random.default_rng(3), "q")
    values = {s.index: s.value for s in q.signals}
    acc = [s.accuracy for s in q.signals]
    lams = [s.lam for s in q.signals]
    for r in range(cfg.n_signals + 1):
        for combo in itertools.combinations(range(cfg.n_signals), r):
            key = subset_key(combo)
            observed = {i: values[i] for i in combo}
            assert q.posterior_table[key] == pytest.approx(
                posterior(observed, acc, lams, cfg.a_z), abs=1e-15
            )


def test_generate_rejection_budget_exhausted_raises():
    # Fully uninformative environment (accuracy 0.5, lam 0): every posterior
    # is exactly 0.5, so the easy band (>=0.85) can never be met.
    cfg = EnvConfig(
        n_signals=3,
        accuracies=[0.5, 0.5, 0.5],
        lams=[0.0, 0.0, 0.0],
        a_z=0.5,
        difficulty="easy",
        max_rejection_tries=50,
    )
    with pytest.raises(ValueError, match="rejection"):
        generate_question(cfg, np.random.default_rng(0), "q")

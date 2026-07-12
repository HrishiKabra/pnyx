"""Generative environment + exact Bayes posterior oracle.

This is the scientific ground truth of Pnyx: every question is drawn from a
known generative process, and the Bayes-optimal posterior P(s=1 | observed
signals) is computed in closed form (no sampling, no approximation) so that
agent / market beliefs can be scored against an exactly-known target.

Generative model
-----------------
Let ``s`` be the binary latent state the question asks about.

* Latent state:            ``s ~ Bernoulli(0.5)``.
* Common latent channel:   ``z = s`` w.p. ``a_z``, else ``z = 1 - s``.
* Signal ``i`` (``i = 0..n-1``):
    - w.p. ``lam_i`` the signal *copies the common channel*: ``x_i = z``;
    - w.p. ``1 - lam_i`` it draws an *independent private channel*
      ``y_i = s`` w.p. ``a_i``, else ``y_i = 1 - s``, and ``x_i = y_i``.

So ``lam_i = 0`` for all ``i`` gives conditionally-independent signals with
accuracies ``a_i``; ``lam_i`` is the correlation knob (``lam_i = 1`` makes the
signal a deterministic copy of ``z``, hence perfectly correlated with every
other copy of ``z``).

Given ``(s, z)`` the signals are mutually independent, and the per-signal
emission probability is::

    P(x_i | s, z) = lam_i * 1{x_i = z} + (1 - lam_i) * P_priv(x_i | s)

    with  P_priv(x_i | s) = a_i      if x_i = s
                          = 1 - a_i  otherwise.

Exact oracle
------------
For an observed subset ``A`` with values ``x_A`` and uniform prior on ``s``::

    P(x_A | s) = sum_{z in {0,1}} P(z | s) * prod_{i in A} P(x_i | s, z)

    P(z | s)   = a_z      if z = s
               = 1 - a_z  otherwise

    P(s=1 | x_A) = P(x_A | s=1) / (P(x_A | s=1) + P(x_A | s=0))

(the equal prior 0.5 cancels). The empty subset returns the prior 0.5.

All functions here are pure and take an explicit ``numpy.random.Generator``;
there is no global RNG and no I/O.
"""

import itertools
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from pnyx.schemas import EnvConfig, QuestionRecord, SignalRecord

__all__ = [
    "EnvConfig",
    "posterior",
    "generate_question",
    "subset_key",
]

# Difficulty target bands for the all-signals posterior on the favored side,
# favored = max(p, 1 - p).
EASY_MIN_FAVORED = 0.85
HARD_MIN_FAVORED = 0.60
HARD_MAX_FAVORED = 0.75


def subset_key(indices: Iterable[int]) -> str:
    """Canonical ``posterior_table`` key for a set of signal indices:
    comma-joined, strictly ascending, deduplicated (``""`` for the empty set).
    Matches the convention documented on ``QuestionRecord``."""
    ordered = sorted(set(int(i) for i in indices))
    return ",".join(str(i) for i in ordered)


def posterior(
    observed: Mapping[int, int],
    accuracies: Sequence[float],
    lams: Sequence[float],
    a_z: float,
) -> float:
    """Exact P(s=1 | observed) under the generative model (see module docstring).

    ``observed`` maps signal index -> observed value in {0, 1}. An empty
    mapping returns the prior 0.5.
    """
    if not observed:
        return 0.5

    def likelihood(s: int) -> float:
        total = 0.0
        for z in (0, 1):
            p_z = a_z if z == s else 1.0 - a_z
            prod = 1.0
            for i, x in observed.items():
                a_i = accuracies[i]
                lam_i = lams[i]
                p_priv = a_i if x == s else 1.0 - a_i
                prod *= lam_i * (1.0 if x == z else 0.0) + (1.0 - lam_i) * p_priv
            total += p_z * prod
        return total

    l1 = likelihood(1)
    l0 = likelihood(0)
    denom = l1 + l0
    if denom == 0.0:
        # Probability-zero observation (only reachable with lam=1 and mutually
        # contradictory copies of z, which the generator never realizes).
        # The prior is the sensible fallback.
        return 0.5
    return l1 / denom


def _difficulty_ok(p_all: float, difficulty: str) -> bool:
    favored = max(p_all, 1.0 - p_all)
    if difficulty == "easy":
        return favored >= EASY_MIN_FAVORED
    # "hard"
    return HARD_MIN_FAVORED <= favored <= HARD_MAX_FAVORED


def _build_posterior_table(
    values: Sequence[int],
    accuracies: Sequence[float],
    lams: Sequence[float],
    a_z: float,
    n: int,
) -> dict[str, float]:
    """Full posterior table over all ``2**n`` subsets of the realized signal
    values, keyed per the ``QuestionRecord`` convention."""
    table: dict[str, float] = {}
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            observed = {i: values[i] for i in combo}
            table[subset_key(combo)] = posterior(observed, accuracies, lams, a_z)
    return table


def generate_question(
    config: EnvConfig,
    rng: np.random.Generator,
    question_id: str,
) -> QuestionRecord:
    """Draw a single question by rejection sampling until the realized draw
    satisfies (a) the difficulty target on the all-signals posterior,
    (b) the single-shard margin for every signal, and (c) union informative
    (the all-signals posterior beats the prior).

    Raises ``ValueError`` if the rejection budget is exhausted.
    """
    n = config.n_signals
    acc = config.accuracies
    lams = config.lams
    a_z = config.a_z

    for _ in range(config.max_rejection_tries):
        s = 1 if rng.random() < 0.5 else 0
        z = s if rng.random() < a_z else 1 - s

        values: list[int] = []
        for i in range(n):
            if rng.random() < lams[i]:
                values.append(z)
            else:
                values.append(s if rng.random() < acc[i] else 1 - s)

        observed_all = {i: values[i] for i in range(n)}
        p_all = posterior(observed_all, acc, lams, a_z)

        # (a) difficulty band
        if not _difficulty_ok(p_all, config.difficulty):
            continue

        # (b) single-shard margin: every single-signal posterior differs from
        #     the all-signals posterior by at least the margin.
        singles = [
            posterior({i: values[i]}, acc, lams, a_z) for i in range(n)
        ]
        if any(
            abs(p_all - p_single) < config.min_single_shard_margin
            for p_single in singles
        ):
            continue

        # (c) union informative: the union of all signals moves off the prior.
        if max(p_all, 1.0 - p_all) <= 0.5:
            continue

        signals = [
            SignalRecord(
                index=i, value=values[i], accuracy=acc[i], lam=lams[i]
            )
            for i in range(n)
        ]
        table = _build_posterior_table(values, acc, lams, a_z, n)
        return QuestionRecord(
            question_id=question_id,
            latent_state=s,
            signals=signals,
            posterior_table=table,
            meta={
                "difficulty": config.difficulty,
                "a_z": a_z,
                "posterior_all": p_all,
            },
        )

    raise ValueError(
        f"generate_question: rejection budget of {config.max_rejection_tries} "
        f"exhausted for difficulty {config.difficulty!r} "
        f"(question_id={question_id!r}); loosen the config "
        "(accuracies, lams, a_z, difficulty band, or margin)."
    )

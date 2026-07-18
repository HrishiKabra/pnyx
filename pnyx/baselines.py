"""Static-pool baselines over Pass-1 (independent) beliefs.

These are the "static pooling" side of H1: mean pool, median pool, an
equal-weight log opinion pool (LOP), and a Platt-style calibrated stack.
Every one of them is scored against the same Bayes-oracle posterior as the
market price (see ``pnyx.env``), so H1 reduces to comparing a market's
posterior gap against these functions' posterior gap on IDENTICAL shards.

Scientific-integrity guard
---------------------------
Global Constraints: "Never compute a baseline from Pass-2 beliefs." Every
function here takes ``BeliefEvent``s and checks ``assert_pass1`` on each one
before touching its probability, raising loudly (``ValueError``, not
``assert`` — this must survive ``python -O``) rather than silently letting a
hand-edited or corrupted log smuggle a market-phase belief into a baseline.
This guard is THE single implementation of that invariant in the codebase:
``pnyx.analysis.pilot`` imports ``assert_pass1`` from here rather than
keeping its own private copy.

Pure functions, no I/O — callers (analysis scripts, ``pnyx.analysis.pilot``)
own reading event logs and pass in the resulting ``list[BeliefEvent]``.

Platt-style calibrated stacking (``calibrated_stack`` / ``loo_calibrated_stack``)
----------------------------------------------------------------------------
Feature per question: the equal-weight mean of its agents' clamped logits
(the same quantity ``log_opinion_pool`` runs through a raw sigmoid). Fit
sigma(a*x + b) by maximizing the Bernoulli log-likelihood with Newton-Raphson
(IRLS): the score (gradient) and Hessian of the log-likelihood both have a
closed form for two parameters, so each Newton step is a 2x2 linear solve.
A small ridge is subtracted from the Hessian diagonal for numerical
stability (avoids a singular solve when the data underdetermines a
parameter, e.g. a single training point, without perceptibly biasing a
well-determined fit). When the training outcomes are degenerate (all 0 or
all 1 — including the vacuous empty-train case) the true MLE is at
infinity: a separable logistic likelihood has no finite maximizer, so
there is nothing for Newton-Raphson to converge to. We fall back to the
uncalibrated sigmoid of the test feature (a=1, b=0, the Newton-Raphson
starting point) rather than returning a diverging or arbitrary fit.
"""

import numpy as np

from pnyx.schemas import BeliefEvent

__all__ = [
    "assert_pass1",
    "mean_pool",
    "median_pool",
    "log_opinion_pool",
    "calibrated_stack",
    "loo_calibrated_stack",
]

# Clamp bound for probabilities before taking a logit (avoids +/-inf at the
# boundary of [0, 1]).
_PROB_LO = 1e-6
_PROB_HI = 1.0 - 1e-6

_MAX_NEWTON_ITERS = 100
_NEWTON_TOL = 1e-10
_RIDGE = 1e-6


def assert_pass1(e: BeliefEvent) -> None:
    """Guard: ``e`` must be a Pass-1 (``phase == "independent"``) belief.

    Raises ``ValueError`` (not ``assert`` — this invariant must survive
    ``python -O``) on a hand-edited/corrupted log that tags a belief event
    ``phase="market"``, rather than silently letting it skew a static
    baseline. See module docstring.
    """
    if e.key.phase != "independent":
        raise ValueError(
            "phase guard violated: belief event has phase="
            f"{e.key.phase!r} (expected 'independent') at key={e.key.as_tuple()}"
        )


def _probs(events: list[BeliefEvent]) -> list[float]:
    """The guard-checked, un-clamped probabilities of ``events``, in order.
    Raises on an empty list — there is nothing to pool."""
    if not events:
        raise ValueError("no belief events to pool")
    for e in events:
        assert_pass1(e)
    return [e.belief.prob for e in events]


def _mean_logit(events: list[BeliefEvent]) -> float:
    """Mean of ``events``' clamped logits — the shared feature computation
    behind ``log_opinion_pool`` and ``calibrated_stack``. Guard-checked and
    raises on an empty list (see ``_probs``)."""
    probs = np.clip(np.array(_probs(events), dtype=float), _PROB_LO, _PROB_HI)
    logits = np.log(probs / (1.0 - probs))
    return float(logits.mean())


def mean_pool(events: list[BeliefEvent]) -> float:
    """Arithmetic mean of ``belief.prob`` over ``events``."""
    probs = _probs(events)
    return sum(probs) / len(probs)


def median_pool(events: list[BeliefEvent]) -> float:
    """Median of ``belief.prob`` over ``events`` (average of the middle two
    for an even count)."""
    probs = sorted(_probs(events))
    n = len(probs)
    mid = n // 2
    if n % 2 == 1:
        return probs[mid]
    return (probs[mid - 1] + probs[mid]) / 2.0


def log_opinion_pool(events: list[BeliefEvent]) -> float:
    """Equal-weight log opinion pool: clamp each prob to
    [1e-6, 1-1e-6], average the logits, sigmoid back."""
    return float(1.0 / (1.0 + np.exp(-_mean_logit(events))))


def _fit_platt(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit sigma(a*x + b) to (x, y) by maximizing Bernoulli log-likelihood
    via Newton-Raphson (IRLS), starting at a=1, b=0. At most
    ``_MAX_NEWTON_ITERS`` steps; stops when max(|delta_a|, |delta_b|) <
    ``_NEWTON_TOL``. A ridge of ``_RIDGE`` is subtracted from the Hessian
    diagonal each step for solve stability.

    Caller's responsibility: ``y`` must not be degenerate (all-0/all-1) —
    that case has no finite MLE and is handled by ``calibrated_stack``
    before this is called.
    """
    a, b = 1.0, 0.0
    for _ in range(_MAX_NEWTON_ITERS):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = np.array([np.sum((y - p) * x), np.sum(y - p)])
        w = p * (1.0 - p)
        # Hessian of the log-likelihood (negative semi-definite); ridge is
        # subtracted from the diagonal (makes it more negative-definite,
        # i.e. Tikhonov-style regularization) so the solve stays stable even
        # when the data underdetermines a or b.
        hessian = np.array(
            [
                [-np.sum(w * x * x) - _RIDGE, -np.sum(w * x)],
                [-np.sum(w * x), -np.sum(w) - _RIDGE],
            ]
        )
        delta = np.linalg.solve(hessian, grad)
        a_new, b_new = a - delta[0], b - delta[1]
        converged = max(abs(a_new - a), abs(b_new - b)) < _NEWTON_TOL
        a, b = a_new, b_new
        if converged:
            break
    return a, b


def calibrated_stack(
    train: list[tuple[list[BeliefEvent], int]],
    test: list[BeliefEvent],
) -> float:
    """Platt-style calibration of the mean logit.

    ``train`` pairs each held-in question's Pass-1 events with its realized
    outcome (0/1); ``test`` is the held-out question's events. Fits
    sigma(a*x + b) on the training questions' (mean-logit, outcome) pairs
    (see ``_fit_platt``) and returns sigma(a*x_test + b). Falls back to the
    uncalibrated sigma(x_test) when the training outcomes are degenerate —
    including an empty ``train`` — since a separable Bernoulli likelihood
    has no finite MLE (see module docstring).
    """
    # Guard every event unconditionally, before the degenerate-outcome
    # fallback branch, so the phase guard is not accidentally bypassed by
    # the fallback for degenerate/empty training data.
    for events, _ in train:
        for e in events:
            assert_pass1(e)
    x_test = _mean_logit(test)

    outcomes = [outcome for _, outcome in train]
    degenerate = not outcomes or all(o == 0 for o in outcomes) or all(o == 1 for o in outcomes)
    if degenerate:
        return float(1.0 / (1.0 + np.exp(-x_test)))

    xs = np.array([_mean_logit(events) for events, _ in train], dtype=float)
    ys = np.array([float(o) for o in outcomes], dtype=float)
    a, b = _fit_platt(xs, ys)
    return float(1.0 / (1.0 + np.exp(-(a * x_test + b))))


def loo_calibrated_stack(
    questions: list[tuple[list[BeliefEvent], int]],
) -> list[float]:
    """Leave-one-question-out calibrated-stack predictions: one prediction
    per entry of ``questions``, each fit on the other n-1 (see
    ``calibrated_stack``)."""
    predictions = []
    for i, (events, _) in enumerate(questions):
        train = questions[:i] + questions[i + 1 :]
        predictions.append(calibrated_stack(train, events))
    return predictions

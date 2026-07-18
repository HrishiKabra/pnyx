"""Pnyx P5 main-run analysis — the core module every later analysis task
(figures, tables, manipulation metrics) consumes.

It turns the P4 event logs under ``data/main/<COND>/`` into three things:

* a per-``(seed, question)`` metrics table (``question_metrics``): the market
  price and each of the four static pools, each scored against the Bayes
  oracle posterior (the *posterior gap*) and the realized outcome (Brier);
* a per-``(seed, agent-pair)`` error-correlation table (``team_rho`` /
  ``team_rho_summary``): the H2 "phase map" x-axis;
* the paired statistics H1/H2/H3 are reported with (``paired_bootstrap``,
  ``wilcoxon_signed_rank``, ``compare``).

Why a separate module from ``pnyx.analysis.pilot``: the pilot aggregates ONE
directory into headline scalars for a matching table; the main run needs a
tidy long-format table keyed by ``(seed, question_id)`` that downstream
figures/tables slice and pair-test. The file-reading primitives are shared
(imported from ``pilot`` — one implementation, per the Task-1 precedent that
moved ``assert_pass1`` to ``pnyx.baselines``).

Design contracts that bind this module
--------------------------------------
* **Phase guard.** Every belief that feeds a pool or a rho is routed through
  ``pnyx.baselines.assert_pass1`` (via ``_pass1_beliefs`` / the pool + rho
  helpers), so a market-phase belief smuggled into a log raises rather than
  silently contaminating a baseline (Global Constraints: "never compute a
  baseline from Pass-2 beliefs").
* **Pass-1 reuse (W/D).** Pass2-only conditions (``D_K*`` reuse ``C``;
  ``W_*`` reuse ``A``) carry no belief events of their own — the CALLER
  passes the source condition's ``ConditionData`` as ``pass1_source`` (mirror
  of ``runner.pass1_source_dir``). The pool columns are then computed
  entirely from the source and are *bit-identical* to
  ``question_metrics(source, None)`` by construction (we compute once, on the
  source, and join). Coverage is asserted the way the runner asserts it: every
  honest agent that traded in the pass2-only condition must have a source
  Pass-1 belief for every settled question, else raise.
* **Honest agents are p0..p5.** Adversary ``adv0`` (condition D) and any
  non-``p{0..5}`` agent are excluded from every pool and rho computation.
* **Determinism.** ``paired_bootstrap`` takes an explicit ``seed`` (default 0)
  and uses ``numpy.random.default_rng`` — same seed, identical output.
* **Purity.** The metric/stat functions do no I/O; only ``load_condition``
  touches the filesystem (through the shared ``pilot`` readers).

Dependencies here are stdlib + numpy + pandas + pydantic only — no
scipy/sklearn (Wilcoxon is implemented from its definition).
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import erf, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pnyx.analysis.pilot import (
    _discover_files,
    _posterior_all,
    _read_events,
    _read_questions,
)
from pnyx.baselines import (
    assert_pass1,
    log_opinion_pool,
    loo_calibrated_stack,
    mean_pool,
    median_pool,
)
from pnyx.schemas import (
    BeliefEvent,
    QuestionRecord,
    SettlementEvent,
    TradeEvent,
)

__all__ = [
    "SeedData",
    "ConditionData",
    "load_condition",
    "question_metrics",
    "team_rho",
    "team_rho_summary",
    "RhoSummary",
    "paired_bootstrap",
    "wilcoxon_signed_rank",
    "compare",
    "ComparisonResult",
    "HONEST_AGENTS",
]

# The six honest agents present in every pool, homogeneous, and mixed team
# (condition D additionally carries an adversary ``adv0``, excluded here).
HONEST_AGENTS: tuple[str, ...] = tuple(f"p{i}" for i in range(6))
_HONEST_SET = frozenset(HONEST_AGENTS)


# ---------------------------------------------------------------------------
# Loaded data model
# ---------------------------------------------------------------------------


@dataclass
class SeedData:
    """One ``(condition, seed)`` slice: its event log split by type and its
    question sidecar keyed by ``question_id``. ``beliefs`` is empty for a
    pass2-only condition (its Pass-1 lives in the source condition's log)."""

    seed: int
    beliefs: list[BeliefEvent] = field(default_factory=list)
    trades: list[TradeEvent] = field(default_factory=list)
    settlements: list[SettlementEvent] = field(default_factory=list)
    questions: dict[str, QuestionRecord] = field(default_factory=dict)


@dataclass
class ConditionData:
    """Every seed of one condition. ``seeds`` maps the integer seed to its
    :class:`SeedData`."""

    condition: str
    seeds: dict[int, SeedData] = field(default_factory=dict)


def _seed_of(name: str, suffix: str) -> int:
    """Extract the integer ``seed`` from a ``<cond>_seed<seed>.<suffix>``
    filename (e.g. ``A_seed2.jsonl`` -> 2)."""
    stem = name[: -len(suffix)] if name.endswith(suffix) else name
    marker = "_seed"
    idx = stem.rfind(marker)
    if idx == -1:
        raise ValueError(f"cannot parse seed from filename {name!r}")
    return int(stem[idx + len(marker):])


def load_condition(run_dir: str | Path) -> ConditionData:
    """Load every ``(seed)`` under one condition directory into a
    :class:`ConditionData`.

    Reads ``<cond>_seed<seed>.jsonl`` event logs and
    ``<cond>_seed<seed>.questions.jsonl`` sidecars (cost ledgers are not
    needed here — subsidy is on the settlement event, parse-fail on the
    belief/trade events). Tolerant of a torn final JSONL line exactly like
    the runner's replay (shared ``pilot`` readers).
    """
    run_dir = Path(run_dir)
    event_paths, sidecar_paths, _ledgers = _discover_files(run_dir)

    seeds: dict[int, SeedData] = {}
    condition: str | None = None

    for p in event_paths:
        seed = _seed_of(p.name, ".jsonl")
        sd = seeds.setdefault(seed, SeedData(seed=seed))
        for e in _read_events(p):
            if e.type == "belief":
                sd.beliefs.append(e)
                if condition is None:
                    condition = e.key.condition
            elif e.type == "trade":
                sd.trades.append(e)
                if condition is None:
                    condition = e.key.condition
            else:  # settlement
                sd.settlements.append(e)
                if condition is None:
                    condition = e.condition

    for p in sidecar_paths:
        seed = _seed_of(p.name, ".questions.jsonl")
        sd = seeds.setdefault(seed, SeedData(seed=seed))
        for rec in _read_questions(p):
            sd.questions[rec.question_id] = rec

    return ConditionData(condition=condition or run_dir.name, seeds=seeds)


# ---------------------------------------------------------------------------
# Phase-guarded honest Pass-1 access
# ---------------------------------------------------------------------------


def _honest_pass1_by_question(sd: SeedData) -> dict[str, list[BeliefEvent]]:
    """``question_id -> [honest Pass-1 beliefs]`` for one seed, guard-checked.

    Filters to ``HONEST_AGENTS`` (drops any adversary) and routes every belief
    through ``assert_pass1`` so a market-phase belief raises. Beliefs are
    returned sorted by ``agent_id`` for deterministic pooling order (the pools
    themselves are order-independent, but stable order keeps any future
    tie-breaking reproducible)."""
    by_q: dict[str, list[BeliefEvent]] = {}
    for e in sd.beliefs:
        if e.key.agent_id not in _HONEST_SET:
            continue
        assert_pass1(e)  # phase guard (raises on a market-phase belief)
        by_q.setdefault(e.key.question_id, []).append(e)
    for beliefs in by_q.values():
        beliefs.sort(key=lambda ev: ev.key.agent_id)
    return by_q


# ---------------------------------------------------------------------------
# Static-pool predictions (mean/median/lop/stack) per (seed, question)
# ---------------------------------------------------------------------------


def _pool_predictions(source: ConditionData) -> dict[tuple[int, str], dict[str, float]]:
    """Every static pool's prediction for each fully-covered, settled question
    of ``source``, keyed by ``(seed, question_id)``.

    "Fully covered" = the question has an honest Pass-1 belief AND a settlement
    (needed for the calibrated stack's outcome). The stack is the leave-one-
    question-out calibrated stack fit *within* each ``(condition, seed)`` over
    that seed's covered questions (``pnyx.baselines.loo_calibrated_stack``);
    mean/median/lop are per-question. Computing this once on the source and
    joining is what makes a pass2-only condition's pool columns bit-identical
    to the source condition's own (see module docstring)."""
    preds: dict[tuple[int, str], dict[str, float]] = {}
    for seed, sd in source.seeds.items():
        by_q = _honest_pass1_by_question(sd)
        outcomes = {s.question_id: s.outcome for s in sd.settlements}
        covered = sorted(q for q in by_q if q in outcomes)
        if not covered:
            continue
        stack_input = [(by_q[q], outcomes[q]) for q in covered]
        stack_preds = loo_calibrated_stack(stack_input)
        for qid, stack_p in zip(covered, stack_preds):
            beliefs = by_q[qid]
            preds[(seed, qid)] = {
                "mean": mean_pool(beliefs),
                "median": median_pool(beliefs),
                "lop": log_opinion_pool(beliefs),
                "stack": stack_p,
            }
    return preds


def _assert_source_coverage(cond: ConditionData, source: ConditionData) -> None:
    """Mirror ``runner._assert_source_covers``: every honest agent that traded
    in the pass2-only ``cond`` must have a source Pass-1 belief for every
    settled question of that seed, else raise. This is the guarantee that the
    reused pool columns are complete (and hence bit-identical to the source's
    own)."""
    missing: list[str] = []
    for seed, sd in cond.seeds.items():
        honest_agents = sorted(
            {t.key.agent_id for t in sd.trades if t.key.agent_id in _HONEST_SET}
        )
        src_sd = source.seeds.get(seed)
        src_beliefs = (
            {(e.key.question_id, e.key.agent_id) for e in src_sd.beliefs}
            if src_sd is not None
            else set()
        )
        for s in sd.settlements:
            for aid in honest_agents:
                if (s.question_id, aid) not in src_beliefs:
                    missing.append(f"seed{seed}/{s.question_id}/{aid}")
    if missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise ValueError(
            f"pass1_source missing full honest Pass-1 coverage for condition "
            f"{cond.condition!r}: {len(missing)} (seed, question, agent) belief(s) "
            f"absent: {preview}"
        )


# ---------------------------------------------------------------------------
# Per-question metrics table
# ---------------------------------------------------------------------------

_POOL_COLS = ("mean", "median", "lop", "stack")


def question_metrics(
    cond: ConditionData, pass1_source: ConditionData | None
) -> pd.DataFrame:
    """One row per ``(seed, question_id)`` of ``cond``'s settled questions.

    Columns: ``seed``, ``question_id``, ``posterior_all``, ``outcome``;
    ``market_prob`` (= ``SettlementEvent.final_price``) with ``market_gap`` /
    ``market_brier``; ``{mean,median,lop,stack}_prob`` with matching ``_gap`` /
    ``_brier`` from the honest agents' Pass-1 pools; ``subsidy``;
    ``n_parse_failed`` (parse-failed belief+trade events for that question).

    ``pass1_source`` is ``None`` for a two-pass condition (pools come from
    ``cond`` itself) and the source condition's :class:`ConditionData` for a
    pass2-only condition (pools come from the source; coverage asserted). The
    posterior gap is ``|prediction - posterior_all|`` where ``posterior_all``
    is the Bayes posterior given ALL signals (the oracle); Brier is
    ``(prediction - outcome)**2``.
    """
    pool_source = pass1_source if pass1_source is not None else cond
    if pass1_source is not None:
        _assert_source_coverage(cond, pool_source)
    pool_preds = _pool_predictions(pool_source)

    rows: list[dict] = []
    for seed in sorted(cond.seeds):
        sd = cond.seeds[seed]
        # parse-failed counts per question (both passes present in this log).
        pf_by_q: dict[str, int] = {}
        for e in sd.beliefs:
            if e.parse_failed:
                pf_by_q[e.key.question_id] = pf_by_q.get(e.key.question_id, 0) + 1
        for t in sd.trades:
            if t.parse_failed:
                pf_by_q[t.key.question_id] = pf_by_q.get(t.key.question_id, 0) + 1

        for s in sorted(sd.settlements, key=lambda ev: ev.question_id):
            qid = s.question_id
            record = sd.questions.get(qid)
            if record is None:
                raise ValueError(
                    f"condition {cond.condition!r} seed {seed}: settled question "
                    f"{qid!r} has no question sidecar record"
                )
            posterior_all = _posterior_all(record)
            if posterior_all is None:
                raise ValueError(
                    f"condition {cond.condition!r} seed {seed} question {qid!r}: "
                    "posterior_table has no all-signals entry (oracle missing)"
                )
            pred = pool_preds.get((seed, qid))
            if pred is None:
                raise ValueError(
                    f"condition {cond.condition!r} seed {seed} question {qid!r}: "
                    "no static-pool prediction available (missing honest Pass-1 "
                    "coverage or settlement in the pool source)"
                )

            outcome = s.outcome
            market_prob = s.final_price
            row = {
                "seed": seed,
                "question_id": qid,
                "posterior_all": posterior_all,
                "outcome": outcome,
                "market_prob": market_prob,
                "market_gap": abs(market_prob - posterior_all),
                "market_brier": (market_prob - outcome) ** 2,
                "subsidy": s.subsidy,
                "n_parse_failed": pf_by_q.get(qid, 0),
            }
            for name in _POOL_COLS:
                p = pred[name]
                row[f"{name}_prob"] = p
                row[f"{name}_gap"] = abs(p - posterior_all)
                row[f"{name}_brier"] = (p - outcome) ** 2
            rows.append(row)

    columns = [
        "seed", "question_id", "posterior_all", "outcome",
        "market_prob", "market_gap", "market_brier",
        "mean_prob", "mean_gap", "mean_brier",
        "median_prob", "median_gap", "median_brier",
        "lop_prob", "lop_gap", "lop_brier",
        "stack_prob", "stack_gap", "stack_brier",
        "subsidy", "n_parse_failed",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Team error correlation (rho)
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length vectors, returning ``nan`` when
    either has zero variance (rather than emitting a numpy divide warning that
    would become an error under ``pytest -W error``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def team_rho(cond: ConditionData) -> pd.DataFrame:
    """One row per ``(seed, agent_a, agent_b)`` — Pearson correlation over the
    questions of two honest agents' Pass-1 *errors*
    ``e_i(q) = prob_i(q) - posterior_all(q)``.

    Pairs are the unordered pairs of honest agents present in the seed
    (15 for the full six). The correlation is taken over questions where both
    agents have a Pass-1 belief and the oracle posterior exists, in sorted
    ``question_id`` order. Phase-guarded like the pools."""
    rows: list[dict] = []
    for seed in sorted(cond.seeds):
        sd = cond.seeds[seed]
        by_q = _honest_pass1_by_question(sd)
        # agent_id -> {question_id: error}
        errors: dict[str, dict[str, float]] = {}
        for qid, beliefs in by_q.items():
            record = sd.questions.get(qid)
            if record is None:
                continue
            posterior_all = _posterior_all(record)
            if posterior_all is None:
                continue
            for e in beliefs:
                errors.setdefault(e.key.agent_id, {})[qid] = (
                    e.belief.prob - posterior_all
                )
        agents = sorted(errors)
        for a, b in combinations(agents, 2):
            common = sorted(set(errors[a]) & set(errors[b]))
            if not common:
                continue
            xa = np.array([errors[a][q] for q in common])
            xb = np.array([errors[b][q] for q in common])
            rows.append(
                {"seed": seed, "agent_a": a, "agent_b": b, "rho": _pearson(xa, xb)}
            )
    return pd.DataFrame(rows, columns=["seed", "agent_a", "agent_b", "rho"])


@dataclass
class RhoSummary:
    """Team-rho reduced for the H2 phase map: the per-seed mean over agent
    pairs, and the mean +/- std of those per-seed means across seeds."""

    per_seed_mean: dict[int, float]
    mean: float
    std: float


def team_rho_summary(cond: ConditionData) -> RhoSummary:
    """Reduce :func:`team_rho`: per seed, the mean rho over its agent pairs;
    then the mean and (population) std of those per-seed means over seeds.
    ``nan`` pair correlations are ignored in the per-seed mean."""
    df = team_rho(cond)
    per_seed: dict[int, float] = {}
    for seed, grp in df.groupby("seed"):
        vals = grp["rho"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        per_seed[int(seed)] = float(vals.mean()) if vals.size else float("nan")
    means = np.array(list(per_seed.values()), dtype=float)
    means = means[~np.isnan(means)]
    overall_mean = float(means.mean()) if means.size else float("nan")
    overall_std = float(means.std()) if means.size else float("nan")
    return RhoSummary(per_seed_mean=per_seed, mean=overall_mean, std=overall_std)


# ---------------------------------------------------------------------------
# Statistics helpers (pure functions on numpy arrays)
# ---------------------------------------------------------------------------


def paired_bootstrap(
    d: np.ndarray,
    question_ids: Sequence,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster (question-level) bootstrap of the mean paired difference.

    ``d`` holds the paired differences over ``(question, seed)`` units;
    ``question_ids`` is the parallel array giving each element's question id.
    The resampling unit is the QUESTION: a resampled question contributes all
    of its seeds' differences, so seed-pairing is respected. Returns
    ``(observed_mean, lo95, hi95)`` where the CI is the 2.5/97.5 percentiles of
    the bootstrap distribution of the mean. Deterministic in ``seed``
    (``numpy.random.default_rng``)."""
    d = np.asarray(d, dtype=float)
    q = np.asarray(question_ids)
    if d.shape[0] != q.shape[0]:
        raise ValueError("d and question_ids must have the same length")
    if d.size == 0:
        raise ValueError("paired_bootstrap: empty difference vector")

    uniq = np.unique(q)
    # Per-question sum and count of differences: a resample's mean is
    # sum(chosen question sums) / sum(chosen question counts).
    q_sum = np.array([d[q == u].sum() for u in uniq], dtype=float)
    q_cnt = np.array([(q == u).sum() for u in uniq], dtype=float)

    rng = np.random.default_rng(seed)
    m = uniq.shape[0]
    chosen = rng.integers(0, m, size=(n_boot, m))
    boot_sum = q_sum[chosen].sum(axis=1)
    boot_cnt = q_cnt[chosen].sum(axis=1)
    boot_means = boot_sum / boot_cnt

    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return float(d.mean()), lo, hi


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks of ``a`` (1-based) with ties assigned their average rank."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=float)
    sorted_a = a[order]
    i = 0
    n = a.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # mean of 0-based positions i..j, +1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def wilcoxon_signed_rank(d: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test via the normal approximation.

    Drops zeros; ranks ``|d|`` with average ranks for ties; ``W`` is the sum of
    ranks of the positive differences. The two-sided p-value uses the normal
    approximation with the tie-corrected variance
    ``sigma^2 = n(n+1)(2n+1)/24 - sum(t^3 - t)/48`` (``t`` the size of each tie
    group of ``|d|``, ``n`` the count after dropping zeros) and a 0.5 continuity
    correction toward the mean ``mu = n(n+1)/4``. Returns ``(W, p)``; an all-zero
    (or empty) input returns ``(0.0, 1.0)``."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0.0]  # drop zeros
    n = d.shape[0]
    if n == 0:
        return 0.0, 1.0

    absd = np.abs(d)
    ranks = _average_ranks(absd)
    W = float(ranks[d > 0.0].sum())

    mu = n * (n + 1) / 4.0
    _vals, counts = np.unique(absd, return_counts=True)
    tie_sum = float(np.sum(counts.astype(float) ** 3 - counts.astype(float)))
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_sum / 48.0
    if var <= 0.0:
        return W, 1.0

    sigma = sqrt(var)
    diff = W - mu
    cc = 0.5 if diff > 0 else (-0.5 if diff < 0 else 0.0)  # continuity correction
    z = (diff - cc) / sigma
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return W, min(1.0, p)


@dataclass
class ComparisonResult:
    """Result of a paired comparison of two metric columns (``col_a - col_b``):
    the mean paired difference, per-seed means, cluster-bootstrap 95% CI,
    Wilcoxon statistic + two-sided p, and the number of paired units."""

    col_a: str
    col_b: str
    mean_diff: float
    per_seed_means: dict[int, float]
    ci: tuple[float, float]
    wilcoxon_W: float
    wilcoxon_p: float
    n: int


def compare(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> ComparisonResult:
    """Paired comparison of two ``question_metrics`` columns over their shared
    ``(seed, question_id)`` rows. The difference is ``col_a - col_b`` (positive
    => ``col_a`` is larger, e.g. ``compare(df, "market_gap", "mean_gap")`` > 0
    means the market has the larger posterior gap). Combines
    :func:`paired_bootstrap` (question-clustered CI) and
    :func:`wilcoxon_signed_rank`."""
    d = (df[col_a] - df[col_b]).to_numpy(dtype=float)
    qids = df["question_id"].to_numpy()
    seeds = df["seed"].to_numpy()

    per_seed = {
        int(s): float(d[seeds == s].mean()) for s in sorted(set(seeds.tolist()))
    }
    _mean, lo, hi = paired_bootstrap(d, qids, n_boot=n_boot, seed=seed)
    W, p = wilcoxon_signed_rank(d)
    return ComparisonResult(
        col_a=col_a,
        col_b=col_b,
        mean_diff=float(d.mean()),
        per_seed_means=per_seed,
        ci=(lo, hi),
        wilcoxon_W=W,
        wilcoxon_p=p,
        n=int(d.shape[0]),
    )

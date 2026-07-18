"""Pnyx P5 tables — Table 1 (accuracy-per-dollar) and Table 2 (parse failures
+ market-maker subsidy), plus ``stats_report``: every pre-registered H1/H2/H3/
H-wealth comparison in one markdown document.

Every number here is either read straight off a ``pnyx.analysis.mainrun``
``question_metrics`` DataFrame column, a straightforward aggregation of one
(a per-seed mean, a grand mean/std of those per-seed means, a sum/max), or a
verbatim consumption of a Task 2/3 statistics call (``mainrun.compare``'s
``ComparisonResult``, ``mainrun.team_rho_summary``'s ``RhoSummary``,
``manipulation.summarize_adversary``'s DataFrame, ``manipulation.
wealth_order_effects``'s comparison dict). Nothing here reimplements a
posterior gap, a Bayes oracle lookup, an adversary metric, or the paired
bootstrap/Wilcoxon machinery — the one exception is H2b's Spearman rank
correlation, which the brief asks to be hand-rolled here (Pearson on average
ranks, reusing ``mainrun.wilcoxon_signed_rank``'s tie-averaging *pattern*,
not its private helper — see ``_average_ranks`` below) rather than importing
scipy or reaching into ``mainrun``'s underscore-prefixed internals.

Design: ``table1`` / ``table2`` / ``stats_report`` are pure functions
(DataFrames/dicts in, a markdown ``str`` out) — no filesystem access. Per
Global Constraints ("I/O only in the CLI layer and the file-writing
helpers"), reading event logs, cost ledgers, and config YAMLs is entirely
the CLI's (``pnyx.cli``'s ``analyze-main`` subcommand) job; it assembles the
``metrics_by_cond`` / ``costs_by_cond`` / ``events_by_cond`` /
``model_by_cond`` inputs these functions consume, then writes the returned
strings to disk itself. This mirrors ``pnyx.analysis.figures``'s
data-prep/plot split, adapted to markdown: there is no separate "data prep"
struct because a markdown table's cells ARE the presentation layer, so the
one function already returns the final string.

``table2``'s signature deliberately has a THIRD parameter
(``model_by_cond: dict[str, dict[str, str]]``) beyond the two
(``metrics_by_cond``, ``events_by_cond``) the brief's one-line signature
sketch names. The brief's own binding definition ("model resolved from the
condition's config YAML agent→model mapping, loaded via
``pnyx.cli.load_config``") requires a YAML load (I/O) that this module must
not perform itself; ``events_by_cond`` (plain ``mainrun.ConditionData``,
which carries agent_ids but no model identity) has nowhere to carry that
mapping, so it is passed as its own argument, built by the CLI via
``pnyx.runner.load_config`` (the same function ``pnyx.cli`` imports and
re-exposes as ``pnyx.cli.load_config``; importing it from ``pnyx.runner``
directly here avoids a ``pnyx.cli`` <-> ``pnyx.analysis.tables`` import
cycle, since ``pnyx.cli`` imports this module).

Determinism: every ``mainrun.compare`` call in ``stats_report`` takes the
caller-supplied ``bootstrap_seed`` (default 0, threaded from the CLI's
``--bootstrap-seed``); no other randomness anywhere in this module. Two
calls with the same inputs and seed produce byte-identical markdown.

Fixed condition order (matches the P4 main-run grid, CLAUDE.md §7):
A, B1, B3, C, D_K1, D_K3, D_K10, W_FIXED, W_SHUFFLED. Condition "B2" of the
pre-registration is condition A (same pool, run once) — noted in Table 1's
caption and used verbatim as an H2a homogeneous arm.

stdlib + numpy + pandas + pydantic only (pydantic only transitively, via the
schemas ``ConditionData`` wraps) — no scipy/sklearn.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from pnyx.analysis.mainrun import ComparisonResult, ConditionData, RhoSummary, compare

__all__ = ["table1", "table2", "stats_report", "PASS1_SOURCE", "MAIN_CONDITIONS"]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Canonical condition order for every table (CLAUDE.md §7's condition grid).
# Public (not underscore-prefixed) so the CLI orchestrator (pnyx.cli's
# analyze-main) can drive its own load_condition/question_metrics loop over
# the SAME nine names rather than maintaining a second copy.
MAIN_CONDITIONS: tuple[str, ...] = (
    "A", "B1", "B3", "C", "D_K1", "D_K3", "D_K10", "W_FIXED", "W_SHUFFLED",
)
_TABLE_CONDITIONS = MAIN_CONDITIONS

# Pass-1 source per pass2-only condition (D_K* reuse C's independent beliefs;
# W_* reuse A's) — CLAUDE.md §7 / the runner's ``pass1_source_dir`` wiring.
# Table 1 uses this to know which rows' mean-pool columns are bit-identical
# to another row (and hence collapsed to "—" rather than repeated).
PASS1_SOURCE: dict[str, str] = {
    "D_K1": "C", "D_K3": "C", "D_K10": "C",
    "W_FIXED": "A", "W_SHUFFLED": "A",
}

_UNIFORM_GUESS_BRIER = 0.25  # Brier of the constant p=0.5 forecast.
_LMSR_B = 40.0  # frozen after the P3 pilot (CLAUDE.md §6.1 / §8 P3).
_SUBSIDY_BOUND = _LMSR_B * math.log(2.0)  # worst-case per-question subsidy, ~27.7259.


def _ordered_conditions(keys: Iterable[str]) -> list[str]:
    """``keys`` reordered to the canonical condition order, with any
    condition not in that fixed list appended alphabetically after (keeps
    small ad-hoc test fixtures — which may not use the real condition names
    — deterministic too)."""
    keys = set(keys)
    ordered = [c for c in _TABLE_CONDITIONS if c in keys]
    extra = sorted(keys - set(_TABLE_CONDITIONS))
    return ordered + extra


def _mean_std_over_seeds(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Grand mean and population std (``ddof=0``, matching
    ``mainrun.RhoSummary`` / ``figures.fig1_data``'s convention) of the
    per-seed means of ``col``."""
    per_seed = df.groupby("seed")[col].mean().to_numpy(dtype=float)
    return float(per_seed.mean()), float(per_seed.std())


# ---------------------------------------------------------------------------
# Table 1 — posterior gap / Brier / accuracy-per-dollar
# ---------------------------------------------------------------------------


def table1(
    metrics_by_cond: dict[str, pd.DataFrame],
    costs_by_cond: dict[str, float],
) -> str:
    """Markdown Table 1: one row per condition present in ``metrics_by_cond``
    (canonical order — see module docstring), each with the market's
    posterior gap and Brier (mean ± std over seeds), the mean-pool-of-
    independent-beliefs baseline's gap/Brier, that condition's own total
    cost (``costs_by_cond``, USD), and accuracy-per-dollar =
    ``(0.25 - mean market Brier) / total cost`` (0.25 = the uniform-guess
    Brier). A zero (or negative, defensively) cost prints
    ``"∞ (free tier)"`` for accuracy-per-dollar rather than dividing by
    zero (B3's nemotron-3-super-120b:free pool costs $0.00).

    A pass2-only condition's (D_K*/W_*) mean-pool columns are bit-identical
    to their Pass-1 source's own row by construction (``mainrun.
    question_metrics``'s ``pass1_source`` reuse) — printed as ``"— (†)"`` /
    ``"— (‡)"`` with a footnote naming the source condition, rather than
    repeating the same numbers, per the brief.
    """
    lines = [
        "## Table 1 — Posterior Gap, Brier, and Accuracy-per-Dollar",
        "",
        (
            "*Market vs. the mean-pool-of-independent-beliefs baseline, "
            "mean ± std over seeds. Accuracy-per-dollar = (0.25 − mean "
            "market Brier) / total cost USD; 0.25 is the uniform-guess "
            "(p=0.5) Brier. Condition \"B2\" of the pre-registration is "
            "condition A (identical pool, run once). D_K\\* and W_\\* are "
            "pass2-only: their mean-pool columns are bit-identical to their "
            "Pass-1 source by construction and are shown as \"—\" "
            "(marked †/‡) to avoid duplicate reporting — see the source "
            "condition's own row below.*"
        ),
        "",
        "| Condition | Market gap | Market Brier | Mean-pool gap | "
        "Mean-pool Brier | Total cost | Accuracy/$ |",
        "|---|---|---|---|---|---|---|",
    ]

    footnote_marks: dict[str, str] = {}
    for cond in _ordered_conditions(metrics_by_cond):
        df = metrics_by_cond[cond]
        gap_mean, gap_std = _mean_std_over_seeds(df, "market_gap")
        brier_mean, brier_std = _mean_std_over_seeds(df, "market_brier")

        source = PASS1_SOURCE.get(cond)
        if source is not None:
            mark = "†" if source == "C" else "‡"
            footnote_marks[mark] = source
            mean_gap_cell = f"— ({mark})"
            mean_brier_cell = f"— ({mark})"
        else:
            mg_mean, mg_std = _mean_std_over_seeds(df, "mean_gap")
            mb_mean, mb_std = _mean_std_over_seeds(df, "mean_brier")
            mean_gap_cell = f"{mg_mean:.3f} ± {mg_std:.3f}"
            mean_brier_cell = f"{mb_mean:.3f} ± {mb_std:.3f}"

        cost = costs_by_cond.get(cond, 0.0)
        acc_per_dollar = (
            "∞ (free tier)"
            if cost <= 0.0
            else f"{(_UNIFORM_GUESS_BRIER - brier_mean) / cost:.3f}"
        )

        lines.append(
            f"| {cond} | {gap_mean:.3f} ± {gap_std:.3f} | "
            f"{brier_mean:.3f} ± {brier_std:.3f} | {mean_gap_cell} | "
            f"{mean_brier_cell} | ${cost:.4f} | {acc_per_dollar} |"
        )

    if footnote_marks:
        lines.append("")
        for mark in sorted(footnote_marks):
            lines.append(
                f"{mark} pass2-only condition; mean-pool columns are "
                f"bit-identical to (reused from) condition "
                f"{footnote_marks[mark]}'s own row above."
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 2 — parse failures per (condition, model) + market-maker subsidy
# ---------------------------------------------------------------------------


def _model_turn_counts(
    cond: ConditionData, model_map: dict[str, str]
) -> dict[str, tuple[int, int]]:
    """``model_id -> (n_parse_failed, n_total)`` over every belief + trade
    event of ``cond`` (its own log only — a pass2-only condition therefore
    counts only its Pass-2 trade turns, since its Pass-1 beliefs live in the
    source condition's own log/row), resolved via ``model_map`` (that
    condition's config-YAML ``agent_id -> model_id`` mapping). An agent
    absent from ``model_map`` (e.g. a mock agent in a synthetic fixture) is
    silently skipped, mirroring ``pilot.analyze_run``'s "unmapped" handling.
    """
    totals: dict[str, int] = {}
    fails: dict[str, int] = {}
    for seed_data in cond.seeds.values():
        for e in list(seed_data.beliefs) + list(seed_data.trades):
            model_id = model_map.get(e.key.agent_id)
            if model_id is None:
                continue
            totals[model_id] = totals.get(model_id, 0) + 1
            if e.parse_failed:
                fails[model_id] = fails.get(model_id, 0) + 1
    return {m: (fails.get(m, 0), n) for m, n in totals.items()}


def table2(
    metrics_by_cond: dict[str, pd.DataFrame],
    events_by_cond: dict[str, ConditionData],
    model_by_cond: dict[str, dict[str, str]],
) -> str:
    """Markdown Table 2: (1) parse-failure count/rate per (condition,
    model) — denominator is that model's total belief+trade LLM turns in
    the condition's own logs, model resolved via ``model_by_cond`` (the
    condition's config-YAML agent→model mapping; see module docstring for
    why this is a separate parameter rather than an in-function YAML
    load); (2) per-condition market-maker subsidy (mean/total/max over every
    settled question, from the ``subsidy`` column ``mainrun.
    question_metrics`` already computes) vs. the theoretical worst-case
    bound ``b·ln 2 ≈ 27.73`` (``b=40``, frozen in P3).
    """
    lines = [
        "## Table 2 — Parse Failures and Market-Maker Subsidy",
        "",
        "### Parse-failure rate per (condition, model)",
        "",
        (
            "*Denominator = that model's total belief + trade LLM turns in "
            "the condition's own logs (post-retry, per Global Constraints "
            "§6.2). Model resolved via the condition's config YAML "
            "agent→model mapping.*"
        ),
        "",
        "| Condition | Model | Parse-fails | Total turns | Rate |",
        "|---|---|---|---|---|",
    ]

    pf_conditions = _ordered_conditions(set(events_by_cond) & set(model_by_cond))
    for cond in pf_conditions:
        counts = _model_turn_counts(events_by_cond[cond], model_by_cond[cond])
        for model_id in sorted(counts):
            n_fail, n_total = counts[model_id]
            rate = n_fail / n_total if n_total else 0.0
            lines.append(
                f"| {cond} | {model_id} | {n_fail} | {n_total} | {rate:.4f} |"
            )

    lines += [
        "",
        "### Market-maker subsidy per condition",
        "",
        f"*Theoretical worst-case bound b·ln 2 ≈ {_SUBSIDY_BOUND:.2f} "
        f"(b={_LMSR_B:.0f}).*",
        "",
        "| Condition | Mean subsidy | Total subsidy | Max subsidy | Bound (b·ln2) |",
        "|---|---|---|---|---|",
    ]
    for cond in _ordered_conditions(metrics_by_cond):
        subsidy = metrics_by_cond[cond]["subsidy"]
        lines.append(
            f"| {cond} | {subsidy.mean():.3f} | {subsidy.sum():.3f} | "
            f"{subsidy.max():.3f} | {_SUBSIDY_BOUND:.2f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# stats_report — every pre-registered H1/H2/H3/H-wealth comparison
# ---------------------------------------------------------------------------

_POOLS: tuple[str, ...] = ("mean", "median", "lop", "stack")
_POOL_LABEL: dict[str, str] = {
    "mean": "Mean pool", "median": "Median pool",
    "lop": "Log opinion pool", "stack": "Calibrated stack",
}


def _fmt_comparison(label: str, cmp: ComparisonResult) -> str:
    """The binding headline format: ``Δ = x.xxx [lo, hi] (95% bootstrap
    CI), Wilcoxon p = 0.xxx, per-seed means: s0/s1/s2`` (per-seed means in
    ascending seed order, slash-joined)."""
    lo, hi = cmp.ci
    per_seed = "/".join(
        f"{cmp.per_seed_means[s]:.3f}" for s in sorted(cmp.per_seed_means)
    )
    return (
        f"- **{label}**: Δ = {cmp.mean_diff:.3f} [{lo:.3f}, {hi:.3f}] "
        f"(95% bootstrap CI), Wilcoxon p = {cmp.wilcoxon_p:.3f}, "
        f"per-seed means: {per_seed}"
    )


def _h1_section(metrics_a: pd.DataFrame, n_boot: int, bootstrap_seed: int) -> list[str]:
    """H1: market vs. each of the 4 static pools on condition A, for both
    the posterior gap and the Brier score (8 comparisons)."""
    lines = ["## H1: Market vs. Static Pools (Condition A)", ""]
    for metric, metric_label in (("gap", "posterior gap"), ("brier", "Brier")):
        lines.append(f"### Market vs. static pools — {metric_label}")
        for pool in _POOLS:
            cmp = compare(
                metrics_a, f"market_{metric}", f"{pool}_{metric}",
                n_boot=n_boot, seed=bootstrap_seed,
            )
            lines.append(
                _fmt_comparison(
                    f"market − {_POOL_LABEL[pool]} ({metric_label})", cmp
                )
            )
        lines.append("")
    return lines


def _paired_merge(df_x: pd.DataFrame, df_y: pd.DataFrame, col: str) -> pd.DataFrame:
    """Inner-join ``df_x``/``df_y`` on ``(seed, question_id)``, keeping only
    ``col`` from each (suffixed ``_x``/``_y``) — the shared-question pairing
    H2a needs to compare two DIFFERENT conditions' market_gap column."""
    merged = df_x[["seed", "question_id", col]].merge(
        df_y[["seed", "question_id", col]],
        on=["seed", "question_id"], suffixes=("_x", "_y"),
    )
    if merged.empty:
        raise ValueError(
            f"no shared (seed, question_id) rows to compare {col!r} on"
        )
    return merged


def _h2a_section(
    c_df: pd.DataFrame,
    homogeneous: dict[str, pd.DataFrame],
    n_boot: int,
    bootstrap_seed: int,
) -> list[str]:
    """H2a: mixed team C's market_gap vs. each homogeneous arm's market_gap,
    paired on the shared ``(seed, question_id)`` (same questions by
    construction — every condition runs the same 40-question file)."""
    lines = ["## H2a: Mixed Team (C) vs. Homogeneous Arms", ""]
    for label, h_df in homogeneous.items():
        merged = _paired_merge(c_df, h_df, "market_gap")
        cmp = compare(merged, "market_gap_x", "market_gap_y", n_boot=n_boot, seed=bootstrap_seed)
        lines.append(_fmt_comparison(f"C − {label} (market_gap)", cmp))
    lines.append("")
    return lines


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Ranks of ``a`` (1-based) with ties assigned their average rank —
    same tie-averaging PATTERN as ``mainrun.wilcoxon_signed_rank``'s rank
    helper, hand-rolled here rather than imported (that helper is a private,
    unexported module internal of ``mainrun``; the brief asks for a
    hand-rolled Spearman, not a reach into another module's underscore
    names)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=float)
    sorted_a = a[order]
    i = 0
    n = a.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation = Pearson correlation of the average ranks
    of ``x`` and ``y``. Returns ``nan`` when either has zero rank-variance
    (e.g. fewer than 2 points, or every value tied)."""
    if x.shape[0] < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    xm = rx - rx.mean()
    ym = ry - ry.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xm * ym).sum() / denom)


_PHASE_MAP_CONDITIONS: tuple[str, ...] = ("A", "B1", "B3", "C")


def _h2b_section(
    rho_by_cond: dict[str, RhoSummary],
    metrics_by_cond: dict[str, pd.DataFrame],
) -> list[str]:
    """H2b: lists every (condition, seed) -> (ρ, market_gap − mean_gap)
    point (the phase-map's own data, ``figures.fig2_data``'s y-axis
    definition) across A/B1/B3/C, then the hand-rolled Spearman rank
    correlation between ρ and the gap difference across all of them
    (n = 4 conditions × seeds; n=12 at 3 seeds → explicitly flagged as
    descriptive, not a hypothesis test, per the brief)."""
    conds = [
        c for c in _PHASE_MAP_CONDITIONS
        if c in rho_by_cond and c in metrics_by_cond
    ]
    lines = [
        "## H2b: Team ρ vs. Market Advantage (Phase-Map Correlation)",
        "",
        "| Condition | Seed | ρ | market_gap − mean_gap |",
        "|---|---|---|---|",
    ]
    rhos: list[float] = []
    gapdiffs: list[float] = []
    for cond in conds:
        rs = rho_by_cond[cond]
        df = metrics_by_cond[cond]
        diff_by_seed = (df["market_gap"] - df["mean_gap"]).groupby(df["seed"]).mean()
        for seed in sorted(rs.per_seed_mean):
            rho_val = rs.per_seed_mean[seed]
            gapdiff = float(diff_by_seed.get(seed, float("nan")))
            lines.append(f"| {cond} | {seed} | {rho_val:.3f} | {gapdiff:.3f} |")
            rhos.append(rho_val)
            gapdiffs.append(gapdiff)

    n = len(rhos)
    spearman = _spearman(np.array(rhos, dtype=float), np.array(gapdiffs, dtype=float))
    lines.append("")
    lines.append(
        f"Spearman rank correlation (ρ vs. market_gap − mean_gap) across "
        f"n={n} (condition, seed) points: r_s = {spearman:.3f} "
        f"(n={n} → descriptive only, not a hypothesis test)."
    )
    lines.append("")
    return lines


def _h3_section(adv_summary: pd.DataFrame) -> list[str]:
    """H3: ``manipulation.summarize_adversary``'s output rendered verbatim
    (its own columns, one row per k, in its own sorted-by-k order).

    Indexes column-by-column (``adv_summary[c].iloc[i]``) rather than via
    ``DataFrame.iterrows()``: ``iterrows()`` returns each row as a Series,
    which — when a DataFrame mixes int and float columns (``k`` is an int,
    every other column a float) — silently upcasts the WHOLE row to a
    common float dtype, turning ``k=1`` into ``1.0`` and defeating the
    int/float formatting branch below. Column-wise indexing preserves each
    column's own dtype.
    """
    lines = ["## H3: Adversary Manipulation", ""]
    cols = list(adv_summary.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for i in range(len(adv_summary)):
        cells = []
        for c in cols:
            v = adv_summary[c].iloc[i]
            if isinstance(v, float):
                cells.append("nan" if math.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


_WEALTH_LABEL: dict[str, str] = {
    "wfixed_vs_a": "W_FIXED − A",
    "wshuffled_vs_a": "W_SHUFFLED − A",
    "wfixed_vs_wshuffled": "W_FIXED − W_SHUFFLED",
}
_WEALTH_KEYS: tuple[str, ...] = ("wfixed_vs_a", "wshuffled_vs_a", "wfixed_vs_wshuffled")
_WEALTH_METRICS: tuple[str, ...] = ("market_gap", "market_brier")


def _h_wealth_section(
    wealth_effects: dict[str, dict[str, ComparisonResult]],
) -> list[str]:
    """H-wealth: the three ``manipulation.wealth_order_effects`` comparisons
    (wfixed_vs_a, wshuffled_vs_a, wfixed_vs_wshuffled), each on market_gap
    and market_brier — 6 comparisons total, rendered verbatim."""
    lines = ["## H-wealth: Persistent Wealth Order Effects", ""]
    for key in _WEALTH_KEYS:
        comp = wealth_effects.get(key)
        if comp is None:
            continue
        label = _WEALTH_LABEL[key]
        for metric in _WEALTH_METRICS:
            lines.append(_fmt_comparison(f"{label} ({metric})", comp[metric]))
    lines.append("")
    return lines


def stats_report(
    metrics_by_cond: dict[str, pd.DataFrame],
    rho_by_cond: dict[str, RhoSummary],
    adv_summary: pd.DataFrame,
    wealth_effects: dict[str, dict[str, ComparisonResult]],
    *,
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
) -> str:
    """The full pre-registered statistics report (§2): H1, H2a, H2b, H3,
    H-wealth, in that order, each under its own ``## H...`` header.

    ``metrics_by_cond`` must contain at least ``"A"`` (H1, and an H2a
    homogeneous arm as "B2") and ``"C"`` (H2a's mixed team, H2b); ``"B1"``/
    ``"B3"`` are included in H2a/H2b when present. ``rho_by_cond`` maps
    A/B1/B3/C to their ``mainrun.team_rho_summary`` output. ``adv_summary``
    is ``manipulation.summarize_adversary``'s DataFrame (H3, verbatim).
    ``wealth_effects`` is ``manipulation.wealth_order_effects``'s output
    (H-wealth, verbatim) — the CALLER must have computed it with the SAME
    ``bootstrap_seed``/``n_boot`` passed here, so every comparison in the
    report shares one deterministic bootstrap draw.

    Every ``mainrun.compare`` call this function makes directly (H1, H2a)
    uses ``n_boot``/``bootstrap_seed`` — determinism: identical inputs and
    seed produce byte-identical markdown.
    """
    if "A" not in metrics_by_cond:
        raise ValueError("stats_report requires condition 'A' in metrics_by_cond (H1)")
    if "C" not in metrics_by_cond:
        raise ValueError("stats_report requires condition 'C' in metrics_by_cond (H2a/H2b)")

    homogeneous: dict[str, pd.DataFrame] = {"A (B2)": metrics_by_cond["A"]}
    for cond in ("B1", "B3"):
        if cond in metrics_by_cond:
            homogeneous[cond] = metrics_by_cond[cond]

    lines = ["# Pnyx P5 Statistics Report", ""]
    lines += _h1_section(metrics_by_cond["A"], n_boot, bootstrap_seed)
    lines += _h2a_section(metrics_by_cond["C"], homogeneous, n_boot, bootstrap_seed)
    lines += _h2b_section(rho_by_cond, metrics_by_cond)
    lines += _h3_section(adv_summary)
    lines += _h_wealth_section(wealth_effects)
    return "\n".join(lines) + "\n"

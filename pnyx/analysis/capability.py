"""Pnyx v2 — capability-tier comparison + herding decomposition.

This module extends the P5 analysis pipeline (``pnyx.analysis.mainrun``) with
the v2 "capability axis": three homogeneous same-question/same-seed conditions
run at increasing model capability — ``flash`` (condition A, deepseek-v4-flash),
``pro`` (A_PRO, deepseek-v4-pro, the same-family knowledge-vs-trading control),
and ``luna`` (A_LUNA, gpt-5.6-luna). It answers two questions:

1. **Does the reversed H1 (market destroys information) move with capability?**
   ``tier_metrics`` / ``cross_tier_compare`` reduce each tier to its market vs.
   mean-pool deficit and pair the tiers on shared ``(seed, question)`` — a thin
   orchestration over ``mainrun.question_metrics`` / ``mainrun.compare``; no
   metric is reimplemented here.
2. **What is the mechanism?** ``herding_weights`` regresses each honest agent's
   in-market stated belief on its own Pass-1 belief and the pre-trade price
   (``y = a + b1·x1 + b2·x2``, OLS via ``numpy.linalg.lstsq``). A falling price
   weight ``b2`` with capability is the "capable traders herd less on the price"
   evidence.

Phase-guard discipline (Global Constraints, extended for v2)
-----------------------------------------------------------
The herding regression is the spec's sanctioned *dynamics analysis* of
in-market beliefs. It reads ``TradeEvent.trade.belief`` (the market-phase
stated belief) as its regression target — this is the ONE place a market-phase
belief is read on purpose. It is therefore documented as **dynamics-only** and
is deliberately kept free of ``pnyx.baselines``: this module imports nothing
from ``pnyx.baselines``, never calls ``assert_pass1`` on the market path, and
never feeds a market belief into a static pool or a ρ estimate. The own-Pass-1
regressor ``x1`` is read only from genuine ``phase=="independent"`` belief
events (filtered here directly), so no market belief can masquerade as Pass-1.
Static baselines and ρ continue to flow exclusively through
``mainrun.question_metrics`` / ``mainrun.team_rho_summary`` (which keep their
own ``assert_pass1`` guard) — this module only consumes their outputs.

Purity: every function here is pure (loaded data / DataFrames in, DataFrame or
markdown ``str`` out). Filesystem I/O is entirely the CLI's job
(``pnyx.cli``'s ``analyze-v2`` subcommand). Dependencies: stdlib + numpy +
pandas + the ``mainrun`` helpers (no scipy/sklearn).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pnyx.analysis.mainrun import (
    HONEST_AGENTS,
    ComparisonResult,
    ConditionData,
    compare,
    question_metrics,
    team_rho_summary,
)

__all__ = [
    "TIER_ORDER",
    "tier_metrics",
    "cross_tier_compare",
    "cross_tier_comparisons",
    "herding_weights",
    "render_tiers_md",
    "render_herding_md",
]

# Fixed tier order everywhere (increasing capability). The v2 conditions map
# flash->A, pro->A_PRO, luna->A_LUNA; this module works purely on the tier
# labels a caller supplies, so the mapping lives in the CLI.
TIER_ORDER: tuple[str, ...] = ("flash", "pro", "luna")

_HONEST_SET = frozenset(HONEST_AGENTS)

# The cross-tier paired comparisons reported in the v2 plan (Context block):
# directional pairs (tier_a - tier_b) on the market gap and on the pool gap.
_CROSS_MARKET_PAIRS: tuple[tuple[str, str], ...] = (
    ("pro", "flash"), ("luna", "pro"), ("luna", "flash"),
)
_CROSS_POOL_PAIRS: tuple[tuple[str, str], ...] = (
    ("pro", "flash"), ("luna", "flash"),
)


# ---------------------------------------------------------------------------
# Tier metrics
# ---------------------------------------------------------------------------


def _grand_mean(df: pd.DataFrame, col: str) -> float:
    """Grand mean of the per-seed means of ``col`` (matches the P5 convention
    used by ``figures.fig1_data`` / ``tables._mean_std_over_seeds``)."""
    return float(df.groupby("seed")[col].mean().mean())


def tier_metrics(
    conds: dict[str, ConditionData],
    *,
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """One row per tier (in ``conds`` iteration order) summarising the market
    vs. mean-pool comparison.

    Columns: ``tier``; ``pool_gap`` (grand-mean mean-pool posterior gap),
    ``market_gap`` (grand-mean market posterior gap), ``market_brier``;
    ``deficit`` = the within-tier ``compare(market_gap, mean_gap)`` mean paired
    difference, with its cluster-bootstrap CI ``deficit_ci_lo`` /
    ``deficit_ci_hi`` and Wilcoxon ``deficit_p`` (the SAME within-tier
    ``mainrun.compare`` call Fig 3(a)'s error bar consumes); ``rho_mean`` /
    ``rho_std`` from ``mainrun.team_rho_summary``; ``n`` = number of
    ``(seed, question)`` units.

    Pure orchestration: no posterior gap, Brier, bootstrap, or Wilcoxon is
    recomputed here — every number is read off ``mainrun.question_metrics`` /
    ``mainrun.compare`` / ``mainrun.team_rho_summary``.
    """
    rows: list[dict] = []
    for tier, cond in conds.items():
        qm = question_metrics(cond, None)
        deficit = compare(qm, "market_gap", "mean_gap",
                          n_boot=n_boot, seed=bootstrap_seed)
        rho = team_rho_summary(cond)
        rows.append({
            "tier": tier,
            "pool_gap": _grand_mean(qm, "mean_gap"),
            "market_gap": _grand_mean(qm, "market_gap"),
            "market_brier": _grand_mean(qm, "market_brier"),
            "deficit": deficit.mean_diff,
            "deficit_ci_lo": deficit.ci[0],
            "deficit_ci_hi": deficit.ci[1],
            "deficit_p": deficit.wilcoxon_p,
            "rho_mean": rho.mean,
            "rho_std": rho.std,
            "n": int(len(qm)),
        })
    return pd.DataFrame(rows, columns=[
        "tier", "pool_gap", "market_gap", "market_brier",
        "deficit", "deficit_ci_lo", "deficit_ci_hi", "deficit_p",
        "rho_mean", "rho_std", "n",
    ])


def cross_tier_compare(
    mets_a: pd.DataFrame,
    mets_b: pd.DataFrame,
    col: str,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> ComparisonResult:
    """Paired comparison of ``col`` between two tiers' ``question_metrics``
    frames, ``tier_a − tier_b``, paired on the shared ``(seed, question_id)``
    (the same questions/seeds run for every tier by construction). Merges the
    two frames and delegates to ``mainrun.compare`` — no new stats code."""
    merged = mets_a[["seed", "question_id", col]].merge(
        mets_b[["seed", "question_id", col]],
        on=["seed", "question_id"], suffixes=("_x", "_y"),
    )
    if merged.empty:
        raise ValueError(f"no shared (seed, question_id) rows to compare {col!r} on")
    return compare(merged, f"{col}_x", f"{col}_y", n_boot=n_boot, seed=seed)


def cross_tier_comparisons(
    mets: dict[str, pd.DataFrame],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[tuple[str, str, str, ComparisonResult]]:
    """The v2 plan's cross-tier comparisons as an ordered list of
    ``(col, tier_a, tier_b, ComparisonResult)`` (tier_a − tier_b): market gap
    for pro−flash, luna−pro, luna−flash; then pool (mean) gap for pro−flash,
    luna−flash. Tiers absent from ``mets`` are skipped."""
    out: list[tuple[str, str, str, ComparisonResult]] = []
    for col, pairs in (("market_gap", _CROSS_MARKET_PAIRS),
                       ("mean_gap", _CROSS_POOL_PAIRS)):
        for a, b in pairs:
            if a not in mets or b not in mets:
                continue
            out.append((col, a, b,
                        cross_tier_compare(mets[a], mets[b], col,
                                           n_boot=n_boot, seed=seed)))
    return out


# ---------------------------------------------------------------------------
# Herding regression (dynamics-only — reads market-phase trade.belief)
# ---------------------------------------------------------------------------


def _honest_pass1_probs(cond: ConditionData) -> dict[tuple[int, str, str], float]:
    """``(seed, question_id, agent_id) -> own Pass-1 prob`` for honest agents,
    read ONLY from genuine ``phase=="independent"`` belief events. This is the
    ``x1`` regressor for the herding OLS. Deliberately does not touch
    ``pnyx.baselines`` (see module docstring) — the phase filter here is a
    plain equality check, so no market-phase belief can leak in as ``x1``."""
    out: dict[tuple[int, str, str], float] = {}
    for seed, sd in cond.seeds.items():
        for e in sd.beliefs:
            if e.key.phase != "independent":
                continue
            if e.key.agent_id not in _HONEST_SET:
                continue
            out[(seed, e.key.question_id, e.key.agent_id)] = e.belief.prob
    return out


def _herding_rows(cond: ConditionData) -> pd.DataFrame:
    """Long-format regression inputs for one condition: one row per honest,
    non-parse-failed market trade that has a matching own Pass-1 belief.

    Columns: ``seed``, ``question_id``, ``round``, ``agent_id``, ``x1`` (own
    Pass-1 prob), ``x2`` (pre-trade price ``price_before``), ``y`` (in-market
    stated belief ``trade.belief`` — the dynamics-only read)."""
    pass1 = _honest_pass1_probs(cond)
    rows: list[dict] = []
    for seed, sd in cond.seeds.items():
        for t in sd.trades:
            if t.parse_failed:
                continue
            aid = t.key.agent_id
            if aid not in _HONEST_SET:
                continue
            x1 = pass1.get((seed, t.key.question_id, aid))
            if x1 is None:
                continue
            rows.append({
                "seed": seed,
                "question_id": t.key.question_id,
                "round": t.key.round,
                "agent_id": aid,
                "x1": x1,
                "x2": t.price_before,
                "y": t.trade.belief,
            })
    return pd.DataFrame(
        rows, columns=["seed", "question_id", "round", "agent_id", "x1", "x2", "y"]
    )


def _fit_ols(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """OLS ``y = a + b1·x1 + b2·x2`` via ``numpy.linalg.lstsq``. Returns
    ``(intercept, b1, b2, r2)``; ``r2`` is ``nan`` when ``y`` has zero
    variance (no total sum of squares to explain)."""
    X = np.column_stack([np.ones(y.shape[0]), x1, x2])
    coef, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    intercept, b1, b2 = float(coef[0]), float(coef[1]), float(coef[2])
    fitted = X @ coef
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return intercept, b1, b2, r2


def _cluster_bootstrap_ci(
    df: pd.DataFrame, n_boot: int, seed: int
) -> tuple[float, float, float, float]:
    """Cluster (question-level) bootstrap 95% CI on ``b1`` and ``b2``:
    resample question ids with replacement, refit the OLS per draw, take the
    2.5/97.5 percentiles. Returns ``(b1_lo, b1_hi, b2_lo, b2_hi)``.
    Deterministic in ``seed`` (``numpy.random.default_rng``)."""
    qids = df["question_id"].to_numpy()
    x1 = df["x1"].to_numpy(dtype=float)
    x2 = df["x2"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    uniq = np.unique(qids)
    groups = {u: np.flatnonzero(qids == u) for u in uniq}
    m = uniq.shape[0]

    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, m, size=(n_boot, m))
    b1s = np.empty(n_boot, dtype=float)
    b2s = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.concatenate([groups[uniq[c]] for c in chosen[i]])
        _a, b1, b2, _r2 = _fit_ols(x1[idx], x2[idx], y[idx])
        b1s[i] = b1
        b2s[i] = b2
    return (
        float(np.percentile(b1s, 2.5)), float(np.percentile(b1s, 97.5)),
        float(np.percentile(b2s, 2.5)), float(np.percentile(b2s, 97.5)),
    )


_HERD_COLUMNS = [
    "scope", "b1", "b2", "intercept", "r2", "n",
    "b1_lo", "b1_hi", "b2_lo", "b2_hi", "drift",
]


def _herding_scope_row(scope: str, df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Fit the herding OLS on one scope's rows and package a result row. An
    empty scope yields an all-nan row (n=0) rather than raising."""
    n = int(len(df))
    if n == 0:
        return {"scope": scope, "b1": float("nan"), "b2": float("nan"),
                "intercept": float("nan"), "r2": float("nan"), "n": 0,
                "b1_lo": float("nan"), "b1_hi": float("nan"),
                "b2_lo": float("nan"), "b2_hi": float("nan"), "drift": float("nan")}
    x1 = df["x1"].to_numpy(dtype=float)
    x2 = df["x2"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    intercept, b1, b2, r2 = _fit_ols(x1, x2, y)
    b1_lo, b1_hi, b2_lo, b2_hi = _cluster_bootstrap_ci(df, n_boot, seed)
    drift = float(np.abs(y - x1).mean())
    return {"scope": scope, "b1": b1, "b2": b2, "intercept": intercept,
            "r2": r2, "n": n, "b1_lo": b1_lo, "b1_hi": b1_hi,
            "b2_lo": b2_lo, "b2_hi": b2_hi, "drift": drift}


def herding_weights(
    cond: ConditionData, *, n_boot: int = 2_000, seed: int = 0
) -> pd.DataFrame:
    """Herding regression for one condition (DYNAMICS-ONLY — reads market-phase
    ``trade.belief``; never feeds a baseline or ρ).

    For every honest, non-parse-failed market trade, regress the in-market
    stated belief ``y`` on the agent's own Pass-1 belief ``x1`` and the
    pre-trade price ``x2``: ``y = a + b1·x1 + b2·x2`` (OLS via
    ``numpy.linalg.lstsq``). Returns one row per scope — ``"pooled"`` (all
    rounds) then ``"1"``/``"2"``/``"3"`` (per market round) — with ``b1``
    (own-information weight), ``b2`` (price weight), ``intercept``, ``r2``,
    ``n``, the question-clustered bootstrap 95% CIs (``b1_lo``/``b1_hi``,
    ``b2_lo``/``b2_hi``; ``n_boot`` draws, ``seed``), and the descriptive drift
    ``mean|y − x1|``.

    Deterministic in ``seed``. Hypothesis the coefficients speak to: ``b2``
    falls (agents herd less on the price) as model capability rises.
    """
    rows_df = _herding_rows(cond)
    result_rows = [_herding_scope_row("pooled", rows_df, n_boot, seed)]
    for rnd in (1, 2, 3):
        scope_df = rows_df[rows_df["round"] == rnd]
        result_rows.append(_herding_scope_row(str(rnd), scope_df, n_boot, seed))
    return pd.DataFrame(result_rows, columns=_HERD_COLUMNS)


# ---------------------------------------------------------------------------
# Markdown renderers (byte-stable — no randomness beyond the seeded bootstrap)
# ---------------------------------------------------------------------------


def _fmt_comparison(label: str, cmp: ComparisonResult) -> str:
    """P5 headline stats format: ``Δ = x.xxx [lo, hi] (95% bootstrap CI),
    Wilcoxon p = 0.xxx, per-seed means: s0/s1/...`` (matches
    ``tables._fmt_comparison`` so v1 and v2 report identically)."""
    lo, hi = cmp.ci
    per_seed = "/".join(
        f"{cmp.per_seed_means[s]:.3f}" for s in sorted(cmp.per_seed_means)
    )
    return (
        f"- **{label}**: Δ = {cmp.mean_diff:.3f} [{lo:.3f}, {hi:.3f}] "
        f"(95% bootstrap CI), Wilcoxon p = {cmp.wilcoxon_p:.3f}, "
        f"per-seed means: {per_seed}"
    )


def render_tiers_md(
    conds: dict[str, ConditionData],
    *,
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
) -> str:
    """Render ``tiers.md``: the per-tier metrics table, the within-tier
    market-vs-mean-pool deficit (P5 stats format), and the cross-tier paired
    comparisons (market gap + pool gap). Deterministic for fixed inputs and
    ``bootstrap_seed`` (byte-stable)."""
    tiers_df = tier_metrics(conds, n_boot=n_boot, bootstrap_seed=bootstrap_seed)
    mets = {tier: question_metrics(cond, None) for tier, cond in conds.items()}
    cross = cross_tier_comparisons(mets, n_boot=n_boot, seed=bootstrap_seed)

    lines = [
        "# Pnyx v2 — Capability-Tier Analysis",
        "",
        "*Three homogeneous tiers on the same 40 questions × 3 seeds: "
        "flash (A), pro (A_PRO, same-family knowledge-vs-trading control), "
        "luna (A_LUNA). Primary metric = posterior gap vs. the Bayes oracle; "
        "deficit = market gap − mean-pool gap (positive ⇒ the market destroys "
        "information relative to independent pooling). Comparisons in the P5 "
        "stats format.*",
        "",
        "## Tier metrics",
        "",
        "| Tier | Pool gap | Market gap | Deficit (market − pool) | "
        "Market Brier | Team ρ | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for _i, row in tiers_df.iterrows():
        lines.append(
            f"| {row['tier']} | {row['pool_gap']:.3f} | {row['market_gap']:.3f} | "
            f"{row['deficit']:.3f} [{row['deficit_ci_lo']:.3f}, "
            f"{row['deficit_ci_hi']:.3f}] (p = {row['deficit_p']:.3f}) | "
            f"{row['market_brier']:.3f} | {row['rho_mean']:.3f} ± "
            f"{row['rho_std']:.3f} | {int(row['n'])} |"
        )

    lines += ["", "## Within-tier market vs. mean pool (deficit)", ""]
    for tier, cond in conds.items():
        cmp = compare(mets[tier], "market_gap", "mean_gap",
                     n_boot=n_boot, seed=bootstrap_seed)
        lines.append(_fmt_comparison(f"{tier}: market − mean pool (posterior gap)", cmp))

    lines += ["", "## Cross-tier comparisons (paired on seed, question)", ""]
    _col_label = {"market_gap": "market gap", "mean_gap": "pool gap"}
    for col, a, b, cmp in cross:
        lines.append(_fmt_comparison(f"{a} − {b} ({_col_label[col]})", cmp))
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt_ci(value: float, lo: float, hi: float) -> str:
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def render_herding_md(herding_by_tier: dict[str, pd.DataFrame]) -> str:
    """Render ``herding.md``: per-tier herding regression tables (pooled +
    per round) with ``b1``/``b2`` and their bootstrap CIs, ``R²``, ``n``, and
    the descriptive drift ``mean|y − x1|``. Deterministic (byte-stable) — the
    only randomness upstream is the seeded cluster bootstrap in
    ``herding_weights``."""
    lines = [
        "# Pnyx v2 — Herding Regression (dynamics-only)",
        "",
        "*In-market stated belief y regressed on the agent's own Pass-1 belief "
        "(x1) and the pre-trade price (x2): y = a + b1·x1 + b2·x2, OLS via "
        "numpy.linalg.lstsq. b1 = own-information weight, b2 = price weight; "
        "95% CIs are question-clustered bootstraps. Drift = mean|y − x1|. This "
        "is the spec's sanctioned dynamics analysis of in-market beliefs — it "
        "never feeds a static baseline or ρ estimate. Hypothesis: b2 falls with "
        "capability tier.*",
        "",
    ]
    _scope_label = {"pooled": "pooled", "1": "round 1", "2": "round 2", "3": "round 3"}
    for tier, hw in herding_by_tier.items():
        lines += [
            f"## {tier}",
            "",
            "| Scope | b1 (own info) | b2 (price) | R² | n | drift mean\\|y−x1\\| |",
            "|---|---|---|---|---|---|",
        ]
        for _i, row in hw.iterrows():
            r2 = row["r2"]
            r2_cell = "nan" if pd.isna(r2) else f"{r2:.3f}"
            lines.append(
                f"| {_scope_label.get(row['scope'], row['scope'])} | "
                f"{_fmt_ci(row['b1'], row['b1_lo'], row['b1_hi'])} | "
                f"{_fmt_ci(row['b2'], row['b2_lo'], row['b2_hi'])} | "
                f"{r2_cell} | {int(row['n'])} | {row['drift']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"

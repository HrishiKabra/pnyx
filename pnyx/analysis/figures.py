"""Pnyx P5 figures — Fig 1 (H1: market vs. static pools) and Fig 2 (H2 phase
map + H3 manipulation panel).

Every plotted number here is either a straightforward aggregation of an
already-computed column from ``pnyx.analysis.mainrun.question_metrics`` (a
per-seed mean, a grand mean/std of those per-seed means, or the literal
``market_gap - mean_gap`` difference Fig 2(a)'s y-axis is *defined* as in the
brief) or comes directly from a Task 2/3 statistics call
(``mainrun.compare``'s ``wilcoxon_p``, ``mainrun.team_rho_summary``'s
``RhoSummary``, ``manipulation.summarize_adversary``'s DataFrame). Nothing
here reimplements a posterior gap, a Bayes oracle lookup, a Wilcoxon test, or
an adversary metric — figures.py is a pure consumer of Tasks 2/3's outputs,
plus presentation-layer arithmetic (means, stds, differences) on top of them.

Design: each figure splits into a pure data-prep helper (``fig1_data`` /
``fig2_data``) returning a plain dict of Python floats/lists — this is what
tests assert on, with exact/approx values computed by hand from small
synthetic fixtures — and a plotting function (``fig1`` / ``fig2``) that
consumes that dict and does no I/O beyond ``savefig``. Uses the non-
interactive Agg backend (set before importing pyplot) so the test suite runs
headless with no display/GUI warnings.

Palette (Okabe–Ito, colorblind-safe; fixed color per entity across every
figure per the project's chart standards):

* market / condition A: ``#0072B2`` (blue) everywhere.
* other conditions (the H2 phase-map / matching-table arms): B1 ``#E69F00``
  (orange), B3 ``#009E73`` (green), C ``#CC79A7`` (pink) — plus a distinct
  marker shape per condition (o/s/^/D) since the #CC79A7/#009E73 pair alone
  is not colorblind-distinguishable.
* Fig 1's four static pools share a muted gray-blue tint family (desaturated
  relative to the market blue) so the market bar visually stands out.
* Fig 2(b)'s two manipulation lines use the two remaining Okabe–Ito colors:
  ``#D55E00`` (vermillion) for flip rate, ``#56B4E9`` (light blue) for
  recovery fraction.

Chart hygiene (both figures): no top/right spines, a light y-grid only, thin
marks, matplotlib's default font, ``figsize=(10, 4)``, ``tight_layout``,
panel letters (a)/(b), axis labels with units, 300 dpi PNG + vector PDF. All
text (axis labels, p-value annotations, direct labels, panel letters) is
near-black/gray (``_TEXT_COLOR``) — never rendered in a series' own color, so
color always carries the SAME meaning (series identity) and text never
competes with it. Fig 2(b) has a single y-axis in [0, 1] (flip rate +
recovery fraction only); mean adversary P&L per k is a small text annotation
under each k tick label, never a second axis — dual axes are banned.

stdlib + numpy + pandas + matplotlib only.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import (headless tests)

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pnyx.analysis.mainrun import RhoSummary, compare

__all__ = ["fig1_data", "fig1", "fig2_data", "fig2"]

# ---------------------------------------------------------------------------
# Palette + layout constants
# ---------------------------------------------------------------------------

MARKET_COLOR = "#0072B2"  # market / condition A, fixed across every figure

# Fixed color + marker per condition (Fig 2(a)'s phase map). A condition
# keeps its color in every figure it appears in; B1/B3/C reuse the
# Okabe-Ito hues not already claimed by the market/condition-A blue.
_CONDITION_COLOR: dict[str, str] = {
    "A": MARKET_COLOR,
    "B1": "#E69F00",
    "B3": "#009E73",
    "C": "#CC79A7",
}
_CONDITION_MARKER: dict[str, str] = {"A": "o", "B1": "s", "B3": "^", "C": "D"}

# Muted gray-blue tints for Fig 1's static pools: same desaturated hue
# family, increasing lightness, deliberately low-saturation relative to
# MARKET_COLOR so the market bar stands out per the brief.
_POOL_COLOR: dict[str, str] = {
    "mean": "#8899AA",
    "median": "#99AABB",
    "lop": "#AABBCC",
    "stack": "#BBCCDD",
}

_FLIP_COLOR = "#D55E00"  # vermillion — Fig 2(b) flip rate
_RECOVERY_COLOR = "#56B4E9"  # light blue — Fig 2(b) recovery fraction

# Every text element (axis labels, annotations, panel letters, direct
# labels) uses this near-black/gray — never a series' own color.
_TEXT_COLOR = "#333333"

_MECHANISMS: tuple[str, ...] = ("market", "mean", "median", "lop", "stack")
_MECH_LABEL: dict[str, str] = {
    "market": "Market",
    "mean": "Mean",
    "median": "Median",
    "lop": "LOP",
    "stack": "Calib. stack",
}

_FIGSIZE = (10.0, 4.0)


# ---------------------------------------------------------------------------
# Fig 1 — H1: market vs. static pools (condition A)
# ---------------------------------------------------------------------------


def _per_seed_means(df: pd.DataFrame, col: str) -> dict[int, float]:
    """``{seed: mean(col) over that seed's rows}`` — the shared reduction
    behind both the bar height (grand mean of these) and the error bar
    (their std)."""
    return {int(s): float(v) for s, v in df.groupby("seed")[col].mean().items()}


def fig1_data(metrics_a: pd.DataFrame) -> dict[str, Any]:
    """Pure data-prep for Fig 1.

    For each of the five mechanisms (market, mean, median, lop, stack) and
    each of the two metrics (posterior gap, Brier), computes:

    * ``bar`` — the grand mean of the 3 per-seed means of
      ``{mechanism}_{metric}`` (equivalently the mean over all 120
      (question, seed) rows, since every seed has the same question count);
    * ``err`` — the population std (``ddof=0``, matching
      ``mainrun.RhoSummary``'s convention) of those 3 per-seed means;
    * ``pvalue`` — the two-sided Wilcoxon p-value of the mechanism's metric
      vs. the market's, taken verbatim from ``mainrun.compare`` (``None``
      for ``market`` itself — a mechanism is never compared against
      itself).

    Returns ``{"mechanisms": [...], "gap": {"bar", "err", "pvalue"}, "brier":
    {...}}``, each of ``bar``/``err``/``pvalue`` a dict keyed by mechanism.
    """
    out: dict[str, Any] = {"mechanisms": list(_MECHANISMS)}
    for metric in ("gap", "brier"):
        bar: dict[str, float] = {}
        err: dict[str, float] = {}
        pvalue: dict[str, float | None] = {}
        for mech in _MECHANISMS:
            col = f"{mech}_{metric}"
            per_seed = _per_seed_means(metrics_a, col)
            vals = np.array(list(per_seed.values()), dtype=float)
            bar[mech] = float(vals.mean())
            err[mech] = float(vals.std())
            if mech == "market":
                pvalue[mech] = None
            else:
                cmp = compare(metrics_a, f"market_{metric}", col)
                pvalue[mech] = cmp.wilcoxon_p
        out[metric] = {"bar": bar, "err": err, "pvalue": pvalue}
    return out


def _style_axis(ax) -> None:
    """Shared chart hygiene: no top/right spines, light y-grid only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.06,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=_TEXT_COLOR,
        va="bottom",
    )


def _bar_colors() -> list[str]:
    return [MARKET_COLOR] + [_POOL_COLOR[m] for m in _MECHANISMS[1:]]


def _plot_bar_panel(ax, panel: dict[str, dict[str, float | None]], ylabel: str) -> None:
    mechs = _MECHANISMS
    x = np.arange(len(mechs))
    heights = [panel["bar"][m] for m in mechs]
    errs = [panel["err"][m] for m in mechs]
    colors = _bar_colors()

    ax.bar(
        x,
        heights,
        yerr=errs,
        color=colors,
        capsize=4,
        edgecolor=_TEXT_COLOR,
        linewidth=0.6,
        ecolor=_TEXT_COLOR,
        error_kw={"elinewidth": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_MECH_LABEL[m] for m in mechs])
    ax.set_ylabel(ylabel)
    _style_axis(ax)

    # Wilcoxon p vs. market, annotated under each static pool's bar (never
    # under market itself, which has no pvalue entry).
    for xi, mech in zip(x, mechs):
        p = panel["pvalue"][mech]
        if p is None:
            continue
        ax.text(
            xi,
            -0.07,
            f"p={p:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=_TEXT_COLOR,
        )


def fig1(metrics_a: pd.DataFrame, out: str | Path) -> None:
    """Fig 1 — H1 on condition A: posterior gap (left, panel a) and Brier
    score (right, panel b) for the market vs. the four static pools. Bars
    are colored market-blue / muted pool tints (see module docstring);
    error bars are the std of the 3 per-seed means; each static pool is
    annotated with its Wilcoxon p vs. market. Saves ``fig1.png`` (300 dpi)
    and ``fig1.pdf`` under ``out`` (created if missing)."""
    data = fig1_data(metrics_a)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE)

    _plot_bar_panel(axes[0], data["gap"], "Posterior gap  |prediction − posterior|")
    axes[0].set_title("Posterior gap", color=_TEXT_COLOR, fontsize=10)
    _panel_letter(axes[0], "a")

    _plot_bar_panel(axes[1], data["brier"], "Brier score")
    axes[1].set_title("Brier score", color=_TEXT_COLOR, fontsize=10)
    _panel_letter(axes[1], "b")

    fig.tight_layout()
    fig.savefig(out / "fig1.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig1.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — H2 phase map + H3 manipulation panel
# ---------------------------------------------------------------------------


def fig2_data(
    rho_by_cond: dict[str, RhoSummary],
    gapdiff_by_cond: dict[str, pd.DataFrame],
    adv_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Pure data-prep for Fig 2.

    ``rho_by_cond`` maps a condition name (A, B1, B3, C) to its
    ``mainrun.team_rho_summary`` output; ``gapdiff_by_cond`` maps the same
    condition names to its ``mainrun.question_metrics`` DataFrame (needs at
    least the ``seed``, ``market_gap``, ``mean_gap`` columns). Per condition,
    per seed, the phase-map y-value is ``mean(market_gap) - mean(mean_gap)``
    over that (condition, seed)'s rows — literally Fig 2(a)'s y-axis
    definition from the brief, not a new metric. ``adv_summary`` is
    ``manipulation.summarize_adversary``'s DataFrame verbatim (one row per
    k), sorted by k.

    Returns ``{"phase_map": {cond: {"seeds", "rho", "gapdiff"}}, "manipulation":
    {"k", "flip_rate", "recovery_fraction", "adv_pnl"}}``.

    Raises ``ValueError`` if ``rho_by_cond`` and ``gapdiff_by_cond`` don't
    share the same condition keys, or if a condition's rho seeds and
    gapdiff seeds don't match — either is a data-alignment bug that would
    silently plot mismatched (rho, gapdiff) pairs.
    """
    if set(rho_by_cond) != set(gapdiff_by_cond):
        raise ValueError(
            "fig2_data: rho_by_cond and gapdiff_by_cond must have the same "
            f"condition keys; got {sorted(rho_by_cond)} vs {sorted(gapdiff_by_cond)}"
        )

    phase_map: dict[str, dict[str, list]] = {}
    for cond, rho_summary in rho_by_cond.items():
        rho_by_seed = rho_summary.per_seed_mean
        gdf = gapdiff_by_cond[cond]
        diff = gdf["market_gap"] - gdf["mean_gap"]
        gapdiff_by_seed = {
            int(s): float(v) for s, v in diff.groupby(gdf["seed"]).mean().items()
        }
        if set(rho_by_seed) != set(gapdiff_by_seed):
            raise ValueError(
                f"fig2_data: condition {cond!r} has mismatched seed sets between "
                f"rho ({sorted(rho_by_seed)}) and gapdiff ({sorted(gapdiff_by_seed)})"
            )
        seeds = sorted(rho_by_seed)
        phase_map[cond] = {
            "seeds": seeds,
            "rho": [rho_by_seed[s] for s in seeds],
            "gapdiff": [gapdiff_by_seed[s] for s in seeds],
        }

    adv_sorted = adv_summary.sort_values("k")
    manipulation = {
        "k": [int(v) for v in adv_sorted["k"]],
        "flip_rate": [float(v) for v in adv_sorted["flip_rate"]],
        "recovery_fraction": [float(v) for v in adv_sorted["recovery_fraction"]],
        "adv_pnl": [float(v) for v in adv_sorted["mean_adv_pnl"]],
    }
    return {"phase_map": phase_map, "manipulation": manipulation}


def _plot_phase_map(ax, phase_map: dict[str, dict[str, list]]) -> None:
    for cond, d in phase_map.items():
        color = _CONDITION_COLOR.get(cond)
        marker = _CONDITION_MARKER.get(cond)
        if color is None or marker is None:
            raise ValueError(
                f"fig2: condition {cond!r} has no fixed color/marker assignment "
                f"(known: {sorted(_CONDITION_COLOR)})"
            )
        ax.scatter(
            d["rho"],
            d["gapdiff"],
            color=color,
            marker=marker,
            s=55,
            label=cond,
            edgecolor=_TEXT_COLOR,
            linewidth=0.4,
            zorder=3,
        )
        cx, cy = float(np.mean(d["rho"])), float(np.mean(d["gapdiff"]))
        ax.annotate(
            cond,
            (cx, cy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=_TEXT_COLOR,
        )

    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1, zorder=1)

    # "market helps"/"hurts" annotated relative to the y=0 line itself (x in
    # axes-fraction, y in data coords) so they sit next to the line
    # regardless of the data's y-range.
    trans = ax.get_yaxis_transform()
    ax.text(
        0.02, -0.01, "market helps (below)", transform=trans,
        ha="left", va="top", fontsize=8, color=_TEXT_COLOR,
    )
    ax.text(
        0.02, 0.01, "market hurts (above)", transform=trans,
        ha="left", va="bottom", fontsize=8, color=_TEXT_COLOR,
    )

    ax.set_xlabel("Measured team error correlation ρ")
    ax.set_ylabel("market_gap − mean_pool_gap")
    _style_axis(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
        labelcolor=_TEXT_COLOR,
    )


def _plot_manipulation(ax, manipulation: dict[str, list]) -> None:
    k = manipulation["k"]
    flip = manipulation["flip_rate"]
    rec = manipulation["recovery_fraction"]
    pnl = manipulation["adv_pnl"]

    ax.plot(k, flip, marker="o", color=_FLIP_COLOR, linewidth=1.5, zorder=3)
    ax.plot(k, rec, marker="s", color=_RECOVERY_COLOR, linewidth=1.5, zorder=3)

    # Direct labels at the right end, near-black text (never series color).
    ax.annotate(
        "flip rate", (k[-1], flip[-1]), textcoords="offset points",
        xytext=(6, 4), fontsize=8, color=_TEXT_COLOR,
    )
    ax.annotate(
        "recovery fraction", (k[-1], rec[-1]), textcoords="offset points",
        xytext=(6, -10), fontsize=8, color=_TEXT_COLOR,
    )

    ax.set_xscale("log")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    # Extra labelpad pushes the axis title below the P&L annotations (added
    # via ax.text, which matplotlib does not account for when auto-placing
    # the title) so the two never overlap.
    ax.set_xlabel("Adversary bankroll multiplier k", labelpad=22)
    ax.set_ylabel("Fraction of questions")
    ax.set_ylim(0.0, 1.0)
    _style_axis(ax)

    # Mean adversary P&L per k: small text annotations under the k tick
    # labels — NEVER a second y-axis (dual axes are banned).
    for ki, p in zip(k, pnl):
        sign = "−" if p < 0 else "+"
        ax.text(
            ki,
            -0.15,
            f"P&L {sign}${abs(p):,.0f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color=_TEXT_COLOR,
        )


def fig2(
    rho_by_cond: dict[str, RhoSummary],
    gapdiff_by_cond: dict[str, pd.DataFrame],
    adv_summary: pd.DataFrame,
    out: str | Path,
) -> None:
    """Fig 2 — (a) phase map: measured team ρ vs. ``market_gap -
    mean_pool_gap``, one marker shape/color per condition (A/B1/B3/C), a
    horizontal line at 0 with "helps"/"hurts" annotations, legend outside
    the axes. (b) manipulation: flip rate + recovery fraction vs. k (log
    scale) on a single y-axis in [0, 1], direct-labeled; mean adversary P&L
    per k as text annotations under the k tick labels (no second axis).
    Saves ``fig2.png`` (300 dpi) and ``fig2.pdf`` under ``out``."""
    data = fig2_data(rho_by_cond, gapdiff_by_cond, adv_summary)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE)

    _plot_phase_map(axes[0], data["phase_map"])
    _panel_letter(axes[0], "a")

    _plot_manipulation(axes[1], data["manipulation"])
    _panel_letter(axes[1], "b")

    fig.tight_layout()
    fig.savefig(out / "fig2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig2.pdf", bbox_inches="tight")
    plt.close(fig)

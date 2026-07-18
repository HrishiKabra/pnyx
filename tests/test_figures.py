"""Tests for pnyx.analysis.figures — Fig 1 (H1: market vs. static pools) and
Fig 2 (H2 phase map + H3 manipulation panel).

Per the task brief, tests assert on the DATA passed to matplotlib — the
plain dicts returned by the pure ``fig1_data`` / ``fig2_data`` helpers —
with exact/approx values computed by hand from small synthetic fixtures,
plus a smoke test that ``fig1`` / ``fig2`` write non-empty PNG + PDF files
to ``tmp_path``. No pixel assertions anywhere.

All fixtures are hand-built in-memory DataFrames / RhoSummary objects (no
event logs, no I/O, no network) — mirrors the fixture style of
tests/test_mainrun.py and tests/test_manipulation.py.
"""

import numpy as np
import pandas as pd
import pytest

from pnyx.analysis import figures
from pnyx.analysis.mainrun import RhoSummary, compare

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _metrics_a_fixture() -> pd.DataFrame:
    """Small synthetic condition-A-shaped ``question_metrics``-like table: 3
    seeds x 4 questions, every ``{mechanism}_{gap,brier}`` column populated
    with a distinct, deterministic value so per-seed means / grand mean /
    std are exactly hand-computable.

    Value formula: ``base[mech] + 0.01*seed + 0.001*qi`` (gap),
    ``0.5*base[mech] + 0.01*seed + 0.001*qi`` (brier); ``qi`` in 0..3, mean
    ``qi`` = 1.5 so the per-seed mean of a column is
    ``base + 0.01*seed + 0.0015``.
    """
    base = {"market": 0.1, "mean": 0.2, "median": 0.25, "lop": 0.3, "stack": 0.35}
    rows = []
    for seed in (0, 1, 2):
        for qi in range(4):
            row = {"seed": seed, "question_id": f"q{qi}"}
            for mech, b in base.items():
                row[f"{mech}_gap"] = b + 0.01 * seed + 0.001 * qi
                row[f"{mech}_brier"] = 0.5 * b + 0.01 * seed + 0.001 * qi
            rows.append(row)
    return pd.DataFrame(rows)


def _rho_summary(per_seed: dict) -> RhoSummary:
    vals = np.array(list(per_seed.values()), dtype=float)
    return RhoSummary(
        per_seed_mean=per_seed, mean=float(vals.mean()), std=float(vals.std())
    )


def _gapdiff_df(seed_to_pairs: dict) -> pd.DataFrame:
    """``{seed: [(market_gap, mean_gap), ...]}`` -> a DataFrame with columns
    ``seed``, ``market_gap``, ``mean_gap`` (the subset of a real
    ``question_metrics`` DataFrame that ``fig2_data`` needs)."""
    rows = []
    for seed, pairs in seed_to_pairs.items():
        for market_gap, mean_gap in pairs:
            rows.append({"seed": seed, "market_gap": market_gap, "mean_gap": mean_gap})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fig1_data
# ---------------------------------------------------------------------------


def test_fig1_data_mechanism_order():
    data = figures.fig1_data(_metrics_a_fixture())
    assert data["mechanisms"] == ["market", "mean", "median", "lop", "stack"]


def test_fig1_data_bar_and_err_are_grand_mean_and_std_of_per_seed_means():
    df = _metrics_a_fixture()
    data = figures.fig1_data(df)

    # hand-computed per mechanism/metric (see _metrics_a_fixture docstring)
    base = {"market": 0.1, "mean": 0.2, "median": 0.25, "lop": 0.3, "stack": 0.35}
    for mech, b in base.items():
        expected_gap_per_seed = [b + 0.01 * s + 0.0015 for s in (0, 1, 2)]
        expected_brier_per_seed = [0.5 * b + 0.01 * s + 0.0015 for s in (0, 1, 2)]
        assert data["gap"]["bar"][mech] == pytest.approx(np.mean(expected_gap_per_seed))
        assert data["gap"]["err"][mech] == pytest.approx(np.std(expected_gap_per_seed))
        assert data["brier"]["bar"][mech] == pytest.approx(
            np.mean(expected_brier_per_seed)
        )
        assert data["brier"]["err"][mech] == pytest.approx(
            np.std(expected_brier_per_seed)
        )


def test_fig1_data_pvalue_matches_mainrun_compare_exactly():
    df = _metrics_a_fixture()
    data = figures.fig1_data(df)
    for mech in ("mean", "median", "lop", "stack"):
        expected_gap_p = compare(df, "market_gap", f"{mech}_gap").wilcoxon_p
        expected_brier_p = compare(df, "market_brier", f"{mech}_brier").wilcoxon_p
        assert data["gap"]["pvalue"][mech] == expected_gap_p
        assert data["brier"]["pvalue"][mech] == expected_brier_p
    # market is not compared against itself
    assert data["gap"]["pvalue"]["market"] is None
    assert data["brier"]["pvalue"]["market"] is None


def test_fig1_writes_png_and_pdf(tmp_path):
    figures.fig1(_metrics_a_fixture(), tmp_path)
    png = tmp_path / "fig1.png"
    pdf = tmp_path / "fig1.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


# ---------------------------------------------------------------------------
# fig2_data
# ---------------------------------------------------------------------------


def test_fig2_data_phase_map_exact_values():
    rho_by_cond = {
        "A": _rho_summary({0: 0.7, 1: 0.6}),
        "B1": _rho_summary({0: 0.2, 1: 0.3}),
    }
    gapdiff_by_cond = {
        "A": _gapdiff_df({0: [(0.1, 0.3)], 1: [(0.2, 0.25)]}),  # diffs: -0.2, -0.05
        "B1": _gapdiff_df({0: [(0.4, 0.2)], 1: [(0.5, 0.1)]}),  # diffs: +0.2, +0.4
    }
    adv_summary = pd.DataFrame(
        {
            "k": [10, 1, 3],
            "flip_rate": [0.4, 0.1, 0.3],
            "recovery_fraction": [0.05, 0.2, 0.1],
            "mean_adv_pnl": [-400.0, -40.0, -120.0],
        }
    )

    data = figures.fig2_data(rho_by_cond, gapdiff_by_cond, adv_summary)

    assert data["phase_map"]["A"]["seeds"] == [0, 1]
    assert data["phase_map"]["A"]["rho"] == pytest.approx([0.7, 0.6])
    assert data["phase_map"]["A"]["gapdiff"] == pytest.approx([-0.2, -0.05])
    assert data["phase_map"]["B1"]["seeds"] == [0, 1]
    assert data["phase_map"]["B1"]["rho"] == pytest.approx([0.2, 0.3])
    assert data["phase_map"]["B1"]["gapdiff"] == pytest.approx([0.2, 0.4])

    # manipulation panel is sorted by k regardless of input order
    assert data["manipulation"]["k"] == [1, 3, 10]
    assert data["manipulation"]["flip_rate"] == pytest.approx([0.1, 0.3, 0.4])
    assert data["manipulation"]["recovery_fraction"] == pytest.approx([0.2, 0.1, 0.05])
    assert data["manipulation"]["adv_pnl"] == pytest.approx([-40.0, -120.0, -400.0])


def test_fig2_data_multi_seed_averages_within_seed():
    # two rows for seed 0 of condition A -> gapdiff must be the MEAN of the
    # two rows' (market_gap - mean_gap), not just the first row's.
    rho_by_cond = {"A": _rho_summary({0: 0.5})}
    gapdiff_by_cond = {"A": _gapdiff_df({0: [(0.10, 0.30), (0.20, 0.30)]})}
    adv_summary = pd.DataFrame(
        {"k": [1], "flip_rate": [0.1], "recovery_fraction": [0.1], "mean_adv_pnl": [-1.0]}
    )
    data = figures.fig2_data(rho_by_cond, gapdiff_by_cond, adv_summary)
    # diffs are -0.2 and -0.1 -> mean -0.15
    assert data["phase_map"]["A"]["gapdiff"] == pytest.approx([-0.15])


def test_fig2_data_condition_key_mismatch_raises():
    rho_by_cond = {"A": _rho_summary({0: 0.5})}
    gapdiff_by_cond = {"B1": _gapdiff_df({0: [(0.1, 0.2)]})}
    adv_summary = pd.DataFrame(
        {"k": [1], "flip_rate": [0.1], "recovery_fraction": [0.1], "mean_adv_pnl": [-1.0]}
    )
    with pytest.raises(ValueError):
        figures.fig2_data(rho_by_cond, gapdiff_by_cond, adv_summary)


def test_fig2_data_seed_mismatch_within_condition_raises():
    rho_by_cond = {"A": _rho_summary({0: 0.5, 1: 0.6})}
    gapdiff_by_cond = {"A": _gapdiff_df({0: [(0.1, 0.2)]})}  # missing seed 1
    adv_summary = pd.DataFrame(
        {"k": [1], "flip_rate": [0.1], "recovery_fraction": [0.1], "mean_adv_pnl": [-1.0]}
    )
    with pytest.raises(ValueError):
        figures.fig2_data(rho_by_cond, gapdiff_by_cond, adv_summary)


# ---------------------------------------------------------------------------
# fig2 smoke test + condition-palette guard
# ---------------------------------------------------------------------------


def _full_phase_map_fixture():
    rho_by_cond = {
        "A": _rho_summary({0: 0.7, 1: 0.6, 2: 0.65}),
        "B1": _rho_summary({0: 0.2, 1: 0.3, 2: 0.25}),
        "B3": _rho_summary({0: 0.4, 1: 0.35, 2: 0.45}),
        "C": _rho_summary({0: 0.5, 1: 0.55, 2: 0.5}),
    }
    gapdiff_by_cond = {
        "A": _gapdiff_df({0: [(0.1, 0.3)], 1: [(0.2, 0.25)], 2: [(0.15, 0.28)]}),
        "B1": _gapdiff_df({0: [(0.4, 0.2)], 1: [(0.5, 0.1)], 2: [(0.45, 0.15)]}),
        "B3": _gapdiff_df({0: [(0.3, 0.25)], 1: [(0.32, 0.28)], 2: [(0.31, 0.29)]}),
        "C": _gapdiff_df({0: [(0.2, 0.22)], 1: [(0.18, 0.2)], 2: [(0.19, 0.21)]}),
    }
    adv_summary = pd.DataFrame(
        {
            "k": [1, 3, 10],
            "flip_rate": [0.1, 0.3, 0.4],
            "recovery_fraction": [0.2, 0.1, 0.05],
            "mean_adv_pnl": [-40.0, -120.0, -400.0],
        }
    )
    return rho_by_cond, gapdiff_by_cond, adv_summary


def test_fig2_writes_png_and_pdf(tmp_path):
    rho_by_cond, gapdiff_by_cond, adv_summary = _full_phase_map_fixture()
    figures.fig2(rho_by_cond, gapdiff_by_cond, adv_summary, tmp_path)
    png = tmp_path / "fig2.png"
    pdf = tmp_path / "fig2.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


def test_fig2_unknown_condition_has_no_fixed_color_raises(tmp_path):
    rho_by_cond = {"Z": _rho_summary({0: 0.5})}
    gapdiff_by_cond = {"Z": _gapdiff_df({0: [(0.1, 0.2)]})}
    adv_summary = pd.DataFrame(
        {"k": [1], "flip_rate": [0.1], "recovery_fraction": [0.1], "mean_adv_pnl": [-1.0]}
    )
    with pytest.raises(ValueError):
        figures.fig2(rho_by_cond, gapdiff_by_cond, adv_summary, tmp_path)

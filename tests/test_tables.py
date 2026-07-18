"""Tests for pnyx.analysis.tables (Table 1, Table 2, stats_report) and the
CLI's ``analyze-main`` wiring.

Table 1 / Table 2 / stats_report unit tests use small hand-built
``question_metrics``-shaped DataFrames (mirrors the fixture style of
tests/test_figures.py and tests/test_manipulation.py) so every cell is
hand-computable. The CLI integration test reuses Task 2's (tests/
test_mainrun.py) synthetic condition builders — ``_make_two_pass_dir`` /
``_make_pass2_only_dir`` and their shared fixture constants — written to
tmp_path as real JSONL event logs/sidecars, plus hand-written cost ledgers
and config YAMLs, to exercise the full ``analyze-main`` pipeline end to end.
No network, no real run, no LLM calls anywhere.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pnyx import cli
from pnyx.analysis import manipulation, tables
from pnyx.analysis.mainrun import ConditionData, RhoSummary, SeedData, compare
from pnyx.schemas import CostEntry, Trade, TradeEvent, TurnKey

from tests.test_mainrun import (
    _BELIEFS,
    _FINAL,
    _OUTCOME,
    _POST,
    _make_pass2_only_dir,
    _make_two_pass_dir,
    _question,
    _settlement,
    _trade,
    _write_run,
)

# ---------------------------------------------------------------------------
# Small hand-built question_metrics-shaped DataFrame fixtures
# ---------------------------------------------------------------------------


def _metrics_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _condition_a_metrics() -> pd.DataFrame:
    """2 seeds x 2 questions, every {mechanism}_{gap,brier} column present,
    hand-computable means/stds."""
    rows = []
    for seed, mult in ((0, 1.0), (1, 1.2)):
        for qi, qbase in enumerate((0.0, 0.02)):
            base = mult * 0.1 + qbase
            rows.append({
                "seed": seed, "question_id": f"q{qi}",
                "market_gap": base, "market_brier": base * 0.5,
                "mean_gap": base + 0.05, "mean_brier": base * 0.5 + 0.05,
                "median_gap": base + 0.06, "median_brier": base * 0.5 + 0.06,
                "lop_gap": base + 0.07, "lop_brier": base * 0.5 + 0.07,
                "stack_gap": base + 0.08, "stack_brier": base * 0.5 + 0.08,
                "subsidy": 1.0 + qi,
            })
    return _metrics_df(rows)


def _pass2_only_metrics(condition_scale: float) -> pd.DataFrame:
    """Only the market_* + subsidy columns (a pass2-only condition still has
    its OWN market_gap/brier — only its mean-pool columns are reused from a
    source; table1 never touches mean_gap/mean_brier for these rows)."""
    rows = []
    for seed in (0, 1):
        for qi in range(2):
            rows.append({
                "seed": seed, "question_id": f"q{qi}",
                "market_gap": condition_scale + 0.01 * qi,
                "market_brier": 0.5 * condition_scale + 0.01 * qi,
                "subsidy": 2.0,
            })
    return _metrics_df(rows)


# ---------------------------------------------------------------------------
# table1
# ---------------------------------------------------------------------------


def test_table1_two_pass_condition_exact_cell():
    df = _condition_a_metrics()
    md = tables.table1({"A": df}, {"A": 2.0})

    # market_gap per-seed means: seed0 -> mean(0.10, 0.12) = 0.11,
    # seed1 -> mean(0.12, 0.14) = 0.13. grand mean = 0.12, std = 0.01.
    assert "| A | 0.120 ± 0.010 |" in md
    # market_brier per-seed means: seed0 -> mean(0.05,0.06)=0.055,
    # seed1 -> mean(0.06,0.07)=0.065. grand mean=0.06, std=0.005.
    assert "0.060 ± 0.005" in md
    # mean_gap per-seed means: seed0 -> 0.16, seed1 -> 0.18 -> mean 0.17 std 0.01
    assert "0.170 ± 0.010" in md
    # accuracy-per-dollar = (0.25 - 0.06) / 2.0 = 0.095
    assert "0.095" in md
    assert "| Condition | Market gap |" in md


def test_table1_zero_cost_prints_infinity():
    df = _condition_a_metrics()
    md = tables.table1({"A": df}, {"A": 0.0})
    assert "∞ (free tier)" in md


def test_table1_missing_cost_defaults_to_zero_and_infinity():
    df = _condition_a_metrics()
    md = tables.table1({"A": df}, {})  # no cost entry at all for A
    assert "∞ (free tier)" in md


def test_table1_pass2_only_footnote_marks():
    md = tables.table1(
        {
            "C": _condition_a_metrics(),
            "D_K1": _pass2_only_metrics(0.2),
            "A": _condition_a_metrics(),
            "W_FIXED": _pass2_only_metrics(0.3),
        },
        {"C": 1.0, "D_K1": 0.3, "A": 2.0, "W_FIXED": 0.2},
    )
    # market_gap per-seed means both seeds -> mean(0.20, 0.21) = 0.205 (std 0);
    # market_brier per-seed means -> mean(0.10, 0.11) = 0.105 (std 0).
    assert "| D_K1 | 0.205 ± 0.000 | 0.105 ± 0.000 | — (†) | — (†) |" in md
    assert "— (‡)" in md
    assert "† pass2-only condition; mean-pool columns are bit-identical to" in md
    assert "condition C's own row" in md
    assert "‡ pass2-only condition" in md
    assert "condition A's own row" in md


def test_table1_condition_canonical_order():
    md = tables.table1(
        {
            "W_SHUFFLED": _pass2_only_metrics(0.1),
            "A": _condition_a_metrics(),
            "C": _condition_a_metrics(),
        },
        {"W_SHUFFLED": 0.1, "A": 2.0, "C": 1.0},
    )
    rows = [l for l in md.splitlines() if l.startswith("| ")]
    # header row (the "|---|" separator has no space after "|", so it's
    # excluded here) + 3 data rows, in canonical order A, C, W_SHUFFLED.
    data_rows = rows[1:]
    assert data_rows[0].startswith("| A |")
    assert data_rows[1].startswith("| C |")
    assert data_rows[2].startswith("| W_SHUFFLED |")


# ---------------------------------------------------------------------------
# table2
# ---------------------------------------------------------------------------


def _belief_event(qid, agent_id, *, condition, seed, parse_failed=False):
    from pnyx.schemas import Belief, BeliefEvent
    return BeliefEvent(
        key=TurnKey(condition=condition, seed=seed, question_id=qid, phase="independent",
                    round=0, agent_id=agent_id),
        belief=Belief(prob=0.6, rationale="r"),
        parse_failed=parse_failed,
        prompt_version="p3-v1",
        ts=0.0,
    )


def _trade_event(qid, agent_id, round_, *, condition, seed, parse_failed=False):
    return TradeEvent(
        key=TurnKey(condition=condition, seed=seed, question_id=qid, phase="market",
                    round=round_, agent_id=agent_id),
        trade=Trade(belief=0.5, action="hold", shares=0.0, rationale="r"),
        executed_shares=0.0, cost=0.0, price_before=0.5, price_after=0.5,
        bankroll_after=100.0, parse_failed=parse_failed, prompt_version="p3-v1", ts=0.0,
    )


def test_table2_parse_fail_counts_exact():
    # condition "X": p0 (model "m1") has 1 belief (parse_failed) + 1 trade
    # (ok) = 2 turns, 1 fail; p1 (model "m2") has 1 belief (ok) = 1 turn, 0 fail.
    beliefs = [
        _belief_event("q0", "p0", condition="X", seed=0, parse_failed=True),
        _belief_event("q0", "p1", condition="X", seed=0, parse_failed=False),
    ]
    trades = [_trade_event("q0", "p0", 1, condition="X", seed=0, parse_failed=False)]
    cond = ConditionData(condition="X", seeds={0: SeedData(seed=0, beliefs=beliefs, trades=trades)})

    md = tables.table2(
        {"X": _pass2_only_metrics(0.1)},
        {"X": cond},
        {"X": {"p0": "m1", "p1": "m2"}},
    )
    assert "| X | m1 | 1 | 2 | 0.5000 |" in md
    assert "| X | m2 | 0 | 1 | 0.0000 |" in md


def test_table2_unmapped_agent_skipped():
    beliefs = [_belief_event("q0", "mock_agent", condition="X", seed=0)]
    cond = ConditionData(condition="X", seeds={0: SeedData(seed=0, beliefs=beliefs)})
    md = tables.table2({"X": _pass2_only_metrics(0.1)}, {"X": cond}, {"X": {}})
    assert "mock_agent" not in md


def test_table2_subsidy_stats_exact():
    df = _condition_a_metrics()  # subsidy column: seed0/1 x q0(1.0)/q1(2.0) -> 4 rows
    cond = ConditionData(condition="A", seeds={})
    md = tables.table2({"A": df}, {"A": cond}, {"A": {}})
    # subsidy values across all 4 rows: [1.0, 2.0, 1.0, 2.0]
    assert "| A | 1.500 | 6.000 | 2.000 | 27.73 |" in md


def test_table2_headers_present():
    md = tables.table2({}, {}, {})
    assert "### Parse-failure rate per (condition, model)" in md
    assert "### Market-maker subsidy per condition" in md
    assert "27.73" in md


# ---------------------------------------------------------------------------
# stats_report — H1
# ---------------------------------------------------------------------------


def test_stats_report_h1_matches_compare_directly():
    df = _condition_a_metrics()
    md = tables.stats_report(
        {"A": df, "C": df}, {}, pd.DataFrame(columns=["k", "flip_rate"]), {},
        n_boot=200, bootstrap_seed=0,
    )
    cmp = compare(df, "market_gap", "mean_gap", n_boot=200, seed=0)
    lo, hi = cmp.ci
    expected = (
        f"Δ = {cmp.mean_diff:.3f} [{lo:.3f}, {hi:.3f}] (95% bootstrap CI), "
        f"Wilcoxon p = {cmp.wilcoxon_p:.3f}"
    )
    assert expected in md
    assert "## H1: Market vs. Static Pools (Condition A)" in md


def test_stats_report_missing_A_raises():
    with pytest.raises(ValueError, match="'A'"):
        tables.stats_report({"C": _condition_a_metrics()}, {}, pd.DataFrame(), {})


def test_stats_report_missing_C_raises():
    with pytest.raises(ValueError, match="'C'"):
        tables.stats_report({"A": _condition_a_metrics()}, {}, pd.DataFrame(), {})


# ---------------------------------------------------------------------------
# stats_report — H2a
# ---------------------------------------------------------------------------


def test_stats_report_h2a_pairs_on_seed_and_question():
    c_df = _metrics_df([
        {"seed": 0, "question_id": "q0", "market_gap": 0.10},
        {"seed": 0, "question_id": "q1", "market_gap": 0.20},
        {"seed": 1, "question_id": "q0", "market_gap": 0.15},
        {"seed": 1, "question_id": "q1", "market_gap": 0.25},
    ])
    b1_df = _metrics_df([
        {"seed": 0, "question_id": "q0", "market_gap": 0.30},
        {"seed": 0, "question_id": "q1", "market_gap": 0.10},
        {"seed": 1, "question_id": "q0", "market_gap": 0.35},
        {"seed": 1, "question_id": "q1", "market_gap": 0.05},
    ])
    a_df = _condition_a_metrics()
    md = tables.stats_report(
        {"A": a_df, "C": c_df, "B1": b1_df}, {},
        pd.DataFrame(columns=["k", "flip_rate"]), {}, n_boot=200, bootstrap_seed=0,
    )
    merged = c_df.merge(b1_df, on=["seed", "question_id"], suffixes=("_x", "_y"))
    cmp = compare(merged, "market_gap_x", "market_gap_y", n_boot=200, seed=0)
    assert f"Δ = {cmp.mean_diff:.3f}" in md
    assert "## H2a: Mixed Team (C) vs. Homogeneous Arms" in md
    assert "C − B1 (market_gap)" in md
    assert "C − A (B2) (market_gap)" in md


# ---------------------------------------------------------------------------
# stats_report — H2b (hand-rolled Spearman)
# ---------------------------------------------------------------------------


def _h2b_row(seed, qid, market_gap, mean_gap):
    """A full ``question_metrics``-shaped row: H2b only reads ``market_gap``/
    ``mean_gap``, but ``stats_report`` unconditionally builds H1 (all 4 pools'
    gap/brier columns) on condition A too, so every row needs the full column
    set — filled with inert 0.0 placeholders here."""
    return {
        "seed": seed, "question_id": qid,
        "market_gap": market_gap, "market_brier": 0.0,
        "mean_gap": mean_gap, "mean_brier": 0.0,
        "median_gap": 0.0, "median_brier": 0.0,
        "lop_gap": 0.0, "lop_brier": 0.0,
        "stack_gap": 0.0, "stack_brier": 0.0,
        "subsidy": 0.0,
    }


def test_stats_report_h2b_spearman_perfect_monotonic():
    # A single condition, 4 seeds, rho and gapdiff both strictly increasing
    # in seed order -> Spearman = +1.0 exactly. rho_by_cond only needs "A"
    # here; metrics_by_cond needs "A" (for H2b) + "C" (required by
    # stats_report itself).
    rho = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    rho_summary = RhoSummary(per_seed_mean=rho, mean=0.25, std=0.11180339887498948)
    gapdiff_df = _metrics_df([
        _h2b_row(s, "q0", g, 0.0) for s, g in zip((0, 1, 2, 3), (1.0, 2.0, 3.0, 4.0))
    ])
    md = tables.stats_report(
        {"A": gapdiff_df, "C": _condition_a_metrics()},
        {"A": rho_summary},
        pd.DataFrame(columns=["k", "flip_rate"]),
        {},
        n_boot=200,
        bootstrap_seed=0,
    )
    assert "r_s = 1.000" in md
    assert "n=4" in md


def test_stats_report_h2b_lists_condition_seed_rows():
    rho_summary = RhoSummary(per_seed_mean={0: 0.5}, mean=0.5, std=0.0)
    gapdiff_df = _metrics_df([_h2b_row(0, "q0", 0.30, 0.10)])
    md = tables.stats_report(
        {"A": gapdiff_df, "C": _condition_a_metrics()},
        {"A": rho_summary},
        pd.DataFrame(columns=["k", "flip_rate"]),
        {},
    )
    assert "| A | 0 | 0.500 | 0.200 |" in md


# ---------------------------------------------------------------------------
# stats_report — H3 (verbatim) + H-wealth
# ---------------------------------------------------------------------------


def test_stats_report_h3_verbatim():
    adv_summary = pd.DataFrame({
        "k": [1, 3],
        "flip_rate": [0.25, 0.5],
        "mean_adv_pnl": [-5.0, -12.5],
    })
    md = tables.stats_report(
        {"A": _condition_a_metrics(), "C": _condition_a_metrics()}, {}, adv_summary, {},
    )
    assert "## H3: Adversary Manipulation" in md
    assert "| k | flip_rate | mean_adv_pnl |" in md
    assert "| 1 | 0.2500 | -5.0000 |" in md
    assert "| 3 | 0.5000 | -12.5000 |" in md


def test_stats_report_h_wealth_verbatim():
    keys = [(0, "q0"), (1, "q0")]
    a_df = _metrics_df([{"seed": s, "question_id": q, "market_gap": 0.10, "market_brier": 0.05}
                         for s, q in keys])
    wf_df = _metrics_df([{"seed": s, "question_id": q, "market_gap": 0.14, "market_brier": 0.07}
                          for s, q in keys])
    ws_df = _metrics_df([{"seed": s, "question_id": q, "market_gap": 0.12, "market_brier": 0.06}
                          for s, q in keys])
    wealth_effects = manipulation.wealth_order_effects(a_df, wf_df, ws_df, n_boot=200, seed=0)
    md = tables.stats_report(
        {"A": _condition_a_metrics(), "C": _condition_a_metrics()}, {},
        pd.DataFrame(columns=["k", "flip_rate"]), wealth_effects,
    )
    assert "## H-wealth: Persistent Wealth Order Effects" in md
    cmp = wealth_effects["wfixed_vs_a"]["market_gap"]
    assert f"Δ = {cmp.mean_diff:.3f}" in md
    assert "W_FIXED − A (market_gap)" in md
    assert "W_SHUFFLED − A (market_brier)" in md
    assert "W_FIXED − W_SHUFFLED (market_gap)" in md


def test_stats_report_all_headers_present():
    md = tables.stats_report(
        {"A": _condition_a_metrics(), "C": _condition_a_metrics()}, {},
        pd.DataFrame(columns=["k", "flip_rate"]), {},
    )
    for header in ("## H1", "## H2a", "## H2b", "## H3", "## H-wealth"):
        assert header in md


def test_stats_report_deterministic_across_calls():
    df = _condition_a_metrics()
    args = ({"A": df, "C": df}, {}, pd.DataFrame(columns=["k", "flip_rate"]), {})
    md1 = tables.stats_report(*args, n_boot=500, bootstrap_seed=0)
    md2 = tables.stats_report(*args, n_boot=500, bootstrap_seed=0)
    assert md1 == md2


# ---------------------------------------------------------------------------
# CLI integration: build a full 9-condition mini-grid and run analyze-main
# ---------------------------------------------------------------------------

_ENV_MIN = {
    "n_signals": 2, "accuracies": [0.75, 0.75], "lams": [0.0, 0.0],
    "a_z": 0.8, "difficulty": "easy", "min_single_shard_margin": 0.05,
    "max_rejection_tries": 100,
}


def _write_min_config(configs_dir: Path, filename: str, condition: str, *,
                       agent_ids: list[str], model_id: str, with_adversary: bool = False) -> None:
    agents = [
        {"agent_id": aid, "shard_indices": [0], "bankroll": 100.0, "kind": "llm", "model": "m"}
        for aid in agent_ids
    ]
    if with_adversary:
        agents.append({
            "agent_id": "adv0", "shard_indices": [], "bankroll": 100.0,
            "kind": "llm", "model": "m", "adversary": True, "adversary_style": "stealthy",
        })
    config = {
        "condition": condition,
        "seeds": [0, 1],
        "n_questions": 2,
        "b": 40.0,
        "n_rounds": 3,
        "wealth_persistent": False,
        "data_dir": f"data/main/{condition}",
        "env": _ENV_MIN,
        "models": {"m": {
            "base_url": "https://example.test", "api_key_env": "TEST_KEY",
            "model_id": model_id, "price_in": 0.01, "price_out": 0.02,
            "rpm_limit": 100, "supports_json_schema": True,
        }},
        "agents": agents,
    }
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / filename).write_text(yaml.safe_dump(config))


def _write_cost_ledger(run_dir: Path, condition: str, total_cost: float, *,
                       model_id: str, n_entries: int = 4) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    per = total_cost / n_entries if n_entries else 0.0
    lines = [
        CostEntry(model_id=model_id, in_tokens=10, out_tokens=10, cost=per).model_dump_json()
        for _ in range(n_entries)
    ]
    (run_dir / f"{condition}.costs.jsonl").write_text("".join(l + "\n" for l in lines))


def _make_honest_pass2_only_dir(tmp_path: Path, condition: str) -> Path:
    """Pass2-only, NO adversary — the W_FIXED/W_SHUFFLED shape (mirrors Task
    2's ``_make_pass2_only_dir`` minus adv0)."""
    run_dir = tmp_path / condition
    for seed in (0, 1):
        events = []
        for qid in ("q000", "q001"):
            for aid in _BELIEFS[qid]:
                for rnd in (1, 2, 3):
                    events.append(_trade(qid, aid, rnd, condition=condition, seed=seed))
            events.append(_settlement(
                qid, _OUTCOME[qid], _FINAL[seed][qid], condition=condition, seed=seed, subsidy=1.0,
            ))
        questions = [_question(qid, posterior_all=_POST[qid], latent_state=_OUTCOME[qid])
                     for qid in ("q000", "q001")]
        _write_run(run_dir, condition, seed, events, questions)
    return run_dir


@pytest.fixture
def mini_grid(tmp_path):
    data_root = tmp_path / "data_main"
    configs_dir = tmp_path / "configs_main"

    _make_two_pass_dir(data_root, "A")
    _make_two_pass_dir(data_root, "B1")
    _make_two_pass_dir(data_root, "B3")
    _make_two_pass_dir(data_root, "C")
    _make_pass2_only_dir(data_root, "D_K1")
    _make_pass2_only_dir(data_root, "D_K3")
    _make_pass2_only_dir(data_root, "D_K10")
    _make_honest_pass2_only_dir(data_root, "W_FIXED")
    _make_honest_pass2_only_dir(data_root, "W_SHUFFLED")

    _write_cost_ledger(data_root / "A", "A", 2.0, model_id="model-a")
    _write_cost_ledger(data_root / "B1", "B1", 1.0, model_id="model-b1")
    # B3 intentionally has NO cost ledger (free-tier condition -> $0.00 ->
    # accuracy-per-dollar prints "infinity (free tier)").
    _write_cost_ledger(data_root / "C", "C", 1.5, model_id="model-c")
    _write_cost_ledger(data_root / "D_K1", "D_K1", 0.3, model_id="model-c")
    _write_cost_ledger(data_root / "D_K3", "D_K3", 0.3, model_id="model-c")
    _write_cost_ledger(data_root / "D_K10", "D_K10", 0.3, model_id="model-c")
    _write_cost_ledger(data_root / "W_FIXED", "W_FIXED", 0.2, model_id="model-a")
    _write_cost_ledger(data_root / "W_SHUFFLED", "W_SHUFFLED", 0.2, model_id="model-a")

    honest_ids = list(_BELIEFS["q000"])  # ["p0", "p1", "p2"]
    _write_min_config(configs_dir, "A.yaml", "A", agent_ids=honest_ids, model_id="model-a")
    _write_min_config(configs_dir, "B1.yaml", "B1", agent_ids=honest_ids, model_id="model-b1")
    _write_min_config(configs_dir, "B3.yaml", "B3", agent_ids=honest_ids, model_id="model-b3")
    _write_min_config(configs_dir, "C.yaml", "C", agent_ids=honest_ids, model_id="model-c")
    _write_min_config(configs_dir, "D_k1.yaml", "D_K1", agent_ids=honest_ids,
                      model_id="model-c", with_adversary=True)
    _write_min_config(configs_dir, "D_k3.yaml", "D_K3", agent_ids=honest_ids,
                      model_id="model-c", with_adversary=True)
    _write_min_config(configs_dir, "D_k10.yaml", "D_K10", agent_ids=honest_ids,
                      model_id="model-c", with_adversary=True)
    _write_min_config(configs_dir, "W_fixed.yaml", "W_FIXED", agent_ids=honest_ids, model_id="model-a")
    _write_min_config(configs_dir, "W_shuffled.yaml", "W_SHUFFLED", agent_ids=honest_ids, model_id="model-a")

    return data_root, configs_dir


def test_analyze_main_cli_writes_all_five_files(mini_grid, tmp_path, capsys):
    data_root, configs_dir = mini_grid
    out_dir = tmp_path / "analysis_out"

    rc = cli.main([
        "analyze-main",
        "--data-root", str(data_root),
        "--configs", str(configs_dir),
        "--out", str(out_dir),
    ])
    assert rc == 0

    for name in ("fig1.png", "fig1.pdf", "fig2.png", "fig2.pdf",
                 "table1.md", "table2.md", "stats.md"):
        p = out_dir / name
        assert p.exists() and p.stat().st_size > 0, f"{name} missing or empty"

    stats_md = (out_dir / "stats.md").read_text()
    for header in ("## H1", "## H2a", "## H2b", "## H3", "## H-wealth"):
        assert header in stats_md

    table1_md = (out_dir / "table1.md").read_text()
    for cond in ("A", "B1", "B3", "C", "D_K1", "D_K3", "D_K10", "W_FIXED", "W_SHUFFLED"):
        assert f"| {cond} |" in table1_md
    assert "∞ (free tier)" in table1_md  # B3's zero-cost row

    table2_md = (out_dir / "table2.md").read_text()
    assert "model-a" in table2_md and "model-c" in table2_md

    out = capsys.readouterr().out
    assert "H1 (condition A)" in out
    assert "H3 flip rates" in out
    assert "Total spend across ledgers" in out


def test_analyze_main_cli_deterministic_across_runs(mini_grid, tmp_path):
    data_root, configs_dir = mini_grid
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    for out_dir in (out1, out2):
        rc = cli.main([
            "analyze-main",
            "--data-root", str(data_root),
            "--configs", str(configs_dir),
            "--out", str(out_dir),
            "--bootstrap-seed", "0",
        ])
        assert rc == 0

    assert (out1 / "stats.md").read_text() == (out2 / "stats.md").read_text()
    assert (out1 / "table1.md").read_text() == (out2 / "table1.md").read_text()
    assert (out1 / "table2.md").read_text() == (out2 / "table2.md").read_text()

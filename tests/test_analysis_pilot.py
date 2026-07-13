"""Tests for pnyx.analysis.pilot — offline aggregation of pilot event logs
+ cost ledgers + question sidecars into the P3 matching-table ingredients.

All fixtures are hand-built synthetic logs written by the tests themselves
(pnyx.runner naming convention: ``{condition}_seed{seed}.jsonl`` +
``{condition}_seed{seed}.questions.jsonl`` + ``{condition}.costs.jsonl``).
No network, no real run, no LLM calls anywhere.
"""

from pathlib import Path

import pytest

from pnyx import cli
from pnyx.analysis import pilot
from pnyx.schemas import (
    Belief,
    BeliefEvent,
    CostEntry,
    QuestionRecord,
    SettlementEvent,
    SignalRecord,
    Trade,
    TradeEvent,
    TurnKey,
)

CONDITION = "PILOT_TEST"
SEED = 0


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _question(qid, *, posterior_all, latent_state, single0=0.6, single1=0.6):
    """A minimal 2-signal QuestionRecord — n_signals doesn't need to be 6 for
    analysis purposes, only ``posterior_table``'s full-subset key (here
    "0,1") and ``latent_state`` matter."""
    return QuestionRecord(
        question_id=qid,
        latent_state=latent_state,
        signals=[
            SignalRecord(index=0, value=1, accuracy=0.7, lam=0.0),
            SignalRecord(index=1, value=1, accuracy=0.7, lam=0.0),
        ],
        posterior_table={"": 0.5, "0": single0, "1": single1, "0,1": posterior_all},
        shards=["shard 0", "shard 1"],
        question_text=f"Does {qid} resolve yes?",
    )


def _key(qid, agent_id, *, phase="independent", round_=0, condition=CONDITION):
    return TurnKey(
        condition=condition, seed=SEED, question_id=qid, phase=phase,
        round=round_, agent_id=agent_id,
    )


def _belief(qid, agent_id, prob, *, phase="independent", round_=0, parse_failed=False,
            condition=CONDITION):
    return BeliefEvent(
        key=_key(qid, agent_id, phase=phase, round_=round_, condition=condition),
        belief=Belief(prob=prob, rationale="r"),
        parse_failed=parse_failed,
        prompt_version="p3-v1",
        ts=0.0,
    )


def _trade(qid, agent_id, round_, *, price_before, price_after, parse_failed=False,
           condition=CONDITION):
    return TradeEvent(
        key=_key(qid, agent_id, phase="market", round_=round_, condition=condition),
        trade=Trade(belief=price_after, action="hold", shares=0.0, rationale="r"),
        executed_shares=0.0,
        cost=0.0,
        price_before=price_before,
        price_after=price_after,
        bankroll_after=100.0,
        parse_failed=parse_failed,
        prompt_version="p3-v1",
        ts=0.0,
    )


def _settlement(qid, outcome, final_price, *, payouts=None, condition=CONDITION):
    return SettlementEvent(
        condition=condition, seed=SEED, question_id=qid, outcome=outcome,
        payouts=payouts or {}, final_price=final_price, subsidy=0.0, ts=0.0,
    )


def _cost(model_id, cost, *, qid=None, agent_id=None, round_=0, phase="independent"):
    key = _key(qid, agent_id, phase=phase, round_=round_) if qid else None
    return CostEntry(model_id=model_id, in_tokens=10, out_tokens=10, cost=cost, key=key)


def _write_jsonl(path: Path, lines_objs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(o.to_jsonl_line() + "\n" for o in lines_objs))


def _write_questions(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))


def _write_costs(path: Path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(e.model_dump_json() + "\n" for e in entries))


def _write_run(
    run_dir: Path, *, events, questions=None, costs=None, condition=CONDITION, seed=SEED
):
    stem = f"{condition}_seed{seed}"
    _write_jsonl(run_dir / f"{stem}.jsonl", events)
    if questions is not None:
        _write_questions(run_dir / f"{stem}.questions.jsonl", questions)
    if costs is not None:
        _write_costs(run_dir / f"{condition}.costs.jsonl", costs)


# ---------------------------------------------------------------------------
# 1. Hand-computed aggregation math (2 questions, 2 agents, one model)
# ---------------------------------------------------------------------------
#
# q000: posterior_all=0.8, outcome=1(yes).  Pass-1: a0=0.55, a1=0.65.
#   pool  = (0.55+0.65)/2 = 0.60          -> pool_gap0   = |0.60-0.8| = 0.20
#                                              pool_brier0 = (0.60-1)^2 = 0.16
#   market final_price = 0.75             -> market_gap0   = |0.75-0.8| = 0.05
#                                              market_brier0 = (0.75-1)^2 = 0.0625
# q001: posterior_all=0.2, outcome=0(no).   Pass-1: a0=0.35, a1=0.25.
#   pool  = (0.35+0.25)/2 = 0.30          -> pool_gap1   = |0.30-0.2| = 0.10
#                                              pool_brier1 = (0.30-0)^2 = 0.09
#   market final_price = 0.30             -> market_gap1   = |0.30-0.2| = 0.10
#                                              market_brier1 = (0.30-0)^2 = 0.09
#
# pool_gap    = mean(0.20, 0.10) = 0.15
# pool_brier  = mean(0.16, 0.09) = 0.125
# market_gap  = mean(0.05, 0.10) = 0.075
# market_brier= mean(0.0625, 0.09) = 0.07625
# degeneracy: neither 0.75 nor 0.30 is <0.05 or >0.95 -> 0/2 = 0.0
#
# standalone gap (model "m1", all 4 Pass-1 beliefs vs. their own question's
# posterior_all): |.55-.8|=.25, |.65-.8|=.15, |.35-.2|=.15, |.25-.2|=.05
#   mean = (.25+.15+.15+.05)/4 = 0.60/4 = 0.15
#
# turns: 4 beliefs + 4 trades (1 round x 2 agents x 2 questions) + 2
# settlements = 10 done; the a1/q001 trade is parse_failed=True ->
# parse-fail rate for "modelid-m1" = 1/8 (beliefs+trades only, not
# settlements).
#
# cost: 8 LLM calls (4 belief + 4 trade) x $0.001 = $0.008 total.


def test_aggregation_math_exact_hand_computed(tmp_path):
    run_dir = tmp_path / "run"
    q000 = _question("q000", posterior_all=0.8, latent_state=1)
    q001 = _question("q001", posterior_all=0.2, latent_state=0)

    events = [
        _belief("q000", "a0", 0.55),
        _belief("q000", "a1", 0.65),
        _belief("q001", "a0", 0.35),
        _belief("q001", "a1", 0.25),
        _trade("q000", "a0", 1, price_before=0.6, price_after=0.75),
        _trade("q000", "a1", 1, price_before=0.75, price_after=0.75),
        _trade("q001", "a0", 1, price_before=0.4, price_after=0.30),
        _trade("q001", "a1", 1, price_before=0.30, price_after=0.30, parse_failed=True),
        _settlement("q000", outcome=1, final_price=0.75),
        _settlement("q001", outcome=0, final_price=0.30),
    ]
    costs = [
        _cost("modelid-m1", 0.001, qid="q000", agent_id="a0"),
        _cost("modelid-m1", 0.001, qid="q000", agent_id="a1"),
        _cost("modelid-m1", 0.001, qid="q001", agent_id="a0"),
        _cost("modelid-m1", 0.001, qid="q001", agent_id="a1"),
        _cost("modelid-m1", 0.001, qid="q000", agent_id="a0", round_=1, phase="market"),
        _cost("modelid-m1", 0.001, qid="q000", agent_id="a1", round_=1, phase="market"),
        _cost("modelid-m1", 0.001, qid="q001", agent_id="a0", round_=1, phase="market"),
        _cost("modelid-m1", 0.001, qid="q001", agent_id="a1", round_=1, phase="market"),
    ]
    _write_run(run_dir, events=events, questions=[q000, q001], costs=costs)

    result = pilot.analyze_run(run_dir)

    assert result.condition == CONDITION
    assert result.n_questions_dataset == 2
    assert result.n_settled == 2
    assert result.unsettled_question_ids == []

    assert result.degeneracy_rate == pytest.approx(0.0)
    assert result.degenerate_count == 0

    assert result.market_gap == pytest.approx(0.075)
    assert result.market_brier == pytest.approx(0.07625)
    assert result.pool_gap == pytest.approx(0.15)
    assert result.pool_brier == pytest.approx(0.125)

    assert result.standalone_gap_by_model == {"modelid-m1": pytest.approx(0.15)}

    fails, total = result.parse_fail_by_model["modelid-m1"]
    assert (fails, total) == (1, 8)

    assert result.cost_by_model == {"modelid-m1": pytest.approx(0.008)}
    assert result.total_cost == pytest.approx(0.008)

    assert result.turns_done == 10  # 4 beliefs + 4 trades + 2 settlements


# ---------------------------------------------------------------------------
# 2. Parse-fail counting across two distinct models
# ---------------------------------------------------------------------------


def test_parse_fail_rate_counting_per_model(tmp_path):
    run_dir = tmp_path / "run"
    q = _question("q000", posterior_all=0.7, latent_state=1)
    events = [
        _belief("q000", "a0", 0.5, parse_failed=True),
        _belief("q000", "a1", 0.5),
        _trade("q000", "a0", 1, price_before=0.5, price_after=0.6, parse_failed=True),
        _trade("q000", "a1", 1, price_before=0.6, price_after=0.6),
        _trade("q000", "b0", 1, price_before=0.6, price_after=0.6),
        _settlement("q000", outcome=1, final_price=0.6),
    ]
    costs = [
        _cost("model-A", 0.0, qid="q000", agent_id="a0"),
        _cost("model-A", 0.0, qid="q000", agent_id="a1"),
        _cost("model-A", 0.0, qid="q000", agent_id="a0", round_=1, phase="market"),
        _cost("model-A", 0.0, qid="q000", agent_id="a1", round_=1, phase="market"),
        _cost("model-B", 0.0, qid="q000", agent_id="b0", round_=1, phase="market"),
    ]
    _write_run(run_dir, events=events, questions=[q], costs=costs)

    result = pilot.analyze_run(run_dir)

    # model-A: 4 belief+trade turns (2 beliefs + 2 trades), 2 parse-failed.
    assert result.parse_fail_by_model["model-A"] == (2, 4)
    # model-B: 1 trade turn, 0 parse-failed.
    assert result.parse_fail_by_model["model-B"] == (0, 1)


# ---------------------------------------------------------------------------
# 3. Price-degeneracy edges: exactly 0.05 / 0.95 are NOT degenerate
# ---------------------------------------------------------------------------


def test_degeneracy_boundary_exact_not_degenerate(tmp_path):
    run_dir = tmp_path / "run"
    prices = {
        "q000": 0.05,   # boundary -> NOT degenerate
        "q001": 0.95,   # boundary -> NOT degenerate
        "q002": 0.049999,  # just below -> degenerate
        "q003": 0.950001,  # just above -> degenerate
    }
    questions = [
        _question(qid, posterior_all=0.5, latent_state=1) for qid in prices
    ]
    events = []
    for qid in prices:
        events.append(_belief(qid, "a0", 0.5))
        events.append(_settlement(qid, outcome=1, final_price=prices[qid]))
    _write_run(run_dir, events=events, questions=questions, costs=[])

    result = pilot.analyze_run(run_dir)

    assert result.degenerate_count == 2
    assert result.degeneracy_rate == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# 4. Phase guard: a market-phase-tagged "belief" event must hard-fail
# ---------------------------------------------------------------------------


def test_phase_guard_direct_unit_raises_on_market_phase():
    contaminated = _belief("q000", "a0", 0.5, phase="market", round_=1)
    with pytest.raises(AssertionError):
        pilot._mean_pool_prob([contaminated])


def test_phase_guard_direct_unit_accepts_independent():
    clean = [_belief("q000", "a0", 0.4), _belief("q000", "a1", 0.6)]
    assert pilot._mean_pool_prob(clean) == pytest.approx(0.5)


def test_phase_guard_fires_end_to_end_on_contaminated_log(tmp_path):
    run_dir = tmp_path / "run"
    q = _question("q000", posterior_all=0.7, latent_state=1)
    events = [
        # A hand-edited/corrupted log: a "belief" event tagged phase="market".
        _belief("q000", "a0", 0.5, phase="market", round_=1),
        _settlement("q000", outcome=1, final_price=0.6),
    ]
    _write_run(run_dir, events=events, questions=[q], costs=[])

    with pytest.raises(AssertionError):
        pilot.analyze_run(run_dir)


# ---------------------------------------------------------------------------
# 5. Incomplete run: unsettled questions skipped from averages, noted
# ---------------------------------------------------------------------------


def test_incomplete_run_skips_unsettled_from_averages_and_notes_them(tmp_path):
    run_dir = tmp_path / "run"
    q000 = _question("q000", posterior_all=0.8, latent_state=1)
    q001 = _question("q001", posterior_all=0.3, latent_state=0)
    q002 = _question("q002", posterior_all=0.5, latent_state=1)  # never reached

    events = [
        # q000: fully done (both agents' Pass-1 + one round of trades + settled).
        _belief("q000", "a0", 0.7),
        _belief("q000", "a1", 0.9),
        _trade("q000", "a0", 1, price_before=0.7, price_after=0.8),
        _trade("q000", "a1", 1, price_before=0.8, price_after=0.8),
        _settlement("q000", outcome=1, final_price=0.8),
        # q001: Pass-1 only done so far, market pass not reached, unsettled.
        _belief("q001", "a0", 0.3),
        _belief("q001", "a1", 0.3),
        # q002: nothing at all done yet.
    ]
    _write_run(run_dir, events=events, questions=[q000, q001, q002], costs=[])

    result = pilot.analyze_run(run_dir)

    assert result.n_questions_dataset == 3
    assert result.n_settled == 1
    assert result.unsettled_question_ids == ["q001", "q002"]
    assert any("unsettled" in n and "q001" in n and "q002" in n for n in result.notes)

    # Only q000 (settled, exact match) feeds the market/degeneracy averages.
    assert result.market_gap == pytest.approx(abs(0.8 - 0.8))
    assert result.degenerate_count == 0
    assert result.degeneracy_rate == pytest.approx(0.0)

    # turns done/expected: 4 beliefs + 2 trades + 1 settlement = 7 done;
    # inferred shape from the log: n_agents=2 (both showed up for q000's
    # Pass-1), n_rounds=1 (one trade round seen) -> expected =
    # 3*2 (beliefs) + 3*2*1 (trades) + 3 (settlements) = 6+6+3 = 15.
    assert result.turns_done == 7
    assert result.turns_expected == 15


def test_run_directory_with_no_files_reports_gracefully(tmp_path):
    run_dir = tmp_path / "never_started"
    result = pilot.analyze_run(run_dir)

    assert result.n_questions_dataset == 0
    assert result.n_settled == 0
    assert result.market_gap is None
    assert result.pool_gap is None
    assert result.degeneracy_rate is None
    assert result.total_cost == pytest.approx(0.0)
    assert any("no event log found" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 6. Cost summing across models, incl. entries with no turn key
# ---------------------------------------------------------------------------


def test_cost_summing_across_models_and_unkeyed_entries(tmp_path):
    run_dir = tmp_path / "run"
    q = _question("q000", posterior_all=0.6, latent_state=1)
    events = [
        _belief("q000", "a0", 0.5),
        _settlement("q000", outcome=1, final_price=0.6),
    ]
    costs = [
        _cost("model-A", 0.01, qid="q000", agent_id="a0"),
        _cost("model-A", 0.02, qid="q000", agent_id="a0"),
        _cost("model-B", 0.05),  # no turn key — still summed into totals
        _cost("model-B", 0.03),
    ]
    _write_run(run_dir, events=events, questions=[q], costs=costs)

    result = pilot.analyze_run(run_dir)

    assert result.cost_by_model == {
        "model-A": pytest.approx(0.03),
        "model-B": pytest.approx(0.08),
    }
    assert result.total_cost == pytest.approx(0.11)
    # model-B never appears in parse_fail_by_model / standalone_gap_by_model
    # (no turn key -> no agent_id -> model_id mapping).
    assert "model-B" not in result.parse_fail_by_model
    assert "model-B" not in result.standalone_gap_by_model


# ---------------------------------------------------------------------------
# 7. Reporting: aligned table + idempotent PROGRESS.md section
# ---------------------------------------------------------------------------


def test_format_report_produces_aligned_table_with_all_runs(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    q = _question("q000", posterior_all=0.6, latent_state=1)
    events_a = [
        _belief("q000", "a0", 0.5, condition="PILOT_A"),
        _settlement("q000", outcome=1, final_price=0.6, condition="PILOT_A"),
    ]
    events_b = [
        _belief("q000", "a0", 0.5, condition="PILOT_B"),
        _settlement("q000", outcome=1, final_price=0.6, condition="PILOT_B"),
    ]
    _write_run(run1, events=events_a, questions=[q], costs=[], condition="PILOT_A")
    _write_run(run2, events=events_b, questions=[q], costs=[], condition="PILOT_B")

    analyses = [pilot.analyze_run(run1), pilot.analyze_run(run2)]
    report = pilot.format_report(analyses)

    assert "PILOT_A" in report
    assert "PILOT_B" in report
    assert str(run1) in report
    assert str(run2) in report
    # Aligned table: header + separator + one row per run at minimum.
    lines = report.splitlines()
    assert lines[0].startswith("run")
    assert set(lines[1].replace(" ", "")) <= {"-"}


def test_write_progress_is_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    q = _question("q000", posterior_all=0.6, latent_state=1)
    events = [_belief("q000", "a0", 0.5), _settlement("q000", outcome=1, final_price=0.6)]
    _write_run(run_dir, events=events, questions=[q], costs=[])
    analyses = [pilot.analyze_run(run_dir)]

    progress = tmp_path / "PROGRESS.md"
    progress.write_text("# Pnyx\n\n## P2 dataset\n\nsome other section\n")

    pilot.write_progress(progress, analyses)
    first = progress.read_text()
    assert first.count("## P3 pilot") == 1
    assert "## P2 dataset" in first  # untouched

    pilot.write_progress(progress, analyses)
    second = progress.read_text()
    assert second.count("## P3 pilot") == 1
    assert second.count("## P2 dataset") == 1


# ---------------------------------------------------------------------------
# 8. CLI wiring
# ---------------------------------------------------------------------------


def test_cli_analyze_pilot_prints_report(tmp_path, capsys):
    run_dir = tmp_path / "run"
    q = _question("q000", posterior_all=0.6, latent_state=1)
    events = [_belief("q000", "a0", 0.5), _settlement("q000", outcome=1, final_price=0.6)]
    _write_run(run_dir, events=events, questions=[q], costs=[])

    rc = cli.main(["analyze-pilot", "--runs", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert CONDITION in out
    assert str(run_dir) in out


def test_cli_analyze_pilot_progress_flag_writes_section(tmp_path, capsys):
    run_dir = tmp_path / "run"
    q = _question("q000", posterior_all=0.6, latent_state=1)
    events = [_belief("q000", "a0", 0.5), _settlement("q000", outcome=1, final_price=0.6)]
    _write_run(run_dir, events=events, questions=[q], costs=[])
    progress = tmp_path / "PROGRESS.md"

    rc = cli.main([
        "analyze-pilot", "--runs", str(run_dir),
        "--progress", "--progress-path", str(progress),
    ])
    assert rc == 0
    assert progress.exists()
    assert "## P3 pilot" in progress.read_text()


def test_no_network_imports_in_analysis_pilot():
    import pnyx.analysis.pilot as pilot_mod

    src = Path(pilot_mod.__file__).read_text()
    for banned in ("import httpx", "import openai", "import requests",
                   "urllib.request", "aiohttp"):
        assert banned not in src, f"pnyx.analysis.pilot must not {banned!r}"

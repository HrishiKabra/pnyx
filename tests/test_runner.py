"""Tests for pnyx.runner — orchestration, event-log persistence, replay
reconstruction, determinism, settlement math, and wealth reset/persist.

These exercise the full mock pipeline end-to-end with ZERO API calls: the
only agents are MockAgent (a pure function of oracle posterior + seeded RNG),
so "no network" holds by construction — there is no provider/httpx import
anywhere on the P1 path (asserted structurally in test_no_network_imports).
"""

import json
from pathlib import Path

import pytest

from pnyx.env import subset_key
from pnyx.market import Market, max_affordable_shares
from pnyx.runner import (
    load_config,
    log_path_for,
    questions_path_for,
    read_events,
    run_experiment,
    ts_stripped_lines,
)
from pnyx.schemas import SettlementEvent, TradeEvent

MOCK_CONFIG = Path(__file__).resolve().parents[1] / "pnyx" / "configs" / "mock.yaml"


def _cfg(tmp_path: Path):
    """Load the shipped mock config but redirect its data dir into a tmp dir."""
    config = load_config(str(MOCK_CONFIG))
    return config.model_copy(update={"data_dir": str(tmp_path / "data")})


# ---------------------------------------------------------------------------
# End-to-end: 5 questions, full log, no network
# ---------------------------------------------------------------------------


def test_end_to_end_five_questions(tmp_path):
    config = _cfg(tmp_path)
    assert config.n_questions == 5
    run_experiment(config)

    seed = config.seeds[0]
    log = log_path_for(config, seed)
    assert log.exists()
    events = read_events(log)

    beliefs = [e for e in events if e.type == "belief"]
    trades = [e for e in events if e.type == "trade"]
    settlements = [e for e in events if e.type == "settlement"]

    n_agents = len(config.agents)
    # Pass 1: one belief per agent per question.
    assert len(beliefs) == config.n_questions * n_agents
    # Pass 2: n_rounds * n_agents trade turns per question (holds logged too).
    assert len(trades) == config.n_questions * config.n_rounds * n_agents
    # One settlement per question.
    assert len(settlements) == config.n_questions

    # Questions were persisted on first generation.
    assert questions_path_for(config, seed).exists()


def test_no_network_imports():
    # ZERO API calls by construction: no HTTP / provider client is importable
    # on the P1 runner path.
    import pnyx.agents
    import pnyx.runner

    for mod in (pnyx.runner, pnyx.agents):
        src = Path(mod.__file__).read_text()
        for banned in ("import httpx", "import openai", "import requests",
                       "urllib.request", "aiohttp"):
            assert banned not in src, f"{mod.__name__} must not {banned!r}"


# ---------------------------------------------------------------------------
# Determinism: two fresh runs -> byte-identical logs after stripping ts
# ---------------------------------------------------------------------------


def test_determinism_identical_logs(tmp_path):
    c1 = _cfg(tmp_path / "run1")
    c2 = _cfg(tmp_path / "run2")
    run_experiment(c1)
    run_experiment(c2)

    seed = c1.seeds[0]
    lines1 = ts_stripped_lines(log_path_for(c1, seed))
    lines2 = ts_stripped_lines(log_path_for(c2, seed))
    assert lines1 == lines2
    assert len(lines1) > 0


# ---------------------------------------------------------------------------
# Replay reconstruction equals live market state
# ---------------------------------------------------------------------------


def test_replay_reconstructs_market(tmp_path):
    config = _cfg(tmp_path)
    run_experiment(config)
    seed = config.seeds[0]
    events = read_events(log_path_for(config, seed))

    settlements = {e.question_id: e for e in events if e.type == "settlement"}

    # Group trades by question, in log order, and re-execute against a fresh
    # Market(b) — every logged price_after must reproduce within 1e-9, and the
    # final reconstructed price must equal the settlement's final_price.
    trades_by_q: dict[str, list[TradeEvent]] = {}
    for e in events:
        if e.type == "trade":
            trades_by_q.setdefault(e.key.question_id, []).append(e)

    for qid, trades in trades_by_q.items():
        market = Market(b=config.b)
        for t in trades:
            assert market.price() == pytest.approx(t.price_before, abs=1e-9)
            if t.trade.action != "hold" and t.executed_shares > 0:
                side = "yes" if t.trade.action == "buy_yes" else "no"
                paid = market.buy(side, t.executed_shares)
                assert paid == pytest.approx(t.cost, abs=1e-9)
            assert market.price() == pytest.approx(t.price_after, abs=1e-9)
        assert market.price() == pytest.approx(settlements[qid].final_price, abs=1e-9)


# ---------------------------------------------------------------------------
# Settlement math: payouts + subsidy recomputed independently from the log
# ---------------------------------------------------------------------------


def test_settlement_math(tmp_path):
    config = _cfg(tmp_path)
    run_experiment(config)
    seed = config.seeds[0]
    events = read_events(log_path_for(config, seed))

    # Reconstruct per-question holdings + market from trades, then verify the
    # SettlementEvent's payouts and subsidy against an independent computation.
    settlements = {e.question_id: e for e in events if e.type == "settlement"}
    trades_by_q: dict[str, list[TradeEvent]] = {}
    for e in events:
        if e.type == "trade":
            trades_by_q.setdefault(e.key.question_id, []).append(e)

    for qid, settle in settlements.items():
        market = Market(b=config.b)
        holdings: dict[str, dict[str, float]] = {}
        for t in trades_by_q.get(qid, []):
            aid = t.key.agent_id
            h = holdings.setdefault(aid, {"yes": 0.0, "no": 0.0})
            if t.trade.action != "hold" and t.executed_shares > 0:
                side = "yes" if t.trade.action == "buy_yes" else "no"
                market.buy(side, t.executed_shares)
                h[side] += t.executed_shares

        true_side = "yes" if settle.outcome == 1 else "no"
        for aid, h in holdings.items():
            expected_payout = h[true_side]  # $1/share on the true side
            assert settle.payouts.get(aid, 0.0) == pytest.approx(expected_payout, abs=1e-9)
        assert settle.subsidy == pytest.approx(market.subsidy(), abs=1e-9)
        assert settle.final_price == pytest.approx(market.price(), abs=1e-9)


# ---------------------------------------------------------------------------
# Wealth reset (default) vs. persistent flag
# ---------------------------------------------------------------------------


def _starting_bankrolls(events, config):
    """Per (question_id, agent_id) starting bankroll, derived from each
    agent's FIRST market turn in the question: start = bankroll_after + cost
    (a hold has cost 0, so start == bankroll_after)."""
    firsts: dict[tuple[str, str], TradeEvent] = {}
    for e in events:
        if e.type == "trade":
            k = (e.key.question_id, e.key.agent_id)
            if k not in firsts:
                firsts[k] = e
    return {k: t.bankroll_after + t.cost for k, t in firsts.items()}


def _ending_wealth(events, qid):
    """Per-agent ending wealth for a question: last trade's bankroll_after
    (cash left) + settlement payout."""
    settle = next(e for e in events if e.type == "settlement" and e.question_id == qid)
    last: dict[str, TradeEvent] = {}
    for e in events:
        if e.type == "trade" and e.key.question_id == qid:
            last[e.key.agent_id] = e  # log order => last write wins
    return {aid: t.bankroll_after + settle.payouts.get(aid, 0.0) for aid, t in last.items()}


def test_wealth_reset_default(tmp_path):
    config = _cfg(tmp_path)
    assert config.wealth_persistent is False
    run_experiment(config)
    events = read_events(log_path_for(config, config.seeds[0]))
    starts = _starting_bankrolls(events, config)
    base = config.agents[0].bankroll
    # Reset: every question starts every agent at the configured bankroll.
    for (_qid, _aid), start in starts.items():
        assert start == pytest.approx(base, abs=1e-9)


def test_wealth_persistent_carries_over(tmp_path):
    config = _cfg(tmp_path).model_copy(update={"wealth_persistent": True})
    run_experiment(config)
    events = read_events(log_path_for(config, config.seeds[0]))

    qids = [q for q in dict.fromkeys(
        e.key.question_id for e in events if e.type == "trade")]
    starts = _starting_bankrolls(events, config)
    base = config.agents[0].bankroll

    # First question: everyone starts at base.
    for aid in (a.agent_id for a in config.agents):
        assert starts[(qids[0], aid)] == pytest.approx(base, abs=1e-9)

    # Subsequent questions: each agent's start == prior question's ending wealth.
    for prev, cur in zip(qids, qids[1:]):
        ending = _ending_wealth(events, prev)
        for aid, wealth in ending.items():
            assert starts[(cur, aid)] == pytest.approx(wealth, abs=1e-9)


def test_reset_and_persistent_differ(tmp_path):
    reset = _cfg(tmp_path / "reset")
    persist = _cfg(tmp_path / "persist").model_copy(update={"wealth_persistent": True})
    run_experiment(reset)
    run_experiment(persist)
    a = ts_stripped_lines(log_path_for(reset, reset.seeds[0]))
    b = ts_stripped_lines(log_path_for(persist, persist.seeds[0]))
    # The two policies produce different trading behaviour after question 1.
    assert a != b


# ---------------------------------------------------------------------------
# Idempotent resume: re-running a completed run changes nothing
# ---------------------------------------------------------------------------


def test_rerun_completed_is_noop(tmp_path):
    config = _cfg(tmp_path)
    run_experiment(config)
    log = log_path_for(config, config.seeds[0])
    before = log.read_text()
    run_experiment(config)  # resume of an already-complete run
    after = log.read_text()
    assert before == after

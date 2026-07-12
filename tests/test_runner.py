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
from pnyx.schemas import QuestionRecord, RunConfig, SettlementEvent, TradeEvent

from tests._fixtures import make_rendered_question, write_questions_file

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
    # on the P1 runner path or the P2 dataset tooling.
    import pnyx.agents
    import pnyx.cli
    import pnyx.dataset
    import pnyx.runner

    for mod in (pnyx.runner, pnyx.agents, pnyx.dataset, pnyx.cli):
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


# ---------------------------------------------------------------------------
# Post-trade bankroll floors at 0.0 despite full-clamp float residue
# ---------------------------------------------------------------------------


def test_bankroll_floors_at_zero_on_full_clamp_residue():
    """max_affordable_shares round-trips through log/exp, so a full-clamp buy
    (requested shares far exceeding affordability) can cost a hair more than
    cash — here ~5.68e-13. Without runner.py's max(0.0, cash - cost) floor,
    bankroll_after would go slightly negative and fail TradeView's ge=0
    validation on the agent's next turn."""
    b, q_yes, q_no = 100.0, 7.476736061608236, 42.44350282216679
    cash, side = 997.8406280338953, "yes"
    market = Market(q_yes, q_no, b)

    # Requested shares far exceed affordability, so clamp returns the full
    # affordable amount (mirrors runner.py's live-execution path).
    executed = market.clamp_shares(side, shares=1_000_000.0, cash=cash)
    cost = market.buy(side, executed)

    # Confirm this scenario actually exercises the residue the fix guards.
    assert cost > cash

    bankroll_after = max(0.0, cash - cost)  # runner.py's post-trade update
    assert bankroll_after >= 0.0
    assert bankroll_after == 0.0


# ---------------------------------------------------------------------------
# questions_file loading path (P2)
# ---------------------------------------------------------------------------


def _file_config(tmp_path: Path, qfile: Path, *, n_questions: int) -> RunConfig:
    base = load_config(str(MOCK_CONFIG))
    return base.model_copy(update={
        "condition": "FILE",
        "n_questions": n_questions,
        "data_dir": str(tmp_path / "data"),
        "questions_file": str(qfile),
    })


def test_questions_file_is_loaded_and_persisted_to_sidecar(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 3)
    config = _file_config(tmp_path, qfile, n_questions=3)
    run_experiment(config)

    seed = config.seeds[0]
    # Sidecar persisted from the dataset file, byte-for-byte record set.
    sidecar = questions_path_for(config, seed)
    assert sidecar.exists()
    loaded = [QuestionRecord.model_validate_json(line)
              for line in sidecar.read_text().splitlines() if line]
    assert [r.question_id for r in loaded] == ["q000", "q001", "q002"]
    # Records carry the rendered content the market runs on.
    for r in loaded:
        assert r.shards is not None and len(r.shards) == 6
        assert r.question_text

    # Event log settles exactly the file's questions (not generated ids).
    events = read_events(log_path_for(config, seed))
    settled = {e.question_id for e in events if e.type == "settlement"}
    assert settled == {"q000", "q001", "q002"}


def test_questions_file_subset_takes_first_n(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 5)
    config = _file_config(tmp_path, qfile, n_questions=2)
    run_experiment(config)
    events = read_events(log_path_for(config, config.seeds[0]))
    settled = {e.question_id for e in events if e.type == "settlement"}
    assert settled == {"q000", "q001"}


def test_questions_file_too_short_fails_loudly(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 2)
    config = _file_config(tmp_path, qfile, n_questions=3)
    with pytest.raises(ValueError, match="at least"):
        run_experiment(config)


def test_questions_file_missing_shards_fails_loudly(tmp_path):
    # A record with no shards must be rejected: the market needs rendered prose.
    rec = make_rendered_question("q000", seed=1).model_copy(update={"shards": None})
    qfile = tmp_path / "ds.jsonl"
    qfile.write_text(rec.model_dump_json() + "\n")
    config = _file_config(tmp_path, qfile, n_questions=1)
    with pytest.raises(ValueError, match="shards"):
        run_experiment(config)


def test_questions_file_missing_question_text_fails_loudly(tmp_path):
    rec = make_rendered_question("q000", seed=1).model_copy(update={"question_text": None})
    qfile = tmp_path / "ds.jsonl"
    qfile.write_text(rec.model_dump_json() + "\n")
    config = _file_config(tmp_path, qfile, n_questions=1)
    with pytest.raises(ValueError, match="question_text"):
        run_experiment(config)


def test_questions_file_deterministic_logs(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 3)
    c1 = _file_config(tmp_path / "a", qfile, n_questions=3)
    c2 = _file_config(tmp_path / "b", qfile, n_questions=3)
    run_experiment(c1)
    run_experiment(c2)
    l1 = ts_stripped_lines(log_path_for(c1, c1.seeds[0]))
    l2 = ts_stripped_lines(log_path_for(c2, c2.seeds[0]))
    assert l1 == l2 and len(l1) > 0

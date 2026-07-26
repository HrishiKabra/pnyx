"""Tests for pnyx.analysis.replay — the (Streamlit-free) replay data layer.

Covers run discovery on a tmp mini-grid, exact-value replay assembly on a
hand-built event sequence (including the condition-D adversary flag and the
D <- C Pass-1 resolution via the sibling source directory), the two derived
views (price series + herding), and the invariant that neither this test
module nor ``pnyx.analysis.replay`` imports Streamlit.

All fixtures are hand-built synthetic logs written by the tests themselves
(pnyx.runner naming convention). No network, no LLM calls, no Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pnyx.analysis import replay
from pnyx.schemas import (
    Belief,
    BeliefEvent,
    QuestionRecord,
    SettlementEvent,
    SignalRecord,
    Trade,
    TradeEvent,
    TurnKey,
)


# ---------------------------------------------------------------------------
# Fixture builders (pnyx.runner naming convention)
# ---------------------------------------------------------------------------


def _question(qid: str, *, posterior_all: float, latent_state: int) -> QuestionRecord:
    return QuestionRecord(
        question_id=qid,
        latent_state=latent_state,
        signals=[
            SignalRecord(index=0, value=1, accuracy=0.7, lam=0.0),
            SignalRecord(index=1, value=1, accuracy=0.7, lam=0.0),
        ],
        posterior_table={"": 0.5, "0": 0.6, "1": 0.6, "0,1": posterior_all},
        shards=["shard 0", "shard 1"],
        question_text=f"Does {qid} resolve yes?",
    )


def _belief(qid, agent_id, prob, *, condition, seed):
    return BeliefEvent(
        key=TurnKey(
            condition=condition, seed=seed, question_id=qid,
            phase="independent", round=0, agent_id=agent_id,
        ),
        belief=Belief(prob=prob, rationale="r"),
        ts=None,
    )


def _trade(
    qid, agent_id, *, condition, seed, round_, action, belief, shares,
    price_before, price_after, cost=0.0, bankroll_after=100.0, parse_failed=False,
):
    return TradeEvent(
        key=TurnKey(
            condition=condition, seed=seed, question_id=qid,
            phase="market", round=round_, agent_id=agent_id,
        ),
        trade=Trade(belief=belief, action=action, shares=shares, rationale="r"),
        executed_shares=shares,
        cost=cost,
        price_before=price_before,
        price_after=price_after,
        bankroll_after=bankroll_after,
        parse_failed=parse_failed,
        ts=None,
    )


def _settlement(qid, *, condition, seed, outcome, final_price, subsidy=1.0):
    return SettlementEvent(
        condition=condition, seed=seed, question_id=qid, outcome=outcome,
        payouts={}, final_price=final_price, subsidy=subsidy, ts=None,
    )


def _write_run(
    root: Path, condition: str, seed: int, events, questions
) -> None:
    """Write one ``(condition, seed)`` event log + question sidecar under
    ``root/<condition>/`` in the runner's filename convention."""
    d = root / condition
    d.mkdir(parents=True, exist_ok=True)
    log = d / f"{condition}_seed{seed}.jsonl"
    log.write_text("".join(e.to_jsonl_line() + "\n" for e in events))
    sidecar = d / f"{condition}_seed{seed}.questions.jsonl"
    sidecar.write_text("".join(q.model_dump_json() + "\n" for q in questions))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_runs_discovers_mini_grid(tmp_path):
    main = tmp_path / "main"
    v2 = tmp_path / "v2"
    q = _question("q000", posterior_all=0.9, latent_state=1)
    for cond, seed in [("A", 0), ("A", 1), ("D_K10", 0)]:
        _write_run(main, cond, seed, [_settlement("q000", condition=cond, seed=seed, outcome=1, final_price=0.7)], [q])
    _write_run(v2, "A_PRO", 0, [_settlement("q000", condition="A_PRO", seed=0, outcome=1, final_price=0.7)], [q])
    # A stray costs ledger must NOT be discovered as a run.
    (main / "A" / "A.costs.jsonl").write_text("")

    runs = replay.list_runs([main, v2])
    got = sorted((r.condition, r.seed) for r in runs)
    assert got == [("A", 0), ("A", 1), ("A_PRO", 0), ("D_K10", 0)]
    a1 = next(r for r in runs if r.condition == "A" and r.seed == 1)
    assert a1.run_dir == main / "A"
    assert a1.root == main


def test_list_runs_skips_missing_root(tmp_path):
    assert replay.list_runs([tmp_path / "does_not_exist"]) == []


# ---------------------------------------------------------------------------
# Replay assembly — two-pass condition (exact values)
# ---------------------------------------------------------------------------


def _two_pass_events(condition="A", seed=0):
    """A 2-agent, 2-round market on one question with hand-set prices."""
    q = _question("q000", posterior_all=0.9, latent_state=1)
    beliefs = [
        _belief("q000", "p0", 0.70, condition=condition, seed=seed),
        _belief("q000", "p1", 0.40, condition=condition, seed=seed),
    ]
    trades = [
        # round 1
        _trade("q000", "p0", condition=condition, seed=seed, round_=1,
               action="buy_yes", belief=0.72, shares=10.0,
               price_before=0.50, price_after=0.60, cost=5.5),
        _trade("q000", "p1", condition=condition, seed=seed, round_=1,
               action="buy_no", belief=0.45, shares=4.0,
               price_before=0.60, price_after=0.55, cost=2.0),
        # round 2
        _trade("q000", "p0", condition=condition, seed=seed, round_=2,
               action="buy_yes", belief=0.80, shares=6.0,
               price_before=0.55, price_after=0.65, cost=3.6),
        _trade("q000", "p1", condition=condition, seed=seed, round_=2,
               action="hold", belief=0.50, shares=0.0,
               price_before=0.65, price_after=0.65, cost=0.0),
    ]
    settlement = _settlement("q000", condition=condition, seed=seed, outcome=1, final_price=0.65, subsidy=2.5)
    return q, beliefs, trades, settlement


def test_question_replay_two_pass_exact(tmp_path):
    q, beliefs, trades, settlement = _two_pass_events()
    _write_run(tmp_path / "main", "A", 0, beliefs + trades + [settlement], [q])
    cond, source = replay.load_run(replay.RunRef("A", 0, tmp_path / "main" / "A"))
    assert source is None  # two-pass condition resolves no external source

    rep = replay.question_replay(cond, 0, "q000", pass1_source=source)
    assert rep.condition == "A"
    assert rep.posterior_all == 0.9
    assert rep.outcome == 1
    assert rep.final_price == 0.65
    assert rep.subsidy == 2.5
    assert rep.question_text == "Does q000 resolve yes?"
    assert rep.has_adversary is False
    assert rep.pre_attack_price is None
    assert rep.pass1_source_condition is None
    assert rep.n_rounds == 2
    assert rep.pass1_by_agent == {"p0": 0.70, "p1": 0.40}

    # Steps in execution (log) order with 0-based step index.
    assert [s.step for s in rep.steps] == [0, 1, 2, 3]
    assert [s.agent_id for s in rep.steps] == ["p0", "p1", "p0", "p1"]
    assert [s.round for s in rep.steps] == [1, 1, 2, 2]
    assert [s.action for s in rep.steps] == ["buy_yes", "buy_no", "buy_yes", "hold"]
    assert rep.steps[0].price_before == 0.50 and rep.steps[0].price_after == 0.60
    assert rep.steps[0].stated_belief == 0.72
    assert all(s.is_adversary is False for s in rep.steps)


def test_price_series_exact():
    q, beliefs, trades, settlement = _two_pass_events()
    cond = replay.ConditionData(condition="A")
    from pnyx.analysis.mainrun import SeedData
    cond.seeds[0] = SeedData(seed=0, beliefs=beliefs, trades=trades,
                             settlements=[settlement], questions={"q000": q})
    rep = replay.question_replay(cond, 0, "q000")
    ps = replay.price_series(rep)
    # step 0 = initial price (first trade's price_before), then price_after each.
    assert [(p.step, p.price) for p in ps.points] == [
        (0, 0.50), (1, 0.60), (2, 0.55), (3, 0.65), (4, 0.65),
    ]
    assert ps.posterior_all == 0.9
    assert ps.outcome == 1
    assert ps.final_price == 0.65
    # Two rounds, each covering two trade steps (point indices 1-2 and 3-4).
    assert [(b.round, b.start_step, b.end_step) for b in ps.round_boundaries] == [
        (1, 1, 2), (2, 3, 4),
    ]


def test_herding_exact():
    q, beliefs, trades, settlement = _two_pass_events()
    cond = replay.ConditionData(condition="A")
    from pnyx.analysis.mainrun import SeedData
    cond.seeds[0] = SeedData(seed=0, beliefs=beliefs, trades=trades,
                             settlements=[settlement], questions={"q000": q})
    rep = replay.question_replay(cond, 0, "q000")
    h = replay.herding(rep)
    assert [a.agent_id for a in h] == ["p0", "p1"]
    p0 = h[0]
    assert p0.pass1_belief == 0.70
    assert [(r.round, r.stated_belief, r.price_at_trade) for r in p0.by_round] == [
        (1, 0.72, 0.50), (2, 0.80, 0.55),
    ]
    p1 = h[1]
    assert p1.pass1_belief == 0.40
    assert [(r.round, r.stated_belief, r.price_at_trade) for r in p1.by_round] == [
        (1, 0.45, 0.60), (2, 0.50, 0.65),
    ]


# ---------------------------------------------------------------------------
# Replay assembly — pass2-only D condition, adversary + source resolution
# ---------------------------------------------------------------------------


def test_question_replay_adversary_and_source_resolution(tmp_path):
    main = tmp_path / "main"
    q = _question("q000", posterior_all=0.85, latent_state=1)

    # Source condition C carries the honest Pass-1 beliefs (two-pass).
    c_beliefs = [
        _belief("q000", "p0", 0.66, condition="C", seed=0),
        _belief("q000", "p1", 0.80, condition="C", seed=0),
    ]
    c_trades = [
        _trade("q000", "p0", condition="C", seed=0, round_=1, action="buy_yes",
               belief=0.66, shares=5.0, price_before=0.5, price_after=0.55),
        _trade("q000", "p1", condition="C", seed=0, round_=1, action="buy_yes",
               belief=0.80, shares=5.0, price_before=0.55, price_after=0.60),
    ]
    c_settlement = _settlement("q000", condition="C", seed=0, outcome=1, final_price=0.60)
    _write_run(main, "C", 0, c_beliefs + c_trades + [c_settlement], [q])

    # Pass2-only D_K10: no belief events; an adversary adv0 pushes toward NO.
    d_trades = [
        _trade("q000", "adv0", condition="D_K10", seed=0, round_=1, action="buy_no",
               belief=0.10, shares=100.0, price_before=0.60, price_after=0.30,
               cost=40.0, bankroll_after=960.0),
        _trade("q000", "p0", condition="D_K10", seed=0, round_=1, action="buy_yes",
               belief=0.66, shares=8.0, price_before=0.30, price_after=0.42, cost=3.0),
        _trade("q000", "p1", condition="D_K10", seed=0, round_=2, action="buy_yes",
               belief=0.80, shares=8.0, price_before=0.42, price_after=0.55, cost=4.0),
    ]
    d_settlement = _settlement("q000", condition="D_K10", seed=0, outcome=1, final_price=0.55)
    _write_run(main, "D_K10", 0, d_trades + [d_settlement], [q])

    cond, source = replay.load_run(replay.RunRef("D_K10", 0, main / "D_K10"))
    assert source is not None and source.condition == "C"

    rep = replay.question_replay(cond, 0, "q000", pass1_source=source)
    # Pass-1 resolved from the C source, NOT from D_K10 (which has none).
    assert rep.pass1_by_agent == {"p0": 0.66, "p1": 0.80}
    assert rep.pass1_source_condition == "C"
    # Adversary flagged; pre-attack price = its first trade's price_before.
    assert rep.has_adversary is True
    assert rep.pre_attack_price == 0.60
    adv = [s for s in rep.steps if s.is_adversary]
    assert len(adv) == 1 and adv[0].agent_id == "adv0" and adv[0].action == "buy_no"
    # Herding lists honest agents only (adversary excluded).
    assert [a.agent_id for a in replay.herding(rep)] == ["p0", "p1"]


def test_question_replay_raises_on_unsettled():
    q = _question("q000", posterior_all=0.9, latent_state=1)
    from pnyx.analysis.mainrun import SeedData
    cond = replay.ConditionData(condition="A")
    cond.seeds[0] = SeedData(seed=0, beliefs=[], trades=[], settlements=[], questions={"q000": q})
    with pytest.raises(ValueError, match="not settled"):
        replay.question_replay(cond, 0, "q000")


# ---------------------------------------------------------------------------
# No-Streamlit invariant
# ---------------------------------------------------------------------------


def test_replay_module_does_not_import_streamlit():
    # Importing the data layer must not pull in streamlit.
    assert "streamlit" not in sys.modules
    # The data layer source contains no streamlit import (AST-level check, so a
    # string literal mentioning the word would not trip it).
    import ast

    tree = ast.parse(Path(replay.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported

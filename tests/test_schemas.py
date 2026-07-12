"""Tests for pnyx/schemas.py — the single source of truth for all P1
Pydantic models (structured LLM outputs, environment records, and the
event log).
"""

import json

import pytest
from pydantic import ValidationError

from pnyx.schemas import (
    Belief,
    BeliefEvent,
    Event,
    QuestionRecord,
    SettlementEvent,
    SignalRecord,
    Trade,
    TradeEvent,
    TurnKey,
    parse_event,
)


# --------------------------------------------------------------------------
# Belief / Trade — field validation bounds
# --------------------------------------------------------------------------


def test_belief_accepts_valid_prob():
    b = Belief(prob=0.5, rationale="fifty fifty")
    assert b.prob == 0.5


def test_belief_prob_above_one_rejected():
    with pytest.raises(ValidationError):
        Belief(prob=1.5, rationale="too high")


def test_belief_prob_below_zero_rejected():
    with pytest.raises(ValidationError):
        Belief(prob=-0.1, rationale="too low")


def test_belief_rationale_over_max_length_rejected():
    with pytest.raises(ValidationError):
        Belief(prob=0.5, rationale="x" * 401)


def test_belief_rationale_at_max_length_accepted():
    b = Belief(prob=0.5, rationale="x" * 400)
    assert len(b.rationale) == 400


def test_trade_action_literal_enforced():
    with pytest.raises(ValidationError):
        Trade(belief=0.5, action="sell_yes", shares=1.0, rationale="bad action")


def test_trade_action_accepts_each_valid_literal():
    for action in ("buy_yes", "buy_no", "hold"):
        t = Trade(belief=0.5, action=action, shares=1.0, rationale="ok")
        assert t.action == action


def test_trade_shares_negative_rejected():
    with pytest.raises(ValidationError):
        Trade(belief=0.5, action="hold", shares=-1.0, rationale="negative shares")


def test_trade_shares_zero_accepted():
    t = Trade(belief=0.5, action="hold", shares=0.0, rationale="zero is fine")
    assert t.shares == 0.0


# --------------------------------------------------------------------------
# SignalRecord
# --------------------------------------------------------------------------


def test_signal_record_accepts_valid_value():
    s = SignalRecord(index=0, value=1, accuracy=0.8, lam=0.0)
    assert s.value == 1


def test_signal_record_value_must_be_zero_or_one():
    with pytest.raises(ValidationError):
        SignalRecord(index=0, value=2, accuracy=0.8, lam=0.0)


def test_signal_record_lam_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        SignalRecord(index=0, value=1, accuracy=0.8, lam=1.5)
    with pytest.raises(ValidationError):
        SignalRecord(index=0, value=1, accuracy=0.8, lam=-0.1)


def test_signal_record_lam_bounds_accepted():
    lo = SignalRecord(index=0, value=1, accuracy=0.8, lam=0.0)
    hi = SignalRecord(index=0, value=1, accuracy=0.8, lam=1.0)
    assert lo.lam == 0.0
    assert hi.lam == 1.0


# --------------------------------------------------------------------------
# QuestionRecord — posterior_table subset-key format
# --------------------------------------------------------------------------


def _make_signals(n: int) -> list[SignalRecord]:
    return [SignalRecord(index=i, value=1, accuracy=0.8, lam=0.0) for i in range(n)]


def test_question_record_accepts_valid_posterior_table_keys():
    qr = QuestionRecord(
        question_id="q1",
        latent_state=1,
        signals=_make_signals(3),
        posterior_table={
            "": 0.5,
            "0": 0.7,
            "0,2": 0.8,
            "0,1,2": 0.9,
        },
    )
    assert qr.posterior_table[""] == 0.5
    assert qr.posterior_table["0,1,2"] == 0.9


def test_question_record_defaults_shards_none_and_meta_empty():
    qr = QuestionRecord(
        question_id="q1",
        latent_state=0,
        signals=_make_signals(1),
        posterior_table={"": 0.5, "0": 0.6},
    )
    assert qr.shards is None
    assert qr.meta == {}


def test_question_record_rejects_unsorted_posterior_key():
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="q1",
            latent_state=1,
            signals=_make_signals(3),
            posterior_table={"2,0": 0.8},
        )


def test_question_record_rejects_duplicate_index_in_posterior_key():
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="q1",
            latent_state=1,
            signals=_make_signals(3),
            posterior_table={"0,0": 0.8},
        )


def test_question_record_rejects_non_numeric_posterior_key():
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="q1",
            latent_state=1,
            signals=_make_signals(3),
            posterior_table={"a,b": 0.8},
        )


def test_question_record_latent_state_must_be_zero_or_one():
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="q1",
            latent_state=2,
            signals=_make_signals(1),
            posterior_table={"": 0.5},
        )


# --------------------------------------------------------------------------
# TurnKey — frozen/hashable, as_tuple()
# --------------------------------------------------------------------------


def _make_turn_key(**overrides) -> TurnKey:
    fields = dict(
        condition="MOCK",
        seed=1,
        question_id="q1",
        phase="independent",
        round=0,
        agent_id="a0",
    )
    fields.update(overrides)
    return TurnKey(**fields)


def test_turn_key_is_frozen():
    key = _make_turn_key()
    with pytest.raises(ValidationError):
        key.condition = "OTHER"


def test_turn_key_is_hashable_and_usable_in_a_set():
    k1 = _make_turn_key()
    k2 = _make_turn_key()
    k3 = _make_turn_key(agent_id="a1")
    assert hash(k1) == hash(k2)
    s = {k1, k2, k3}
    assert len(s) == 2


def test_turn_key_as_tuple():
    key = _make_turn_key()
    assert key.as_tuple() == ("MOCK", 1, "q1", "independent", 0, "a0")


def test_turn_key_phase_literal_enforced():
    with pytest.raises(ValidationError):
        _make_turn_key(phase="bogus")


# --------------------------------------------------------------------------
# Event log — JSONL round-trip, discriminated union dispatch
# --------------------------------------------------------------------------


def _belief_event(ts: float | None = 123.456) -> BeliefEvent:
    return BeliefEvent(
        key=_make_turn_key(),
        belief=Belief(prob=0.62, rationale="signal favors yes"),
        ts=ts,
    )


def _trade_event(ts: float | None = 789.0) -> TradeEvent:
    return TradeEvent(
        key=_make_turn_key(phase="market", round=1),
        trade=Trade(belief=0.62, action="buy_yes", shares=5.0, rationale="cheap yes"),
        executed_shares=3.2,
        cost=1.75,
        price_before=0.5,
        price_after=0.58,
        bankroll_after=98.25,
        ts=ts,
    )


def _settlement_event(ts: float | None = 999.0) -> SettlementEvent:
    return SettlementEvent(
        condition="MOCK",
        seed=1,
        question_id="q1",
        outcome=1,
        payouts={"a0": 5.0, "a1": 0.0},
        final_price=0.83,
        subsidy=4.2,
        ts=ts,
    )


def test_belief_event_jsonl_roundtrip():
    e = _belief_event()
    line = e.to_jsonl_line()
    parsed = parse_event(line)
    assert parsed == e
    assert isinstance(parsed, BeliefEvent)


def test_trade_event_jsonl_roundtrip():
    e = _trade_event()
    line = e.to_jsonl_line()
    parsed = parse_event(line)
    assert parsed == e
    assert isinstance(parsed, TradeEvent)


def test_settlement_event_jsonl_roundtrip():
    e = _settlement_event()
    line = e.to_jsonl_line()
    parsed = parse_event(line)
    assert parsed == e
    assert isinstance(parsed, SettlementEvent)


def test_belief_event_jsonl_roundtrip_with_ts_none():
    e = _belief_event(ts=None)
    parsed = parse_event(e.to_jsonl_line())
    assert parsed == e
    assert parsed.ts is None


def test_discriminated_union_dispatches_on_type_field():
    belief_line = _belief_event().to_jsonl_line()
    trade_line = _trade_event().to_jsonl_line()
    settlement_line = _settlement_event().to_jsonl_line()

    assert isinstance(parse_event(belief_line), BeliefEvent)
    assert isinstance(parse_event(trade_line), TradeEvent)
    assert isinstance(parse_event(settlement_line), SettlementEvent)


def test_parse_event_rejects_unknown_type():
    bogus = json.dumps({"type": "bogus", "ts": None})
    with pytest.raises(ValidationError):
        parse_event(bogus)


def test_belief_event_parse_failed_defaults_false():
    e = _belief_event()
    assert e.parse_failed is False


# --------------------------------------------------------------------------
# Canonical serialization: sorted keys at every nesting level, so logs are
# byte-comparable across runs once the wall-clock `ts` field is stripped.
# --------------------------------------------------------------------------


def test_to_jsonl_line_has_sorted_keys_at_every_level():
    line = _trade_event().to_jsonl_line()
    parsed = json.loads(line)

    def assert_sorted(obj):
        if isinstance(obj, dict):
            keys = list(obj.keys())
            assert keys == sorted(keys)
            for value in obj.values():
                assert_sorted(value)

    assert_sorted(parsed)


def test_to_jsonl_line_is_single_line_no_trailing_whitespace():
    line = _belief_event().to_jsonl_line()
    assert "\n" not in line
    assert line == line.strip()


def test_events_identical_except_ts_are_byte_identical_once_ts_stripped():
    e1 = _belief_event(ts=1.0)
    e2 = _belief_event(ts=2.0)

    d1 = json.loads(e1.to_jsonl_line())
    d2 = json.loads(e2.to_jsonl_line())
    d1.pop("ts")
    d2.pop("ts")

    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


# --------------------------------------------------------------------------
# Event type alias sanity (used as a discriminated-union type elsewhere)
# --------------------------------------------------------------------------


def test_event_type_alias_is_exported():
    # Event is a type alias (Annotated Union), not a class — just confirm
    # it's importable and usable as documented.
    assert Event is not None

"""Tests for pnyx.agents.LLMAgent — prompt building, the one-retry parse
contract, billing of every call, and park/error propagation.

Zero network: the provider is a fake object with an async ``complete`` matching
``Provider.complete``'s signature. Billing is captured via a list-appending
callback (the runner supplies a ledger-writing one).
"""

import asyncio

import pytest

from pnyx.agents import LLMAgent
from pnyx.providers import ProviderError, SchemaParseError, TurnParked
from pnyx.schemas import AgentSpec, Belief, ModelSpec, Trade, TradeView, BeliefView, Usage

MODEL = ModelSpec(
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_KEY",
    model_id="deepseek/deepseek-v4-flash",
    price_in=0.077, price_out=0.154, rpm_limit=300, supports_json_schema=True,
)


class FakeProvider:
    """Scripted provider: each entry in ``script`` is either a
    ``(result_obj, Usage)`` tuple to return, or an Exception to raise. Records
    the messages/schema of every call for assertions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def complete(self, messages, schema, model_spec, temperature, max_tokens):
        self.calls.append(
            dict(messages=messages, schema=schema, model_spec=model_spec,
                 temperature=temperature, max_tokens=max_tokens)
        )
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _spec(agent_id="a0", persona="contrarian"):
    return AgentSpec(agent_id=agent_id, shard_indices=[0], kind="llm",
                     model="m", persona=persona)


def _belief_view():
    return BeliefView(
        question_id="q0", signal_values={0: 1}, persona="contrarian",
        question_text="Will it rain?", shard_texts=["Clouds gathered."],
    )


def _trade_view():
    return TradeView(
        question_id="q0", signal_values={0: 1}, persona="contrarian",
        question_text="Will it rain?", shard_texts=["Clouds gathered."],
        price=0.4, round=1, n_rounds=3, bankroll=100.0,
        max_affordable_yes=50.0, max_affordable_no=40.0,
    )


def _agent(provider):
    return LLMAgent(_spec(), MODEL, "contrarian", provider,
                    temperature=0.7, max_tokens=256)


def _bills():
    billed = []
    return billed, (lambda u: billed.append(u))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_elicit_belief_success_bills_once():
    usage = Usage(prompt_tokens=10, completion_tokens=5)
    prov = FakeProvider([(Belief(prob=0.7, rationale="ok"), usage)])
    billed, bill = _bills()
    res = asyncio.run(_agent(prov).elicit_belief(_belief_view(), bill=bill))
    assert res.parse_failed is False
    assert res.belief.prob == 0.7
    assert billed == [usage]
    # Correct schema + prompt content passed through.
    call = prov.calls[0]
    assert call["schema"] is Belief
    assert call["max_tokens"] == 256
    assert any("Will it rain?" in m["content"] for m in call["messages"])


def test_decide_trade_passes_pass1_prob_into_prompt():
    usage = Usage(prompt_tokens=8, completion_tokens=4)
    prov = FakeProvider([(Trade(belief=0.6, action="buy_yes", shares=3.0, rationale="x"), usage)])
    billed, bill = _bills()
    res = asyncio.run(_agent(prov).decide_trade(_trade_view(), pass1_prob=0.81, bill=bill))
    assert res.parse_failed is False
    assert res.trade.action == "buy_yes"
    text = "\n".join(m["content"] for m in prov.calls[0]["messages"])
    assert "0.81" in text and "prior analysis concluded" in text


# ---------------------------------------------------------------------------
# One-retry parse contract
# ---------------------------------------------------------------------------


def test_parse_failure_retries_once_then_succeeds():
    u1 = Usage(prompt_tokens=10, completion_tokens=0)
    u2 = Usage(prompt_tokens=12, completion_tokens=6)
    prov = FakeProvider([
        SchemaParseError(usage=u1, content="not json", error="invalid json"),
        (Belief(prob=0.55, rationale="second try"), u2),
    ])
    billed, bill = _bills()
    res = asyncio.run(_agent(prov).elicit_belief(_belief_view(), bill=bill))
    assert res.parse_failed is False
    assert res.belief.prob == 0.55
    # BOTH calls billed (failed attempt + successful retry).
    assert billed == [u1, u2]
    # Retry carried the validation error text forward.
    retry_msgs = prov.calls[1]["messages"]
    assert any("invalid json" in m["content"] for m in retry_msgs)
    assert len(retry_msgs) == len(prov.calls[0]["messages"]) + 1


def test_second_parse_failure_yields_hold_fallback_flagged():
    u1 = Usage(prompt_tokens=10, completion_tokens=0)
    u2 = Usage(prompt_tokens=11, completion_tokens=0)
    prov = FakeProvider([
        SchemaParseError(usage=u1, content="x", error="e1"),
        SchemaParseError(usage=u2, content="y", error="e2"),
    ])
    billed, bill = _bills()
    res = asyncio.run(_agent(prov).decide_trade(_trade_view(), pass1_prob=0.5, bill=bill))
    assert res.parse_failed is True
    assert res.trade.action == "hold"
    assert res.trade.belief == 0.5
    assert res.trade.shares == 0.0
    assert billed == [u1, u2]  # both failures billed


def test_belief_second_parse_failure_yields_half_prob():
    prov = FakeProvider([
        SchemaParseError(usage=Usage(), content="x", error="e1"),
        SchemaParseError(usage=Usage(), content="y", error="e2"),
    ])
    billed, bill = _bills()
    res = asyncio.run(_agent(prov).elicit_belief(_belief_view(), bill=bill))
    assert res.parse_failed is True
    assert res.belief.prob == 0.5


# ---------------------------------------------------------------------------
# Park / provider-error propagation
# ---------------------------------------------------------------------------


def test_turn_parked_propagates_unbilled():
    prov = FakeProvider([TurnParked("parked")])
    billed, bill = _bills()
    with pytest.raises(TurnParked):
        asyncio.run(_agent(prov).elicit_belief(_belief_view(), bill=bill))
    assert billed == []  # nothing to bill on a park


def test_provider_error_bills_usage_then_propagates():
    usage = Usage(prompt_tokens=7, completion_tokens=0)
    prov = FakeProvider([ProviderError("malformed 200", usage=usage)])
    billed, bill = _bills()
    with pytest.raises(ProviderError):
        asyncio.run(_agent(prov).decide_trade(_trade_view(), pass1_prob=0.5, bill=bill))
    assert billed == [usage]  # billed the incurred usage before re-raising


def test_provider_error_on_retry_bills_both():
    u1 = Usage(prompt_tokens=5, completion_tokens=0)
    u2 = Usage(prompt_tokens=6, completion_tokens=0)
    prov = FakeProvider([
        SchemaParseError(usage=u1, content="x", error="e1"),
        ProviderError("500 on retry", usage=u2),
    ])
    billed, bill = _bills()
    with pytest.raises(ProviderError):
        asyncio.run(_agent(prov).elicit_belief(_belief_view(), bill=bill))
    assert billed == [u1, u2]

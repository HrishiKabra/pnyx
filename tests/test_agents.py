"""Tests for pnyx.agents — the Agent protocol + MockAgent (zero-intelligence
trader).

MockAgent is a pure function of (oracle posterior, RNG state): given the
same seed it must reproduce identical beliefs and trades. The trade-rule
tests below pin the exact belief MockAgent will draw for a fixed seed by
replaying the same RNG call the implementation makes (one
``rng.normal(0.0, sigma)`` draw) on an independently-constructed generator
with the same seed — the same "two independently-seeded generators agree"
pattern used in tests/test_env.py — and then choose prices relative to that
known value so the trade-rule assertions are exact, not probabilistic.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from pnyx.agents import Agent, AgentSpec, MockAgent
from pnyx.schemas import Belief, BeliefView, Trade, TradeView

SIGMA = MockAgent.BELIEF_SIGMA
BAND = MockAgent.TRADE_BAND


def _probe_belief(p_i: float, seed: int) -> float:
    """Replay MockAgent's first RNG draw (the belief noise) independently,
    so tests can pick prices relative to the exact value MockAgent will
    produce for a given (oracle posterior, seed) without hand-decoding
    numpy's PRNG stream."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA)
    return float(np.clip(p_i + noise, 0.01, 0.99))


def _belief_view(**overrides) -> BeliefView:
    fields = dict(question_id="q1", signal_values={0: 1, 2: 0})
    fields.update(overrides)
    return BeliefView(**fields)


def _trade_view(price: float, **overrides) -> TradeView:
    fields = dict(
        question_id="q1",
        signal_values={0: 1, 2: 0},
        price=price,
        round=1,
        n_rounds=3,
        bankroll=100.0,
        max_affordable_yes=50.0,
        max_affordable_no=40.0,
    )
    fields.update(overrides)
    return TradeView(**fields)


# --------------------------------------------------------------------------
# AgentSpec
# --------------------------------------------------------------------------


def test_agent_spec_defaults():
    spec = AgentSpec(agent_id="a0", shard_indices=[0, 2])
    assert spec.bankroll == 100.0
    assert spec.kind == "mock"
    assert spec.shard_indices == [0, 2]


def test_agent_spec_kind_literal_enforced():
    with pytest.raises(ValidationError):
        AgentSpec(agent_id="a0", shard_indices=[0], kind="llm")


def test_agent_spec_bankroll_negative_rejected():
    with pytest.raises(ValidationError):
        AgentSpec(agent_id="a0", shard_indices=[0], bankroll=-1.0)


def test_agent_spec_custom_bankroll_accepted():
    spec = AgentSpec(agent_id="a0", shard_indices=[], bankroll=250.0)
    assert spec.bankroll == 250.0


# --------------------------------------------------------------------------
# BeliefView / TradeView (additive schemas.py models)
# --------------------------------------------------------------------------


def test_belief_view_defaults_persona_empty():
    v = BeliefView(question_id="q1", signal_values={0: 1, 2: 0})
    assert v.persona == ""
    assert v.signal_values == {0: 1, 2: 0}


def test_belief_view_persona_can_be_set():
    v = BeliefView(question_id="q1", signal_values={}, persona="skeptic")
    assert v.persona == "skeptic"


def test_trade_view_is_a_belief_view_plus_market_fields():
    v = _trade_view(price=0.5)
    assert isinstance(v, BeliefView)
    assert v.question_id == "q1"
    assert v.signal_values == {0: 1, 2: 0}
    assert v.price == 0.5
    assert v.round == 1
    assert v.n_rounds == 3
    assert v.bankroll == 100.0
    assert v.max_affordable_yes == 50.0
    assert v.max_affordable_no == 40.0


def test_trade_view_price_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        _trade_view(price=1.5)
    with pytest.raises(ValidationError):
        _trade_view(price=-0.1)


def test_trade_view_round_must_be_at_least_one():
    with pytest.raises(ValidationError):
        _trade_view(price=0.5, round=0)


def test_trade_view_max_affordable_negative_rejected():
    with pytest.raises(ValidationError):
        _trade_view(price=0.5, max_affordable_yes=-1.0)


# --------------------------------------------------------------------------
# Agent protocol conformance
# --------------------------------------------------------------------------


def test_mock_agent_satisfies_agent_protocol():
    agent = MockAgent(0.7, np.random.default_rng(0))
    assert isinstance(agent, Agent)


def test_mock_agent_elicit_belief_returns_belief_model():
    agent = MockAgent(0.7, np.random.default_rng(0))
    belief = agent.elicit_belief(_belief_view())
    assert isinstance(belief, Belief)


def test_mock_agent_decide_trade_returns_trade_model():
    agent = MockAgent(0.7, np.random.default_rng(0))
    trade = agent.decide_trade(_trade_view(price=0.5))
    assert isinstance(trade, Trade)


# --------------------------------------------------------------------------
# Determinism (hard requirement): same seed => same outputs, every call.
# --------------------------------------------------------------------------


def test_elicit_belief_deterministic_given_seed():
    a1 = MockAgent(0.7, np.random.default_rng(42))
    a2 = MockAgent(0.7, np.random.default_rng(42))
    for _ in range(5):
        b1 = a1.elicit_belief(_belief_view())
        b2 = a2.elicit_belief(_belief_view())
        assert b1 == b2


def test_decide_trade_deterministic_given_seed():
    a1 = MockAgent(0.7, np.random.default_rng(123))
    a2 = MockAgent(0.7, np.random.default_rng(123))
    for _ in range(5):
        t1 = a1.decide_trade(_trade_view(price=0.5))
        t2 = a2.decide_trade(_trade_view(price=0.5))
        assert t1 == t2


def test_full_sequence_deterministic_given_seed():
    # Mixed sequence of belief + trade calls: full trajectories must match.
    def run(seed):
        agent = MockAgent(0.55, np.random.default_rng(seed))
        out = [agent.elicit_belief(_belief_view())]
        for r in range(1, 4):
            out.append(agent.decide_trade(_trade_view(price=0.5, round=r)))
        return out

    assert run(99) == run(99)


def test_different_seeds_produce_different_beliefs():
    a1 = MockAgent(0.7, np.random.default_rng(1))
    a2 = MockAgent(0.7, np.random.default_rng(2))
    b1 = a1.elicit_belief(_belief_view())
    b2 = a2.elicit_belief(_belief_view())
    assert b1.prob != b2.prob


def test_rng_state_advances_across_calls():
    # Repeated calls on the same agent must not return an identical cached
    # value every time (i.e. the RNG is actually being consumed).
    agent = MockAgent(0.5, np.random.default_rng(0))
    draws = {agent.elicit_belief(_belief_view()).prob for _ in range(10)}
    assert len(draws) > 1


# --------------------------------------------------------------------------
# Belief bounds: clip(p_i + Normal(0, 0.08), 0.01, 0.99)
# --------------------------------------------------------------------------


def test_belief_clipped_within_bounds_near_one():
    agent = MockAgent(0.999, np.random.default_rng(0))
    for _ in range(500):
        belief = agent.elicit_belief(_belief_view())
        assert 0.01 <= belief.prob <= 0.99


def test_belief_clipped_within_bounds_near_zero():
    agent = MockAgent(0.001, np.random.default_rng(1))
    for _ in range(500):
        belief = agent.elicit_belief(_belief_view())
        assert 0.01 <= belief.prob <= 0.99


def test_belief_matches_probe_exactly_for_fixed_seed():
    p_i, seed = 0.62, 7
    expected = _probe_belief(p_i, seed)
    agent = MockAgent(p_i, np.random.default_rng(seed))
    belief = agent.elicit_belief(_belief_view())
    assert belief.prob == pytest.approx(expected, abs=1e-15)


def test_belief_rationale_is_short_fixed_string():
    agent = MockAgent(0.5, np.random.default_rng(0))
    b1 = agent.elicit_belief(_belief_view())
    b2 = agent.elicit_belief(_belief_view())
    assert b1.rationale == b2.rationale
    assert len(b1.rationale) <= 400
    assert isinstance(b1.rationale, str) and b1.rationale != ""


def test_belief_centered_near_oracle_posterior_statistically():
    # Sanity check on the noise model (not exact-value): with sigma=0.08 and
    # p_i=0.6 (far from the clip boundaries), the empirical mean over many
    # draws should land close to p_i.
    agent = MockAgent(0.6, np.random.default_rng(2))
    samples = [agent.elicit_belief(_belief_view()).prob for _ in range(3000)]
    assert np.mean(samples) == pytest.approx(0.6, abs=0.02)
    assert np.std(samples) == pytest.approx(SIGMA, abs=0.02)


# --------------------------------------------------------------------------
# Trade rule: exact thresholds on (belief, price), pinned via _probe_belief.
# --------------------------------------------------------------------------


def test_trade_buy_yes_when_belief_exceeds_price_plus_band():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    price = expected_belief - BAND - 0.01  # belief > price + band
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(_trade_view(price=price))
    assert trade.action == "buy_yes"
    assert trade.belief == pytest.approx(expected_belief, abs=1e-15)
    assert trade.shares > 0.0


def test_trade_buy_no_when_belief_below_price_minus_band():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    price = expected_belief + BAND + 0.01  # belief < price - band
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(_trade_view(price=price))
    assert trade.action == "buy_no"
    assert trade.belief == pytest.approx(expected_belief, abs=1e-15)
    assert trade.shares > 0.0


def test_trade_hold_when_price_equals_belief_exactly():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(_trade_view(price=expected_belief))
    assert trade.action == "hold"
    assert trade.shares == 0.0


def test_trade_hold_within_band_just_inside():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    # price such that belief is exactly at the edge (price + band): band is
    # a strict inequality (`belief > price + band`), so equality => hold.
    price = expected_belief - BAND
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(_trade_view(price=price))
    assert trade.action == "hold"
    assert trade.shares == 0.0


def test_trade_buy_yes_just_outside_band():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    price = expected_belief - BAND - 1e-6
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(_trade_view(price=price))
    assert trade.action == "buy_yes"


def test_trade_belief_field_matches_computed_belief_not_oracle():
    # trade.belief must be the agent's *noisy* belief, not the raw oracle
    # posterior it was constructed with.
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(_trade_view(price=0.5))
    assert trade.belief == pytest.approx(expected_belief, abs=1e-15)


# --------------------------------------------------------------------------
# Requested shares = frac * max_affordable_<chosen side>, frac ~ U(0.2, 0.6)
# --------------------------------------------------------------------------


def test_buy_yes_shares_within_frac_band_of_max_affordable_yes():
    agent = MockAgent(0.95, np.random.default_rng(0))
    for _ in range(200):
        trade = agent.decide_trade(_trade_view(price=0.5, max_affordable_yes=80.0, max_affordable_no=1.0))
        assert trade.action == "buy_yes"
        assert 0.2 * 80.0 - 1e-9 <= trade.shares <= 0.6 * 80.0 + 1e-9


def test_buy_no_shares_within_frac_band_of_max_affordable_no():
    agent = MockAgent(0.05, np.random.default_rng(0))
    for _ in range(200):
        trade = agent.decide_trade(_trade_view(price=0.5, max_affordable_yes=1.0, max_affordable_no=80.0))
        assert trade.action == "buy_no"
        assert 0.2 * 80.0 - 1e-9 <= trade.shares <= 0.6 * 80.0 + 1e-9


def test_buy_yes_shares_use_yes_side_not_no_side():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    price = expected_belief - BAND - 0.01
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(
        _trade_view(price=price, max_affordable_yes=10.0, max_affordable_no=10_000.0)
    )
    assert trade.action == "buy_yes"
    assert trade.shares <= 0.6 * 10.0 + 1e-9  # not derived from the huge NO-side number


def test_hold_shares_always_zero_even_with_affordability():
    p_i, seed = 0.6, 11
    expected_belief = _probe_belief(p_i, seed)
    agent = MockAgent(p_i, np.random.default_rng(seed))
    trade = agent.decide_trade(
        _trade_view(price=expected_belief, max_affordable_yes=999.0, max_affordable_no=999.0)
    )
    assert trade.action == "hold"
    assert trade.shares == 0.0


# --------------------------------------------------------------------------
# Construction guards / zero-network sanity
# --------------------------------------------------------------------------


def test_mock_agent_rejects_out_of_bounds_oracle_posterior():
    with pytest.raises(ValueError):
        MockAgent(1.5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        MockAgent(-0.1, np.random.default_rng(0))


def test_mock_agent_accepts_boundary_oracle_posteriors():
    MockAgent(0.0, np.random.default_rng(0))
    MockAgent(1.0, np.random.default_rng(0))


def test_agents_module_has_no_network_imports():
    import pnyx.agents as agents_mod

    src = agents_mod.__file__
    with open(src) as f:
        text = f.read()
    for forbidden in ("httpx", "openai", "requests", "urllib", "socket", "langchain"):
        assert forbidden not in text, f"unexpected network/provider import: {forbidden}"

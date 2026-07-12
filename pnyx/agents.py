"""Pnyx agents — the Agent protocol and the deterministic zero-intelligence
MockAgent trader.

P1 scope only: no LLM providers, no network calls, no API plumbing. Real
LLM-backed agents arrive in P3 (CLAUDE.md §8 "P3 — Pilot"); Global
Constraints requires P1 to make ZERO API calls. MockAgent is a pure
function of (oracle posterior, RNG state): the same seed always reproduces
the same beliefs and trades.

This module never imports pnyx.market: affordability
(``max_affordable_yes`` / ``max_affordable_no``) is computed by the runner
(Task 5) via market.py and handed to the agent on ``TradeView``; the
requested share count is clamped server-side regardless of what the agent
asks for (Global Constraints — "never trust the agent's number").
"""

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from pnyx.schemas import Belief, BeliefView, Trade, TradeView

__all__ = ["AgentSpec", "Agent", "MockAgent"]


class AgentSpec(BaseModel):
    """Minimal P1 agent configuration: identity, shard assignment, starting
    bankroll, and kind.

    ``kind`` is a closed Literal with a single P1 value; real LLM agent
    kinds (with model/persona/provider fields) arrive in P3, so adding one
    later is a deliberate, visible schema change rather than a silent
    string.
    """

    agent_id: str
    shard_indices: list[int]
    bankroll: float = Field(default=100.0, ge=0.0)
    kind: Literal["mock"] = "mock"


@runtime_checkable
class Agent(Protocol):
    """Structural protocol every agent kind (MockAgent today, LLM-backed
    agents in P3) must satisfy.

    Two turns per question per the two-pass elicitation protocol: an
    independent belief (Pass 1, no market context) and, once per market
    round, a trade decision (Pass 2).
    """

    def elicit_belief(self, view: BeliefView) -> Belief:
        """Pass-1 independent belief: the agent's shard + persona only, no
        prices, no other agents, no market context."""
        ...

    def decide_trade(self, view: TradeView) -> Trade:
        """Pass-2 market-round trade decision given the current price and
        the agent's own affordability at this instant."""
        ...


class MockAgent:
    """Zero-intelligence trader: deterministic given its oracle posterior
    and a seeded ``numpy.random.Generator``. Makes zero network/API calls.

    Constructed with the exact oracle posterior for its own shard subset
    (``p_i``, a float the runner reads from
    ``QuestionRecord.posterior_table`` for this agent's ``shard_indices``)
    and an ``numpy.random.Generator`` the runner seeds per
    ``(seed, agent_id, question_id)`` — that seeding is what makes a full
    run reproducible (Global Constraints, "Determinism").

    Belief rule
    -----------
    ``clip(p_i + Normal(0, sigma=BELIEF_SIGMA), BELIEF_LO, BELIEF_HI)`` —
    the oracle posterior perturbed by mean-zero Gaussian noise, clipped away
    from the extremes (a ZI trader should never claim total certainty).

    Trade rule (per round)
    -----------------------
    Draw a fresh noisy belief (same rule as above). Compare to the current
    price with a dead-band of ``TRADE_BAND``:

    * ``belief > price + TRADE_BAND``  -> ``buy_yes``
    * ``belief < price - TRADE_BAND``  -> ``buy_no``
    * otherwise                        -> ``hold`` (0 shares requested)

    On a buy, the requested share count is
    ``frac * max_affordable_<chosen side>`` with
    ``frac ~ Uniform(FRAC_LO, FRAC_HI)`` drawn from the same RNG. The
    runner still clamps this server-side before executing.
    """

    BELIEF_SIGMA = 0.08
    BELIEF_LO = 0.01
    BELIEF_HI = 0.99
    TRADE_BAND = 0.02
    FRAC_LO = 0.2
    FRAC_HI = 0.6
    BELIEF_RATIONALE = "mock agent: oracle posterior plus Gaussian noise"
    TRADE_RATIONALE = "mock agent: threshold rule on belief vs. price"

    def __init__(self, oracle_posterior: float, rng: np.random.Generator) -> None:
        if not (0.0 <= oracle_posterior <= 1.0):
            raise ValueError(
                f"oracle_posterior must be in [0, 1], got {oracle_posterior}"
            )
        self.oracle_posterior = float(oracle_posterior)
        self.rng = rng

    def _noisy_belief(self) -> float:
        """One fresh noisy-belief draw, consuming exactly one RNG normal
        sample. Shared by elicit_belief and decide_trade so both quote the
        same rule."""
        noise = self.rng.normal(0.0, self.BELIEF_SIGMA)
        return float(
            np.clip(self.oracle_posterior + noise, self.BELIEF_LO, self.BELIEF_HI)
        )

    def elicit_belief(self, view: BeliefView) -> Belief:
        return Belief(prob=self._noisy_belief(), rationale=self.BELIEF_RATIONALE)

    def decide_trade(self, view: TradeView) -> Trade:
        belief = self._noisy_belief()

        action: Literal["buy_yes", "buy_no", "hold"]
        if belief > view.price + self.TRADE_BAND:
            action = "buy_yes"
            max_affordable = view.max_affordable_yes
        elif belief < view.price - self.TRADE_BAND:
            action = "buy_no"
            max_affordable = view.max_affordable_no
        else:
            action = "hold"
            max_affordable = 0.0

        if action == "hold":
            shares = 0.0
        else:
            frac = self.rng.uniform(self.FRAC_LO, self.FRAC_HI)
            shares = float(frac * max_affordable)

        return Trade(
            belief=belief, action=action, shares=shares, rationale=self.TRADE_RATIONALE
        )
